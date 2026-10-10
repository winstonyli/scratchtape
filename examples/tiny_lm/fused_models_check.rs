// K tiny_lm models trained at once on the eGPU, in the same launches
// (horizontal fusion, HFTA: Wang et al. 2021, arXiv 2102.02344). The step
// is dispatch-bound (~145 launches whatever the batch), so K models cost
// far less than K runs. Model m is training_recipe_check's run at seed
// seed + m: the same init, the same batch order and the same recipe
// (batch, lr, warmup, momentum, weight decay on every parameter, dropout).
// Dropout masks differ from a solo run's (the hash sees a model's
// elements at shifted indices), so with dropout only the statistics match.
//
// `alpha` > 0 makes the models learn from each other as well: deep mutual
// learning (Zhang et al. 2018, arXiv 1706.00384), online codistillation
// without staleness (Anil et al. 2018, arXiv 1804.03235). Each model's
// loss gains alpha · KL(peers' mean prediction ‖ its own) per position,
// the peers held constant (DeviceTape::with_distill).
//
// `sync_every_steps` > 0 averages the K models' parameters every that many
// steps and at the end (local SGD; the DiLoCo family, arXiv 2311.08105,
// minus its outer optimizer). It needs `shared_init` = 1: all K models start
// from seed's init and differ only in their batches. Momentum buffers stay
// per model. `lr_decay_frac` > 0 decays the lr linearly to 0 over that fraction
// of the steps at the end; the model size args default to the 128/8/256/4 model.
// `groups` > 1 splits the K models into that many groups, each with its own
// init (seed + 1000 g) that averages only within itself, so the groups can be
// ensembled (needs shared_init = 1 and sync_every > 0).
// Resumable (LONG_RUNS.md): every `checkpoint_secs` (default 600) the step,
// each model's RNG state, the parameters and the optimizer state go to
// runs/<name>.resume; relaunching the same command continues from it, and a
// killed-and-resumed run reproduces an uninterrupted one. The file is
// removed once the final checkpoints are written. (The final "training ms/step"
// then covers only the last launch.)
// `outer_lr` > 0 replaces the plain mean with DiLoCo's outer step
// (Nesterov momentum `outer_mu` on the pseudo-gradient, DeviceParams::outer_step).
//
//   fused_models_check <name> <k> <batch> <lr> [windows] [seed] [weight_decay] [dropout] [warmup_windows] [momentum] [alpha] [alpha_ramp_windows] [sync_every_steps] [shared_init] [outer_lr] [outer_mu] [corpus] [lr_decay_frac] [d_model] [heads] [d_ff] [blocks] [groups] [checkpoint_secs] [profile]
//   profile=gpu|host charges 50 steps (after 20 of warm-up) to launch sites and prints the top 40 (GPU time, or host queueing time), then exits.
//   (any of them also as key=value, e.g. `... 4 32 0.05 windows=4096000 dropout=0.1 d_model=384`)
//
// Every 8000 windows (with averaging, only at sync points) it prints each
// model's held-out CE over every non-overlapping 64-byte window of the
// corpus's held-out split (common::split_corpus; 371 windows for
// aesops_fables), their mean, and the ensemble's: the CE of the models'
// averaged probabilities, from per-row losses read back at evals only. At
// the end it prints the train-probe CE (the same number of windows from
// the start of the training text), then each model is re-scored alone
// (K = 1 on the device) and must match its fused score to 1e-4; parameters
// go to runs/<name>_m<m>.ckpt (flatten_all order).
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, host_profile_cut, host_profile_start, host_profile_take, pack, profile_start, profile_take, read, upload_f32};
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use std::io::Write;
use std::time::{Duration, Instant};

#[path = "../common/mod.rs"]
mod common;
use common::{sample_window, split_corpus};

const SEQ_LEN: usize = 64;
const VOCAB: usize = 256;
/// Held-out windows per model per evaluation launch.
const EVAL_CHUNK: usize = 32;

/// training_recipe_check's init, drawn from `rng` in the same order.
fn init(rng: &mut Rng, cfg: &Config) -> Vec<f32> {
    let token_emb = Embedding::new(rng, cfg.vocab, cfg.d);
    let pos_emb = Embedding::new(rng, cfg.t, cfg.d);
    let blocks: Vec<_> = (0..cfg.n_blocks).map(|_| TransformerBlock::new(rng, cfg.d, cfg.heads, cfg.d_ff)).collect();
    let final_ln = LayerNorm::new(cfg.d);
    let output_proj = Linear::new(rng, cfg.d, cfg.vocab);
    pack(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj)
}

/// Each of `dev`'s models' mean CE over the corpus's non-overlapping
/// windows, and the ensemble's (the mean over rows of -log of the models'
/// mean probability of the target).
fn device_ce(dev: &DeviceParams, cfg: &Config, corpus: &[usize]) -> (Vec<f64>, f64) {
    let k = dev.models.k;
    let starts: Vec<usize> = (0..corpus.len() - SEQ_LEN).step_by(SEQ_LEN).collect();
    let (mut each, mut ens) = (vec![0.0f64; k], 0.0f64);
    for chunk in starts.chunks(EVAL_CHUNK) {
        let (mut ids, mut targets) = (vec![], vec![]);
        for &s in chunk {
            ids.extend(&corpus[s..s + SEQ_LEN]);
            targets.extend(&corpus[s + 1..s + SEQ_LEN + 1]);
        }
        let rows = ids.len();
        let (ids, targets) = (ids.repeat(k), targets.repeat(k));
        let mut dt = DeviceTape::new(dev);
        let (_, _, loss) = model_forward(&mut dt, cfg, &ids, &targets, k * chunk.len());
        let rl = read(dt.row_losses(loss));
        for r in 0..rows {
            let l: Vec<f64> = (0..k).map(|m| rl[m * rows + r] as f64).collect();
            for m in 0..k {
                each[m] += l[m];
            }
            // -log mean exp(-l), shifted by the smallest loss
            let lo = l.iter().cloned().fold(f64::INFINITY, f64::min);
            ens += lo - (l.iter().map(|x| (lo - x).exp()).sum::<f64>() / k as f64).ln();
        }
    }
    let n = (starts.len() * SEQ_LEN) as f64;
    (each.iter().map(|s| s / n).collect(), ens / n)
}

/// The command-line parameters, in positional order, with defaults ("" = required).
const PROFILE_WARM: usize = 20;
const PROFILE_STEPS: usize = 50;

const PARAMS: [(&str, &str); 25] = [
    ("name", ""),
    ("k", ""),
    ("batch", ""),
    ("lr", ""),
    ("windows", "64000"),
    ("seed", "1"),
    ("weight_decay", "0"),
    ("dropout", "0"),
    ("warmup_windows", "0"),
    ("momentum", "0"),
    ("alpha", "0"),
    ("alpha_ramp_windows", "0"),
    ("sync_every_steps", "0"),
    ("shared_init", "0"),
    ("outer_lr", "0"),
    ("outer_mu", "0.9"),
    ("corpus", "aesops_fables"),
    ("lr_decay_frac", "0"),
    ("d_model", "128"),
    ("heads", "8"),
    ("d_ff", "256"),
    ("blocks", "4"),
    ("groups", "1"),
    ("checkpoint_secs", "600"),
    ("profile", "0"),
];

/// Arguments after the program name: bare values fill PARAMS in order, `key=value` sets one by name
/// (the two mix freely). Each parameter is set at most once; unknown keys and missing required ones panic.
struct Params(std::collections::HashMap<&'static str, String>);

impl Params {
    fn parse(args: &[String]) -> Params {
        let usage = || {
            format!(
                "usage: fused_models_check {}  (bare values in this order, or key=value)",
                PARAMS.iter().map(|(n, d)| if d.is_empty() { format!("<{n}>") } else { format!("[{n}]") }).collect::<Vec<_>>().join(" ")
            )
        };
        let mut set: std::collections::HashMap<&'static str, String> = Default::default();
        let mut next = 0;
        for arg in args {
            let (name, value) = match arg.split_once('=') {
                Some((key, value)) => (
                    PARAMS
                        .iter()
                        .find(|(n, _)| *n == key)
                        .unwrap_or_else(|| {
                            panic!(
                                "unknown parameter {key}
{}",
                                usage()
                            )
                        })
                        .0,
                    value,
                ),
                None => {
                    let slot = PARAMS.get(next).unwrap_or_else(|| {
                        panic!(
                            "too many arguments
{}",
                            usage()
                        )
                    });
                    next += 1;
                    (slot.0, arg.as_str())
                }
            };
            assert!(set.insert(name, value.to_string()).is_none(), "{name} given twice");
        }
        for (name, default) in PARAMS {
            if !set.contains_key(name) {
                assert!(
                    !default.is_empty(),
                    "missing {name}
{}",
                    usage()
                );
                set.insert(name, default.to_string());
            }
        }
        Params(set)
    }

    fn get<T: std::str::FromStr>(&self, name: &str) -> T {
        self.0[name].parse().unwrap_or_else(|_| panic!("bad value {:?} for {name}", self.0[name]))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let p = Params::parse(&args);
    let name: String = p.get("name");
    let (k, batch, lr): (usize, usize, f32) = (p.get("k"), p.get("batch"), p.get("lr"));
    let windows: usize = p.get("windows");
    let seed: u64 = p.get("seed");
    assert!(seed != 0, "seed 0 is xorshift's fixed point");
    let weight_decay: f32 = p.get("weight_decay");
    let dropout: f32 = p.get("dropout");
    let warmup_windows: usize = p.get("warmup_windows");
    let momentum: f32 = p.get("momentum");
    let alpha: f32 = p.get("alpha");
    // alpha rises linearly from 0 over this many windows (0: constant)
    let ramp = (p.get::<usize>("alpha_ramp_windows") / batch).max(1);
    let sync_every: usize = p.get("sync_every_steps");
    let shared_init = p.get::<String>("shared_init") == "1";
    assert!(sync_every == 0 || shared_init, "averaging models needs shared_init = 1");
    let outer_lr: f32 = p.get("outer_lr");
    let outer_mu: f32 = p.get("outer_mu");
    let corpus_name: String = p.get("corpus");
    let lr_decay_frac: f32 = p.get("lr_decay_frac");
    let cfg = Config { vocab: VOCAB, d: p.get("d_model"), heads: p.get("heads"), d_ff: p.get("d_ff"), t: SEQ_LEN, n_blocks: p.get("blocks"), softmax1: false };
    let groups: usize = p.get("groups");
    assert!(k % groups == 0 && (groups == 1 || (shared_init && sync_every > 0 && outer_lr == 0.0)), "groups needs K % groups == 0, shared_init, sync_every and no outer step");
    let warmup = (warmup_windows / batch).max(1);
    let steps = windows / batch;
    let eval_every = (8000 / batch).max(1);

    let (train, held_out) = split_corpus(&corpus_name);
    let (train, held_out) = (&train[..], &held_out[..]);

    // Model m's generator: its init, then its batches (training_recipe_check's order).
    let mut rngs: Vec<Rng> = (0..k as u64).map(|m| Rng::new(seed + m)).collect();
    let flat: Vec<f32> = if shared_init {
        (0..groups as u64).flat_map(|g| init(&mut Rng::new(seed + 1000 * g), &cfg).repeat(k / groups)).collect()
    } else {
        rngs.iter_mut().flat_map(|r| init(r, &cfg)).collect()
    };
    let config = format!(
        "k={k} batch={batch} lr={lr} windows={windows} seeds={seed}..{} wd={weight_decay} dropout={dropout} warmup={warmup_windows} momentum={momentum} alpha={alpha} ramp={ramp} sync_every={sync_every} shared_init={shared_init} outer_lr={outer_lr} outer_mu={outer_mu} corpus={corpus_name} lr_decay_frac={lr_decay_frac} model d={} heads={} d_ff={} blocks={} groups={groups}",
        seed + k as u64 - 1,
        cfg.d,
        cfg.heads,
        cfg.d_ff,
        cfg.n_blocks
    );
    println!("run {name}: pid {} {config} steps={steps}", std::process::id());
    let _lease = gpu_lease::hold(Kind::Shared, &format!("scratchtape fused_models_check {name}"), Duration::from_secs(4 * 3600));
    // Resume file: header "<config> | <step> <rng state per model>", then
    // the parameters, then the velocity (momentum), then the outer anchor
    // and velocity (outer step), one line each.
    let checkpoint_secs: u64 = p.get("checkpoint_secs");
    let profile: String = p.get("profile");
    let resume_path = format!("runs/{name}.resume");
    let (mut flat, mut first) = (flat, 0);
    let mut saved: std::vec::IntoIter<Vec<f32>> = vec![].into_iter();
    if let Ok(text) = std::fs::read_to_string(&resume_path) {
        let mut lines = text.trim_end().lines();
        let (header, at) = lines.next().unwrap().split_once(" | ").unwrap();
        assert_eq!(header, config, "{resume_path} is from a different config");
        let mut at = at.split_whitespace();
        first = at.next().unwrap().parse().unwrap();
        for rng in rngs.iter_mut() {
            *rng = Rng::new(at.next().unwrap().parse().unwrap());
        }
        let mut vecs: Vec<Vec<f32>> = lines.map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect()).collect();
        flat = vecs.remove(0);
        saved = vecs.into_iter();
        println!("resumed from {resume_path} at step {first}");
    }
    let dev = DeviceParams::upload_models(&flat, k);
    let ones = (weight_decay > 0.0).then(|| upload_f32(&vec![1.0; flat.len()]));
    let stride = flat.len() / k;
    let velocity = (momentum > 0.0).then(|| upload_f32(&saved.next().unwrap_or_else(|| vec![0.0; flat.len()])));
    let outer = (outer_lr > 0.0).then(|| match (saved.next(), saved.next()) {
        (Some(anchor), Some(v)) => (upload_f32(&anchor), upload_f32(&v)),
        _ => (upload_f32(&flat[..stride]), upload_f32(&vec![0.0; stride])),
    });
    assert!(saved.next().is_none(), "{resume_path} has more state than this config uses");
    println!("columns: step | windows seen per model | held-out CE per model | mean | ensemble | elapsed | training ms/step since last row");
    std::fs::create_dir_all("runs").unwrap();
    let start = Instant::now();
    let mut evaluating = Duration::ZERO;
    let mut segment = (Instant::now(), first);
    let mut last = (vec![], 0.0);
    let mut last_save = Instant::now();
    for step in first..=steps {
        // `profile=gpu|host`: after PROFILE_WARM steps (compilation), charge PROFILE_STEPS steps to launch sites, print, stop.
        if step == first + PROFILE_WARM + PROFILE_STEPS && profile != "0" {
            let (sites, what) = if profile == "gpu" { (profile_take(), "GPU") } else { (host_profile_take(), "host") };
            let total: f64 = sites.iter().map(|s| s.2).sum();
            for (site, n, secs) in sites.iter().take(40) {
                println!("{:>7.3} ms/step {:>5.1}%  {:>3} launches/step  {site}", secs * 1e3 / PROFILE_STEPS as f64, 100.0 * secs / total, n / PROFILE_STEPS);
            }
            println!("{what} time, all launches: {:.2} ms/step over {} sites", total * 1e3 / PROFILE_STEPS as f64, sites.len());
            return;
        }
        if step == first + PROFILE_WARM {
            dev.read(&dev.params); // drain the queue so compilation is not charged
            match profile.as_str() {
                "gpu" => profile_start(),
                "host" => host_profile_start(),
                _ => {}
            }
        }
        // Save before this step's work; `step` is the next one to run.
        if step > first && step < steps && last_save.elapsed().as_secs() >= checkpoint_secs {
            let line = |v: Vec<f32>| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            let states: Vec<String> = rngs.iter().map(|r| r.state().to_string()).collect();
            let mut text = format!("{config} | {step} {}\n{}", states.join(" "), line(dev.read(&dev.params)));
            if let Some(v) = &velocity {
                text += &format!("\n{}", line(dev.read(v)));
            }
            if let Some((anchor, v)) = &outer {
                text += &format!("\n{}\n{}", line(read(anchor)[..stride].to_vec()), line(read(v)[..stride].to_vec()));
            }
            let tmp = format!("{resume_path}.tmp");
            std::fs::write(&tmp, text).unwrap();
            std::fs::rename(&tmp, &resume_path).unwrap();
            last_save = Instant::now();
        }
        // With averaging, only evaluate right after a sync, where the models are the averaged one.
        if (step % eval_every == 0 && (sync_every == 0 || step % sync_every == 0)) || step == steps {
            gpu_lease::pause_while_exclusive();
            let t = Instant::now();
            let ms_per_step = match step - segment.1 {
                0 => "-".to_string(),
                n => format!("{:.2}", segment.0.elapsed().as_secs_f64() * 1e3 / n as f64),
            };
            let (each, ens) = device_ce(&dev, &cfg, held_out);
            let mean = each.iter().sum::<f64>() / k as f64;
            let each_s: Vec<String> = each.iter().map(|c| format!("{c:.4}")).collect();
            println!("{step:>6} | {:>7} | {} | {mean:.4} | {ens:.4} | {:.0}s | {ms_per_step}", step * batch, each_s.join(" "), start.elapsed().as_secs_f32());
            if each.iter().any(|c| c.is_nan()) {
                println!("diverged to NaN before step {step}");
                return;
            }
            std::io::stdout().flush().unwrap();
            last = (each, ens);
            evaluating += t.elapsed();
            segment = (Instant::now(), step);
        }
        if step == steps {
            break;
        }
        let mut lr = lr * ((step + 1) as f32 / warmup as f32).min(1.0);
        if lr_decay_frac > 0.0 {
            lr *= ((steps - step) as f32 / (lr_decay_frac * steps as f32)).min(1.0);
        }
        let (mut input, mut target) = (Vec::with_capacity(k * batch * SEQ_LEN), Vec::with_capacity(k * batch * SEQ_LEN));
        for rng in &mut rngs {
            for _ in 0..batch {
                let (i, t) = sample_window(rng, train, SEQ_LEN);
                input.extend(i);
                target.extend(t);
            }
        }
        dev.zero_grads();
        let dt = if dropout > 0.0 { DeviceTape::with_dropout(&dev, dropout, (seed as u32).wrapping_mul(0x85eb_ca6b) ^ step as u32) } else { DeviceTape::new(&dev) };
        let mut dt = dt.with_distill(alpha * ((step + 1) as f32 / ramp as f32).min(1.0));
        let (_, _, loss) = model_forward(&mut dt, &cfg, &input, &target, k * batch);
        dt.backward(loss);
        match &velocity {
            Some(v) => dev.momentum(lr, momentum, v, ones.as_ref().map(|m| (lr * weight_decay, m))),
            None => {
                if let Some(ones) = &ones {
                    dev.decay(lr * weight_decay, ones);
                }
                dev.sgd(lr)
            }
        }
        if sync_every > 0 && ((step + 1) % sync_every == 0 || step + 1 == steps) {
            match &outer {
                Some((anchor, v)) => dev.outer_step(anchor, v, outer_lr, outer_mu),
                None => dev.average_groups(k / groups),
            }
        }
        if profile == "host" {
            host_profile_cut();
        }
    }
    let _ = std::fs::remove_file(&resume_path);
    let training = start.elapsed() - evaluating;

    // Train-probe: the first held-out-sized stretch of the training text, to
    // see the train/held-out gap per arm.
    let (probe_each, probe_ens) = device_ce(&dev, &cfg, &train[..held_out.len()]);
    let probe_s: Vec<String> = probe_each.iter().map(|c| format!("{c:.4}")).collect();
    println!("train-probe CE per model {} | mean {:.4} | ensemble {probe_ens:.4}", probe_s.join(" "), probe_each.iter().sum::<f64>() / k as f64);

    // Each model alone, from its fused parameters: the fused evaluation's
    // check.
    let params = dev.read(&dev.params);
    for (m, p) in params.chunks(cfg.len()).enumerate() {
        let solo = device_ce(&DeviceParams::upload(p), &cfg, held_out).0[0];
        assert!((solo - last.0[m]).abs() < 1e-4, "model {m}: alone {solo:.5} vs fused {:.5}", last.0[m]);
        let text = p.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
        std::fs::write(format!("runs/{name}_m{m}.ckpt"), text).unwrap();
    }
    println!(
        "each model re-scored alone matches; saved runs/{name}_m*.ckpt ({:.0}s training = {:.2} ms/step, {:.2} ms/step per model)",
        training.as_secs_f32(),
        training.as_secs_f64() * 1e3 / steps.max(1) as f64,
        training.as_secs_f64() * 1e3 / (steps.max(1) * k) as f64
    );
}
