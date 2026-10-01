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
//   fused_models_check <name> <k> <batch> <lr> [windows] [seed] [weight_decay] [dropout] [warmup_windows] [momentum] [alpha] [alpha_ramp_windows] [sync_every_steps] [shared_init] [outer_lr] [outer_mu] [corpus] [lr_decay_frac] [d_model] [heads] [d_ff] [blocks] [groups] [checkpoint_secs]
//
// Every 8000 windows it prints each model's held-out CE (training_recipe_check's
// deterministic 371 windows), their mean, and the ensemble's: the CE of
// the models' averaged probabilities, from per-row losses read back at
// evals only. At the end each model is re-scored alone (K = 1 on the
// device) and must match its fused score to 1e-4; parameters go to
// runs/<name>_m<m>.ckpt (flatten_all order). No resume: a 1M-window run
// at K 4 takes minutes.
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, pack, read, upload_f32};
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 5, "usage: fused_models_check <name> <k> <batch> <lr> [windows] [seed] [weight_decay] [dropout] [warmup_windows] [momentum] [alpha] [alpha_ramp_windows] [sync_every_steps] [shared_init] [outer_lr] [outer_mu] [corpus] [lr_decay_frac] [d_model] [heads] [d_ff] [blocks] [groups] [checkpoint_secs]");
    let name = &args[1];
    let k: usize = args[2].parse().unwrap();
    let batch: usize = args[3].parse().unwrap();
    let lr: f32 = args[4].parse().unwrap();
    let arg = |i: usize, default: &str| args.get(i).cloned().unwrap_or(default.into());
    let windows: usize = arg(5, "64000").parse().unwrap();
    let seed: u64 = arg(6, "1").parse().unwrap();
    assert!(seed != 0, "seed 0 is xorshift's fixed point");
    let weight_decay: f32 = arg(7, "0").parse().unwrap();
    let dropout: f32 = arg(8, "0").parse().unwrap();
    let warmup_windows: usize = arg(9, "0").parse().unwrap();
    let momentum: f32 = arg(10, "0").parse().unwrap();
    let alpha: f32 = arg(11, "0").parse().unwrap();
    // alpha rises linearly from 0 over this many windows (0: constant)
    let ramp = (arg(12, "0").parse::<usize>().unwrap() / batch).max(1);
    let sync_every: usize = arg(13, "0").parse().unwrap();
    let shared_init = arg(14, "0") == "1";
    assert!(sync_every == 0 || shared_init, "averaging models needs shared_init = 1");
    let outer_lr: f32 = arg(15, "0").parse().unwrap();
    let outer_mu: f32 = arg(16, "0.9").parse().unwrap();
    let corpus_name = arg(17, "aesops_fables");
    let lr_decay_frac: f32 = arg(18, "0").parse().unwrap();
    let size = |i: usize, default: &str| -> usize { arg(i, default).parse().unwrap() };
    let cfg = Config { vocab: VOCAB, d: size(19, "128"), heads: size(20, "8"), d_ff: size(21, "256"), t: SEQ_LEN, n_blocks: size(22, "4"), softmax1: false };
    let groups = size(23, "1");
    assert!(k % groups == 0 && (groups == 1 || (shared_init && sync_every > 0 && outer_lr == 0.0)), "groups needs K % groups == 0, shared_init, sync_every and no outer step");
    let warmup = (warmup_windows / batch).max(1);
    let steps = windows / batch;
    let eval_every = (8000 / batch).max(1);

    let (train, held_out) = split_corpus(&corpus_name);
    let (train, held_out) = (&train[..], &held_out[..]);

    // Model m's generator: its init, then its batches (training_recipe_check's order).
    let mut rngs: Vec<Rng> = (0..k as u64).map(|m| Rng::new(seed + m)).collect();
    let flat: Vec<f32> = if shared_init { (0..groups as u64).flat_map(|g| init(&mut Rng::new(seed + 1000 * g), &cfg).repeat(k / groups)).collect() } else { rngs.iter_mut().flat_map(|r| init(r, &cfg)).collect() };
    let config = format!("k={k} batch={batch} lr={lr} windows={windows} seeds={seed}..{} wd={weight_decay} dropout={dropout} warmup={warmup_windows} momentum={momentum} alpha={alpha} ramp={ramp} sync_every={sync_every} shared_init={shared_init} outer_lr={outer_lr} outer_mu={outer_mu} corpus={corpus_name} lr_decay_frac={lr_decay_frac} model d={} heads={} d_ff={} blocks={} groups={groups}", seed + k as u64 - 1, cfg.d, cfg.heads, cfg.d_ff, cfg.n_blocks);
    println!("run {name}: pid {} {config} steps={steps}", std::process::id());
    let _lease = gpu_lease::hold(Kind::Shared, &format!("scratchtape fused_models_check {name}"), Duration::from_secs(4 * 3600));
    // Resume file: header "<config> | <step> <rng state per model>", then
    // the parameters, then the velocity (momentum), then the outer anchor
    // and velocity (outer step), one line each.
    let checkpoint_secs: u64 = arg(24, "600").parse().unwrap();
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
        let dt = if dropout > 0.0 {
            DeviceTape::with_dropout(&dev, dropout, (seed as u32).wrapping_mul(0x85eb_ca6b) ^ step as u32)
        } else {
            DeviceTape::new(&dev)
        };
        let mut dt = dt.with_distill(alpha * ((step + 1) as f32 / ramp as f32).min(1.0));
        let (_, _, loss) = model_forward(&mut dt, &cfg, &input, &target, k * batch);
        dt.backward(loss);
        if let Some(ones) = &ones {
            dev.decay(lr * weight_decay, ones);
        }
        match &velocity {
            Some(v) => dev.momentum(lr, momentum, v),
            None => dev.sgd(lr),
        }
        if sync_every > 0 && ((step + 1) % sync_every == 0 || step + 1 == steps) {
            match &outer {
                Some((anchor, v)) => dev.outer_step(anchor, v, outer_lr, outer_mu),
                None => dev.average_groups(k / groups),
            }
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
