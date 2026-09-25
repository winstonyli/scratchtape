// K tiny_lm models trained at once on the eGPU, in the same launches
// (horizontal fusion, HFTA: Wang et al. 2021, arXiv 2102.02344). The step
// is dispatch-bound (~145 launches whatever the batch), so K models cost
// far less than K runs. Model m is training_recipe_check's run at seed
// seed + m: the same init, the same batch order and the same recipe
// (batch, lr, warmup, momentum, weight decay on every parameter, dropout).
// Dropout masks differ from a solo run's (the hash sees a model's
// elements at shifted indices), so with dropout only the statistics match.
//
//   fused_models_check <name> <k> <batch> <lr> [windows] [seed] [weight_decay] [dropout] [warmup_windows] [momentum]
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
use common::{encode_bytes, sample_window};

const D_MODEL: usize = 128;
const N_HEADS: usize = 8;
const D_FF: usize = 256;
const SEQ_LEN: usize = 64;
const N_BLOCKS: usize = 4;
const VOCAB: usize = 256;
/// Held-out windows per model per evaluation launch.
const EVAL_CHUNK: usize = 32;

/// training_recipe_check's init, drawn from `rng` in the same order.
fn init(rng: &mut Rng) -> Vec<f32> {
    let token_emb = Embedding::new(rng, VOCAB, D_MODEL);
    let pos_emb = Embedding::new(rng, SEQ_LEN, D_MODEL);
    let blocks: Vec<_> = (0..N_BLOCKS).map(|_| TransformerBlock::new(rng, D_MODEL, N_HEADS, D_FF)).collect();
    let final_ln = LayerNorm::new(D_MODEL);
    let output_proj = Linear::new(rng, D_MODEL, VOCAB);
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
    assert!(args.len() >= 5, "usage: fused_models_check <name> <k> <batch> <lr> [windows] [seed] [weight_decay] [dropout] [warmup_windows] [momentum]");
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
    let warmup = (warmup_windows / batch).max(1);
    let steps = windows / batch;
    let eval_every = (8000 / batch).max(1);

    let full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let split = (full.len() as f32 * 0.9) as usize;
    let (train, held_out) = full.split_at(split);

    // Model m's generator: its init, then its batches (training_recipe_check's order).
    let mut rngs: Vec<Rng> = (0..k as u64).map(|m| Rng::new(seed + m)).collect();
    let flat: Vec<f32> = rngs.iter_mut().flat_map(init).collect();
    let config = format!("k={k} batch={batch} lr={lr} windows={windows} seeds={seed}..{} wd={weight_decay} dropout={dropout} warmup={warmup_windows} momentum={momentum}", seed + k as u64 - 1);
    println!("run {name}: pid {} {config} steps={steps}", std::process::id());
    let _lease = gpu_lease::hold(Kind::Shared, &format!("scratchtape fused_models_check {name}"), Duration::from_secs(4 * 3600));
    let dev = DeviceParams::upload_models(&flat, k);
    let cfg = Config { vocab: VOCAB, d: D_MODEL, heads: N_HEADS, d_ff: D_FF, t: SEQ_LEN, n_blocks: N_BLOCKS, softmax1: false };
    let ones = (weight_decay > 0.0).then(|| upload_f32(&vec![1.0; flat.len()]));
    let velocity = (momentum > 0.0).then(|| upload_f32(&vec![0.0; flat.len()]));
    println!("columns: step | windows seen per model | held-out CE per model | mean | ensemble | elapsed | training ms/step since last row");
    std::fs::create_dir_all("runs").unwrap();
    let start = Instant::now();
    let mut evaluating = Duration::ZERO;
    let mut segment = (Instant::now(), 0);
    let mut last = (vec![], 0.0);
    for step in 0..=steps {
        if step % eval_every == 0 || step == steps {
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
        let lr = lr * ((step + 1) as f32 / warmup as f32).min(1.0);
        let (mut input, mut target) = (Vec::with_capacity(k * batch * SEQ_LEN), Vec::with_capacity(k * batch * SEQ_LEN));
        for rng in &mut rngs {
            for _ in 0..batch {
                let (i, t) = sample_window(rng, train, SEQ_LEN);
                input.extend(i);
                target.extend(t);
            }
        }
        dev.zero_grads();
        let mut dt = if dropout > 0.0 {
            DeviceTape::with_dropout(&dev, dropout, (seed as u32).wrapping_mul(0x85eb_ca6b) ^ step as u32)
        } else {
            DeviceTape::new(&dev)
        };
        let (_, _, loss) = model_forward(&mut dt, &cfg, &input, &target, k * batch);
        dt.backward(loss);
        if let Some(ones) = &ones {
            dev.decay(lr * weight_decay, ones);
        }
        match &velocity {
            Some(v) => dev.momentum(lr, momentum, v),
            None => dev.sgd(lr),
        }
    }
    let training = start.elapsed() - evaluating;

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
