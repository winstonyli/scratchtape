// Why does tiny_lm_corpus.rs's transformer trail a Kneser-Ney 7-gram by
// 0.15-0.20 nats (ngram_baseline.rs: 1.655 vs 1.80-1.86)? Its recipe is
// one 64-byte window per step, plain SGD at lr 0.3, no regularization -
// and it overfits (train ~1.25 vs held-out ~1.8). This varies the recipe
// one factor at a time and scores every run on the same deterministic
// held-out CE (all 371 non-overlapping 64-byte windows, ngram_baseline's
// windowing) instead of the training log's noisy 20-window sample.
//
// One condition per process (crash-isolated, launchable at low priority,
// per LONG_RUNS.md):
//   training_recipe_check <name> <softmax1 0|1> <batch> <lr> [windows] [checkpoint_secs] [cpu|gpu] [seed] [weight_decay] [all|weights] [dropout] [warmup_windows] [momentum]
// `windows` is the total training budget in 64-byte windows (default
// 64000, tiny_lm_corpus.rs's), so batch size changes steps, not data:
// steps = windows / batch. Progress streams to stdout as it happens; the
// final model is saved to runs/<name>.ckpt.
//
// Resumable (LONG_RUNS.md): every `checkpoint_secs` (default 600) the
// step, RNG state and parameters go to runs/<name>.resume. Relaunching
// the same command continues from there, bit-identical to an unbroken
// run; the file is removed once the final .ckpt is written.
//
// `gpu` runs each training step on the eGPU (gpu_step's device tape, same
// init, batches and SGD) with no readback per step, and evaluates there
// too (`device_ce`); the parameters come back to the CPU only for
// checkpoints. At the end the CPU re-scores the final model's held-out CE
// and the run fails if the two differ by 1e-3. Holds a shared GPU lease
// and pauses at evaluations while another job holds an exclusive one. WGPU_BACKEND=dx12 as gpu_step::client. The GPU run is not the CPU
// run bit for bit: about once a step a ReLU input lands within rounding
// of 0 and the two disagree (docs/gpu_step_design.md, milestone 5).
//
// `seed` (default 1) drives both the init and the batch order.
// `weight_decay` (default 0, GPU only) shrinks parameters by
// (1 - lr * weight_decay) each step before the gradient step: every one
// (`all`, the default), or only the weights and embeddings (`weights`:
// LayerNorm and biases exempt, Config::decay_mask). `dropout` (default
// 0, GPU only) is the rate on each block's attention output and FFN
// hidden layer (gpu_step::tape::block_forward), training steps only.
// `warmup_windows` (default 0) ramps lr linearly from lr/k to lr over
// the first k = warmup_windows / batch steps, so warmup covers the same
// data at any batch size (Goyal et al. 2017, for large-batch lr scaling).
// `momentum` (default 0, GPU only) is heavy-ball momentum mu: v = mu*v + g,
// p -= lr*v, so the effective lr is lr / (1 - mu). Its velocity is
// checkpointed too (a third line, in gpu_step's pack order).
//
// Same architecture and init stream as tiny_lm_corpus.rs (seed 1), so
// batch=1 lr=0.3 reproduces attention_uniformity_check.rs's plain (1.852)
// and softmax1 (1.803) checkpoints' recipe, though not their batch order
// (that example also draws eval windows from the training rng).
//
// Results, round 1: batch size at lr 0.3, 64000 windows (2026-09-24;
// 4 concurrent runs at BelowNormal on a CPU shared with other jobs,
// ~4.5-5.5 h each). Final train-probe / held-out:
//   plain_b8     8000 steps  1.390 / 1.858   (batch 1: 1.852)
//   softmax1_b8  8000 steps  1.375 / 1.846   (batch 1: 1.803)
//   plain_b32    2000 steps  1.806 / 2.128
//   softmax1_b32 2000 steps  1.813 / 2.115
// Batch 8 ties batch 1 with 8x fewer steps; batch 32 at unscaled lr is
// under-stepped. softmax1's edge over plain shrinks to ~0.012 under
// batching. At batch 8 the train/held-out gap is 0.47 and still
// widening: overfitting, not optimization, is what's limiting.
//
// GPU, same recipe (2026-09-25; eGPU on Vulkan, idle otherwise):
//   gpu_plain_b8     1.357 / 1.857   (CPU above: 1.390 / 1.858)
//   gpu_softmax1_b8  1.378 / 1.860   (CPU above: 1.375 / 1.846)
// ~20 s per run instead of ~4.5 h, uncontended: ~1.9-2.0 ms/step once
// the kernels are compiled (the first 1000 steps include ~0.7-1.5 s of
// compilation), plus ~0.3 s per evaluation on the device (125 s in all
// when evaluation ran on the CPU). Other sessions' jobs on the eGPU
// (a render loop, self-play) made steps 2.6-18 ms at times; the column
// shows it. Reruns, and a run killed and resumed, reproduce the
// checkpoint byte for byte. GPU and CPU runs are chaotic twins (see
// above), and their gap is itself a measure of seed-level noise: 0.001
// held-out for plain but 0.014 for softmax1, with softmax1 behind plain
// on the GPU. So round 1's ~0.012 softmax1 edge at batch 8 is within
// that noise. Seeds 1-5 (2026-09-25), held-out mean +- sd:
//   plain     1.8521 +- 0.0108   (1.8573 1.8448 1.8652 1.8555 1.8379)
//   softmax1  1.8481 +- 0.0091   (1.8603 1.8553 1.8405 1.8403 1.8443)
// Paired by seed, plain - softmax1 = 0.004, 95% CI [-0.015, 0.023]: no
// detectable difference at batch 8.
//
// Weight decay, plain, held-out mean over seeds 1-5: wd 0 1.8521, 1e-4
// 1.8353, 2e-4 1.8334, 3e-4 1.8353; 5e-4 1.898 and 1e-3+ underfit (seeds
// 1-2). On fresh seeds 6-10, 2e-4 vs 0: 1.8405 vs 1.8708, paired gain
// 0.030, 95% CI [0.007, 0.054]. Seeds 1-2, held-out mean (wd 0: 1.8511,
// all 2e-4: 1.8381): weights-only 2e-4 1.8412, 5e-4 1.8508, 1e-3 1.8766,
// 2e-3 1.9567, so exempting LayerNorm and biases allows no larger rate;
// dropout 0.05 1.8683, 0.1 1.9034, 0.2 1.9817, worse at every rate.
//
// Batch 1, 64000 steps, held-out over seeds 1-5 (GPU):
//   plain     1.8247 +- 0.0138   (1.8072 1.8227 1.8174 1.8332 1.8429)
//   softmax1  1.8037 +- 0.0089   (1.7989 1.8050 1.7923 1.8062 1.8162)
// Paired, plain - softmax1 = 0.021, 95% CI [0.011, 0.031], 5 of 5 seeds.
// Batch 1 beats batch 8 by 0.027 (plain) and 0.044 (softmax1).
// Batch 8 at lr 1.2 with 6400 warmup windows matches batch 1 (plain,
// seeds 1-5): 1.8187 +- 0.0091 vs 1.8247, paired 0.006, 95% CI [-0.014,
// 0.026], at ~6x the windows per second. lr 0.6: 1.8241. Without warmup
// lr 0.6 ends at 1.864 and 1.2 diverges. Batch 16 (lr 1.2: 1.846) and 32
// (lr 1.2: 1.923) can't catch up: lr 1.7 / 2.4 diverge.
// Momentum 0.9 + warmup, plain, seeds 1-5, held-out (vs batch 1 1.8247):
//   batch 8,  lr 0.12  1.8088 +- 0.0089
//   batch 16, lr 0.24  1.8110 +- 0.0115
//   batch 32, lr 0.48  1.8175 +- 0.0155   (paired 0.007, CI [-0.016, 0.030])
// No divergence up to lr 0.96 (worse CE). softmax1 at batch 8, SGD lr 1.2
// + warmup: 1.8263 vs plain 1.8187 (its batch-1 lead doesn't survive).
// Long runs, batch 32 + momentum, seeds 1-3, held-out final (best):
// none 256k windows 2.163 (1.809 at ~64-88k, then memorizes); dropout 0.1
// + wd 2e-4 256k 1.743 (1.711); dropout 0.3 + wd 2e-4 1M 1.686 (1.663);
// dropout 0.4 + wd 2e-4 1M 1.677 (1.670). The 7-gram is 1.655.
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, pack, read_f32, upload_f32};
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::io::Write;
use std::time::{Duration, Instant};

#[path = "../common/mod.rs"]
mod common;
use common::{Model, apply_grad, encode_bytes, flatten_all, reconstruct, sample_window};

const D_MODEL: usize = 128;
const N_HEADS: usize = 8;
const D_FF: usize = 256;
const SEQ_LEN: usize = 64;
const N_BLOCKS: usize = 4;
const VOCAB: usize = 256;

/// full_ce on the device, from the parameters in `dev`. Windows go in
/// chunks of up to 64: the row kernels launch one cube per softmax row
/// (batch * heads * T of them), and a launch dimension stops at 65535.
fn device_ce(dev: &DeviceParams, cfg: &Config, corpus: &[usize]) -> f32 {
    let starts: Vec<usize> = (0..corpus.len() - SEQ_LEN).step_by(SEQ_LEN).collect();
    let mut total = 0.0f64;
    for chunk in starts.chunks(64) {
        let (mut ids, mut targets) = (vec![], vec![]);
        for &s in chunk {
            ids.extend(&corpus[s..s + SEQ_LEN]);
            targets.extend(&corpus[s + 1..s + SEQ_LEN + 1]);
        }
        let mut dt = DeviceTape::new(dev);
        let (_, _, loss) = model_forward(&mut dt, cfg, &ids, &targets, chunk.len());
        total += read_f32(dt.value(loss)) as f64 * chunk.len() as f64;
    }
    (total / starts.len() as f64) as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 5,
        "usage: training_recipe_check <name> <softmax1 0|1> <batch> <lr> [windows] [checkpoint_secs] [cpu|gpu] [seed] [weight_decay] [all|weights] [dropout] [warmup_windows] [momentum]"
    );
    let name = &args[1];
    let softmax1 = args[2] == "1";
    let batch: usize = args[3].parse().unwrap();
    let lr: f32 = args[4].parse().unwrap();
    let windows: usize = args.get(5).map(|w| w.parse().unwrap()).unwrap_or(64000);
    let checkpoint_secs: u64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(600);
    let gpu = match args.get(7).map(String::as_str) {
        None | Some("cpu") => false,
        Some("gpu") => true,
        Some(other) => panic!("device must be cpu or gpu, not {other}"),
    };
    let seed: u64 = args.get(8).map(|s| s.parse().unwrap()).unwrap_or(1);
    assert!(seed != 0, "seed 0 is xorshift's fixed point");
    let weight_decay: f32 = args.get(9).map(|s| s.parse().unwrap()).unwrap_or(0.0);
    assert!(weight_decay == 0.0 || gpu, "weight_decay is GPU only");
    let decay_all = match args.get(10).map(String::as_str) {
        None | Some("all") => true,
        Some("weights") => false,
        Some(other) => panic!("decay applies to all or weights, not {other}"),
    };
    let dropout: f32 = args.get(11).map(|s| s.parse().unwrap()).unwrap_or(0.0);
    assert!(dropout == 0.0 || gpu, "dropout is GPU only");
    let warmup_windows: usize = args.get(12).map(|s| s.parse().unwrap()).unwrap_or(0);
    let warmup = (warmup_windows / batch).max(1);
    let momentum: f32 = args.get(13).map(|s| s.parse().unwrap()).unwrap_or(0.0);
    assert!(momentum == 0.0 || gpu, "momentum is GPU only");
    let steps = windows / batch;
    let eval_every = (8000 / batch).max(1);

    let full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let split = (full.len() as f32 * 0.9) as usize;
    let (train, held_out) = full.split_at(split);
    // A fixed 371-window slice of train, for a like-for-like overfitting gap.
    let train_probe = &train[..held_out.len()];

    let mut rng = Rng::new(seed);
    let mut m = Model {
        token_emb: Embedding::new(&mut rng, VOCAB, D_MODEL),
        pos_emb: Embedding::new(&mut rng, SEQ_LEN, D_MODEL),
        blocks: (0..N_BLOCKS).map(|_| TransformerBlock::new(&mut rng, D_MODEL, N_HEADS, D_FF)).collect(),
        final_ln: LayerNorm::new(D_MODEL),
        output_proj: Linear::new(&mut rng, D_MODEL, VOCAB),
    };

    // Resume: header "<config> | <step> <rng state>", then the parameters,
    // then (momentum runs) the velocity.
    let config = format!(
        "softmax1={softmax1} batch={batch} lr={lr} windows={windows} seed={seed} wd={weight_decay} decay_all={decay_all} dropout={dropout} warmup={warmup_windows} momentum={momentum}"
    );
    let resume_path = format!("runs/{name}.resume");
    let mut first = 0;
    let mut resumed_velocity = None;
    if let Ok(text) = std::fs::read_to_string(&resume_path) {
        let mut lines = text.split('\n');
        let (header, params) = (lines.next().unwrap(), lines.next().unwrap());
        let (saved, at) = header.split_once(" | ").unwrap();
        assert_eq!(saved, config, "{resume_path} is from a different config");
        resumed_velocity = lines.next().map(|v| v.split_whitespace().map(|x| x.parse::<f32>().unwrap()).collect::<Vec<_>>());
        assert_eq!(resumed_velocity.is_some(), momentum > 0.0, "{resume_path}: velocity line must match momentum");
        let (step, state) = at.split_once(' ').unwrap();
        first = step.parse().unwrap();
        rng = Rng::new(state.parse().unwrap());
        let flat: Vec<f32> = params.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let (token_emb, pos_emb, blocks, final_ln, output_proj) = reconstruct(&flat, VOCAB, D_MODEL, SEQ_LEN, N_BLOCKS, N_HEADS, D_FF);
        m = Model { token_emb, pos_emb, blocks, final_ln, output_proj };
    }

    println!("run {name}: pid {} {config} steps={steps} device={}", std::process::id(), if gpu { "gpu" } else { "cpu" });
    // On the GPU the parameters live in `dev`; `m` is refreshed from it
    // (`sync`) only to save them.
    let _lease = gpu.then(|| gpu_lease::hold(Kind::Shared, &format!("scratchtape training_recipe_check {name}"), Duration::from_secs(4 * 3600)));
    let dev = gpu.then(|| DeviceParams::upload(&pack(&m.token_emb, &m.pos_emb, &m.blocks, &m.final_ln, &m.output_proj)));
    let cfg = Config { vocab: VOCAB, d: D_MODEL, heads: N_HEADS, d_ff: D_FF, t: SEQ_LEN, n_blocks: N_BLOCKS, softmax1 };
    let decay_mask = (weight_decay > 0.0).then(|| {
        let mask = if decay_all { vec![1.0; cfg.len()] } else { cfg.decay_mask() };
        upload_f32(&mask)
    });
    let velocity = (momentum > 0.0).then(|| upload_f32(&resumed_velocity.unwrap_or_else(|| vec![0.0; cfg.len()])));
    let sync = |m: &mut Model| {
        if let Some(dev) = &dev {
            let (token_emb, pos_emb, blocks, final_ln, output_proj) = reconstruct(&dev.read(&dev.params), VOCAB, D_MODEL, SEQ_LEN, N_BLOCKS, N_HEADS, D_FF);
            *m = Model { token_emb, pos_emb, blocks, final_ln, output_proj };
        }
    };
    let mut held_out_ce = f32::NAN;
    if first > 0 {
        println!("resumed from {resume_path} at step {first}");
    }
    println!("columns: step | windows seen | train-probe CE | held-out CE (deterministic) | elapsed | training ms/step since last row");
    std::fs::create_dir_all("runs").unwrap();
    let start = Instant::now();
    let mut evaluating = Duration::ZERO;
    // Training time since the previous evaluation, and its first step.
    let mut segment = (Instant::now(), first);
    let mut last_save = Instant::now();
    for step in first..=steps {
        // Save before this step's work; `step` is the next one to run.
        let save = step > first && last_save.elapsed().as_secs() >= checkpoint_secs;
        let eval = step % eval_every == 0 || step == steps;
        if save || step == steps {
            sync(&mut m);
        }
        if eval && gpu {
            gpu_lease::pause_while_exclusive();
        }
        if save {
            let flat = flatten_all(&m.token_emb, &m.pos_emb, &m.blocks, &m.final_ln, &m.output_proj);
            let params = flat.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            let tmp = format!("{resume_path}.tmp");
            let mut text = format!("{config} | {step} {}\n{params}", rng.state());
            if let (Some(dev), Some(v)) = (&dev, &velocity) {
                text += "\n";
                text += &dev.read(v).iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            }
            std::fs::write(&tmp, text).unwrap();
            std::fs::rename(&tmp, &resume_path).unwrap();
            last_save = Instant::now();
        }
        if eval {
            let t = Instant::now();
            let ms_per_step = match step - segment.1 {
                0 => "-".to_string(),
                n => format!("{:.2}", segment.0.elapsed().as_secs_f64() * 1e3 / n as f64),
            };
            let (probe_ce, held) = match &dev {
                Some(dev) => (device_ce(dev, &cfg, train_probe), device_ce(dev, &cfg, held_out)),
                None => (m.full_ce(train_probe, SEQ_LEN, softmax1), m.full_ce(held_out, SEQ_LEN, softmax1)),
            };
            held_out_ce = held;
            println!("{step:>6} | {:>6} | {probe_ce:.4} | {held:.4} | {:.0}s | {ms_per_step}", step * batch, start.elapsed().as_secs_f32());
            if held.is_nan() {
                println!("diverged to NaN before step {step}");
                return;
            }
            std::io::stdout().flush().unwrap();
            evaluating += t.elapsed();
            segment = (Instant::now(), step);
        }
        if step == steps {
            break;
        }
        let lr = lr * ((step + 1) as f32 / warmup as f32).min(1.0);
        let mut input = Vec::with_capacity(batch * SEQ_LEN);
        let mut target = Vec::with_capacity(batch * SEQ_LEN);
        for _ in 0..batch {
            let (i, t) = sample_window(&mut rng, train, SEQ_LEN);
            input.extend(i);
            target.extend(t);
        }
        if let Some(dev) = &dev {
            dev.zero_grads();
            let mut dt = if dropout > 0.0 {
                // A fresh mask every step, reproducible from (seed, step).
                DeviceTape::with_dropout(dev, dropout, (seed as u32).wrapping_mul(0x85eb_ca6b) ^ step as u32)
            } else {
                DeviceTape::new(dev)
            };
            let (_, _, dloss) = model_forward(&mut dt, &cfg, &input, &target, batch);
            dt.backward(dloss);
            match &velocity {
                Some(v) => dev.momentum(lr, momentum, v, decay_mask.as_ref().map(|m| (lr * weight_decay, m))),
                None => {
                    if let Some(mask) = &decay_mask {
                        dev.decay(lr * weight_decay, mask);
                    }
                    dev.sgd(lr)
                }
            }
            continue;
        }
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = m.forward(&mut tape, &input, batch, softmax1);
        let loss = tape.cross_entropy(logits, &target);
        if tape.value(loss).data[0].is_nan() {
            println!("diverged to NaN at step {step}");
            return;
        }
        tape.backward(loss);
        apply_grad(&tape, &out, &mut m.token_emb, &mut m.pos_emb, &mut m.blocks, &mut m.final_ln, &mut m.output_proj, &Sgd { lr });
    }

    let flat = flatten_all(&m.token_emb, &m.pos_emb, &m.blocks, &m.final_ln, &m.output_proj);
    let text = flat.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    std::fs::write(format!("runs/{name}.ckpt"), text).unwrap();
    let _ = std::fs::remove_file(&resume_path);
    let total = start.elapsed();
    if gpu {
        // The device's CE against the CPU's on the final model: the same
        // numbers up to float summation order.
        let cpu = m.full_ce(held_out, SEQ_LEN, softmax1);
        println!("held-out CE of the final model on the CPU: {cpu:.4} (device {held_out_ce:.4})");
        assert!((cpu - held_out_ce).abs() < 1e-3, "device CE disagrees with the CPU's");
    }
    let training = total - evaluating;
    println!(
        "saved runs/{name}.ckpt ({:.0}s total: {:.0}s evaluating, {:.0}s training = {:.2} ms/step)",
        total.as_secs_f32(),
        evaluating.as_secs_f32(),
        training.as_secs_f32(),
        training.as_secs_f64() * 1e3 / (steps - first).max(1) as f64
    );
}
