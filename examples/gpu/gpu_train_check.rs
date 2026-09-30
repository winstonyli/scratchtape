// Milestone 5 of docs/gpu_step_design.md: does the device tape train like
// the CPU tape, and is its step faster?
//   gpu_train_check [softmax1 0|1] [steps=200] [control [index] | time | profile | host]
// training_recipe_check's setup (tiny_lm, batch 8, lr 0.3, seed 1): the
// same init and the same batches go to both tapes, in lockstep. Prints
// both training losses as it goes, then both models' train-probe and
// held-out CE (evaluated on the CPU; the GPU's parameters are read back
// once), then step times and launches per step.
// `control` runs no GPU: the second column is the CPU tape again, from an
// init with one weight (default mid-vector, a block weight) nudged by 1e-6. That's how far two
// runs drift from a rounding-sized difference alone, the yardstick for
// the GPU's drift (its ReLUs disagree with the CPU's on a unit or so a
// step; see device_training_tracks_cpu_over_steps).
// `time` runs GPU steps only, back to back (no CPU step in between to let
// the GPU clock down), and reports best/median step ms, how much of the
// step the host spends queueing it (before the loss readback blocks), and
// the step time when steps are pipelined (no readback until the last
// step, as a training loop that logs the loss rarely would run). `profile` does the
// same with gpu_step's device-timestamp profiler on and prints each launch
// site's GPU time per step. `host` does the same with the host-side
// profiler (gpu_step::host_profile_start): host time per launch site
// while queueing a step, readback excluded.
// Holds an exclusive GPU lease; runs at Normal CPU priority (LONG_RUNS.md:
// a BelowNormal GPU feeder starves under load).
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, LAUNCHES, client, host_profile_cut, host_profile_start, host_profile_take, pack, profile_start, profile_take, read_f32};
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::io::Write;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

#[path = "../common/mod.rs"]
mod common;
use common::{Model, apply_grad, encode_bytes, reconstruct, sample_window};

const D_MODEL: usize = 128;
const N_HEADS: usize = 8;
const D_FF: usize = 256;
const SEQ_LEN: usize = 64;
const VOCAB: usize = 256;
const N_BLOCKS: usize = 4;
const BATCH: usize = 8;
const LR: f32 = 0.3;

/// One CPU SGD step; returns the loss.
fn cpu_step(m: &mut Model, input: &[usize], target: &[usize], softmax1: bool, opt: &Sgd) -> f32 {
    let mut tape = Tape::with_capacity(2000);
    let (logits, out) = m.forward(&mut tape, input, BATCH, softmax1);
    let loss = tape.cross_entropy(logits, target);
    tape.backward(loss);
    apply_grad(&tape, &out, &mut m.token_emb, &mut m.pos_emb, &mut m.blocks, &mut m.final_ln, &mut m.output_proj, opt);
    tape.value(loss).data[0]
}

fn model_from_flat(flat: &[f32]) -> Model {
    let (token_emb, pos_emb, blocks, final_ln, output_proj) = reconstruct(flat, VOCAB, D_MODEL, SEQ_LEN, N_BLOCKS, N_HEADS, D_FF);
    Model { token_emb, pos_emb, blocks, final_ln, output_proj }
}

/// (best, median) in ms.
fn summary(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0] * 1e3, v[v.len() / 2] * 1e3)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let softmax1 = args.get(1).is_some_and(|a| a == "1");
    let steps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(200);
    let cfg = Config { vocab: VOCAB, d: D_MODEL, heads: N_HEADS, d_ff: D_FF, t: SEQ_LEN, n_blocks: N_BLOCKS, softmax1 };

    let full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let split = (full.len() as f32 * 0.9) as usize;
    let (train, held_out) = full.split_at(split);
    let train_probe = &train[..held_out.len()];

    let mut rng = Rng::new(1);
    let mut m = Model {
        token_emb: Embedding::new(&mut rng, VOCAB, D_MODEL),
        pos_emb: Embedding::new(&mut rng, SEQ_LEN, D_MODEL),
        blocks: (0..N_BLOCKS).map(|_| TransformerBlock::new(&mut rng, D_MODEL, N_HEADS, D_FF)).collect(),
        final_ln: LayerNorm::new(D_MODEL),
        output_proj: Linear::new(&mut rng, D_MODEL, VOCAB),
    };
    let opt = Sgd { lr: LR };

    if args.get(3).is_some_and(|a| a == "control") {
        let mut flat = pack(&m.token_emb, &m.pos_emb, &m.blocks, &m.final_ln, &m.output_proj);
        // default: a block weight (token 0 never occurs in the corpus)
        let mid = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(flat.len() / 2);
        flat[mid] += 1e-6;
        let mut m2 = model_from_flat(&flat);
        println!("gpu_train_check control: cpu vs cpu with flat[{mid}] += 1e-6; softmax1={softmax1} steps={steps}");
        println!("columns: step | cpu loss | nudged cpu loss | difference");
        let (mut a_tail, mut b_tail) = (0.0f64, 0.0f64);
        for step in 0..steps {
            let (mut input, mut target) = (Vec::with_capacity(BATCH * SEQ_LEN), Vec::with_capacity(BATCH * SEQ_LEN));
            for _ in 0..BATCH {
                let (i, t) = sample_window(&mut rng, train, SEQ_LEN);
                input.extend(i);
                target.extend(t);
            }
            let (a, b) = (cpu_step(&mut m, &input, &target, softmax1, &opt), cpu_step(&mut m2, &input, &target, softmax1, &opt));
            if step + 20 >= steps {
                a_tail += a as f64 / 20.0;
                b_tail += b as f64 / 20.0;
            }
            if step % 20 == 0 || step + 1 == steps {
                println!("{step:>4} | {a:.4} | {b:.4} | {:+.1e}", b - a);
                std::io::stdout().flush().unwrap();
            }
        }
        println!("mean training loss, last 20 steps: cpu {a_tail:.4}, nudged {b_tail:.4}");
        println!("train-probe CE: cpu {:.4}, nudged {:.4}", m.full_ce(train_probe, SEQ_LEN, softmax1), m2.full_ce(train_probe, SEQ_LEN, softmax1));
        println!("held-out CE:    cpu {:.4}, nudged {:.4}", m.full_ce(held_out, SEQ_LEN, softmax1), m2.full_ce(held_out, SEQ_LEN, softmax1));
        return;
    }

    let _lease = gpu_lease::hold(Kind::Exclusive, "scratchtape gpu_train_check (milestone 5)", Duration::from_secs(15 * 60));
    client();
    let dev = DeviceParams::upload(&pack(&m.token_emb, &m.pos_emb, &m.blocks, &m.final_ln, &m.output_proj));
    println!("gpu_train_check: pid {} softmax1={softmax1} batch={BATCH} lr={LR} steps={steps}", std::process::id());
    let profile = args.get(3).is_some_and(|a| a == "profile");
    let host = args.get(3).is_some_and(|a| a == "host");
    if profile || host || args.get(3).is_some_and(|a| a == "time") {
        let mut batches = |n: usize| -> Vec<(Vec<usize>, Vec<usize>)> {
            (0..n)
                .map(|_| {
                    let (mut input, mut target) = (Vec::with_capacity(BATCH * SEQ_LEN), Vec::with_capacity(BATCH * SEQ_LEN));
                    for _ in 0..BATCH {
                        let (i, t) = sample_window(&mut rng, train, SEQ_LEN);
                        input.extend(i);
                        target.extend(t);
                    }
                    (input, target)
                })
                .collect()
        };
        // Queues one step; returns the loss's handle without reading it.
        let queue_step = |input: &[usize], target: &[usize]| {
            dev.zero_grads();
            let mut dt = DeviceTape::new(&dev);
            let (_, _, dloss) = model_forward(&mut dt, &cfg, input, target, BATCH);
            dt.backward(dloss);
            dev.sgd(LR);
            dt.value(dloss).clone()
        };
        let (mut t, mut t_queue) = (vec![], vec![]);
        for (step, (input, target)) in batches(steps).iter().enumerate() {
            let t0 = Instant::now();
            let loss = queue_step(input, target);
            let queued = t0.elapsed().as_secs_f64();
            if host {
                host_profile_cut();
            }
            let gl = read_f32(&loss);
            if step > 0 {
                t.push(t0.elapsed().as_secs_f64());
                t_queue.push(queued);
            } else if profile {
                profile_start(); // after compilation
            } else if host {
                host_profile_start();
            }
            assert!(gl.is_finite(), "GPU loss diverged at step {step}");
        }
        if profile {
            let sites = profile_take();
            let per = |x: f64| x * 1e3 / (steps - 1) as f64;
            let total: f64 = sites.iter().map(|s| s.2).sum();
            for (site, n, secs) in sites {
                println!("{:>7.3} ms/step {:>5.1}%  {:>3} launches/step  {site}", per(secs), 100.0 * secs / total, n / (steps - 1));
            }
            println!("GPU time, all launches: {:.2} ms/step", per(total));
        }
        if host {
            let sites = host_profile_take();
            let per = |x: f64| x * 1e3 / (steps - 1) as f64;
            let total: f64 = sites.iter().map(|s| s.2).sum();
            for (site, n, secs) in sites {
                println!("{:>7.3} ms/step {:>5.1}%  {:>3} launches/step  {:>5.1} us each  {site}", per(secs), 100.0 * secs / total, n / (steps - 1), secs * 1e6 / n as f64);
            }
            println!("host time, all launches: {:.2} ms/step", per(total));
        }
        let ((b, m), (qb, qm)) = (summary(t), summary(t_queue));
        println!("gpu only, {steps} steps back to back: step ms best {b:.2} / median {m:.2}; host queueing best {qb:.2} / median {qm:.2}");
        if !profile && !host {
            let mut runs = vec![];
            for _ in 0..3 {
                let batches = batches(steps);
                let t0 = Instant::now();
                let mut loss = None;
                for (input, target) in &batches {
                    loss = Some(queue_step(input, target));
                }
                assert!(read_f32(&loss.unwrap()).is_finite(), "GPU loss diverged");
                runs.push(t0.elapsed().as_secs_f64() / steps as f64);
            }
            let (b, m) = summary(runs);
            println!("pipelined, {steps} steps, one readback, 3 runs: step ms best {b:.2} / median {m:.2}");
        }
        return;
    }
    println!("columns: step | cpu loss | gpu loss | gpu - cpu");

    let (mut t_cpu, mut t_gpu, mut launches) = (vec![], vec![], 0);
    let (mut cpu_tail, mut gpu_tail) = (0.0f64, 0.0f64);
    for step in 0..steps {
        let (mut input, mut target) = (Vec::with_capacity(BATCH * SEQ_LEN), Vec::with_capacity(BATCH * SEQ_LEN));
        for _ in 0..BATCH {
            let (i, t) = sample_window(&mut rng, train, SEQ_LEN);
            input.extend(i);
            target.extend(t);
        }

        // GPU step: zero, forward, backward, SGD, read the loss back.
        let (t0, l0) = (Instant::now(), LAUNCHES.load(Ordering::Relaxed));
        dev.zero_grads();
        let mut dt = DeviceTape::new(&dev);
        let (_, _, dloss) = model_forward(&mut dt, &cfg, &input, &target, BATCH);
        dt.backward(dloss);
        dev.sgd(LR);
        let gl = read_f32(dt.value(dloss));
        let gpu_time = t0.elapsed().as_secs_f64();
        launches = LAUNCHES.load(Ordering::Relaxed) - l0;

        let t1 = Instant::now();
        let cl = cpu_step(&mut m, &input, &target, softmax1, &opt);
        let cpu_time = t1.elapsed().as_secs_f64();

        if step > 0 {
            // step 0 includes kernel compilation
            t_gpu.push(gpu_time);
            t_cpu.push(cpu_time);
        }
        if step + 20 >= steps {
            cpu_tail += cl as f64 / 20.0;
            gpu_tail += gl as f64 / 20.0;
        }
        if step % 20 == 0 || step + 1 == steps {
            println!("{step:>4} | {cl:.4} | {gl:.4} | {:+.1e}   (step ms: cpu {:.0}, gpu {:.1})", gl - cl, cpu_time * 1e3, gpu_time * 1e3);
            std::io::stdout().flush().unwrap();
        }
        assert!(gl.is_finite(), "GPU loss diverged at step {step}");
    }

    let g = model_from_flat(&dev.read(&dev.params));
    println!("mean training loss, last 20 steps: cpu {cpu_tail:.4}, gpu {gpu_tail:.4}");
    println!("train-probe CE: cpu {:.4}, gpu {:.4}", m.full_ce(train_probe, SEQ_LEN, softmax1), g.full_ce(train_probe, SEQ_LEN, softmax1));
    println!("held-out CE:    cpu {:.4}, gpu {:.4}", m.full_ce(held_out, SEQ_LEN, softmax1), g.full_ce(held_out, SEQ_LEN, softmax1));
    let ((cb, cm), (gb, gm)) = (summary(t_cpu), summary(t_gpu));
    println!("step ms (best / median, steps 1..): cpu {cb:.1} / {cm:.1}, gpu {gb:.2} / {gm:.2}; speedup {:.1}x / {:.1}x", cb / gb, cm / gm);
    println!("gpu launches per step: {launches}");
}

