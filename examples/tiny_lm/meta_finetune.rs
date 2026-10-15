// Meta-trained adaptability (docs/tiers_design.md, "Meta-trained adaptability"): fine-tune a checkpoint so that the per-chunk
// SGD step `tier_eval dyn=<lr> dyn_every=2` takes at test time helps more. First-order MAML: a run is `k` consecutive chunks of
// two windows (spaced seq - warm bytes apart, as the scored windows are); chunk t is scored at the fast weights theta_{t-1}
// (adapted on chunks 1..t-1 by plain SGD at `inner_lr`, no dropout) on its last seq - warm rows only, the rows `tier_eval` scores
// (rows the earlier steps already trained on are ignored: target = vocab); that gradient g_t is kept. Then the chunk's full-row
// gradient, as `tier_eval` takes it, gives the inner step. The outer update is momentum SGD on the mean of the g_t over `runs` runs. `inner_lr=0` is the control: the same data and
// the same code with every gradient taken at the slow weights, i.e. ordinary fine-tuning.
//   meta_finetune <ckpt_in> <ckpt_out> [updates=2000] [runs=4] [k=4] [inner_lr=0.04] [lr=0.01] [momentum=0.9] [wd=0.0002]
//                 [seed=1] [corpus=novels22] [seq=128] [warm=96]
// Logs the mean scored-row loss of the training chunks by position t in the run: if adaptation works, later chunks are cheaper.
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, read};
use scratchtape::nn::Rng;
use std::time::{Duration, Instant};

#[path = "../common/mod.rs"]
mod common;
use common::split_corpus;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (ckpt_in, ckpt_out) = (&args[0], &args[1]);
    let opt: std::collections::HashMap<&str, &str> = args[2..].iter().map(|a| a.split_once('=').expect("key=value")).collect();
    let get = |k: &str, d: &str| opt.get(k).copied().unwrap_or(d).to_string();
    let (updates, runs, k, seed): (usize, usize, usize, u64) = (get("updates", "2000").parse().unwrap(), get("runs", "4").parse().unwrap(), get("k", "4").parse().unwrap(), get("seed", "1").parse().unwrap());
    let (inner_lr, lr, mu, wd): (f32, f32, f32, f32) = (get("inner_lr", "0.04").parse().unwrap(), get("lr", "0.01").parse().unwrap(), get("momentum", "0.9").parse().unwrap(), get("wd", "0.0002").parse().unwrap());
    let (seq, warm): (usize, usize) = (get("seq", "128").parse().unwrap(), get("warm", "96").parse().unwrap());
    let cfg = Config { vocab: 256, d: 256, heads: 8, d_ff: 512, t: seq, n_blocks: 4, softmax1: false };
    let mut theta: Vec<f32> = std::fs::read_to_string(ckpt_in).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(theta.len(), cfg.len(), "{ckpt_in} does not match the model config");
    let (train, _) = split_corpus(&get("corpus", "novels22"));
    let unit = seq - warm;
    let span = 2 * k * unit + seq + 1;
    let mut rng = Rng::new(seed);
    let mut velocity = vec![0.0f32; theta.len()];
    let _lease = gpu_lease::hold(Kind::Shared, "scratchtape meta_finetune", Duration::from_secs(2 * 3600));
    println!("meta_finetune pid {} {ckpt_in} -> {ckpt_out}: updates={updates} runs={runs} k={k} inner_lr={inner_lr} lr={lr} mu={mu} wd={wd} seed={seed}", std::process::id());
    let (start, mut loss_by_t, mut logged) = (Instant::now(), vec![0.0f64; k], 0usize);
    for u in 1..=updates {
        gpu_lease::pause_while_exclusive();
        let mut gsum = vec![0.0f32; theta.len()];
        for r in 0..runs {
            let s0 = (rng.next_f32() * (train.len() - span) as f32) as usize;
            let fast = DeviceParams::upload(&theta);
            for t in 0..k {
                let (mut ids, mut targets) = (vec![], vec![]);
                for j in 0..2 {
                    let s = s0 + (2 * t + j) * unit;
                    ids.extend(&train[s..s + seq]);
                    targets.extend(&train[s + 1..s + seq + 1]);
                }
                // The query: this chunk's new rows at the weights adapted so far.
                let query: Vec<usize> = targets.iter().enumerate().map(|(i, &x)| if i % seq < warm { 256 } else { x }).collect();
                fast.zero_grads();
                let mut dt = DeviceTape::new(&fast);
                let (_, _, loss) = model_forward(&mut dt, &cfg, &ids, &query, 2);
                if r == 0 {
                    loss_by_t[t] += read(dt.row_losses(loss)).iter().map(|&x| x as f64).sum::<f64>() / (2 * unit) as f64;
                }
                dt.backward(loss);
                drop(dt);
                for (a, b) in gsum.iter_mut().zip(fast.read(&fast.grads)) {
                    *a += b * (seq as f32 / unit as f32); // the mean was over all rows
                }
                if inner_lr > 0.0 && t + 1 < k {
                    fast.zero_grads();
                    let mut dt = DeviceTape::new(&fast);
                    let (_, _, loss) = model_forward(&mut dt, &cfg, &ids, &targets, 2);
                    dt.backward(loss);
                    drop(dt);
                    fast.sgd(inner_lr);
                }
            }
        }
        let scale = 1.0 / (runs * k) as f32;
        for i in 0..theta.len() {
            velocity[i] = mu * velocity[i] + gsum[i] * scale;
            theta[i] = theta[i] * (1.0 - lr * wd) - lr * velocity[i];
        }
        logged += 1;
        if u % 50 == 0 || u == updates {
            let by_t: Vec<String> = loss_by_t.iter().map(|l| format!("{:.4}", l / logged as f64)).collect();
            println!("{u:>6} | train loss by chunk position {} | {:.0} ms/update", by_t.join(" "), start.elapsed().as_secs_f64() * 1e3 / u as f64);
            loss_by_t.fill(0.0);
            logged = 0;
        }
        if u % 500 == 0 || u == updates {
            std::fs::write(ckpt_out, theta.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")).unwrap();
        }
    }
    println!("saved {ckpt_out} after {:.0}s", start.elapsed().as_secs_f32());
}
