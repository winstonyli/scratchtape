// Null baseline for tiny_lm_corpus.rs's per-head "top sink target" metric,
// run with NO model at all: what does that metric report if attention is
// exactly uniform over the visible (causal) keys?
//
// Why it matters: softmax1+QK-norm bounds every score to
// [-1,1]/sqrt(d_k) = +/-0.25 (d_k=16), so within a row no weight can exceed
// another by more than e^0.5 =~ 1.65x - attention is near-uniform by
// construction. The sink metric averages weight per (query byte, key byte)
// pair over one FIXED set of sampled windows (window_seed 777, shared by
// every seed and head), then takes each query byte's argmax key. When the
// weights carry almost no content signal, that argmax is decided by where
// each key byte happened to sit in those fixed windows (early keys are
// visible to more queries, each at a larger 1/(qi+2) weight) plus small-
// sample noise for rare bytes - not by anything the model learned.
//
// Result: uniform attention alone reproduces the reported numbers - same
// 19953 usable windows, self-attention rate ~0.72 (reported 0.64-0.78),
// and '\u{bc}' as the top sink with 10 votes (reported 9-11). See the
// README/doc correction: the "all 32 heads converge on '\u{bc}'" finding
// was a measurement artifact of near-uniform attention, not learned
// geometry.
//
// The 23 noise-excluded bytes are hardcoded from the tiny_lm_corpus.rs run
// (bottom-20% accumulated |gradient|); deriving them would need a training
// run, and the null needs no model.
use scratchtape::nn::Rng;
use std::collections::HashMap;

#[path = "../common/mod.rs"]
mod common;
use common::{encode_bytes, sample_window};

fn main() {
    let full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let split = (full.len() as f32 * 0.9) as usize;
    let train = &full[..split];
    let seq_len = 64;

    let excluded: Vec<usize> =
        [b'Z' as usize, 0x81, 0x82, 0x83, 0x84, 0x85, 0x89, 0x8d, 0x91, 0xa2, 0xa3, 0xaa, 0xae, 0xb0, 0xb1, 0xb3, 0xb5, 0xb6, 0xb8, 0xb9, 0xba, 0xbb, 0xbe].to_vec();
    let mut distinct: Vec<usize> = train.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let filtered: Vec<usize> = distinct.into_iter().filter(|b| !excluded.contains(b)).collect();
    let byte_index: HashMap<usize, usize> = filtered.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let n = filtered.len();

    for (name, plus_one) in [("softmax1-uniform", 2.0f32), ("softmax-uniform", 1.0f32)] {
        let mut sum = vec![0.0f32; n * n];
        let mut count = vec![0u32; n * n];
        let mut rng = Rng::new(777);
        let mut used = 0;
        for _ in 0..20000 {
            let (window, _) = sample_window(&mut rng, train, seq_len);
            if window.iter().any(|b| !byte_index.contains_key(b)) {
                continue;
            }
            used += 1;
            for qi in 0..seq_len {
                let q = byte_index[&window[qi]];
                for ki in 0..seq_len {
                    let k = byte_index[&window[ki]];
                    if ki <= qi {
                        sum[q * n + k] += 1.0 / (qi as f32 + plus_one);
                    }
                    count[q * n + k] += 1;
                }
            }
        }
        let (mut self_count, mut covered) = (0, 0);
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            let best = (0..n).filter(|&j| count[i * n + j] > 0).max_by(|&a, &b| {
                let ma = sum[i * n + a] / count[i * n + a] as f32;
                let mb = sum[i * n + b] / count[i * n + b] as f32;
                ma.partial_cmp(&mb).unwrap()
            });
            let Some(top) = best else { continue };
            covered += 1;
            if top == i {
                self_count += 1;
            } else {
                *votes.entry(top).or_insert(0) += 1;
            }
        }
        let mut ranked: Vec<(usize, usize)> = votes.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let top: Vec<String> = ranked.iter().take(5).map(|&(j, c)| format!("{:?}({c})", (filtered[j] as u8) as char)).collect();
        println!("{name}: {used} windows used, self-attention rate = {:.2}, top sink targets: {}", self_count as f32 / covered as f32, top.join(" "));
    }
}
