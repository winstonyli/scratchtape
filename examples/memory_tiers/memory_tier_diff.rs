// Weight-drift diffing: turns this project's existing loss-based
// forgetting measurement into a weight-based one. Loads the phase-1
// baseline checkpoint plus both phase-2 outcomes (memory_tier_load.rs's
// with-replay result, memory_tier_load_no_replay.rs's no-replay result)
// and computes per-layer L2 drift from baseline for each - does replay
// specifically protect certain layers, or shrink drift roughly uniformly
// across all of them?
//
// Segment boundaries are found by running the SAME from_flat sequence
// memory_tier_load.rs uses on the baseline checkpoint, capturing each
// call's offset range - reuses already-correct reconstruction logic
// instead of hand-deriving each struct's flat length separately, and the
// resulting objects are otherwise unused (only their offset bookkeeping
// matters here).
use scratchtape::nn::{Embedding, LayerNorm, Linear, TransformerBlock};
use std::fs;

const CHECKPOINT_PATH: &str = "memory_tier_checkpoint.txt";
const AFTER_REPLAY_CHECKPOINT_PATH: &str = "memory_tier_checkpoint_after_replay.txt";
const AFTER_NOREPLAY_CHECKPOINT_PATH: &str = "memory_tier_checkpoint_after_noreplay.txt";

const D_MODEL: usize = 32;
const N_HEADS: usize = 4;
const D_FF: usize = 64;
const SEQ_LEN: usize = 16;
const N_BLOCKS: usize = 2;
const VOCAB_SIZE: usize = 256;

fn read_flat(path: &str) -> Vec<f32> {
    let text = fs::read_to_string(path).unwrap_or_else(|_| panic!("couldn't read {path} - run memory_tier_save, memory_tier_load, and memory_tier_load_no_replay first"));
    text.split_whitespace().map(|s| s.parse().expect("bad float in checkpoint")).collect()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

fn main() {
    let baseline = read_flat(CHECKPOINT_PATH);
    let after_replay = read_flat(AFTER_REPLAY_CHECKPOINT_PATH);
    let after_noreplay = read_flat(AFTER_NOREPLAY_CHECKPOINT_PATH);
    assert_eq!(baseline.len(), after_replay.len(), "checkpoint size mismatch - architecture must match across all three files");
    assert_eq!(baseline.len(), after_noreplay.len(), "checkpoint size mismatch - architecture must match across all three files");

    let mut offset = 0usize;
    let mut segments: Vec<(String, usize, usize)> = Vec::new();

    let start = offset;
    let _ = Embedding::from_flat(&baseline, &mut offset, VOCAB_SIZE, D_MODEL);
    segments.push(("token_emb".to_string(), start, offset));

    let start = offset;
    let _ = Embedding::from_flat(&baseline, &mut offset, SEQ_LEN, D_MODEL);
    segments.push(("pos_emb".to_string(), start, offset));

    for i in 0..N_BLOCKS {
        let start = offset;
        let _ = TransformerBlock::from_flat(&baseline, &mut offset, D_MODEL, N_HEADS, D_FF);
        segments.push((format!("block {i}"), start, offset));
    }

    let start = offset;
    let _ = LayerNorm::from_flat(&baseline, &mut offset, D_MODEL);
    segments.push(("final_ln".to_string(), start, offset));

    let start = offset;
    let _ = Linear::from_flat(&baseline, &mut offset, D_MODEL, VOCAB_SIZE);
    segments.push(("output_proj".to_string(), start, offset));
    assert_eq!(offset, baseline.len(), "checkpoint had leftover/missing floats - architecture mismatch");

    println!("per-layer L2 weight drift from the phase-1 baseline (replay | no-replay | replay's fraction of no-replay drift):");
    let (mut total_replay, mut total_noreplay) = (0.0f32, 0.0f32);
    for (name, start, end) in &segments {
        let d_replay = l2_distance(&baseline[*start..*end], &after_replay[*start..*end]);
        let d_noreplay = l2_distance(&baseline[*start..*end], &after_noreplay[*start..*end]);
        let ratio = if d_noreplay == 0.0 { 0.0 } else { d_replay / d_noreplay };
        total_replay += d_replay * d_replay;
        total_noreplay += d_noreplay * d_noreplay;
        println!("  {name}: {d_replay:.4} | {d_noreplay:.4} | {ratio:.3}");
    }
    let overall_ratio = if total_noreplay == 0.0 { 0.0 } else { total_replay.sqrt() / total_noreplay.sqrt() };
    println!("  overall (all params): {:.4} | {:.4} | {overall_ratio:.3}", total_replay.sqrt(), total_noreplay.sqrt());
}
