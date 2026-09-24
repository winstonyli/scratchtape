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
//   training_recipe_check <name> <softmax1 0|1> <batch> <lr> [windows]
// `windows` is the total training budget in 64-byte windows (default
// 64000, tiny_lm_corpus.rs's), so batch size changes steps, not data:
// steps = windows / batch. Progress streams to stdout as it happens; the
// final model is saved to runs/<name>.ckpt.
//
// Same architecture and init stream as tiny_lm_corpus.rs (seed 1), so
// batch=1 lr=0.3 reproduces attention_uniformity_check.rs's plain (1.852)
// and softmax1 (1.803) checkpoints' recipe.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use std::io::Write;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, encode_bytes, flatten_all, sample_window};

const D_MODEL: usize = 128;
const N_HEADS: usize = 8;
const D_FF: usize = 256;
const SEQ_LEN: usize = 64;
const N_BLOCKS: usize = 4;
const VOCAB: usize = 256;

struct Model {
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    final_ln: LayerNorm,
    output_proj: Linear,
}

/// `input_ids` holds `batch` windows stacked row-wise; positions restart at
/// 0 for each window (see tiny_lm_batched.rs).
fn forward(tape: &mut Tape, m: &Model, input_ids: &[usize], batch: usize, softmax1: bool) -> (Var, ForwardOut) {
    let positions: Vec<usize> = (0..batch).flat_map(|_| 0..SEQ_LEN).collect();
    let tok_out = m.token_emb.forward(tape, input_ids);
    let pos_out = m.pos_emb.forward(tape, &positions);
    let mut x = tape.add(tok_out.y, pos_out.y);
    let mut block_outs = Vec::with_capacity(m.blocks.len());
    for block in &m.blocks {
        let out = block.forward_full(tape, x, batch, softmax1);
        x = out.y;
        block_outs.push(out);
    }
    let ln_out = m.final_ln.forward(tape, x);
    let proj_out = m.output_proj.forward(tape, ln_out.y);
    (proj_out.y, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
}

/// Mean CE over every non-overlapping 64-byte window of `corpus` - the
/// same windowing ngram_baseline.rs and attention_uniformity_check.rs use.
fn full_ce(m: &Model, corpus: &[usize], softmax1: bool) -> f32 {
    let starts: Vec<usize> = (0..corpus.len() - SEQ_LEN).step_by(SEQ_LEN).collect();
    let mut total = 0.0;
    for &s in &starts {
        let mut tape = Tape::new();
        let (logits, _) = forward(&mut tape, m, &corpus[s..s + SEQ_LEN], 1, softmax1);
        let loss = tape.cross_entropy(logits, &corpus[s + 1..s + SEQ_LEN + 1]);
        total += tape.value(loss).data[0];
    }
    total / starts.len() as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 5, "usage: training_recipe_check <name> <softmax1 0|1> <batch> <lr> [windows]");
    let name = &args[1];
    let softmax1 = args[2] == "1";
    let batch: usize = args[3].parse().unwrap();
    let lr: f32 = args[4].parse().unwrap();
    let windows: usize = args.get(5).map(|w| w.parse().unwrap()).unwrap_or(64000);
    let steps = windows / batch;
    let eval_every = (8000 / batch).max(1);

    let full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let split = (full.len() as f32 * 0.9) as usize;
    let (train, held_out) = full.split_at(split);
    // A fixed 371-window slice of train, for a like-for-like overfitting gap.
    let train_probe = &train[..held_out.len()];

    let mut rng = Rng::new(1);
    let mut m = Model {
        token_emb: Embedding::new(&mut rng, VOCAB, D_MODEL),
        pos_emb: Embedding::new(&mut rng, SEQ_LEN, D_MODEL),
        blocks: (0..N_BLOCKS).map(|_| TransformerBlock::new(&mut rng, D_MODEL, N_HEADS, D_FF)).collect(),
        final_ln: LayerNorm::new(D_MODEL),
        output_proj: Linear::new(&mut rng, D_MODEL, VOCAB),
    };
    let opt = Sgd { lr };

    println!("run {name}: pid {} softmax1={softmax1} batch={batch} lr={lr} windows={windows} steps={steps}", std::process::id());
    println!("columns: step | windows seen | train-probe CE | held-out CE (deterministic) | elapsed");
    let start = Instant::now();
    for step in 0..=steps {
        if step % eval_every == 0 || step == steps {
            println!(
                "{step:>6} | {:>6} | {:.4} | {:.4} | {:.0}s",
                step * batch,
                full_ce(&m, train_probe, softmax1),
                full_ce(&m, held_out, softmax1),
                start.elapsed().as_secs_f32()
            );
            std::io::stdout().flush().unwrap();
        }
        if step == steps {
            break;
        }
        let mut input = Vec::with_capacity(batch * SEQ_LEN);
        let mut target = Vec::with_capacity(batch * SEQ_LEN);
        for _ in 0..batch {
            let (i, t) = sample_window(&mut rng, train, SEQ_LEN);
            input.extend(i);
            target.extend(t);
        }
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &m, &input, batch, softmax1);
        let loss = tape.cross_entropy(logits, &target);
        if tape.value(loss).data[0].is_nan() {
            println!("diverged to NaN at step {step}");
            return;
        }
        tape.backward(loss);
        apply_grad(&tape, &out, &mut m.token_emb, &mut m.pos_emb, &mut m.blocks, &mut m.final_ln, &mut m.output_proj, &opt);
    }

    std::fs::create_dir_all("runs").unwrap();
    let flat = flatten_all(&m.token_emb, &m.pos_emb, &m.blocks, &m.final_ln, &m.output_proj);
    let text = flat.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    std::fs::write(format!("runs/{name}.ckpt"), text).unwrap();
    println!("saved runs/{name}.ckpt ({:.0}s total)", start.elapsed().as_secs_f32());
}
