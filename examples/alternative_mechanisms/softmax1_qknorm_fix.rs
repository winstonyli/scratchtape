// Tests the fix softmax1_divergence_diagnosis.rs named but didn't build:
// QK-norm (L2-normalizing Q and K per position, before the dot product)
// - used in several real transformer architectures specifically to
// bound logit growth. That file found softmax1 causes a genuine,
// accelerating runaway (block0's Q/K weight norms and their own
// gradient grow together, unlike plain softmax's bounded gradient from
// identical seeds/data) that NaNs by step 775 at lr=0.03. If unbounded
// raw-score growth is really the proximate mechanism, normalizing Q/K
// to unit length before the dot product bounds every score to
// [-1/sqrt(d_k), 1/sqrt(d_k)] regardless of how large the underlying
// weight norms get - a direct structural fix, not a schedule/warm-up
// change (consistent with that file's finding that this is a
// continuous runaway, not a discontinuous jump a ramp could smooth).
//
// TransformerBlock's own fields (q_heads/k_heads/etc, not just its Out
// struct) are still private - exposing them was a bigger, separate
// architectural question than this specific test needed, so this
// builds a small local block struct instead, using the same public
// Linear/LayerNorm layers and low-level tape ops (batched_matmul,
// softmax1, concat) TransformerBlock::forward_full itself is built
// from - not a new engine capability, just this file's own composition
// of existing ones, self-contained rather than requiring another
// src/nn.rs change to test one hypothesis.
use scratchtape::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use scratchtape::tensor::NdArray;

#[path = "../common/mod.rs"]
mod common;
use common::{encode_bytes, sample_window};

struct CustomBlock {
    ln1: LayerNorm,
    q_heads: Vec<Linear>,
    k_heads: Vec<Linear>,
    v_heads: Vec<Linear>,
    out_proj: Linear,
    ln2: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
}

struct CustomBlockOut {
    y: Var,
    ln1_out: LayerNormOut,
    q_outs: Vec<LinearOut>,
    k_outs: Vec<LinearOut>,
    v_outs: Vec<LinearOut>,
    out_proj_out: LinearOut,
    ln2_out: LayerNormOut,
    ffn1_out: LinearOut,
    ffn2_out: LinearOut,
}

impl CustomBlock {
    fn new(rng: &mut Rng, d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        let d_k = d_model / n_heads;
        Self {
            ln1: LayerNorm::new(d_model),
            q_heads: (0..n_heads).map(|_| Linear::new(rng, d_model, d_k)).collect(),
            k_heads: (0..n_heads).map(|_| Linear::new(rng, d_model, d_k)).collect(),
            v_heads: (0..n_heads).map(|_| Linear::new(rng, d_model, d_k)).collect(),
            out_proj: Linear::new(rng, d_model, d_model),
            ln2: LayerNorm::new(d_model),
            ffn1: Linear::new(rng, d_model, d_ff),
            ffn2: Linear::new(rng, d_ff, d_model),
        }
    }

    /// L2-normalizes each row (position) of `x` to unit length - the QK-
    /// norm step under test. A small eps before sqrt guards a
    /// theoretical all-zero row (never observed, but matches this
    /// project's existing eps-before-sqrt/log safety convention rather
    /// than assuming it can't happen).
    fn l2_normalize_rows(tape: &mut Tape, x: Var) -> Var {
        let sq = tape.mul(x, x);
        let sum_sq = tape.sum_last_axis(sq);
        let eps = tape.leaf(NdArray::new(vec![1e-8; tape.value(sum_sq).data.len()], tape.value(sum_sq).shape.clone()));
        let sum_sq_eps = tape.add(sum_sq, eps);
        let norm = tape.sqrt(sum_sq_eps);
        tape.div(x, norm)
    }

    fn forward(&self, tape: &mut Tape, x: Var, use_softmax1: bool, use_qknorm: bool, d_k: usize) -> CustomBlockOut {
        let seq_len = tape.value(x).shape[0];
        let mask = tape.leaf(causal_mask(seq_len));

        let ln1_out = self.ln1.forward(tape, x);
        let normed1 = ln1_out.y;

        let mut q_outs = Vec::with_capacity(self.q_heads.len());
        let mut k_outs = Vec::with_capacity(self.k_heads.len());
        let mut v_outs = Vec::with_capacity(self.v_heads.len());
        let mut head_outputs = Vec::with_capacity(self.q_heads.len());
        for h in 0..self.q_heads.len() {
            let q_out = self.q_heads[h].forward(tape, normed1);
            let k_out = self.k_heads[h].forward(tape, normed1);
            let v_out = self.v_heads[h].forward(tape, normed1);

            let (q_for_scores, k_for_scores) =
                if use_qknorm { (Self::l2_normalize_rows(tape, q_out.y), Self::l2_normalize_rows(tape, k_out.y)) } else { (q_out.y, k_out.y) };

            // batch_size=1 throughout - batched_matmul(..., transpose_b=true)
            // does Q@K^T in one call, same as TransformerBlock::forward_full,
            // avoiding a separate transpose+matmul double-borrow.
            let scores = tape.batched_matmul(q_for_scores, k_for_scores, 1, true);
            let scaled = tape.scale(scores, 1.0 / (d_k as f32).sqrt());
            let masked = tape.add(scaled, mask);
            let weights = if use_softmax1 { tape.softmax1(masked) } else { tape.softmax(masked) };
            head_outputs.push(tape.batched_matmul(weights, v_out.y, 1, false));

            q_outs.push(q_out);
            k_outs.push(k_out);
            v_outs.push(v_out);
        }

        let concat = tape.concat(&head_outputs);
        let out_proj_out = self.out_proj.forward(tape, concat);
        let x1 = tape.add(x, out_proj_out.y);

        let ln2_out = self.ln2.forward(tape, x1);
        let ffn1_out = self.ffn1.forward(tape, ln2_out.y);
        let hidden = tape.relu(ffn1_out.y);
        let ffn2_out = self.ffn2.forward(tape, hidden);
        let y = tape.add(x1, ffn2_out.y);

        CustomBlockOut { y, ln1_out, q_outs, k_outs, v_outs, out_proj_out, ln2_out, ffn1_out, ffn2_out }
    }

    fn apply_grad(&mut self, tape: &Tape, out: &CustomBlockOut, opt: &Sgd) {
        self.ln1.apply_grad(tape, &out.ln1_out, opt);
        for h in 0..self.q_heads.len() {
            self.q_heads[h].apply_grad(tape, &out.q_outs[h], opt);
            self.k_heads[h].apply_grad(tape, &out.k_outs[h], opt);
            self.v_heads[h].apply_grad(tape, &out.v_outs[h], opt);
        }
        self.out_proj.apply_grad(tape, &out.out_proj_out, opt);
        self.ln2.apply_grad(tape, &out.ln2_out, opt);
        self.ffn1.apply_grad(tape, &out.ffn1_out, opt);
        self.ffn2.apply_grad(tape, &out.ffn2_out, opt);
    }

    fn q_k_l2_norm(&self) -> f32 {
        self.q_heads.iter().chain(self.k_heads.iter()).map(|l| l2_norm(&l.w.data)).sum()
    }
}

struct ForwardOut {
    tok_out: EmbeddingOut,
    pos_out: EmbeddingOut,
    block_outs: Vec<CustomBlockOut>,
    ln_out: LayerNormOut,
    proj_out: LinearOut,
}

fn forward(
    tape: &mut Tape,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[CustomBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    input_ids: &[usize],
    use_softmax1: bool,
    use_qknorm: bool,
    d_k: usize,
) -> (Var, ForwardOut) {
    let positions: Vec<usize> = (0..input_ids.len()).collect();
    let tok_out = token_emb.forward(tape, input_ids);
    let pos_out = pos_emb.forward(tape, &positions);
    let mut x = tape.add(tok_out.y, pos_out.y);

    let mut block_outs = Vec::with_capacity(blocks.len());
    for block in blocks {
        let out = block.forward(tape, x, use_softmax1, use_qknorm, d_k);
        x = out.y;
        block_outs.push(out);
    }

    let ln_out = final_ln.forward(tape, x);
    let proj_out = output_proj.forward(tape, ln_out.y);
    let logits = proj_out.y;
    (logits, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
}

fn apply_grad(
    tape: &Tape,
    out: &ForwardOut,
    token_emb: &mut Embedding,
    pos_emb: &mut Embedding,
    blocks: &mut [CustomBlock],
    final_ln: &mut LayerNorm,
    output_proj: &mut Linear,
    opt: &Sgd,
) {
    token_emb.apply_grad(tape, &out.tok_out, opt);
    pos_emb.apply_grad(tape, &out.pos_out, opt);
    for (block, block_out) in blocks.iter_mut().zip(out.block_outs.iter()) {
        block.apply_grad(tape, block_out, opt);
    }
    final_ln.apply_grad(tape, &out.ln_out, opt);
    output_proj.apply_grad(tape, &out.proj_out, opt);
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Causal mask: -1e9 above the diagonal (blocks attending to future
/// positions), 0 on/below it. TransformerBlock::forward_full builds and
/// RefCell-caches its own version internally (private to nn.rs) - this
/// file can't reuse it, so it's rebuilt here directly. Correctness
/// matters more than the caching optimization for a diagnostic run
/// (leafed fresh every block/step - fine at this seq_len=64, n_blocks=4
/// scale, no attempt to match forward_full's cached-mask performance).
fn causal_mask(seq_len: usize) -> NdArray {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = -1e9;
        }
    }
    NdArray::new(data, vec![seq_len, seq_len])
}

fn run(label: &str, use_softmax1: bool, use_qknorm: bool, d_model: usize, n_heads: usize, d_ff: usize, seq_len: usize, n_blocks: usize, vocab_size: usize, corpus: &[usize], lr: f32, max_steps: usize) {
    let d_k = d_model / n_heads;
    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<CustomBlock> = (0..n_blocks).map(|_| CustomBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr };

    println!("\n=== {label} (softmax1={use_softmax1}, qknorm={use_qknorm}) ===");
    println!("columns: step | loss | block0 Q+K weight L2 | block0 Q+K grad L2");

    for step in 0..max_steps {
        let (input, target) = sample_window(&mut rng, corpus, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input, use_softmax1, use_qknorm, d_k);
        let loss = tape.cross_entropy(logits, &target);
        let loss_val = tape.value(loss).data[0];
        tape.backward(loss);

        if step % 50 == 0 || loss_val.is_nan() {
            let qk_norm = blocks[0].q_k_l2_norm();
            let qk_grad_norm: f32 = blocks[0]
                .q_heads
                .iter()
                .chain(blocks[0].k_heads.iter())
                .zip(out.block_outs[0].q_outs.iter().chain(out.block_outs[0].k_outs.iter()))
                .filter_map(|(_, o)| tape.grad(o.w))
                .map(|g| l2_norm(&g.data))
                .sum();
            println!("{step:>4} | {loss_val:.6} | {qk_norm:.3} | {qk_grad_norm:.3}");
            if loss_val.is_nan() {
                println!("NaN at step {step} - stopping");
                break;
            }
        }

        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    println!("=== {label}: completed {max_steps} steps without NaN ===");
}

fn main() {
    let corpus = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    let lr = 0.03;
    let max_steps = 2000;

    println!("QK-norm fix test: d_model={d_model} n_heads={n_heads} n_blocks={n_blocks} lr={lr}");
    println!("Does L2-normalizing Q/K before the dot product (bounding every score to [-1,1]/sqrt(d_k)) prevent softmax1_divergence_diagnosis.rs's runaway?");

    run("softmax1, no qknorm (reproduce the known NaN)", true, false, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, &corpus, lr, max_steps);
    run("softmax1 + qknorm (the fix under test)", true, true, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, &corpus, lr, max_steps);
}
