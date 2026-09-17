use engine::nn::Linear;
use engine::tape::Tape;
use engine::tensor::NdArray;

/// Multi-head attention over the same associative-recall setup as
/// attention_recall.rs, split into 2 heads of 2 dims each.
///
/// Important correction versus the naive expectation: this does NOT
/// reduce to the single-head result. Each head computes its own dot
/// product over only ITS OWN 2 dims, so each head's softmax normalizes
/// over a different (smaller) score distribution than a single head
/// scoring across all 4 dims would - splitting the scoring itself changes
/// what gets computed, it isn't just a reparameterization of the same
/// computation. So this demo shows each head's own (legitimately
/// different) attention pattern, not an equivalence check.
///
/// Per-head Q/K/V here are hand-set identity-slice projections (head 0
/// reads dims 0-1, head 1 reads dims 2-3) rather than h separate learned
/// Linear layers - kept minimal since nothing here is trained; the point
/// is to exercise the split -> per-head-attention -> Concat plumbing, not
/// to demonstrate learned per-head specialization.
fn main() {
    let d_model = 4usize;
    let n_heads = 2usize;
    let d_k = d_model / n_heads;

    let keys = NdArray::new(
        vec![
            4.0, 0.0, 0.0, 0.0, //
            0.0, 4.0, 0.0, 0.0, //
            0.0, 0.0, 4.0, 0.0, //
            0.0, 0.0, 0.0, 4.0,
        ],
        vec![4, 4],
    );
    let values = NdArray::new(
        vec![
            10.0, 0.0, 0.0, 0.0, //
            0.0, 20.0, 0.0, 0.0, //
            0.0, 0.0, 30.0, 0.0, //
            0.0, 0.0, 0.0, 40.0,
        ],
        vec![4, 4],
    );
    let queries = NdArray::new(
        vec![
            0.0, 0.0, 1.0, 0.0, // exact match: key 2
            0.9, 0.1, 0.0, 0.1, // noisy version of key 0
            0.0, 0.5, 0.0, 0.5, // ambiguous blend of keys 1 and 3
        ],
        vec![3, 4],
    );

    let mut tape = Tape::new();
    let q = tape.leaf(queries.clone());
    let k = tape.leaf(keys.clone());
    let v = tape.leaf(values.clone());

    let identity_slice = |start: usize| -> Linear {
        let mut w = vec![0.0f32; d_model * d_k];
        for i in 0..d_k {
            w[(start + i) * d_k + i] = 1.0;
        }
        Linear { w: NdArray::new(w, vec![d_model, d_k]), b: NdArray::zeros(vec![d_k]) }
    };

    let mut head_outputs = Vec::with_capacity(n_heads);
    let mut head_weights = Vec::with_capacity(n_heads);

    for head in 0..n_heads {
        let start = head * d_k;
        let proj = identity_slice(start);
        let q_h = proj.forward(&mut tape, q).y;
        let k_h = proj.forward(&mut tape, k).y;
        let v_h = proj.forward(&mut tape, v).y;

        let kt = tape.transpose(k_h);
        let scores = tape.matmul(q_h, kt);
        let scaled = tape.scale(scores, 1.0 / (d_k as f32).sqrt());
        let weights = tape.softmax(scaled);
        let out = tape.matmul(weights, v_h);

        head_weights.push(weights);
        head_outputs.push(out);
    }

    let combined = tape.concat(&head_outputs);

    let labels = ["exact match (key 2)", "noisy query (~key 0)", "ambiguous (keys 1 & 3)"];
    for head in 0..n_heads {
        let w = tape.value(head_weights[head]);
        println!("head {head} (reads dims {}..{}):", head * d_k, head * d_k + d_k);
        for (i, label) in labels.iter().enumerate() {
            let row = &w.data[i * 4..i * 4 + 4];
            println!("  {label}: {row:?}");
        }
    }

    let out = tape.value(combined);
    println!("\ncombined (concatenated) retrieval per query:");
    for (i, label) in labels.iter().enumerate() {
        let row = &out.data[i * d_model..i * d_model + d_model];
        println!("  {label}: {row:?}");
    }
}
