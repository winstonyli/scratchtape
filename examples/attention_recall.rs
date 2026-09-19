use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

/// Pure forward-pass associative recall - no training, no gradient descent.
/// Directly tests the Modern Hopfield Networks claim flagged earlier in
/// this project's design log: softmax(QKᵀ/sqrt(d_k))V is mathematically one
/// step of continuous Hopfield energy-descent retrieval. 4 fixed memory
/// slots (key, value) pairs; query with a noisy version of one stored key;
/// check attention retrieves the matching clean value.
fn main() {
    // 4 memory slots, d_k = d_v = 4. Each key is a one-hot pattern scaled to
    // magnitude 4 (not 1) - unit-norm one-hot keys give only a 1-vs-0
    // dot-product gap, which the 1/sqrt(d_k) scaling then shrinks further,
    // leaving softmax nearly uniform and the retrieval unreadable. A real
    // trained Q/K projection would learn an appropriately separated scale;
    // here that scale is picked by hand for the same reason.
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

    // Queries: exact key 2, a noisy/partial version of key 0, and an
    // ambiguous 50/50 blend of keys 1 and 3 - tests exact recall,
    // noise-robust recall, and genuine ambiguity respectively.
    let queries = NdArray::new(
        vec![
            0.0, 0.0, 1.0, 0.0, // exact match: key 2
            0.9, 0.1, 0.0, 0.1, // noisy version of key 0
            0.0, 0.5, 0.0, 0.5, // ambiguous blend of keys 1 and 3
        ],
        vec![3, 4],
    );

    let d_k = 4.0f32;
    let mut tape = Tape::new();
    let q = tape.leaf(queries.clone());
    let k = tape.leaf(keys.clone());
    let v = tape.leaf(values.clone());

    let kt = tape.transpose(k);
    let scores = tape.matmul(q, kt);
    let scaled = tape.scale(scores, 1.0 / d_k.sqrt());
    let weights = tape.softmax(scaled);
    let out = tape.matmul(weights, v);

    let w = tape.value(weights);
    let retrieved = tape.value(out);

    let labels = ["exact match (key 2)", "noisy query (~key 0)", "ambiguous (keys 1 & 3)"];
    for (i, label) in labels.iter().enumerate() {
        let row = &w.data[i * 4..i * 4 + 4];
        let out_row = &retrieved.data[i * 4..i * 4 + 4];
        println!("{label}");
        println!("  attention weights: {row:?}");
        println!("  retrieved value:   {out_row:?}\n");
    }
}
