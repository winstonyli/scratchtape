use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

/// Tests Evan Miller's "Attention Is Off By One" claim directly, not just
/// citing it: does softmax1 actually let attention express "nothing here,"
/// visible as lower total attention weight, versus plain softmax's forced
/// near-uniform distribution when nothing genuinely matches?
///
/// Same 4-slot memory as attention_recall.rs, but that demo's three cases
/// (exact match, noisy match, ambiguous-but-tied) all have genuinely
/// relevant content - softmax and softmax1 should behave similarly on all
/// three. The new "no good match" case here is the one that actually
/// exercises the distinguishing behavior.
fn main() {
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
            0.0, 0.5, 0.0, 0.5, // ambiguous blend of keys 1 and 3 - still relevant, just tied
            -0.3, -0.3, -0.3, -0.3, // no good match: negative dot product with every key
        ],
        vec![4, 4],
    );

    let d_k = 4.0f32;
    let labels = ["exact match (key 2)", "noisy query (~key 0)", "ambiguous (keys 1 & 3, still relevant)", "no good match (negative vs every key)"];

    let mut tape = Tape::new();
    let q = tape.leaf(queries.clone());
    let k = tape.leaf(keys.clone());
    let v = tape.leaf(values.clone());
    let kt = tape.transpose(k);
    let scores = tape.matmul(q, kt);
    let scaled = tape.scale(scores, 1.0 / d_k.sqrt());
    let weights_softmax = tape.softmax(scaled);
    let out_softmax = tape.matmul(weights_softmax, v);

    let mut tape1 = Tape::new();
    let q1 = tape1.leaf(queries.clone());
    let k1 = tape1.leaf(keys.clone());
    let v1 = tape1.leaf(values.clone());
    let kt1 = tape1.transpose(k1);
    let scores1 = tape1.matmul(q1, kt1);
    let scaled1 = tape1.scale(scores1, 1.0 / d_k.sqrt());
    let weights_softmax1 = tape1.softmax1(scaled1);
    let out_softmax1 = tape1.matmul(weights_softmax1, v1);

    let w = tape.value(weights_softmax);
    let w1 = tape1.value(weights_softmax1);
    let o = tape.value(out_softmax);
    let o1 = tape1.value(out_softmax1);

    for (i, label) in labels.iter().enumerate() {
        let row = &w.data[i * 4..i * 4 + 4];
        let row1 = &w1.data[i * 4..i * 4 + 4];
        let total: f32 = row.iter().sum();
        let total1: f32 = row1.iter().sum();
        println!("{label}");
        println!("  softmax:  weights {row:?}  (sum={total:.4})  ->  {:?}", &o.data[i * 4..i * 4 + 4]);
        println!("  softmax1: weights {row1:?}  (sum={total1:.4})  ->  {:?}", &o1.data[i * 4..i * 4 + 4]);
        println!();
    }
}
