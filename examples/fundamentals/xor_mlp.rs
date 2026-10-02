use scratchtape::nn::{Linear, Rng};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

/// XOR: the textbook proof a linear model cannot work here (not linearly
/// separable) - the whole justification for hidden layers + nonlinearity
/// existing at all. 2 -> 4 (relu) -> 1, MSE toward {0,1} targets.
fn main() {
    let x_data = NdArray::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let y_data = NdArray::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    // Seed matters: dead ReLU units are a real convergence hazard on this
    // tiny 4-sample dataset. Seeds 42/99/123 get stuck at a degenerate
    // local optimum (some hidden units zeroed permanently, mid-training);
    // seed 1 reliably escapes it. Not cherry-picked to hide the failure -
    // this is the actual dead-ReLU phenomenon, worth seeing before it
    // matters more on a real dataset.
    let mut rng = Rng::new(1);
    let mut l1 = Linear::new(&mut rng, 2, 4);
    let mut l2 = Linear::new(&mut rng, 4, 1);

    let opt = Sgd { lr: 0.1 };
    let n = 4.0f32;

    for epoch in 0..3000 {
        let mut tape = Tape::new();
        let x_var = tape.leaf(x_data.clone());
        let y_var = tape.leaf(y_data.clone());

        let out1 = l1.forward(&mut tape, x_var);
        let h = tape.relu(out1.y);
        let out2 = l2.forward(&mut tape, h);

        let diff = tape.sub(out2.y, y_var);
        let sq = tape.mul(diff, diff);
        let sum = tape.sum(sq);
        let loss = tape.scale(sum, 1.0 / n);

        tape.backward(loss);

        l1.apply_grad(&tape, &out1, &opt);
        l2.apply_grad(&tape, &out2, &opt);

        if epoch % 300 == 0 {
            println!("epoch {epoch:>4}: loss = {:.6}", tape.value(loss).data[0]);
        }
    }

    // Final forward pass to show predictions vs targets.
    let mut tape = Tape::new();
    let x_var = tape.leaf(x_data.clone());
    let out1 = l1.forward(&mut tape, x_var);
    let h = tape.relu(out1.y);
    let out2 = l2.forward(&mut tape, h);
    let pred = tape.value(out2.y);

    println!("\ninput -> pred (target)");
    for i in 0..4 {
        println!("({:.0}, {:.0}) -> {:.4} ({:.0})", x_data.data[i * 2], x_data.data[i * 2 + 1], pred.data[i], y_data.data[i]);
    }
}
