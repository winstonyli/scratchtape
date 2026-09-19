use scratchtape::nn::{Linear, Rng};
use scratchtape::optim::{Adam, AdamState};
use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

/// Same XOR task as xor_mlp.rs, driven by Adam instead of SGD - a direct
/// comparison point. Drives l.w/l.b directly (bypasses Linear::apply_grad,
/// which is Sgd-specific) since Linear stays optimizer-agnostic.
fn main() {
    let x_data = NdArray::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let y_data = NdArray::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    let mut rng = Rng::new(1);
    let mut l1 = Linear::new(&mut rng, 2, 4);
    let mut l2 = Linear::new(&mut rng, 4, 1);

    let adam = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut w1_state = AdamState::zeros_like(&l1.w);
    let mut b1_state = AdamState::zeros_like(&l1.b);
    let mut w2_state = AdamState::zeros_like(&l2.w);
    let mut b2_state = AdamState::zeros_like(&l2.b);

    let n = 4.0f32;

    for epoch in 0..500 {
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

        adam.step(&mut l1.w, tape.grad(out1.w).unwrap(), &mut w1_state);
        adam.step(&mut l1.b, tape.grad(out1.b).unwrap(), &mut b1_state);
        adam.step(&mut l2.w, tape.grad(out2.w).unwrap(), &mut w2_state);
        adam.step(&mut l2.b, tape.grad(out2.b).unwrap(), &mut b2_state);

        if epoch % 50 == 0 {
            println!("epoch {epoch:>4}: loss = {:.6}", tape.value(loss).data[0]);
        }
    }

    let mut tape = Tape::new();
    let x_var = tape.leaf(x_data.clone());
    let out1 = l1.forward(&mut tape, x_var);
    let h = tape.relu(out1.y);
    let out2 = l2.forward(&mut tape, h);
    let pred = tape.value(out2.y);

    println!("\ninput -> pred (target)");
    for i in 0..4 {
        println!(
            "({:.0}, {:.0}) -> {:.4} ({:.0})",
            x_data.data[i * 2],
            x_data.data[i * 2 + 1],
            pred.data[i],
            y_data.data[i]
        );
    }
}
