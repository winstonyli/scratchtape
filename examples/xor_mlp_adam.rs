use scratchtape::nn::{Linear, Rng};
use scratchtape::optim::{Adam, Optimizer};
use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

/// Same XOR task as xor_mlp.rs, driven by Adam instead of SGD - a direct
/// comparison point. Uses Linear::apply_grad_with, generic over Optimizer,
/// rather than hand-rolling adam.step per w/b field.
fn main() {
    let x_data = NdArray::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let y_data = NdArray::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    let mut rng = Rng::new(1);
    let mut l1 = Linear::new(&mut rng, 2, 4);
    let mut l2 = Linear::new(&mut rng, 4, 1);

    let adam = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut w1_state = Adam::new_state(&l1.w.shape);
    let mut b1_state = Adam::new_state(&l1.b.shape);
    let mut w2_state = Adam::new_state(&l2.w.shape);
    let mut b2_state = Adam::new_state(&l2.b.shape);

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

        l1.apply_grad_with(&tape, &out1, &adam, &mut w1_state, &mut b1_state);
        l2.apply_grad_with(&tape, &out2, &adam, &mut w2_state, &mut b2_state);

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
