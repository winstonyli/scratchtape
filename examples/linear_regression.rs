use scratchtape::nn::Rng;
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

/// Ground truth: y = 3x + 2, plus small noise. Simplest possible task that
/// exercises the whole stack: MatMul, broadcast Add (bias), Sub, Mul, Sum, SGD.
fn main() {
    let n = 64usize;
    let mut rng = Rng::new(12345);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.next_f32() * 10.0 - 5.0;
        let noise = (rng.next_f32() - 0.5) * 0.1;
        xs.push(x);
        ys.push(3.0 * x + 2.0 + noise);
    }

    let x_data = NdArray::new(xs, vec![n, 1]);
    let y_data = NdArray::new(ys, vec![n, 1]);

    let mut w = NdArray::new(vec![0.0], vec![1, 1]);
    let mut b = NdArray::new(vec![0.0], vec![1]);

    let opt = Sgd { lr: 0.01 };

    for epoch in 0..200 {
        // Fresh tape every step - dynamic/define-by-run graph, matches how
        // PyTorch rebuilds its graph each forward pass rather than reusing one.
        let mut tape = Tape::new();
        let x_var = tape.leaf(x_data.clone());
        let y_var = tape.leaf(y_data.clone());
        let w_var = tape.leaf(w.clone());
        let b_var = tape.leaf(b.clone());

        let mm = tape.matmul(x_var, w_var);
        let pred = tape.add(mm, b_var);
        let diff = tape.sub(pred, y_var);
        let sq = tape.mul(diff, diff);
        let sum = tape.sum(sq);
        let loss = tape.scale(sum, 1.0 / n as f32);

        tape.backward(loss);

        let w_grad = tape.grad(w_var).unwrap().clone();
        let b_grad = tape.grad(b_var).unwrap().clone();
        opt.step(&mut w, &w_grad);
        opt.step(&mut b, &b_grad);

        if epoch % 20 == 0 {
            println!(
                "epoch {epoch:>3}: loss = {:.6}, w = {:.4}, b = {:.4}",
                tape.value(loss).data[0],
                w.data[0],
                b.data[0]
            );
        }
    }

    println!("final: w = {:.4} (target 3.0), b = {:.4} (target 2.0)", w.data[0], b.data[0]);
}
