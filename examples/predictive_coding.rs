use engine::nn::Rng;
use engine::optim::{Adam, AdamState};
use engine::tape::Tape;
use engine::tensor::NdArray;

/// Predictive Coding (Whittington & Bogacz 2017 formulation): bypasses Tape
/// entirely - not because it needs zero derivatives (that's ES's reason),
/// but because every derivative here is local and closed-form, never
/// chained through a global reverse-mode graph. Architecture mirrors
/// xor_mlp.rs (2 -> 4 (relu) -> 1) minus bias terms, kept out to keep the
/// closed-form update rules clean rather than cluttered with extra terms -
/// XOR is still solvable without bias (the (0,0)->0 case is trivially
/// satisfied by linearity through the origin).
///
/// e1 = x1 - relu(x0 @ W0)   (hidden layer's own prediction error)
/// e2 = x2 - x1 @ W1         (output error; x2 is clamped to the target)
/// F  = 0.5*(sum(e1^2) + sum(e2^2))
///
/// Inference (relaxation): only x1 is free. x0, x2 stay clamped.
///   dF/dx1 = e1 - e2 @ W1ᵀ   (output has no relu, so f'(z1) = 1, drops out)
///   x1 -= lr_inference * dF/dx1, repeated T times per training step.
///
/// Weight update, after settling:
///   dF/dW0 = -(x0ᵀ @ (e1 * relu'(z0)))
///   dF/dW1 = -(x1ᵀ @ e2)
fn relu_grad(z: &NdArray) -> NdArray {
    NdArray { data: z.data.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect(), shape: z.shape.clone() }
}

/// Settles x1 for `steps` relaxation iterations, then returns the resulting
/// weight gradients (dF/dW0, dF/dW1) at the settled point.
fn pc_grads(x0: &NdArray, target: &NdArray, w0: &NdArray, w1: &NdArray, steps: usize, lr_inf: f32) -> (NdArray, NdArray) {
    let z0 = x0.matmul(w0);
    let pred1 = z0.relu();
    let mut x1 = pred1.clone(); // seed via one feedforward pass, not zeros/random

    for _ in 0..steps {
        let e1 = x1.sub(&pred1);
        let z1 = x1.matmul(w1);
        let e2 = target.sub(&z1);
        let d_f_dx1 = e1.sub(&e2.matmul(&w1.transpose()));
        x1 = x1.sub(&d_f_dx1.scale(lr_inf));
    }

    let e1 = x1.sub(&pred1);
    let z1 = x1.matmul(w1);
    let e2 = target.sub(&z1);
    let d_f_dw0 = x0.transpose().matmul(&e1.mul(&relu_grad(&z0))).scale(-1.0);
    let d_f_dw1 = x1.transpose().matmul(&e2).scale(-1.0);
    (d_f_dw0, d_f_dw1)
}

/// Exact backprop gradient via the existing, already-verified Tape - the
/// ground truth PC's relaxation is supposed to approximate as T grows.
/// Loss = 0.5*sum((pred-target)^2), matching F's e2-term scale exactly so
/// the two gradients are numerically comparable, not just directionally.
fn backprop_grads(x0: &NdArray, target: &NdArray, w0: &NdArray, w1: &NdArray) -> (NdArray, NdArray) {
    let mut tape = Tape::new();
    let x0v = tape.leaf(x0.clone());
    let w0v = tape.leaf(w0.clone());
    let w1v = tape.leaf(w1.clone());
    let tv = tape.leaf(target.clone());
    let z0 = tape.matmul(x0v, w0v);
    let h = tape.relu(z0);
    let pred = tape.matmul(h, w1v);
    let diff = tape.sub(pred, tv);
    let sq = tape.mul(diff, diff);
    let sum = tape.sum(sq);
    let loss = tape.scale(sum, 0.5);
    tape.backward(loss);
    (tape.grad(w0v).unwrap().clone(), tape.grad(w1v).unwrap().clone())
}

fn rel_error(a: &NdArray, b: &NdArray) -> f32 {
    let diff = a.sub(b);
    let num = diff.mul(&diff).sum().data[0].sqrt();
    let den = b.mul(b).sum().data[0].sqrt();
    num / den.max(1e-8)
}

fn main() {
    let x0 = NdArray::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let target = NdArray::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    let mut rng = Rng::new(1);
    let limit0 = (6.0f32 / 2.0).sqrt();
    let w0_init = NdArray::new((0..8).map(|_| (rng.next_f32() * 2.0 - 1.0) * limit0).collect(), vec![2, 4]);
    let limit1 = (6.0f32 / 4.0).sqrt();
    let w1_init = NdArray::new((0..4).map(|_| (rng.next_f32() * 2.0 - 1.0) * limit1).collect(), vec![4, 1]);

    // Flagship experiment: does PC's gradient converge toward backprop's
    // exact gradient as relaxation step count T grows? Same fixed weights,
    // same input/target, only T varies - directly tests the literature's
    // core equivalence claim with real numbers instead of citing it.
    //
    // MEASURED RESULT DOES NOT MATCH THE CLAIM AS STATED: relative error is
    // SMALLEST at T=1 (~0.27 for dW1) and gets WORSE, then plateaus, as T
    // grows (~0.75 by T=20+) - the opposite direction from "converges to
    // backprop as T->infinity". The closed-form update rules above were
    // hand-rederived independently and match this code, so this isn't a
    // sign/transpose bug. Best working explanation: backprop's gradient
    // implicitly assumes x1 = relu(x0@W0) exactly (e1 = 0, no drift), while
    // PC's relaxation deliberately lets x1 drift away from that to reduce
    // e2 - more relaxation steps means MORE drift, not less, which would
    // explain a growing (not shrinking) gap. The precise equivalence in the
    // literature most likely depends on precision-weighting the error terms
    // (each error scaled by an inverse-variance term), which this minimal
    // version omits. Parked as an open question rather than force-fit -
    // genuine finding, not glossed over.
    let (bp_w0, bp_w1) = backprop_grads(&x0, &target, &w0_init, &w1_init);
    println!("backprop gradient (ground truth):");
    println!("  dW0 = {:?}", bp_w0.data);
    println!("  dW1 = {:?}", bp_w1.data);
    println!("\nPC gradient relative error vs backprop, by relaxation step count T:");
    for &t in &[1usize, 5, 20, 50, 100, 300] {
        let (pc_w0, pc_w1) = pc_grads(&x0, &target, &w0_init, &w1_init, t, 0.1);
        println!(
            "  T={t:>3}: dW0 rel err = {:.6}, dW1 rel err = {:.6}",
            rel_error(&pc_w0, &bp_w0),
            rel_error(&pc_w1, &bp_w1)
        );
    }

    // Secondary: actually train XOR via repeated relax-then-update steps,
    // reusing the same pc_grads mechanics. T=20 per step - enough for a
    // reasonable approximation without paying for full convergence every step.
    let mut w0 = w0_init.clone();
    let mut w1 = w1_init.clone();
    let adam = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut w0_state = AdamState::zeros_like(&w0);
    let mut w1_state = AdamState::zeros_like(&w1);

    println!("\ntraining XOR via predictive coding (T=20 relaxation steps/epoch):");
    for epoch in 0..1000 {
        let (d_w0, d_w1) = pc_grads(&x0, &target, &w0, &w1, 20, 0.1);
        adam.step(&mut w0, &d_w0, &mut w0_state);
        adam.step(&mut w1, &d_w1, &mut w1_state);

        if epoch % 100 == 0 {
            let pred = x0.matmul(&w0).relu().matmul(&w1);
            let diff = pred.sub(&target);
            let loss = 0.5 * diff.mul(&diff).sum().data[0];
            println!("  epoch {epoch:>4}: loss = {loss:.6}");
        }
    }

    let pred = x0.matmul(&w0).relu().matmul(&w1);
    println!("\ninput -> pred (target)");
    for i in 0..4 {
        println!("  ({:.0}, {:.0}) -> {:.4} ({:.0})", x0.data[i * 2], x0.data[i * 2 + 1], pred.data[i], target.data[i]);
    }
}
