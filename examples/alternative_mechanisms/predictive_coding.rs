use scratchtape::nn::Rng;
use scratchtape::optim::{Adam, AdamState};
use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

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

/// Precision-weighted revisit of the gap above: each layer's error term
/// gets its own fixed scalar precision (inverse-variance) weight, per the
/// literature's free-energy formulation. Fixed hyperparameters here, not
/// learned/jointly-trained per-unit precision (the real free-energy PC
/// formulation) - proportionate scope for testing whether precision-
/// weighting explains the gap at all, before committing to that bigger
/// build. Parked: revisit with genuinely learned precision later if this
/// fixed-value test is inconclusive rather than decisive either way.
fn pc_grads_precision(x0: &NdArray, target: &NdArray, w0: &NdArray, w1: &NdArray, steps: usize, lr_inf: f32, sigma1_inv: f32, sigma2_inv: f32) -> (NdArray, NdArray) {
    let z0 = x0.matmul(w0);
    let pred1 = z0.relu();
    let mut x1 = pred1.clone();

    for _ in 0..steps {
        let e1 = x1.sub(&pred1);
        let z1 = x1.matmul(w1);
        let e2 = target.sub(&z1);
        let d_f_dx1 = e1.scale(sigma1_inv).sub(&e2.matmul(&w1.transpose()).scale(sigma2_inv));
        x1 = x1.sub(&d_f_dx1.scale(lr_inf));
    }

    let e1 = x1.sub(&pred1);
    let z1 = x1.matmul(w1);
    let e2 = target.sub(&z1);
    let d_f_dw0 = x0.transpose().matmul(&e1.mul(&relu_grad(&z0))).scale(-sigma1_inv);
    let d_f_dw1 = x1.transpose().matmul(&e2).scale(-sigma2_inv);
    (d_f_dw0, d_f_dw1)
}

/// Same sigma2_inv scaling applied to backprop's reference loss - keeps the
/// comparison on the same footing (a constant multiplier, doesn't change
/// backprop's gradient direction) rather than comparing differently-scaled
/// quantities.
fn backprop_grads_precision(x0: &NdArray, target: &NdArray, w0: &NdArray, w1: &NdArray, sigma2_inv: f32) -> (NdArray, NdArray) {
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
    let loss = tape.scale(sum, 0.5 * sigma2_inv);
    tape.backward(loss);
    (tape.grad(w0v).unwrap().clone(), tape.grad(w1v).unwrap().clone())
}

/// Same relaxation as pc_grads_precision, but also returns the settled
/// error terms e1/e2 - needed to update precision itself via its closed-
/// form MLE. pc_grads_precision's own signature stays unchanged (the fixed-
/// value sweep above already depends on its existing 2-tuple return); this
/// is a separate function, not a refactor, matching this file's existing
/// pattern of pc_grads/pc_grads_precision as separate top-level functions
/// rather than one sharing a helper for the relaxation loop.
fn pc_grads_and_errors(
    x0: &NdArray,
    target: &NdArray,
    w0: &NdArray,
    w1: &NdArray,
    steps: usize,
    lr_inf: f32,
    sigma1_inv: f32,
    sigma2_inv: f32,
) -> (NdArray, NdArray, NdArray, NdArray) {
    let z0 = x0.matmul(w0);
    let pred1 = z0.relu();
    let mut x1 = pred1.clone();

    for _ in 0..steps {
        let e1 = x1.sub(&pred1);
        let z1 = x1.matmul(w1);
        let e2 = target.sub(&z1);
        let d_f_dx1 = e1.scale(sigma1_inv).sub(&e2.matmul(&w1.transpose()).scale(sigma2_inv));
        x1 = x1.sub(&d_f_dx1.scale(lr_inf));
    }

    let e1 = x1.sub(&pred1);
    let z1 = x1.matmul(w1);
    let e2 = target.sub(&z1);
    let d_f_dw0 = x0.transpose().matmul(&e1.mul(&relu_grad(&z0))).scale(-sigma1_inv);
    let d_f_dw1 = x1.transpose().matmul(&e2).scale(-sigma2_inv);
    (d_f_dw0, d_f_dw1, e1, e2)
}

/// MAP precision estimate under a Gamma(alpha, beta) prior (the conjugate
/// prior for a zero-mean Gaussian's precision) instead of raw MLE - the
/// fix surveyed but not built when this file was parked. Prior mode fixed
/// at 1.0 (matching sigma_inv's own pre-warmup default, no new "what
/// should the prior believe" hyperparameter needed) with `pseudo_n`
/// pseudo-observations worth of confidence in that belief:
///   Pi_MAP = (pseudo_n + N) / (pseudo_n + sum(e^2))
/// Derivation: Gamma(alpha=pseudo_n/2+1, beta=pseudo_n/2) has mode
/// (alpha-1)/beta = 1.0 by construction; the posterior after N Gaussian
/// observations with sum-of-squares ss is Gamma(alpha+N/2, beta+ss/2),
/// whose mode is exactly the formula above (same MLE derivation
/// update_precision's original doc comment used, just with a prior in the
/// numerator and denominator instead of none).
///
/// This is a fundamentally different fix from the old clamp, not a
/// tuned replacement: it's self-bounding rather than hard-walled. As
/// ss -> 0 (near-perfect fit), Pi_MAP -> (pseudo_n+N)/pseudo_n, a finite
/// ceiling the formula approaches smoothly - no discontinuous wall to hit,
/// so no single-step jump when crossing whatever threshold a clamp would
/// have imposed. As ss grows, Pi_MAP behaves like a softened MLE (the
/// +pseudo_n floor in the denominator prevents it from crashing quite as
/// fast toward zero as raw MLE would), so it still tempers - not
/// eliminates - the large-error collapse risk that motivated the original
/// warm-up. Kept EMA-smoothed for the same reason as before: a fresh
/// single-step estimate would chase noise, not the trend.
fn update_precision_map(e: &NdArray, ema_ssq: &mut f32, decay: f32, pseudo_n: f32) -> f32 {
    let ssq = e.mul(e).sum().data[0];
    *ema_ssq = decay * *ema_ssq + (1.0 - decay) * ssq;
    let n = e.data.len() as f32;
    (pseudo_n + n) / (pseudo_n + *ema_ssq)
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
    // MEASURED RESULT DOES NOT MATCH THE CLAIM AS STATED: dW1's relative
    // error is SMALLEST at T=1 (~0.27) and gets WORSE, then plateaus, as T
    // grows (~0.75 by T=20+) - the opposite direction from "converges to
    // backprop as T->infinity" (dW0 behaves differently - its error actually
    // shrinks with T even here, from 0.90 to a 0.67 plateau - a real, mixed
    // result, not uniformly wrong). The closed-form update rules above were
    // hand-rederived independently and match this code, so this isn't a
    // sign/transpose bug. Best working explanation: backprop's gradient
    // implicitly assumes x1 = relu(x0@W0) exactly (e1 = 0, no drift), while
    // PC's relaxation deliberately lets x1 drift away from that to reduce
    // e2 - more relaxation steps means MORE drift, not less.
    //
    // REVISITED with precision-weighting below (each error scaled by a
    // fixed inverse-variance term) - the hypothesized missing piece. Real,
    // substantial, but PARTIAL confirmation: a large sigma1_inv (penalizing
    // x1's drift heavily) collapses dW0's error plateau from 0.67 to 0.17
    // (exactly 0 at T=1), and reverses dW1's wrong-direction pattern - it
    // now shrinks from 0.267 (T=1) to a 0.224 plateau instead of growing.
    // Neither fully converges to zero error even at T=300, though - both
    // plateau at a persistent nonzero gap, not the clean "->0 as T->infinity"
    // the literature claims. Precision-weighting is real and directionally
    // correct, substantially shrinks the gap and fixes the wrong-direction
    // pattern, but a fixed hand-set value doesn't fully close it. Genuinely
    // learned/jointly-trained per-unit precision (the real free-energy PC
    // formulation, not this fixed-hyperparameter version) is parked for a
    // future revisit if the fixed-value result stays unsatisfying.
    let (bp_w0, bp_w1) = backprop_grads(&x0, &target, &w0_init, &w1_init);
    println!("backprop gradient (ground truth):");
    println!("  dW0 = {:?}", bp_w0.data);
    println!("  dW1 = {:?}", bp_w1.data);
    println!("\nPC gradient relative error vs backprop, by relaxation step count T:");
    for &t in &[1usize, 5, 20, 50, 100, 300] {
        let (pc_w0, pc_w1) = pc_grads(&x0, &target, &w0_init, &w1_init, t, 0.1);
        println!("  T={t:>3}: dW0 rel err = {:.6}, dW1 rel err = {:.6}", rel_error(&pc_w0, &bp_w0), rel_error(&pc_w1, &bp_w1));
    }

    // Precision-weighted revisit: theoretically-motivated first test, not
    // an arbitrary grid sweep. The original hypothesis for the gap was that
    // x1 drifts further from its feedforward value (relu(x0@W0)) the longer
    // relaxation runs, while backprop implicitly assumes zero drift. A
    // LARGE sigma1_inv heavily penalizes exactly that drift, directly
    // targeting the hypothesized mechanism rather than guessing blindly.
    let (bp_w0_p, bp_w1_p) = backprop_grads_precision(&x0, &target, &w0_init, &w1_init, 1.0);
    println!("\nprecision-weighted revisit (sigma1_inv=10.0, sigma2_inv=1.0 - heavily penalizes x1 drift):");
    for &t in &[1usize, 5, 20, 50, 100, 300] {
        let (pc_w0, pc_w1) = pc_grads_precision(&x0, &target, &w0_init, &w1_init, t, 0.1, 10.0, 1.0);
        println!("  T={t:>3}: dW0 rel err = {:.6}, dW1 rel err = {:.6}", rel_error(&pc_w0, &bp_w0_p), rel_error(&pc_w1, &bp_w1_p));
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

    // Genuinely learned precision - the real free-energy formulation the
    // fixed-value revisit above was proportionate scope ahead of. Per-layer
    // scalar (same granularity as the fixed-value test, not per-unit -
    // still the smaller step), EMA-smoothed and recomputed each training
    // step from the settled error, not a hand-swept constant.
    //
    // Previously parked after three instabilities: raw MLE crashed toward
    // zero on large early error (self-reinforcing collapse), a warm-up
    // fixed that but exposed the mirror failure (unbounded MLE exploding
    // toward infinity once error got small), and a [0.1, 100] clamp still
    // NaN'd - diagnosed as the clamp's ceiling still being a 100x jump in
    // one step right at the warm-up boundary, not the bound's value.
    // Fixed here with both surveyed remedies together: update_precision_map
    // replaces raw MLE with a MAP estimate under a Gamma prior (self-
    // bounding - no discontinuous wall to hit, see its own doc comment),
    // and a linear ramp blends sigma_inv from the frozen 1.0 toward that
    // estimate over `ramp_epochs` instead of switching in one step,
    // directly targeting the diagnosed mechanism (the jump itself). Result:
    // no NaN at any epoch, precision converges to a stable (5.0, 2.0) and
    // stays there, XOR trains correctly throughout.
    // Precision needs real training experience before it means anything -
    // sigma_inv=1.0 at epoch 0 is an arbitrary starting point, not a claim -
    // so the diagnostic T-sweep below runs AFTER training, at whatever
    // weights+precision actually emerged, mirroring how the fixed-value
    // sweep ran at a fixed weight snapshot rather than at initialization.
    let mut w0_lp = w0_init.clone();
    let mut w1_lp = w1_init.clone();
    let adam_lp = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut w0_lp_state = AdamState::zeros_like(&w0_lp);
    let mut w1_lp_state = AdamState::zeros_like(&w1_lp);
    let mut sigma1_inv = 1.0f32;
    let mut sigma2_inv = 1.0f32;
    let mut ema1 = 1.0f32;
    let mut ema2 = 1.0f32;
    let decay = 0.95;
    // Mitigation for a real failure mode found on the first attempt (no
    // warm-up): precision crashed almost immediately (output error is
    // naturally large before weights have trained at all), and once
    // sigma2_inv is tiny, dF/dW1 is scaled by that same tiny value - W1
    // stops being effectively corrected, e2 stays large, precision stays
    // crashed - a self-reinforcing collapse, not a bug (Pi* = N/sum(e^2) is
    // the correct closed-form MLE; naive joint MLE precision learning is a
    // known-unstable pattern in the literature, "precision/variance
    // collapse"). Freezing precision at 1.0 until weights have had a chance
    // to reduce the error on their own removes the race condition entirely
    // - cheapest of the surveyed mitigations, no new hyperparameter beyond
    // an epoch count. EMA seeded from real settled error the moment warm-up
    // ends, not the arbitrary 1.0 default, so the first post-warmup
    // precision estimate reflects where training actually is.
    let warmup_epochs = 200;
    // Diagnosed root cause of the earlier NaN, stated plainly in the
    // parked commit: "not about the bound's value, the discontinuity
    // itself" - precision jumped from the frozen 1.0 straight to whatever
    // the estimator produced in a single step the instant warm-up ended,
    // and Adam's moment estimates (calibrated to 200 epochs of one
    // gradient scale) overshot on the sudden regime change. A linear ramp
    // over `ramp_epochs` blends sigma_inv from 1.0 toward the MAP estimate
    // instead of switching in one step - bounds the largest possible
    // per-step CHANGE in precision to roughly 1/ramp_epochs of the total
    // gap, however large that gap turns out to be, addressing the actual
    // diagnosed mechanism (the jump size) rather than re-guessing at a
    // bound (the old clamp's failed approach).
    let ramp_epochs = 50.0;
    // pseudo_n=4.0: matches N (this error term always has 4 elements,
    // seq unbatched over the 4 XOR rows) - "one real batch's worth" of
    // prior confidence, the same order-of-magnitude reasoning as the old
    // floor/ceiling picks, but now feeding a principled formula instead of
    // a hand-tuned wall (see update_precision_map's doc comment).
    let pseudo_n = 4.0;

    println!(
        "\ntraining XOR with LEARNED precision ({warmup_epochs}-epoch warm-up, {ramp_epochs}-epoch ramp, MAP estimate with pseudo_n={pseudo_n}, T=20 relaxation steps/epoch):"
    );
    for epoch in 0..1000 {
        let (d_w0, d_w1, e1, e2) = pc_grads_and_errors(&x0, &target, &w0_lp, &w1_lp, 20, 0.1, sigma1_inv, sigma2_inv);
        adam_lp.step(&mut w0_lp, &d_w0, &mut w0_lp_state);
        adam_lp.step(&mut w1_lp, &d_w1, &mut w1_lp_state);

        if epoch >= warmup_epochs {
            if epoch == warmup_epochs {
                ema1 = e1.mul(&e1).sum().data[0];
                ema2 = e2.mul(&e2).sum().data[0];
            }
            let map1 = update_precision_map(&e1, &mut ema1, decay, pseudo_n);
            let map2 = update_precision_map(&e2, &mut ema2, decay, pseudo_n);
            let ramp_frac = ((epoch - warmup_epochs) as f32 / ramp_epochs).min(1.0);
            sigma1_inv = (1.0 - ramp_frac) * 1.0 + ramp_frac * map1;
            sigma2_inv = (1.0 - ramp_frac) * 1.0 + ramp_frac * map2;
        }

        if epoch % 100 == 0 {
            let pred = x0.matmul(&w0_lp).relu().matmul(&w1_lp);
            let diff = pred.sub(&target);
            let loss = 0.5 * diff.mul(&diff).sum().data[0];
            println!("  epoch {epoch:>4}: loss = {loss:.6}, sigma1_inv = {sigma1_inv:.4}, sigma2_inv = {sigma2_inv:.4}");
        }
    }

    // Same diagnostic as the fixed-value revisit, now at weights+precision
    // that emerged from actual training - no collapse this time (see the
    // header comment above). Measured: does NOT close the persistent
    // nonzero-plateau gap the fixed-value test also couldn't fully close -
    // dW1's error is actually larger at low T here than the fixed-value
    // sweep showed (1.56 at T=1 vs that sweep's 0.267), though both settle
    // to a similar-magnitude plateau by T=20+. Different question than
    // "does it explode": stability was the thing that was broken and is
    // now fixed; full convergence to backprop as T grows was never fully
    // achieved by any variant tried in this file, fixed-value or learned.
    let (bp_w0_lp, bp_w1_lp) = backprop_grads_precision(&x0, &target, &w0_lp, &w1_lp, sigma2_inv);
    println!("\nlearned-precision T-sweep at trained weights (sigma1_inv={sigma1_inv:.4}, sigma2_inv={sigma2_inv:.4}):");
    for &t in &[1usize, 5, 20, 50, 100, 300] {
        let (pc_w0, pc_w1) = pc_grads_precision(&x0, &target, &w0_lp, &w1_lp, t, 0.1, sigma1_inv, sigma2_inv);
        println!("  T={t:>3}: dW0 rel err = {:.6}, dW1 rel err = {:.6}", rel_error(&pc_w0, &bp_w0_lp), rel_error(&pc_w1, &bp_w1_lp));
    }

    let pred_lp = x0.matmul(&w0_lp).relu().matmul(&w1_lp);
    println!("\ninput -> pred (target) [learned precision]");
    for i in 0..4 {
        println!("  ({:.0}, {:.0}) -> {:.4} ({:.0})", x0.data[i * 2], x0.data[i * 2 + 1], pred_lp.data[i], target.data[i]);
    }
}
