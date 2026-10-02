use scratchtape::nn::{Linear, Rng};
use scratchtape::tensor::NdArray;

/// Plain NdArray math, no Tape/Var anywhere - proves ES needs zero autograd
/// machinery, not just forward-only by convention.
fn forward(l1: &Linear, l2: &Linear, x: &NdArray) -> NdArray {
    let h = x.matmul(&l1.w).add(&l1.b).relu();
    h.matmul(&l2.w).add(&l2.b)
}

fn mse_loss(pred: &NdArray, target: &NdArray) -> f32 {
    let diff = pred.sub(target);
    let sq = diff.mul(&diff);
    sq.sum().data[0] / pred.numel() as f32
}

fn randn_like(rng: &mut Rng, shape: &[usize]) -> NdArray {
    let n: usize = shape.iter().product();
    NdArray::new((0..n).map(|_| rng.next_gaussian()).collect(), shape.to_vec())
}

/// OpenAI-ES (2017): estimate the loss gradient via population sampling
/// instead of backprop. grad ~= (1/(2*N*sigma)) * sum((F(t+se)-F(t-se)) * e)
/// - a Monte Carlo score-function estimator, not a true gradient, but used
/// the same way in an update step. Antithetic pairs (+e/-e per sample)
/// cancel odd-moment noise bias for free - no extra noise draws needed.
fn main() {
    let x_data = NdArray::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let y_data = NdArray::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    let mut rng = Rng::new(1);
    let mut l1 = Linear::new(&mut rng, 2, 4);
    let mut l2 = Linear::new(&mut rng, 4, 1);

    let population = 100usize;
    let sigma = 0.1f32;
    let lr = 0.1f32;
    let iterations = 300usize;
    let mut forward_evals = 0usize;

    for iter in 0..iterations {
        let mut grad_w1 = NdArray::zeros(l1.w.shape.clone());
        let mut grad_b1 = NdArray::zeros(l1.b.shape.clone());
        let mut grad_w2 = NdArray::zeros(l2.w.shape.clone());
        let mut grad_b2 = NdArray::zeros(l2.b.shape.clone());

        for _ in 0..population {
            let eps_w1 = randn_like(&mut rng, &l1.w.shape);
            let eps_b1 = randn_like(&mut rng, &l1.b.shape);
            let eps_w2 = randn_like(&mut rng, &l2.w.shape);
            let eps_b2 = randn_like(&mut rng, &l2.b.shape);

            let plus = Linear::from_parts(l1.w.add(&eps_w1.scale(sigma)), l1.b.add(&eps_b1.scale(sigma)));
            let plus2 = Linear::from_parts(l2.w.add(&eps_w2.scale(sigma)), l2.b.add(&eps_b2.scale(sigma)));
            let f_pos = mse_loss(&forward(&plus, &plus2, &x_data), &y_data);

            let minus = Linear::from_parts(l1.w.sub(&eps_w1.scale(sigma)), l1.b.sub(&eps_b1.scale(sigma)));
            let minus2 = Linear::from_parts(l2.w.sub(&eps_w2.scale(sigma)), l2.b.sub(&eps_b2.scale(sigma)));
            let f_neg = mse_loss(&forward(&minus, &minus2, &x_data), &y_data);

            forward_evals += 2;

            let coeff = (f_pos - f_neg) / (2.0 * population as f32 * sigma);
            grad_w1 = grad_w1.add(&eps_w1.scale(coeff));
            grad_b1 = grad_b1.add(&eps_b1.scale(coeff));
            grad_w2 = grad_w2.add(&eps_w2.scale(coeff));
            grad_b2 = grad_b2.add(&eps_b2.scale(coeff));
        }

        // Minimizing loss, so descend the estimated gradient (same
        // convention as SGD/Adam - not maximizing fitness as the original
        // OpenAI-ES paper frames it).
        l1.w = l1.w.sub(&grad_w1.scale(lr));
        l1.b = l1.b.sub(&grad_b1.scale(lr));
        l2.w = l2.w.sub(&grad_w2.scale(lr));
        l2.b = l2.b.sub(&grad_b2.scale(lr));

        if iter % 30 == 0 {
            let loss = mse_loss(&forward(&l1, &l2, &x_data), &y_data);
            println!("iter {iter:>4}: loss = {:.6}, forward evals so far = {forward_evals}", loss);
        }
    }

    let pred = forward(&l1, &l2, &x_data);
    println!("\ninput -> pred (target)");
    for i in 0..4 {
        println!("({:.0}, {:.0}) -> {:.4} ({:.0})", x_data.data[i * 2], x_data.data[i * 2 + 1], pred.data[i], y_data.data[i]);
    }
    println!("\ntotal forward evals: {forward_evals} (backprop: ~150-300 forward+backward pairs to solve the same task)");
}
