use scratchtape::nn::{Linear, Rng};
use scratchtape::optim::{Adam, AdamState};
use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

/// Diagonal linear SSM (S4D-style, not full S4/Mamba - no HiPPO init, no
/// selectivity): h_t = a*h_{t-1} + b*u_t, elementwise per channel. Copy-task
/// benchmark, the standard way long-range memory is tested in the
/// SSM/RNN literature: a value at t=0, distractor noise in between, a query
/// marker at the final step asking the network to reconstruct it.
///
/// Not the same task as attention_recall.rs on purpose - attention's memory
/// slots were an unordered SET (no recurrence involved at all); an SSM only
/// means something over an ORDERED sequence, so forcing the identical task
/// would test nothing real. This is the actual standard SSM-vs-RNN
/// benchmark instead.
fn main() {
    let d_content = 4usize;
    let d_in = d_content + 1; // + 1 marker channel signaling "recall now"
    let d_state = 8usize;
    let seq_len = 8usize; // t=0 content, t=1..=6 distractors, t=7 query
    let batch = 4usize; // one training example per possible content pattern

    let mut rng = Rng::new(1);

    let contents = NdArray::new(
        vec![
            4.0, 0.0, 0.0, 0.0, //
            0.0, 4.0, 0.0, 0.0, //
            0.0, 0.0, 4.0, 0.0, //
            0.0, 0.0, 0.0, 4.0,
        ],
        vec![batch, d_content],
    );

    let mut inputs: Vec<NdArray> = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let mut data = vec![0.0f32; batch * d_in];
        if t == 0 {
            for b in 0..batch {
                for c in 0..d_content {
                    data[b * d_in + c] = contents.data[b * d_content + c];
                }
            }
        } else if t == seq_len - 1 {
            for b in 0..batch {
                data[b * d_in + d_content] = 1.0; // marker channel
            }
        } else {
            for b in 0..batch {
                for c in 0..d_content {
                    data[b * d_in + c] = rng.next_gaussian() * 0.1; // distractor noise
                }
            }
        }
        inputs.push(NdArray::new(data, vec![batch, d_in]));
    }

    // embed's w/b get leafed ONCE per epoch and reused across all seq_len
    // steps (weight-tying, as any real recurrent parameter must be) -
    // Linear::forward can't be used here since it re-leafs fresh w/b on
    // every call, which would silently break weight-tying and the
    // cross-timestep gradient accumulation this demo specifically measures.
    // readout is only ever called once (at the final step), so it has no
    // such problem and uses Linear::forward normally.
    let embed = Linear::new(&mut rng, d_in, d_state);
    let mut readout = Linear::new(&mut rng, d_state, d_content);
    let mut embed_w = embed.w;
    let mut embed_b = embed.b;
    let mut a = NdArray::new(vec![0.9; d_state], vec![1, d_state]);
    let mut b_gain = NdArray::new(vec![0.5; d_state], vec![1, d_state]);

    let adam = Adam { lr: 0.02, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut embed_w_state = AdamState::zeros_like(&embed_w);
    let mut embed_b_state = AdamState::zeros_like(&embed_b);
    let mut readout_w_state = AdamState::zeros_like(&readout.w);
    let mut readout_b_state = AdamState::zeros_like(&readout.b);
    let mut a_state = AdamState::zeros_like(&a);
    let mut b_state = AdamState::zeros_like(&b_gain);

    // Snapshot BEFORE training - the vanishing-gradient measurement needs to
    // happen where loss is still large. After training converges, gradient
    // is trivially ~0 everywhere near the optimum, which would measure
    // nothing about the recurrence's actual information-carrying capacity.
    let init_embed_w = embed_w.clone();
    let init_embed_b = embed_b.clone();
    let init_a = a.clone();
    let init_b_gain = b_gain.clone();
    let init_readout = Linear::from_parts(readout.w.clone(), readout.b.clone());

    let epochs = 400;
    let denom = 1.0 / (batch * d_content) as f32;

    for epoch in 0..epochs {
        let mut tape = Tape::new();
        let a_var = tape.leaf(a.clone());
        let b_var = tape.leaf(b_gain.clone());
        let w_var = tape.leaf(embed_w.clone());
        let b_bias_var = tape.leaf(embed_b.clone());
        let target = tape.leaf(contents.clone());

        let mut h = tape.leaf(NdArray::zeros(vec![batch, d_state]));
        for t in 0..seq_len {
            let x_t = tape.leaf(inputs[t].clone());
            let mm = tape.matmul(x_t, w_var);
            let u_t = tape.add(mm, b_bias_var);
            let decayed = tape.mul(a_var, h);
            let driven = tape.mul(b_var, u_t);
            h = tape.add(decayed, driven);
        }

        let out = readout.forward(&mut tape, h);
        let diff = tape.sub(out.y, target);
        let sq = tape.mul(diff, diff);
        let sum = tape.sum(sq);
        let loss = tape.scale(sum, denom);

        tape.backward(loss);

        adam.step(&mut embed_w, tape.grad(w_var).unwrap(), &mut embed_w_state);
        adam.step(&mut embed_b, tape.grad(b_bias_var).unwrap(), &mut embed_b_state);
        readout.apply_grad_with(&tape, &out, &adam, &mut readout_w_state, &mut readout_b_state);
        adam.step(&mut a, tape.grad(a_var).unwrap(), &mut a_state);
        adam.step(&mut b_gain, tape.grad(b_var).unwrap(), &mut b_state);

        if epoch % 40 == 0 {
            println!("epoch {epoch:>4}: loss = {:.6}", tape.value(loss).data[0]);
        }
    }

    // Forward+backward with TRAINED params - reports final recall accuracy.
    fn run(
        embed_w: &NdArray,
        embed_b: &NdArray,
        a: &NdArray,
        b_gain: &NdArray,
        readout: &Linear,
        inputs: &[NdArray],
        contents: &NdArray,
        batch: usize,
        d_state: usize,
        denom: f32,
    ) -> (Tape, Vec<scratchtape::tape::Var>, scratchtape::tape::Var) {
        let mut tape = Tape::new();
        let a_var = tape.leaf(a.clone());
        let b_var = tape.leaf(b_gain.clone());
        let w_var = tape.leaf(embed_w.clone());
        let b_bias_var = tape.leaf(embed_b.clone());
        let target = tape.leaf(contents.clone());

        let mut h_vars = vec![tape.leaf(NdArray::zeros(vec![batch, d_state]))];
        for x in inputs {
            let x_t = tape.leaf(x.clone());
            let mm = tape.matmul(x_t, w_var);
            let u_t = tape.add(mm, b_bias_var);
            let h_prev = *h_vars.last().unwrap();
            let decayed = tape.mul(a_var, h_prev);
            let driven = tape.mul(b_var, u_t);
            h_vars.push(tape.add(decayed, driven));
        }
        let out = readout.forward(&mut tape, *h_vars.last().unwrap());
        let diff = tape.sub(out.y, target);
        let sq = tape.mul(diff, diff);
        let sum = tape.sum(sq);
        let loss = tape.scale(sum, denom);
        tape.backward(loss);
        (tape, h_vars, out.y)
    }

    let (tape, _, out_y) = run(&embed_w, &embed_b, &a, &b_gain, &readout, &inputs, &contents, batch, d_state, denom);
    let pred = tape.value(out_y);
    println!("\ncontent -> recalled (target), trained params");
    for b in 0..batch {
        let p = &pred.data[b * d_content..b * d_content + d_content];
        let t = &contents.data[b * d_content..b * d_content + d_content];
        println!("  {t:?} -> {p:?}");
    }

    // Forward+backward with the pre-training INITIAL params - this is the
    // actual vanishing-gradient measurement. Loss is still large here, so
    // the gradient's shape across timesteps reflects the recurrence's raw
    // information-carrying capacity, not the trivial "near a minimum" case.
    let (init_tape, init_h_vars, _) = run(&init_embed_w, &init_embed_b, &init_a, &init_b_gain, &init_readout, &inputs, &contents, batch, d_state, denom);
    println!("\ngradient norm |d(loss)/d(h_t)| at each timestep, AT INITIALIZATION (t=0 is earliest, farthest from the loss):");
    for (t, &hv) in init_h_vars.iter().enumerate() {
        match init_tape.grad(hv) {
            Some(g) => println!("  t={t}: {:.6}", g.mul(g).sum().data[0].sqrt()),
            None => println!("  t={t}: (no gradient reached this node)"),
        }
    }
}
