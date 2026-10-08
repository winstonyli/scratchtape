# Gating the words tier's weight by uncertainty (offline fit)

Follow-up to `2026-10-07-uncertainty-gating-design.md`. That experiment gated the kNN weight and found about +0.001 nats
over the best constant (below the 0.002 bar). This one asks the same question of the words tier, whose weight is also a
fixed constant (`words:1:0.25`).

## Question

Does choosing the words weight per position (from the model's uncertainty, how much evidence the word list has, and the
kNN neighbours' distance) lower held-out cross-entropy by more than 0.002 nats per position over the best constant words
weight? A constant is the right baseline: a gate can reproduce any constant, so gain over 0.5-style defaults would be
retuning, not gating.

## What is gated

The stack is `lexicon:0.3 + words:1:0.25 + knn:256:0.4:15`, evaluated as before (warm=32, stride=1, online memory,
store=100000, checkpoint `runs/nov_big_k1_d0.1_8m_m0.ckpt`). Only the words weight lambda_w varies; lexicon (0.3) and the kNN weight mu
(0.4) stay fixed.

Words at an inside-word position (non-empty prefix, `total > 0`, `non_letter > 0`; the only positions where it is applied,
since `first` letters are off) mixes `p_w = (1 - lambda_w) p_in + lambda_w q`, where `p_in` is the lexicon's output and q
the word-list distribution (letters by count; non-letters get the word-end mass times `p_in` renormalised over non-letters).
The final probability of the target is `(1 - mu) p_w,t + mu k_t`, with `k_t` the kNN distribution's probability of the target.
Everything else is untouched, so for a position `i` the loss at weight `l` is

    loss_i(l) = -ln((1 - mu) ((1 - l) p_in,t + l q_t) + mu k_t)        (applied positions)
    loss_i    = -ln((1 - mu) p_in,t + mu k_t)                          (unapplied positions, constant)

Valid offline because words only sees `p_in` (earlier tiers) and the kNN search does not depend on words or on mu
(it uses hidden states and the online memory, which holds bytes). If a position has no kNN neighbours the kNN tier leaves p alone
(`k_t` would equal `p_w,t`); with the train keys always visible this does not occur in the online run, so `gate_fit_words`
asserts there are none (rows with `d0 == 0 && dk == 0`) instead of handling them.

## Dumps

Two files, one row per scored position of the last spec, in scoring order (so rows align by index):

- the existing kNN dump (`dump=`; f32 x 6: chunk, p_t, k_t, H, d0, dk), where `p_t` is the target probability **after** words at 0.25;
- a new words dump (`wdump=`; f32 x 8): `chunk, applied (0/1), p_in_t, q_t, H_in, ln(1+total), bigram (0/1), prefix_len`.
  `H_in` is the natural-log entropy of `p_in`. For unapplied rows `q_t = p_in_t` and the last three features are 0.

`wdump=` is valid only when the last spec has a `words:` part before a final `knn:` part, like `dump=`; both options are
required together. Same atomic write every 20 chunks.

## Offline tool

New example `words_gate_fit` (`examples/tiny_lm/words_gate_fit.rs`) reads both dumps and `mu=0.4 logged=<CE>`:

- **Gate 0:** mean loss at lambda_w = 0.25 and mu = 0.4 over all rows equals the logged CE to 1e-4.
- **Gate 0b:** at every row (unapplied rows have `q_t = p_in_t`) `(1 - 0.25) p_in,t + 0.25 q_t` equals the kNN dump's `p_t` to
  1e-5 (the join and the formula agree); chunk ids and row counts match.
- Splits as in `gate_fit`: even/odd chunks and early/late halves. Fit on one half, score on the other.
- Baselines: constant lambda_w on a 0.01 grid, best on the fit half, scored on the score half.
- Gates, fitted on applied rows of the fit half:
  - binned: 2 (bigram) x 4 (quantile bins of `H_in`), lambda_w on a 0.02 grid per bin;
  - parametric: lambda_w = sigmoid(w . [1, z_H_in, z_ln(1+total), bigram, z_prefix_len, z_ln(1+d0)]) with the same full-batch gradient
    descent with backtracking, started at the best constant (clamped to [0.01, 0.99]); `d0` is the kNN dump's nearest distance.
- Gains are reported per scored position (all positions, applied or not): baseline CE minus gated CE on the score half, versus
  the best constant lambda_w and versus the logged lambda_w = 0.25.
- Prints the binned table and the fitted weights.

## Success and kill

- **Success:** parametric gain over the best constant is greater than 0.002 nats on both splits. Then follow up by wiring the
  gate into the stack (it needs the kNN search before words; the search is independent of words, so reordering is safe) and
  checking the whole-stack CE.
- **Kill:** binned gain under 0.002 on either split. Record it and stop gating the tiers; the gain is in the memory's content.
- In between: record, no wiring.

## Cost and risks

- One full run (about 8-15 min GPU) for the dumps; no new GPU code. The same quiet/contended rules as before apply; CE and the
  fit do not depend on timing.
- Words applies at about 72% of positions, so a gain per applied position of 0.003 is about 0.002 overall.
- Even/odd halves share books, so early/late is the stricter split; the verdict needs both.
- `d0` is a cross-tier feature: the deployed order would have to run the kNN search first.

## Out of scope

Gating the lexicon, retuning mu jointly with lambda_w, between-word (`first`) positions, order-0 words, any change to the stack's
default specs.
