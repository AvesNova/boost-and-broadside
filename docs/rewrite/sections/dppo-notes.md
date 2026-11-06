# DPPO notes (Qi et al., "Rethinking the Trust Region in LLM Reinforcement Learning")

arXiv 2602.04879, ICML 2026; code github.com/sail-sg/Stable-RL; verl `LOSS_MODE=dppo_tv`
/ `dppo_kl`. Read 2026-11-06 by a reader agent (sections 1-7 and appendix A verbatim;
appendices B-F from a summary, lower confidence; figures not read).

## The objective (their Eq. 12)

mu is the behaviour (rollout) policy, pi the trained one, r_t = pi(a_t|s_t) / mu(a_t|s_t),
A_t the advantage.

    L = E_mu[ sum_t M_t * r_t * A_t ]
    M_t = 0 if (A_t > 0 and r_t > 1 and D > delta) or (A_t < 0 and r_t < 1 and D > delta)
          1 otherwise
    D = D(mu(.|s_t) || pi(.|s_t)), total variation or KL, of the whole distribution

- A hard one-sided gradient mask, not a clip or a penalty: masked samples get zero
  gradient; unmasked ones keep the unclipped r * A. Moves back toward ratio 1 are never
  blocked. With D replaced by |r - 1| it is PPO's clip.
- Thresholds: delta 0.15 (TV), 0.05 (KL); some runs 0.2 (TV). Swept 0.10-0.20 (TV),
  reported insensitive. (Pinsker: KL >= 2 TV^2, so TV 0.15 ~ KL 0.045.)

## The argument

The ratio of the sampled action is a one-sample estimate of TV (TV = 1/2 E_mu|r - 1|).
Clipping on it over-constrains rare actions and under-constrains dominant ones:
mu 1e-4 -> pi 1e-2 has ratio 100 (clipped) but moves 0.0099 of mass; mu 0.99 -> pi 0.80 has
ratio 0.81 (not clipped) but moves 0.19. DAPO's clip-higher (an asymmetric ratio bound)
treats the symptom; in their clip-relaxation study, relaxing only upward for rare actions
gave no significant gain, relaxing both ways was best.

## Approximations (for vocabularies; not needed for 42 joint actions)

- Binary: TV = |mu(a_t) - pi(a_t)|; KL of the two-way split {a_t, rest}.
- Top-K (K = 20) plus an "other" bucket.
- Both are lower bounds of the full divergence, so exact TV masks more at the same delta:
  re-tune delta for the exact form.

## Results and caveats

- Beats GRPO with clip-higher and CISPO on AIME (Qwen3 MoE and dense) and RLHF
  (AlpacaEval 2.0 80.9 length-controlled); no margins in the text. Binary and Top-K similar.
- Much of the motivation is training-inference mismatch (vLLM vs trainer), which a single
  engine like ours barely has; the efficiency argument may transfer, the stability one less.
- Not discussed: a cap on r for unmasked rare actions with small D (large unclipped
  weights); factored action heads (ours: compute the 42-way joint exactly); storing mu's
  full distribution at rollout time.
