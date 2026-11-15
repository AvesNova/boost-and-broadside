# Run order (proposal for the author)

The order of the Yes runs in [arm-list.md](arm-list.md), with the reasoning, and where the Maybe runs would fit. Each decision follows [decision-rule.md](decision-rule.md).

## The choice: one chain, or waves

"Build on the winner" read strictly is one long chain: each arm waits for the previous decision. That's 14 runs in a row, about 4–5 hours each with evaluation, so roughly 3 days with the box mostly idle.

**Proposal: waves.**
- Within a wave, every challenger is run against the same incumbent, in parallel, one GPU each.
- Between waves, the winners are adopted together, and the combination is trained once as the next wave's incumbent.
- That combination run is itself checked against the old incumbent with the decision rule, so two winners that interfere are caught.
- On 8 GPUs this takes about 4–5 waves, about 1–1.5 days.

**Why it's sound:**
- Every comparison is still one setting against a fixed incumbent, at the same seeds and length.
- The only thing given up is an arm in a wave seeing an earlier arm's win from that same wave. Wave boundaries are placed where that matters: data and game choices before model choices, and A14 before A13.
- The extra cost is one incumbent run per wave, about 4 hours, and only in waves that have a winner.

## Waves

**Wave 0, the dry run (1 GPU):**
- R0 at seed 0: the baseline, the pipeline check, the compile caches and the micro-batch probe.
- If it's clean, it is run 1, and R5 (the averaging suite) follows on its per-update files.
- **Why first:** every later number is read against R0, and the dry run's version-21 curve confirms the 100M budget before anything else starts.

**Wave 1, data and game choices (3 GPUs), all against R0:**
- R0 at seed 1;
- A1 variable maps;
- A2 variable counts.

**Why these together, and first:**
- A1 and A2 change BC's data and what every later arm trains on, so they have to be settled before the model arms.
- Both may fail to train, which is why the second seed of a non-R0 arm waits.
- R0's second seed runs here so seed noise (and so δ) is known before the first close decision.

**After wave 1:**
- Set δ from R0's seed pair.
- The second seed of the first arm that trains (A1 or A2) starts at once.
- If either won, train the new incumbent (R0 plus the winner, or both). If both won, decide the combination against each single winner.

**Wave 2, rewards on the RL fork (2 GPUs; up to 8 with the Maybe forks):**
- A7 premium 2:1;
- A7 premium 2:1 annealed.

Both fork from the wave-1 incumbent's BC.

**Why here:**
- The premium changes what RL optimises, and the model arms' comparisons should be made under the reward we'll keep.
- They're cheap forks, so the wave is short (about 2 hours).
- The Maybe RL forks (A4, A5, A6, A8, A9, A10) share this BC and fit in the same wave if chosen.

**Wave 3, the model (5 GPUs, plus up to 2 for A19 and A22 if chosen), all against the wave-2 incumbent:**
- A14 belief input;
- A15 layout (spatial only);
- A16 GRU;
- A17 raw scalars;
- R1 MLP baseline.

**Why together:**
- They're separate parts of the model, so there's no ordering among them except A13, which waits for A14.
- R1 is a reference and doesn't feed the chain.

**Wave 4, after A14 (up to 4 GPUs):**
- A13 fewer auxiliary heads: all removed but next-state if A14 showed the belief helps, otherwise all removed. Against the wave-3 incumbent.
- A24 RL from scratch, on the wave-3 incumbent's configuration, so the paper's BC-against-scratch claim is made on the near-final model.
- The Maybe late runs: A23's width trio needs the final configuration, so it can't start until A13 is decided (a wave 5). R3 and R4 also need the final model.

**Wave 5 (if needed):** the final incumbent's run, if wave 4 had a winner; then A23, R3 and R4 if chosen; then the 500M publication run.

## Summary

| wave | runs (Yes) | GPUs | about |
|---|---|---|---|
| 0 | R0 s0 (+ R5) | 1 | 4–6 h |
| 1 | R0 s1, A1, A2 (+ the first trained arm's s1) | 3–4 | 4–5 h |
| 2 | A7 2:1, A7 annealed | 2 | 2–3 h |
| 3 | A14, A15, A16, A17, R1 | 5 | 4–5 h |
| 4 | A13, A24 | 2 | 4–6 h |

Add one incumbent run between waves that had a winner. **Total wall-clock** on up to 8 GPUs: about 1–1.5 days, against about 3 days for a strict chain.

## Open for the author

- Waves against a strict chain.
- A24 in wave 4 (on the near-final model) against running it in parallel from wave 1 (on R0's configuration).
- Which Maybe runs join waves 2, 3 and 5.
