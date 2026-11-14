# Ideas and candidate arms

These are candidate experiments and features, numbered so they can be referred to. Each is a candidate, not a decision; the author decides. They were collected on 2026-11-14 from the weekend-arms planning.

## Training setup and optimisation

1. Batch size: 2048 against 4096 environments.
2. Rollout segment length (128 ticks), longer or shorter.
3. PPO minibatches: 32 against 256. Run early, as RL-only forks from one BC checkpoint.
4. Learning rate: current against ×2. KL is about 0.007 against a target of 0.02, so the updates are conservative.
5. Epochs per update.
6. Clip range and target KL.
7. Entropy bonus or schedule. Entropy fell to 0.4 late in rl-v20.
8. LR schedule: constant against warm-up plus decay.
9. Gradient-clip norm, Adam betas, weight decay.
10. Value-loss coefficient.
11. Old log-probabilities: from the rollout against recomputed.
12. Return-scale decay and floor.
13. Per-level γ/λ against a unified γ/λ.

## BC and the handoff

14. Handoff point: BC to 50M against BC to its plateau (about 85M).
15. A KL-to-BC penalty during early RL.
16. BC teacher temperature and floor.
17. BC data mix: self-play against scripted games.
18. BC stop rule: plateau against a fixed budget.
19. RL starting from BC's live weights against its averaged weights.
20. BC→RL against RL from scratch, with the same rewards and no helper shaping, at 2–3× the steps.

## Rewards

21. Reward level weights.
22. Team reward against individual reward mixing.
23. Dense shaping levels (damage, kill/death) on against off.
24. Deathmatch outcome: kill share against kill lead. Done: no difference, so kill share stays.

## Critic

25. A centralised critic that sees the full state, against a critic with the actor's view.
26. A separate critic network against the shared trunk.
27. Two-hot distributional value against a scalar value.

## Model

28. Attention heads: 2 against 4.
29. Width: 128 against 192. Run last, compared at equal steps and at equal cost, with an LR re-check.
30. Layout: 2 Yemong blocks against 6 spatial blocks.
31. Temporal block: Griffin against GRU.
32. Qwen gated attention, elementwise (`--set yemong.gate=elementwise`, 59987b0a; about 1% cost).
33. Feed-forward width ratio.
34. RMSNorm against LayerNorm.
35. Input encoding: HL-Gauss codes against raw scalars.
36. Position encoding: rotary against the colour code against both.
37. Ego-centric frame against world frame.
38. Map tokens against a rasterised map with a CNN.
39. Presence radius.
40. Next-state head: delta against absolute.
41. Belief decay constants.
42. Belief input: the next-step heads' outputs for hidden ships against the temporal layers alone.
43. A simple MLP baseline, as a reference rather than a pairwise arm.

## Auxiliary heads

44. All auxiliary heads removed.
45. Each auxiliary head removed in turn: next-state, correction, density, enemy action, shot, geometry.
74. **(2026-11-14)** A full bullet-trajectory head against an "if I shot now" head. The second predicts the damage a shot fired now would do to each enemy and each ally.

## Data and symmetry

46. Rotation and reflection augmentation.
47. A seat-symmetry check.
48. Fog of war: always on against a mix.

## League and opponents

49. Self-play only against a league.
50. League size and snapshot cadence.
51. Opponent sampling: uniform against prioritised.
52. The scripted opponents' share in RL.
53. More varied scripted opponents.

## Environment distribution

54. Map sizes: fixed against variable (bc-v21 was started, then stopped for the shutdown).
55. Ship and field counts: fixed against variable. Absent ships and fields are removed from the game and their slots masked, as in evaluation. Deathmatch and elimination go down to 1v1; frontline to 4v4 (3v3 needs a decision). Masking is free on the fused kernel since 6985a342.
56. Ship-type lineups: always YYJJJ against mixed or random lineups.
57. Mode mix shares.
58. Episode length and respawn time.

## Game physics

75. **(2026-11-14)** Realistic bullet drag. Bullets already have quadratic drag, k = 8e-4 per metre at the floor (`Rules.bullet_drag`, physics/bullets.py), close to a 20 mm round's ~6.5e-4 per real metre. To check: whether the unit is a real or a standard metre (1/5.82 m) at real scale, and whether a rise near the speed of sound is worth modelling.

## Evaluation questions

59. Exploitability: train an exploiter against the frozen final model.
60. Specialists against the generalist (single-mode against all-mode).
61. Generalisation to unseen lineups, map sizes and team sizes.
62. Greedy against sampled actions at evaluation.
63. Live weights against the exponential average against the uniform average.
64. Elo against steps and against cost (scaling curves).
65. Comparison with the v1 and v2.1 champions.
66. Human play.
67. Cost per Elo point.

## Process

68. Noise calibration: the same config with two seeds (a seed twin of rl-v20 gives it with one run).
69. The final model with 2–3 seeds.
70. Equal steps against equal compute for capacity arms.
71. A bf16 against fp32 spot check.
72. Matched random streams across arms. The 2026-11-14 audit found almost everything on torch's global generator. Proposed fixes:
    - give model initialisation its own forked stream;
    - draw maps from a hash of (seed, environment, episode);
    - always draw the arena-scale slot, so turning map sizes on or off doesn't shift other draws;
    - give actions, the scripted agent, seats and stagger their own streams;
    - key each module's initial weights by its parameter path.
73. A round-robin rating of every arm's final policy, from the stored full-outcome records.

## Product (not arms)

- Docs in three tiers: the repository site, a one-page white paper, and a full paper (IEEE CoG, AIIDE, or a NeurIPS/ICLR workshop). Fix the evaluation figures before running the experiments.
- Gymnasium (vector and single-agent) and PettingZoo (parallel) wrappers: a few days of work.
- A browser replay viewer for the first release. A Rust/WASM port of the game comes later, with a parity harness, then multiplayer through an authoritative server.
- Hugging Face model cards, ONNX exports, a dataset of tournament records, and a Space.
- `CITATION.cff`, a changelog and releases; a demo video; a compute and cost section.
