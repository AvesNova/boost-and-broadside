# Overnight log, 2026-11-14

This log records the changes and decisions made while the author was away, with the evidence for each. Times are KST. The newest entries are at the bottom.

## 02:00 — mandate

1. Finish bc-v19 so it yields data for balancing the geometry head.
2. Build Yemong v8 with the knee raised.
3. Decide whether to make settings tensors.
4. BC to about 1000 Elo against Baekho, with a budget of at most 150M steps.
5. Then calibrate RL from the BC run, aiming at about 2000 Elo.

## 02:00 — decision: settings stay baked-in numbers (not tensors)

- **What the change would buy:** about 7 minutes of compile saved per run that differs only in numbers. On 8 GPUs the arms compile in parallel, so that is about 7 minutes of wall-clock time per round, under an hour for 10 rounds.
- **What it would cost:** plumbing through the returns, losses, map sampling and play tuning, plus tests. Values that switch work on or off, such as a zero weight skipping a head, can't be tensors without computing both paths. CUDA graphs need in-place updates.
- **Verdict:** the bug risk outweighs about an hour of compile. I'll revisit if compile time becomes a real bottleneck, for example cold compiles on shared-CPU hosts.

## 02:00 — geometry head: total weight kept at 0.007

- **The design rule:** set in commit e334d032, the weight targets a trunk gradient norm of about 0.02, the level the next-state head reaches in trained BC.
- **bc-v19 measurements:**
  - `gradient/norm/trunk/geometry` is 0.0205 at update 10 and 0.0218 at update 20.
  - next_state is 0.0224 and 0.0132; density 0.076 and 0.092; policy 0.029 and 0.072.
- **Verdict:** on target, so it stays.
- **The balance of its five outputs needs per-output gradients.** Those are being added to the diagnostics in v8 and measured on a short CPU run, using the method from learner.md.

## 02:00 — bc-v19 flags at update 17 (Sonnet triage): none is a run killer

- **`sigma_miscalibrated`:** a monitoring artifact. It is a mean of error²/σ² over near-constant quantities. Robust calibration is fine, with 60–80% of errors within 1σ. The fix to the trigger goes into v8.
- **Next-state loss:** peaked during the combat ramp-up and is now falling.
- **Correction loss:** rising slowly and flattening; I'm watching it.
- **Damage critic explained variance:** recovering; 0.19 at update 11, 0.44 at update 17.
- **To watch:** BC KL has been flat at 0.46–0.47 since update 10 (v1 reached 0.40). The stick head is the weak part, at 76% agreement.

## 02:00 — Yemong v8 (agent, in progress)

- **Velocity code:** knee 100, spacing 5, about 97 bins. Reach stays about 1006 m/s.
- **Diagnostics:** the per-output gradient split for the geometry head.
- **Alert:** the `sigma_miscalibrated` trigger made robust.

## Live Elo (averaged policy against Baekho = 1000)

| step | deathmatch | elimination | frontline |
|---|---|---|---|
| 10M | 548 ± 35 | lost every game | lost every game |
