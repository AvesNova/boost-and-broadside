# Fight plan (one- or two-circle, held; hard turn after the pass): progress

- 01:15 worktree ../v2-fightplan (branch scripted-fight-plan from core 17e6cf3b). Read the agent,
  the ghosts notes (the head-on stalemate diagnosis: the plan flips every pass, the two-circle
  cap folds full stick into middle). Plan: memory column `plan` (plan x angles off at its choice),
  held for the engagement (same target), switched only when angles off grew by `switch_loss`
  since the plan was set; a per-match lean drawn in `fresh_memory` (like `side`) added to the
  energy advantage, so the yardstick keeps a mix too; no stick cap while the committed sense is
  held. Next: sibling sanity bench, before numbers on core. ETA before numbers 01:40.
- 01:18 change committed (baca2b9a), quick tier scripted+game 142 passed, ruff clean; pushed.
- 01:25 sanity before/after (sanity.py, 256 games each). Worked: plan switches per engagement
  1v1 1.75 -> 0.17, 4v4 0.22 -> 0.09; one-circle plans chosen 3-13% -> 22-28%. Didn't: 1v1
  without fields stalls more (draws 41 -> 78, head-on passes per kill 2.35 -> 5.19, kills per
  environment-minute 1.66 -> 0.99); 4v4 about unchanged.
- Ablations (scratchpad copies, same bench): hold only (old two-circle cap kept): 1v1 draws 31,
  kpm 1.93; full stick only (plan flipping as before): draws 27, kpm 1.84. Each alone helps;
  together they stall. Trying a stall rule (switch after N passes without gain) as a measured
  option, not as the default.
