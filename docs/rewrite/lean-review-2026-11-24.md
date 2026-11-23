# Independent review of lean, round 1 (2026-11-24, core at b5fc9bc8)

Read-only review by an agent that built none of it. CPU checks only.

**Verdict.** The core training path is sound: the action delay, seats, GAE chaining across
segments, hit-label rows, the next-state mask, the BC/teacher gate and the drain are consistent.
One training bug (B1), one evaluation bug (B2), a version-hash gap (B3); the rest is coupling and
draft-era leftovers. `pytest lean -m "not slow"`: 239 passed.

## Bugs
- **B1 (high)** Retiring league slots stop after one tick: `League.draw` (pool.py:249-250, via
  rollout.py:259-264 `draw_members`, called every tick) gives every environment
  `member_slot[member.clamp(min=0)]`; a RETIRED game (member -1) reads the scripted agent's -1,
  so the scripted agent takes over the rest of the game and the slot frees at the next prepare.
  Shown by scratchpad/r/retiring.py. Fix: keep the slot where the game did not end.
- **B2 (medium)** Adaptive allocation breaks for pairings with no games (evaluator.py:271 NaN
  score into rating.allocate): `allocate([(nan,0)]*5, 100)` gives [34, 34, 32, 0, 0]. Fix: pass
  (wins + draws/2, games).
- **B3 (medium)** The store's game version (store.py:55-63) hashes lean/game, lean/scripted and
  contract.py, but not the network inference path (lean/model, eval/runner.py, the learner
  helpers it uses), and does hash the analysis-only game/tuning.py.
- **B4 (low-medium)** Checkpoints are written in place (weights.py:62), not tmp-then-replace.
- **B5 (low)** Evaluation forces bf16 for networks (runner.py:125) instead of the policy's own
  precision setting.
- **B6 (low)** `--start` is silently ignored unless `--stage rl` (train.py:305).
- **B7 (low, tool)** The sandbox places ships by writing the state in place (a second spawn
  rule; world.py:76, 108), and the opponent's spawn speed ignores the medium.

## Coupling
- eval/runner.py imports learner internals (map_fields, reset_carry, sample_joint, seat_view,
  whole; Stepper, stepper_weights); one tick is written three times (rollout act/advance,
  runner.tick, World.advance).
- Circular import: rollout.py:539 imports EPISODE_CONSTANT from store.py, store.py:59 imports
  SEAT_AXIS from rollout.
- Process-global state: duck-shape flag set in two places; runner.play calls torch.manual_seed;
  the ship card mutated process-wide (tuning.py:414, 610, 756; the tool needs an RLock and a
  child process per session); `_with_row` does not clear core._SHIP_CARDS.
- Wide signatures: act returns an 11-tuple, advance takes 16 parameters, slot_actions 11.

## Bloat
- Production code used only by tests: game.reset_done, scripted.renew_memory,
  scripted.marginals, returns.support_usage.
- Stale docs: lean/LAPTOP_PROMPT.md; lean/README.md's first-cut text and v2-parity record;
  CONTRACT.md "First cut".
- symlog defined twice (game/core.py:1494, model/policy.py:184); the default 1,000 RL updates
  in train.py rather than LearnConfig; persona_spread misplaced in config.py; checkpoints pickle
  carry.next_episodes, which collect redraws.
- ruff: 532 findings (529 line length).

## Speed (not measured)
- The runner idles slots until the next segment when more than E games end in one segment.
- The scripted decision runs for every ship in evaluation even when a seat is a network.
