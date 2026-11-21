# Briefing: preparing the 5090 pairwise arms (handover, 2026-11-15)

**Superseded in part (2026-11-15, session 3):** break-even now fixes the weaker side at 20 (no 30-ship cap, no floor) and frontline break-even is scored; the open items below were answered there. [arms-todo.md](arms-todo.md) is current.

This hands the work over to a new session (a remote server; the laptop is shutting down). Branch `prep/arms`, from `v2` at 87532adf, which now holds everything merged so far (integrate/next-runs fast-forwarded into v2).

Read these first:
- [arms-todo.md](arms-todo.md), the tracking document: **Open for the author** (current), every dated decision, the work in progress and what's left to implement. Update and commit it whenever something is decided or changes.
- [arm-review.md](arm-review.md): every candidate arm (A1–A24, R1–R6) with its question, purpose, cost and importance.
- [evaluation-suite.md](evaluation-suite.md): how an arm pair is decided (`bnb evaluate`) and the design pilot.
- [sections/mode-timing/results.md](sections/mode-timing/results.md): the game-length measurements behind the freeze (deathmatch kill rate, elimination wipe times by map size and by shrink, frontline small teams).
- [sections/frontline-lock/results.md](sections/frontline-lock/results.md): the frontline lock re-tune to a random walk.

## The goal

Get everything ready for the pairwise-arm runs on rented 5090s:
1. Freeze the game (clocks, tickets, the elimination shrink, minimum team sizes; then one game-version bump).
2. Settle the evaluation and decision method. Done; the numbers are being re-measured on fixed maps.
3. Choose the arms that fit the budget.

## Where the discussion stands

**The game freeze**, the active topic. All numbers are suggestions, not locked in.
- **Frontline:** win at ±2 (was ±3) with a 10-minute clock. With the front now a random walk (lock 6 s/√km, P(reversal) 0.49), reaching ±k takes about k² captures; at 5v5, ±2 ends 79% of games by capture within 10 minutes, ±3 only 47%. Smaller teams capture more slowly (3v3 68% and 2v2 36% at ±2 in 12 minutes), so the minimum is 3v3.
- **Deathmatch:** 40 tickets with 10 minutes. Imugi kills 4.57 per team per minute at 5v5, so 40 tickets end about 80% of games by tickets (±5 tickets).
- **Elimination:** the author wants the collapsing map to decide games more often, not necessarily by killing ships directly. What the sweeps showed (Imugi, 5v5, 256 games per row):
  - On a fixed map, the radius hardly matters down to about 2.5 km: deaths per minute 4.5 to 7.4, median game 58 to 42 s. Below about 2 km the kill rate rises roughly as one over the radius (1.8 km 10.5, 1.2 km 16, 800 m 27, 500 m 42 per minute), and the boundary's share of deaths goes from under 5% to 17–40%.
  - So today's linear shrink from 7.8 km acts only in its last quarter: across 5 start times × 4 durations the median stayed at 51–58 s, and only the tail and mean moved.
  - **Proposed next step (awaiting the author):** a "fast, then slow" shrink law (fast to about 2–2.5 km, then slowly to zero) and a sweep of its shapes. Smaller alternatives: a smaller starting map; a start at 0 with a short shrink (moves the median to 47 s, but mostly by boundary kills).
- **More freeze data offered, not yet chosen:** trained agents (rl-v20 self-play) instead of only Imugi; kill rates and wipe times at 1v1, 2v2 and 10v10 on the fixed maps (deathmatch tickets scale per ship today); frontline at ±3 with faster captures.

**Fixed-size maps (the author's firm rule, 2026-11-15).** Every map is its preset's own size at every team size until an agent trains successfully on variable-size maps (arm A1); if that fails, fixed-size full stop. Merged: training's map-size sampling is off by default (opt-in `arena_scale` for A1), and evaluation and break-even use the preset maps (`--map-rule density` opt-in).

**Evaluation (decided method, numbers being re-measured).**
- Win-probability scoring: each mode a third, the conditions equal within a mode.
- A group-sequential test: K = 4 looks, δ = 0.02, α = 0.05, power 0.8, with cost-aware allocation, a no-regression gate and the challenger rule.
- The first pilot (on density maps) gave about 0.6–0.85 laptop engine-hours per decision; seed noise (about 110 Elo between seeds), not games, limits decisions, and arm A3 sets δ from it.
- The pilot is being re-run on the fixed maps (see below).
- Break-even now goes up to 30 of our ships against 20 Imugi; the author questioned why weaker-than-Imugi agents matter (answered; open item 2).
- Frontline break-even is unscored by default; Claude recommends scoring it again now that it plays on the preset map (open item 1).

**Other results today.**
- Exploit check: rl-v20 (171M) beats all six cheese agents except the frontline zone rusher (0.55); Imugi beats the rusher 0.75, so it's a cycle, a blind spot of a policy trained without cheese agents. Recheck on the first arm.
- Rating layout probe (laptop): the GPU is saturated at 2 blocks; keep 2 × 96; probe again on the 5090.

**Merged today into v2:**
- `bnb evaluate` and the design tools;
- the training defaults (kill-lead deathmatch outcome, mirrored 50/50 lineups, mode mix 6/5/5, (1−p)² league sampling with no floor, the snapshot schedules, BC fixed at 100M);
- per-update policy files off by default (on only for the averaging arm);
- the Imugi rename (recorded identities stay `scripted…`);
- fixed maps.

## Open tickets

See **Open for the author** in [arms-todo.md](arms-todo.md) (8 items). In short:
1. score frontline break-even again;
2. break-even up to 30;
3. league retention;
4. the freeze values and the elimination shrink;
5. which extra freeze data to collect;
6. the run budget and the arm list;
7. the averaging sweep variants;
8. the older unmerged branches.

## Work interrupted by the shutdown

- **The fixed-map design pilot** (`evaluate-pilot-fixed` unit on the laptop, started about 09:45) dies with the laptop. Re-run it where the checkpoints are:

  ```bash
  RUNS=<runs directory with rl-v20 and rl-mix-kl>; STEP=policies/step_75497472/exponential_average
  uv run --no-sync bnb evaluate $RUNS/rl-v20/$STEP $RUNS/rl-mix-kl/$STEP \
    --out <dir>/pilot-fixed/rl-mix-kl --device cuda --games 512 --unscored
  ```

  `--unscored` with no names plays frontline break-even too. Then `bnb evaluate decide <dir> --peek` for the per-condition variances and costs, and `bnb evaluate design <dir> --looks 2 3 4 5 --simulations 400`. Record the result in evaluation-suite.md's "Design pilot". It replaces the density-map numbers for 1v1, 2v2, 10v10 and break-even, and measures what 30-a-side break-even games cost. On the laptop the checkpoints were under `/home/vizia/avesnova/boost-and-broadside/data/box-5090-backup/runs/`; the server needs a copy.
- Nothing else is running.

## Working agreements (from the author)

- **Questions are questions:** "what do you think" asks for an opinion, never a go-ahead. Answer, propose, and wait for agreement before launching anything not already agreed.
- **Decisions only when explicit.** Most of the author's ideas are suggestions; be critical of them and say so. The dictation is imperfect: double-check odd words ("illumination" means elimination, "Imuji" means Imugi).
- **Features on by default**, with a setting to turn each off for ablations. **No legacy code** or shims.
- **Tracking:** every decision goes into arms-todo.md, committed and pushed. "Open for the author" lists go in a reply only at transitions or on request.
- **Commit and push as you go.** Commits end with the `Co-Authored-By` trailer; never bare `git stash`.
- **Runs:** `uv run --no-sync`, never bare python. One GPU job at a time.
- **Tests:** quick tier `-m "not slow"` on the touched files only; no whole-tree suites. Then docs, then commits.
- **Reporting:** ETAs as clock times ("ETA 14:45, worst 15:05"). Long jobs persist every batch to disk, stream progress and have a bound. Record the power profile before any timing.
- **Fixing bugs:** fix bugs (performance ones too) without asking, and report them.
- **Rented boxes:** no credentials on rented boxes; never destroy or reboot one.
