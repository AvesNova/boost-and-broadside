# Brief: why RL makes the BC clone worse

The first RL run warm-started from a good BC clone and made it worse, fast. Find out why,
with evidence, and fix it.

## What we know

- **The clone:** `/home/vizia/avesnova/runs-v2/2026-11-04-mixed-wandering-quill/policy`.
  Its closed-loop score against the scripted agent was about 0.40 over 1,000-game windows
  (BC, 100M steps, mixed preset, fog on).
- **The RL run:** `/home/vizia/avesnova/runs-v2/2026-11-04-mixed-tidy-harbour` (PPO,
  `--init-from` the clone, mixed preset, stopped by Ctrl-C at about update 237 of 381;
  read-only: its checkpoints, `metrics.jsonl`, `config.json`). Its launch log is
  `/home/vizia/avesnova/runs-v2/rl-launch.log`.
- **A metrics sweep** of both runs: `/tmp/bnb-scratch-run-analysis/findings.md`, with
  per-series summaries (`rl_summ.tsv`, `bc_summ.tsv`) and charts. Its main points:
  - `pool/score/scripted` falls to about 0.04 from update ~26 on, and the live policy
    loses 0.16 to its own 20-update weight average;
  - joint entropy rises from 0.76 to 2.07 (power +0.51, stick +0.40, shoot +0.28);
    the fire share goes from 2.5% to 22.7%, the hit rate from 7.7% to 3.2%, and the
    damage per ship-second from 3.2 to 1.2;
  - the KL sits at its 0.02 target every update, and 62% of updates stop their epochs
    early; the printed `drift` is 0.25–0.5 from update 1;
  - the entropy bonus's gradient is about zero; the policy term dominates. So the policy
    gradient itself points the wrong way: suspect the learning signal, not the tuning;
  - fog and the game mix are the same in BC and RL.
- **Earlier bugs of this shape:** a seat-symmetry bug once made team 1 unplayable (the
  learner's view of the second seat was wrong). Self-play is half the RL data, with two
  learner views per game.

## Suspects, in the order to check

1. **The advantage's sign or alignment per seat or view:** the rewards per team (zero-sum
   events), the value bootstrap at episode ends and respawns, GAE across segment
   boundaries, which view's reward a token gets.
2. **The action log-probabilities:** rollout against learner (masks such as a gun on
   cooldown, the factor order, the temperature). What is `drift`, and why is it 0.25 at
   update 1, before any update?
3. **The rewards:** an event that pays for firing, or no cost for wasted shots. Check the
   reward-event shares and the weights against the game.
4. **The warm start:** the critic and the return scale carried from the clone; a fresh
   optimizer at a learning rate of 3e-4 after a 1M-step warm-up.

## How

- **Measure where strength falls:** rate the clone and the RL run's checkpoints offline
  against the scripted agent with enough games for a standard error under 0.03
  (`bnb ratings` / `bnb head-to-head`; read their docs).
- **Read the code paths** above, and write a CPU test for each suspect you can pin down: a
  hand-built case whose correct advantage, reward or log-probability is known.
- **Short RL ablations** from the clone, each long enough to see the collapse (it shows by
  update ~25 in the fire share, the entropy and the score against scripted). Run the
  control first. Make the score against scripted measurable each update with many games if
  you can (for example a pool against the scripted agent), and say how. Candidates: no
  self-play, the entropy coefficient at 0, a lower learning rate, the critic re-fitted
  before the policy moves. Choose by what the code reading suggests.
- **Fix** what you find on your branch, with a test, and show with a short run that the
  clone no longer degrades (its score against scripted holds or rises over ~50 updates).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-rl-degradation`, branch `fix/rl-degradation`
  (from v2). Work only there. Scratch and your runs in `/tmp/bnb-scratch-rl-degradation`
  (`--runs-directory`). Read the two run directories and the analysis scratch above; do not
  read any other directory under `/home/vizia/avesnova`.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **GPU:** you are the only GPU job. Hold the benchmark lock for every GPU command:
  `uv run --no-sync python ~/.local/share/bnb/locked.py <command...>`. Never run two GPU
  jobs at once.
- **W&B:** no `--wandb` on your runs; local metrics only. Credentials are in `~/.netrc`;
  never print them.

## Done

- The cause, or causes, named with evidence; a fix with a test; a short run showing the
  clone holds its strength. If you cannot find it, report what you ruled out and how.
- Commit to `fix/rl-degradation` and push. Author Julian Kizanis <julianzia@gmail.com>; end
  every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message, one page: the strength curve of the checkpoints, each suspect with its
  verdict and evidence, the fix and its commit, and the ablations' results in a table.
