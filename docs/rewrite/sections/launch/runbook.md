# Launch runbook: a rented box, from bootstrap to the leaderboard

The scripts in this directory run the final experiments on one rented machine of any number
of GPUs (vast.ai, `/workspace` a volume): the 1-GPU dry run, the 8-GPU Stage A box, and the
1-3 GPU box the winners continue on. Every job goes through `bnb queue` with one runner per
GPU (supervisor programs), so the box resumes on its own after a restart.

| file | what |
|---|---|
| `launch.toml` | the final config every arm shares, the lineup (two Yongma, three Jujak), the box's paths, the probe's and the tournament's settings |
| `arms.toml` | the arms (a proposal): the mainline, Stage A, the ablations |
| `bootstrap.sh REF [CACHE]` | the box: clone, `uv sync --frozen`, caches on `/workspace`, supervisor (heartbeat, a runner per GPU) |
| `env.sh` | every job's environment: the cache paths, compile threads and CPU threads per job |
| `cache.sh export\|import ARCHIVE` | the compile caches as one archive, with a manifest (GPU, torch, Triton, commit) |
| `launch.py` | `check`, `probe`, `enqueue`, `status`, `tournament`, `leaderboard`, `resume-to` |
| `replays.py` | whole matches of a policy as play-shell replays, headless, and `matches.jsonl` |
| `reward_check.py` | a run's reward, play and episode series by window, and the reward-hacking flags |
| `pull.sh` | the laptop's pull of results over the box's direct address (ssh/rsync) |
| `hub.sh` | the same transfers through the Hugging Face hub: the cache archive, the runs (checkpoints included), the laptop's download |

## The dry run on one 5090, in short

```bash
# box (root), once the game tree is frozen and launch.toml's lineup is set and pushed
bash /workspace/bootstrap.sh REF
cd /workspace/boost-and-broadside; L=docs/rewrite/sections/launch; source $L/env.sh
.venv/bin/python $L/launch.py check
.venv/bin/python $L/launch.py probe --group mainline        # compiles: fills the cache
bash $L/cache.sh export /workspace/compile-cache.tar.zst    # for the 8x box (same GPU model)
.venv/bin/python $L/launch.py enqueue --group mainline      # the chain to 300M, on runner-0
.venv/bin/python $L/launch.py status
# laptop: pull.sh (section 3); after the run: section 5 (tournament, replays, reward check)
```

Export the cache again after the run's first updates (the run compiles a few graphs the
probe does not: the RL warm start, the lanes' snapshots); the archive replaces the old one.

## Two transfer paths

- **ssh/rsync over the box's direct address** (`pull.sh`, `scp -P PORT root@IP:...`):
  needs nothing but the key; the relay address (`sshN.vast.ai`) drops anything longer than
  about 2 minutes, so never copy through it.
- **The Hugging Face hub** (`hub.sh`, repo `julianzia/boost-and-broadside-runs`, public,
  the project's run store, under `final/`): uploads resume where they stopped and a box
  reaches the hub at datacentre speed; the laptop downloads with no token. The box needs a
  write token in `/workspace/.env` (`HF_TOKEN=hf_...`, never committed); with it,
  bootstrap adds the `hub-sync` program, which uploads every run directory whole (the
  training checkpoints too, so any box can continue a run) every 30 minutes.

## 1. Rent and bootstrap

**Host memory for cold compiles.** A process compiling the learner cold holds about 12 GB of
host memory (measured 2026-11-12 on gpubox6, `compile-split/progress.md`; three at once
reached a 32 GB box's limit and the OOM killer). The queue's warm key staggers only the jobs
of one model (a second job of a key waits for the first one's first update); jobs of
different models (Stage A's arms) start cold at once, one per GPU, and nothing caps how many.
So rent for RAM: at least 16 GB per GPU beyond the runs' own use (the 8-GPU box: 128 GB
free; check `free -g` after bootstrap). On a box short of it, enqueue fewer arms than GPUs
until the first ones have logged an update (`launch.py status`), then the rest.

1. Rent the box. Note its **direct** ssh address (vast's "Direct ssh connect": `root@IP -p
   PORT`). The relay (`sshN.vast.ai`) drops any transfer longer than about 2 minutes: use it
   only for a shell, never for copies.
2. Freeze the tree: the game version and code the run will use, pushed to GitHub (the
   compile cache is keyed by the code, so build on the final game tree only). Set the
   lineup in `launch.toml` and push.
3. On the box, as root:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/AvesNova/boost-and-broadside/REF/docs/rewrite/sections/launch/bootstrap.sh -o /workspace/bootstrap.sh
   echo HF_TOKEN=hf_... > /workspace/.env; chmod 600 /workspace/.env   # for the hub path
   bash /workspace/bootstrap.sh REF                                  # first box
   bash /workspace/bootstrap.sh REF /workspace/compile-cache.tar.zst # a box given the cache by scp
   bash /workspace/bootstrap.sh REF hub:compile-cache-5090.tar.zst   # ... or from the hub
   ```

   It logs to `/workspace/bootstrap.log`; at the end `supervisorctl status` shows
   `heartbeat` and `runner-0` ... `runner-<N-1>` RUNNING (an empty queue: they wait).

## 2. Check, probe, queue

```bash
cd /workspace/boost-and-broadside
L=docs/rewrite/sections/launch; source $L/env.sh
.venv/bin/python $L/launch.py check            # refuses the placeholder lineup; prints each arm's differences
.venv/bin/python $L/launch.py probe --group mainline   # or --group A on the 8x box
.venv/bin/python $L/launch.py enqueue --group mainline --dry-run
.venv/bin/python $L/launch.py enqueue --group mainline
```

- `probe` runs `bnb train --probe` (setup and one update) per distinct model and stage, one
  GPU each, from the largest of `[probe]`'s micro-batch rows down, and keeps the largest
  whose reserved peak is under 90% of the GPU. Micro-batches add up to the minibatch, so
  the choice changes memory and speed, not the update. Results:
  `/workspace/runs/launch/micro_batch.json`; logs beside it. The probes compile, so they
  also fill the compile cache with the shapes the runs then use.
- `enqueue` adds each run as a queue job: a chain (`bnb train --objective chain`) with the
  probed rows per stage, priority = the arm's cost (the longest start first), and a warm
  key per model (a second job of the same model waits until the first has logged its first
  update, so it reads the first one's compiles).
- What a chain does: BC (no rating lanes, `bc_set`'s `lanes.environments=0`) stops on its
  plateau or at its 100M cap, never on its score stop, whose windows need lanes; RL then
  starts from BC's exponential average (`policy-exponential/`{.plain}, the same weights as
  `policy/`{.plain}), always, and its live rating starts at 0, since a BC run without
  lanes rates nothing. RL's training pool is its snapshots, both averages and the
  full-strength scripted agent: no weaker rung and no random player.

## 3. Watch

```bash
.venv/bin/python $L/launch.py status     # per job: stage, chain steps, SPS, ETA (KST), KL, lane scores
.venv/bin/bnb queue list --queue-file /workspace/queue/queue.json
.venv/bin/bnb queue log 1 --queue-file /workspace/queue/queue.json
tail /workspace/heartbeat.log            # a gap is downtime
```

On the laptop, a pull every 20 minutes (records, metrics, logs, the tournament's records
and the policies its field names; not training checkpoints):

```bash
systemd-run --user --unit bnb-pull -p MemoryMax=2G \
  docs/rewrite/sections/launch/pull.sh ~/avesnova/runs-5090 root@IP PORT 20
cat ~/avesnova/runs-5090/status.txt
# or from the hub (what hub-sync has uploaded; no checkpoints unless --checkpoints):
docs/rewrite/sections/launch/hub.sh pull-runs ~/avesnova/runs-5090-hub mainline-s0-bc mainline-s0-rl ratings
```

## 4. After a restart

Nothing to do: supervisor starts the runners at boot; each settles its own slot's jobs
that were running (marked interrupted) and queues them again as resumes from their newest
checkpoints (a chain resumes in the stage it was in). `supervisorctl stop runner-K` stops
slot K's job gracefully (after its update; up to 15 minutes) and queues it again.

## 5. Tournament, replays, reward check

```bash
.venv/bin/python $L/launch.py tournament --target 50 --shards 1   # round 1: coarse
.venv/bin/python $L/launch.py tournament --target 25 --shards 3   # later: tighter, 3 GPUs a preset
.venv/bin/python $L/launch.py leaderboard                         # every round's games, one fit
```

- A round's field is every run's policy files at `[tournament].steps` (chain steps, BC
  included) reached so far, the averaged agent (`uniform_average`) and live, labelled
  `<run> <steps>M averaged|live (<stage> <exact steps>M)`, plus the scripted agent (1000).
  Each preset is rated on its own. The round's jobs go before waiting training jobs
  (priority 100), each reusing every earlier round's records, so a later round plays only
  where the errors are still above its target. With `--shards N` each preset runs as N
  jobs (own seed, own record, each at target x sqrt N); `leaderboard` fits them together.
- Every player flies `launch.toml`'s lineup. A record does not keep the lineup: reuse only
  records of the same one (a new lineup: a new runs directory).

```bash
P=/workspace/runs/mainline-s0-rl/policies/step_<N>/uniform_average
.venv/bin/python $L/replays.py --policy $P --out /workspace/runs/replays/mainline --matches 3 \
  --lineup Yongma Yongma Jujak Jujak Jujak
.venv/bin/python $L/reward_check.py /workspace/runs mainline-s0 \
  --matches /workspace/runs/replays/mainline/matches.jsonl > /workspace/runs/reward-check.md
```

On the laptop, open a replay with `bnb play --replay DIR`.

## 6. Keep the compile cache, continue a run

```bash
bash $L/cache.sh export /workspace/compile-cache.tar.zst
bash $L/hub.sh push-cache /workspace/compile-cache.tar.zst compile-cache-5090.tar.zst   # hub path
# next box: bootstrap.sh REF hub:compile-cache-5090.tar.zst (downloads, checks the SHA-256, imports)
# or by scp over the direct addresses, through the laptop:
#   scp -P PORT root@IP:/workspace/compile-cache.tar.zst .
#   scp -P PORT2 compile-cache.tar.zst root@IP2:/workspace/; bootstrap.sh REF /workspace/compile-cache.tar.zst
```

The learning rate is a warm-up then constant on environment steps, so a chain that ended at
its budget continues as the same run: `launch.py resume-to 500000000 mainline-s0` raises
the chain's combined budget and queues its resume (from its final checkpoint, which the box
keeps). On another box: `hub.sh pull-runs /workspace/hub mainline-s0 mainline-s0-bc
mainline-s0-rl --checkpoints`, move `/workspace/hub/final/runs/*` into `/workspace/runs/`,
then `resume-to` (or copy the run directories over the direct addresses).
