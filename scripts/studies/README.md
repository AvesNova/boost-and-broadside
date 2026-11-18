# Studies behind the v1 results

The fan charts, the crossover by size, averaged against final live weights, game length
by ratio, the survivor chart and the two `wild-universe-731` clips in the README and
`docs/` come from the scripts here, not from a `bnb` mode, so they have no artifact under
`checkpoints/wild-universe-731/artifacts/`. This page is their record: the commands, the
seeds, the games per point and the code they ran against.

- `fan/` measures win rates of fixed learned-team sizes against larger scripted teams.
- `clips/` plays seeded headless batches, then re-renders chosen games as clips.

Every launcher can be started from anywhere; it runs from the repository root, reads the
checkpoints under `checkpoints/wild-universe-731/` and writes under `out/studies/`
(`FAN_DIR` overrides the fan directory). The laptop launchers take a GPU lock
(`GPU_LOCK`, default `out/gpu.lock`) and run each job in a 10 GB `systemd-run --user`
memory scope. All of it needs a CUDA GPU.

The results reach the docs through `scripts/docs_values.py --pull [DIR]`, which copies the
merged data and light charts from `out/studies/` (or `DIR`, or `$BNB_STUDIES_DIR`) into
`docs/results/v1-731/`, then rebuilds `docs/values.json`.

## Code version

The studies ran on 2026-11-17. Every laptop job (RTX 4070 Laptop) ran from a clean
checkout of `main` at `d017f0b0`. The 16- and 32-ship seed-1 sweeps ran on a rented
RTX 5090 from a fresh clone of `main` made that day; its logs do not record the commit, but
`main` has been at `d017f0b0` since 2026-09-07, so the clone was that commit. The scripts
here are the ones that ran, with three kinds of change since: absolute paths replaced by
the defaults above, lint fixes (variable renames, line wraps), and, in `fan/extras.py`,
chart notes that read the game counts from the data and end labels that no longer overlap.
The data files are unchanged by the redraw.

## Fan study

`fan/fan_eval.py CKPT OUT.jsonl --trained N --scripted S... --games G --seed K` plays `G`
games of `N` learned ships (team 0) against `S` stochastic scripted ships (team 1) for each
`S`, in chunks of at most 1,024 parallel environments, under the checkpoint's own ship and
environment settings (for this run the same as `main`'s physics). It seeds PyTorch once
with `K` at the start, then writes one JSON line per point with the wins, losses and ties,
the ties that hit the episode cap, and the mean game length in ticks. Ties count as
non-wins in `win_rate`.

The grid is `N, 1.25N, 1.5N, 1.75N, 2N` scripted ships for `N = 4, 8, 16, 32`, at 9 frozen
ladder checkpoints, the final live checkpoint `step_000671846400.pt` and the final averaged
checkpoint `avg_step_000671846400.pt`: 55 points per size.

| Ships | Seed 1 sweep, 1,024 games per point | Seed 2 extra games (final live and averaged only) | Games per final point |
|---:|---|---|---:|
| 4 | laptop; the same `fan_eval.py` call as `run_n8_laptop.sh` with `--trained 4 --scripted 4 5 6 7 8` (no launcher kept) | `run_extra_laptop.sh`, +3,072 | 4,096 |
| 8 | `run_n8_laptop.sh` | `run_final_laptop.sh`, +3,072 | 4,096 |
| 16 | `run_box.sh` | `run_final_laptop.sh`, +3,072 | 4,096 |
| 32 | `run_box.sh` | `run_final_laptop.sh`, +2,048 | 3,072 |

Every ladder point stays at its 1,024 seed-1 games. `run_extra_laptop.sh` also played the
seed-2 games for one ladder checkpoint at 4 ships before it was stopped;
`merge_extra.py` leaves ladder rows of the extra file out.

To reproduce, run the sweeps, rename each seed-1 output `fan_731_nN.jsonl` to
`fan_731_nN.base.jsonl`, run the extra games, then merge and draw:

```bash
scripts/studies/fan/run_n8_laptop.sh            # and the 4-ship equivalent
scripts/studies/fan/run_box.sh                  # 16 and 32 ships
for n in 4 8 16 32; do mv out/studies/fan/fan_731_n$n.jsonl out/studies/fan/fan_731_n$n.base.jsonl; done
scripts/studies/fan/run_extra_laptop.sh         # 4 ships (stop it after the 4-ship size)
scripts/studies/fan/run_final_laptop.sh         # 8, 16 and 32 ships
scripts/studies/fan/finalize.sh                 # merge, fan charts, crossover, extras
uv run --no-sync python scripts/docs_values.py --pull
```

`finalize.sh` runs `merge_extra.py` (sums the counts per checkpoint and point, recomputes the
rates), `plot_fan.py` (one fan chart per size; a dotted wins-plus-half-the-ties line per
ratio when ties pass 5% of games at some point), `crossover.py`, `extras.py` and
`win_rate_by_size.py`. `extras.py` interpolates the 50% crossover ratio between grid
points, with a 95% interval from 4,000 parametric bootstrap resamples (NumPy seed 0).

## Clips and survivors

`clips/hero.py batch` plays one batch of `--envs` games per seed with the final averaged
checkpoint, never resetting a finished game, and logs each game's winner, survivors,
length and ship counts every 8 decisions. `hero.py render` replays a batch and films the
picked games; the same batch on the same GPU reproduces every game, and each rendered game
is checked against its logged winner, survivors and length (all matched). `hero.py`
refuses a checkpoint whose physics differs from `main`'s.

| Batch | Scenario | Seeds | Games per seed | Games | Used for |
|---|---|---|---:|---:|---|
| 8 v 15 | vs scripted | 1001–1008 | 512 | 4,096 | survivor chart (`survivors_8v15.json`) |
| 4 v 7 | vs scripted | 2001–2002 | 512 | 1,024 | README hero clip: seed 2001, game 406 |
| 32 v 50 | vs scripted | 4001–4002 | 128 | 256 | large-battle clip: seed 4001, game 57 |

```bash
scripts/studies/clips/run_batch.sh 8 15 vs_scripted out/studies/clips/logs/games_8v15.jsonl 1001 1002 1003 1004 1005 1006 1007 1008
scripts/studies/clips/run_batch.sh 4 7 vs_scripted out/studies/clips/logs/games_4v7.jsonl 2001 2002
ENVS=128 scripts/studies/clips/run_batch.sh 32 50 vs_scripted out/studies/clips/logs/games_32v50.jsonl 4001 4002
uv run --no-sync python scripts/studies/clips/analyze.py out/studies/clips/logs/games_8v15.jsonl --survivors-out out/studies/clips/survivors_8v15.json
uv run --no-sync python scripts/studies/clips/plot_survivors.py out/studies/clips/survivors_8v15.json out/studies/clips/survivors_8v15
scripts/studies/clips/render.sh 4 7 vs_scripted out/studies/clips/4v7_against_odds 4v7 out/studies/clips/logs/games_4v7.jsonl 2001:406
ENVS=128 scripts/studies/clips/render.sh 32 50 vs_scripted out/studies/clips/32v50_large_battle 32v50 out/studies/clips/logs/games_32v50.jsonl 4001:57
```

The two GIFs were copied into `docs/results/v1-731/clips/` as
`hero_4v7_seed2001_env406.gif` and `large_32v50_seed4001_env057.gif`.
