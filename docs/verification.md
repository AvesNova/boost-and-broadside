# Documentation verification report

- Verification date: 2026-07-28
- Audited code baseline: `2592ebdce35269c908c191a70725743886f96b93`
- Landmark result set: `resilient-resonance-682` (`chpl40cj`)

## Outcome

The root README and supporting documentation form one consistent narrative around
zero-shot team-size transfer. Technical depth is routed into focused pages, while raw
artifacts and claim-level evidence remain linked at the point where they are discussed.

The active documentation stack is:

- [`README.md`](../README.md): showcase, central result, system overview, and routing;
- [`getting-started.md`](getting-started.md): setup, commands, checkpoints, and development;
- [`environment.md`](environment.md): simulation, actions, physics, and validation;
- [`architecture.md`](architecture.md): features, attention, recurrence, and heads;
- [`training.md`](training.md): recurrent PPO, rewards, opponents, ratings, and checkpoints;
- [`evaluation.md`](evaluation.md): crossover, ELO, diagnostics, methods, and limitations;
- [`replays.md`](replays.md): curated qualitative evidence and capture provenance;
- [`evidence.md`](evidence.md): claim map and deferred asset ledger;
- [`engineering/memory-optimization.md`](engineering/memory-optimization.md): retained
  engineering analysis, moved from the top-level docs directory.

The former `docs/game_design.md` was replaced by `environment.md` after correcting its
projectile-velocity and ship-collision descriptions. The former `docs/mem_profiling.md`
was retained under the engineering path.

## Checks performed

| Check | Result | Notes |
|---|---|---|
| Local Markdown targets | Pass | Every relative Markdown link/image target across repository Markdown resolves to an existing file or directory. Fragment targets in the new stack were also reviewed against their headings. |
| Referenced media | Pass | All README/supporting-page PNG and GIF paths exist. Curated replay dimensions, durations, and sizes were inspected with `file`/`ffprobe`. |
| Hero replay outcome | Pass with provenance caveat | The 8-vs-11 terminal frame has three blue learned-policy survivors and no red scripted ships. The GIF lacks a checkpoint/capture sidecar. |
| Chart reproduction | Pass | The history, crossover, and fleet-scale renderers ran against included artifacts with outputs directed to `/tmp`; all nine regenerated PNGs were byte-identical to the tracked files. |
| Crossover numbers | Pass | README/evaluation values were checked against `docs/crossover/crossover.json`, including adjacent winning/losing boundary points. |
| ELO numbers | Pass | Live, frozen-checkpoint, scripted, standard-error, game-count, tie-convention, and shared-anchor values were checked against `elo_calibrated.json`. |
| Fleet-scale ratings | Pass | Directed outcomes conserve every completed game at all seven sizes; aggregate win/tie matrices reproduce those outcomes, and all three anchor views derive from the same stored counts. |
| Run/configuration numbers | Pass | Training scale, model size, environment count, step target, runtime, throughput, and hardware were checked against the W&B export. |
| CLI import/help | Pass | `main.py --help` imports and lists the modes and flags documented by the new pages. |
| Test suite | Pass | 362 passed; six hardware-specific tests skipped; 72 expected CPU-visible CUDA-autocast warnings. |
| Ruff | Pass | `ruff check .` reports no issues. |
| Diff whitespace | Pass | `git diff --check` reports no tracked-file whitespace errors. |

The sandbox's normal home cache is read-only, so Python commands used a temporary uv
cache under `/tmp`. This changes cache placement only, not the command behavior under
test.

## Terminology consistency

The reader-facing stack consistently distinguishes:

- **policy/controller:** one learned recurrent network acting for a whole team;
- **policy-controlled ship:** one ship receiving one action emitted by that network;
- **scripted-controlled ship:** one ship acted for by the scripted controller;
- **4-vs-4 training:** eight total ships in the landmark environment;
- **zero-shot team-size transfer:** evaluation at unseen team sizes with unchanged weights;
- **online ELO:** the sequential in-training estimate;
- **calibrated ELO:** the post-hoc Bradley-Terry reconstruction used for results.

The crossover renderer and rasters use “policy-controlled ships” and
“scripted-controlled ships.” Policy-controlled ships are on x, scripted-controlled ships
are on y, and equal 0–64 data scales make the parity line 45°.

## Commands not exercised

The following were intentionally not executed during a read/documentation-focused audit:

- `uv sync` in a clean clone, because it can require package/network access;
- `main.py --mode rl --smoke`, because it writes training/checkpoint output;
- interactive watch mode;
- GPU replay regeneration;
- the underlying crossover and ELO tournaments, which are expensive and already have raw
  stored artifacts.

Accordingly, the setup page does not claim clean-install CI coverage or that every GPU
workflow was rerun during this documentation change.

## Remaining documentation debt

These items do not block the current stack:

1. add source commit, checkpoint hash, seed policy, and uncertainty to crossover output
   (new runs now preserve wins, losses, ties, games, and mean episode length per cell);
2. add replay JSON sidecars and poster frames;
3. curate a boundary/failure replay;
4. replace the tall policy raster with a wide, count-agnostic system diagram;
5. add a compact uncertainty-aware ELO figure;
6. add a `LICENSE`, contribution guide, and CI workflow before making corresponding
   open-source/contribution promises.

The detailed priorities and rationale remain in the
[deferred asset and analysis ledger](evidence.md#deferred-asset-and-analysis-ledger).
