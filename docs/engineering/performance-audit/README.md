# Performance experiment handoff — start here

Branch: `perf/realtime-experiment-handoff`. The next agent's self-contained task
is [performance-handoff-prompt.md](../performance-handoff-prompt.md).
The user is now plugged in and authorizes intensive experiments. **Never change
power profiles, governors, GPU power limits, sleep, or keep-awake settings.**
Deadline: **2026-09-20 22:00 Asia/Seoul (13:00 UTC)**. No experiment runner or
restart scheduler was installed during handoff preparation.

## Reading order

1. This index: provenance, conclusions, and reproduction.
2. [Audit report](../realtime-and-training-audit.md): source analysis, alternatives,
   measurements, limitations, and acceptance criteria.
3. [Launch prompt](../performance-handoff-prompt.md): bounded task and agent roles.
4. [File inventory](INVENTORY.md) and [integrity manifest](manifest.json): every
   archived file, original location, and SHA-256 checksum.

## Established evidence

| Evidence | Result | Implication |
|---|---|---|
| 50v50 CPU, two policies, full frame | Roughly 1.6 seconds; compiled LOS about 0.68 seconds | Dense visibility is costly, but fixing it alone misses real time |
| 50v50 CUDA projectile LOS | 5.494 ms reference; 0.271 ms compiled real; 0.545 ms Warp | Alternatives passed sampled boolean parity, not complete-game parity |
| 50v50 CUDA full-frame comparisons | Roughly 200 ms; neither LOS substitution materially helped | Optimize more of the complete path |
| 50v50 drawing phases | Roughly 92–109 ms | Renderer needs independent attention |
| CUDA environment-only B=1 to B=960 | Roughly 67–73 ms per batch step | Fixed overhead is substantial; excludes policy/PPO/storage |
| Actual PPO, interrupted | Warmup 578.93 s; one measured update 215.94 s, two epochs | No completed aggregate or repeatability claim |

The report contains precise aggregation and raw links. These are different
workloads; do not add phase medians or infer speedups across unrelated runs.
No tested complete-frame configuration reached 30 Hz. Short random-policy runs
do not certify trained combat, long-run tails, or 60 FPS.

## Archive layout

| Location | Contents | Status |
|---|---|---|
| `results/` | 22 fresh CPU/CUDA audit JSON files | Completed measurements; metadata evolved during the audit |
| `historical/` | 40 original repository benchmark JSON files | Older/separate inference, training, concurrency, real-time and rendering runs |
| `logs/` | 18 console logs/manifests, including partial PPO | Supporting provenance; not all represent completed measurements |
| `manifest.json` | Original paths, checksums, baseline revision, source hashes | Handoff integrity; not an invented measurement-time source snapshot |

All raw files are byte-preserved. Embedded paths refer to original locations.
Write new outputs elsewhere; never rewrite historical measurements.

Historical groups:
- `realtime/` and `realtime_render/`: CPU/CUDA, 5v5/50v50, one/two policy sides;
  these help explain the user's original screenshot.
- `concurrent-realtime*`: buffered sequential, streams, host threads, interleaved
  trials; see [concurrency analysis](../concurrent-rollout-plan.md).
- `inference_*`, `train_*`, `presence_density.json`, `ppo-overlap.json`: earlier
  model/training studies; see [historical throughput investigation](../rl-throughput.md).

Some historical files lack timestamps, source hashes, or full hardware metadata.
Do not treat them as today's baseline. Old `sustainable_hz` fields may mean inverse
median latency; newer `measured_hz` is measured window throughput.

## Source provenance

Audit base: `7f9cc421af80d59d04baa9fa27af7b14df74e920` on
`feat/yemong-spatial-relational`, with existing interactive/concurrency changes.
Commit `cddb8af` preserves those changes separately on this branch:
`benchmarks/realtime_latency.py`, `src/boost_and_broadside/modes/interactive.py`,
`tests/modes/test_interactive.py`, and `docs/engineering/concurrent-rollout-plan.md`.
NN actions have a one-tick delay; human/scripted actions remain immediate.

The audit harness evolved during measurement. Raw metadata includes the old
revision and sometimes `git status`, not a complete dirty diff per run. Manifest
source hashes identify final handoff code, not every historical harness revision.
Re-establish a small paired baseline before assessing new changes.

An audit fix removes an evaluator call from `primary_only` in
[`rl_kernel_profile.py`](../../../benchmarks/rl_kernel_profile.py). Previously
primary-only included evaluation and the combined phase evaluated twice.
A fresh timed run of the corrected profiler was cancelled, not completed.

## Environment and workload

- i7-13620H, RTX 4070 Laptop, 8,188 MiB VRAM; Python 3.13.11,
  Torch 2.13.0+cu130, NVIDIA driver 595.84.
- Profile `power-saver`, platform `quiet`, governor `powersave`; never changed.
  AC power can affect effective performance even with the same profile. Compare
  new alternatives under the same current conditions.
- Audit harness: one Torch CPU thread. Actual PPO profiler: ten threads.
- CUDA was hidden in the sandbox but available through permitted host execution.
  A sandbox availability failure does not establish no GPU.
- 5v5: 10 ships, 10 fields; 50v50: 100 ships, 72 fields; 10 projectile slots/ship.
  Full-frame trials used two distinct random policies and 900×900 offscreen drawing.
- Runtime schema: `frontline_shields_v9`. The inspected older
  `icy-energy-741/best_training.pt` used `recursive_belief_v8` and was not loaded.
- Training normally excludes projectile perception; rendered comparisons include
  it. Disabling it for play changes the workload and must be labeled an ablation.

## Reproduction

Run from repository root using the existing `.venv`; a fresh checkout needs
dependencies from `pyproject.toml`/`uv.lock`. Avoid changing versions mid-comparison.

| Harness | Purpose |
|---|---|
| [`performance_audit.py`](../../../benchmarks/performance_audit.py) | Component profile, LOS parity/timing, full-frame A/B, environment throughput |
| [`performance_audit_warp.py`](../../../benchmarks/performance_audit_warp.py) | Optional Warp LOS; no production dependency |
| [`realtime_latency.py`](../../../benchmarks/realtime_latency.py) | Two-policy loop, buffered actions, rendering/concurrency controls |
| [`rl_pipeline_profile.py`](../../../benchmarks/rl_pipeline_profile.py) | Actual rollout/storage/evaluator/PPO pipeline |
| [`rl_kernel_profile.py`](../../../benchmarks/rl_kernel_profile.py) | Corrected primary/evaluator kernel attribution |

```bash
# Inspect the baseline harness; explicitly select sequential execution for A/B.
.venv/bin/python benchmarks/realtime_latency.py --help

# Short CPU component profile.
.venv/bin/python benchmarks/performance_audit.py --mode profile \
  --scenario 5v5 --steps 10 --out artifacts/deadline-experiment/profile-5.json

# CUDA full-frame reference versus compiled real LOS; requires GPU access.
TORCHINDUCTOR_CACHE_DIR=/tmp/bnb-audit-inductor .venv/bin/python \
  benchmarks/performance_audit.py --device cuda --mode realtime \
  --scenario 50v50 --steps 100 --pairs 3 --policy-compile default \
  --out artifacts/deadline-experiment/realtime-50.json

# Environment-only, NOT complete training throughput.
.venv/bin/python benchmarks/performance_audit.py --device cuda \
  --mode throughput --scenario 5v5 --batch 128 --steps 100 \
  --out artifacts/deadline-experiment/env-b128.json
```

[Completed GPU command order and durations](logs/gpu-runs.json) and
[CPU follow-up commands](logs/followup.log) preserve original output paths;
change `--out` for future runs. The interrupted training command was:

```bash
.venv/bin/python benchmarks/rl_pipeline_profile.py --updates 2 --warmup 1 \
  --timing wall --no-checkpoint --checkpoint-dir /tmp/bnb-audit-checkpoints \
  --out artifacts/deadline-experiment/ppo.json
```

It stopped after one measured update. `--no-checkpoint` did not suppress an
internal best-training checkpoint; no weights are archived here. Do not rerun
solely to recreate the missing aggregate.

Optional Warp was installed separately without changing project dependencies:

```bash
uv pip install --python .venv/bin/python --target /tmp/bnb-audit-warp warp-lang==1.17.0
```

Pass `--warp-path /tmp/bnb-audit-warp` and, for full-path comparisons,
`--backend warp`. Temporary dependencies/caches are not guaranteed to persist.

## Harness limitations and unrun work

- **Throughput mode**: `--backend compiled` means eager Torch reference, not a
  compiled tick/LOS. Newer metadata says `los_backend: eager_reference`.
  Collision compilation is disabled in this harness.
- Throughput sample arrays are host enqueue times. Use completion-timed
  `window_mean_ms` and derived transitions/s for headline throughput.
- LOS checks establish sampled boolean parity, not equivalence at every tangent.
- `--mode render` and `--mode tick-compile` were written but never run. Renderer
  ablations, CPU Warp LOS, extra perception suite runs, and the corrected primary
  profile were cancelled before starting. They remain unvalidated experiments.
- Transient queue scripts and empty logs were not promoted to this branch. No
  jobs resume automatically. Original ignored artifacts remain locally; this
  archive is the versioned evidence source.

## Handoff checks

Preparation verified all 80 archive checksums, parsed every archived JSON,
checked local documentation links and handoff source hashes, and ran Ruff on
the two audit harnesses and corrected kernel profiler. The preserved interactive
tests passed: `tests/modes/test_interactive.py`, 7 tests. No new performance
measurements were run during branch preparation.
