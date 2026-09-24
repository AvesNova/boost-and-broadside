# Decision runtime Phase 5 final validation

Date: 2026-09-24

Phase 5 closes the action-timing/runtime migration begun at
`138f540beaa8e4abe504b7808c057f7dfdd0db76` and performance-validated at
`918aa9cd9e23a1a022befa68cc2f2656bf1fd05c`. No known production execution path
retains an independent version of autonomous action timing, pending-action privacy,
or reset/respawn cancellation.

## Final test record

- Complete repository suite: **1,696 passed**, 7 warnings, 582.70 seconds.
- Post-suite focused validation after formatting and benchmark repair: **224 passed**, 2
  warnings, 12.81 seconds.
- Repository-wide Ruff formatting and lint checks: passed.
- `git diff --check`: passed.
- CUDA was available to the complete suite; its compiled-tick, policy-I/O, checkpoint,
  PPO stream, and benchmark coverage was not skipped.
- The Phase 4 CUDA performance runs and deadline-subphase smoke remain recorded in
  [`runtime-phase4-performance.md`](runtime-phase4-performance.md).

The warnings are the existing CPU `vmap` fallback for flash attention, Matplotlib
`tight_layout` compatibility, and TorchInductor complex-operator performance warning.
They are not semantic failures.

## Final execution-path audit

Production controller paths now resolve as follows:

| Consumer | Authoritative transition mechanism |
| --- | --- |
| Match evaluation, tournaments, capture, feature/AR/noise diagnostics | `MatchRunner` |
| Continuous Elo | `PendingActionState` in `EloEvaluator` |
| PPO, BC teacher data, self-play, league, auxiliary scales | Batched `PendingActionState` scheduler |
| Interactive/watch neural, scripted, and random control | Interactive `PendingActionState` scheduler |
| Human control | Explicit immediate override on the interactive scheduler |

The final source search found no production mode manually writing the serialized
pending-action field. Its raw accesses are limited to:

- `runtime/actions.py`, the authoritative perspective-aware writer;
- `evaluation/next_state.py`, a read-only auxiliary decoder;
- `env/observation.py`, the observation accessor.

An architecture regression test now scans production and benchmark Python sources:

- a file that both runs a controller and advances an environment must declare
  `PendingActionState`, `advance_autonomous_decision`, or `MatchRunner`;
- raw pending-action observation access is restricted to the explicit owners above.

Direct environment stepping remains in fixed-action physics, renderer, parity, and
component-throughput probes. Those probes intentionally measure an environment primitive
rather than a controller decision process and are not alternative game runtimes.

## Diagnostic migrations completed during Phase 5

The final audit found older controller-bearing trajectories in these benchmark tools:

- presence-density scene generation;
- scripted play throughput and the Frontline overhaul throughput arm;
- Frontline density and fog suites;
- neural inference scaling;
- learned and constant-velocity belief diagnostics;
- scripted statistical playtests;
- compiled-belief trace generation.

They now use the same delayed autonomous transition primitives as evaluation. Neural
diagnostics write the canonical pending queue into their observations before inference,
including team privacy and respawn cancellation.

Bounded smoke runs passed for every migrated tool. This included a CUDA 5v5 neural
inference run and an exact two-step reference/candidate perception parity run.

The deadline experiment also had two stale assumptions repaired:

- its reconstructed team view represented enemy action state as neutral; it now uses
  the authoritative private-action category writer;
- it monkeypatched a perception symbol no longer used by the wrapper; the real-time
  harness now exposes an explicit benchmark-only perception override instead.

## Documentation/test cleanup

The formerly failing legacy-CLI documentation test scanned immutable engineering records
for benchmark-script flags such as `performance_audit.py --mode throughput` and treated
them as obsolete application CLI. It now checks reader-facing documentation while
allowing engineering records to preserve reproducible historical commands.

## Performance conclusion

The Phase 4 relative gate passed: no substantial regression was measured in batched
training or low-N execution, and CUDA-stream overlap remains active. The established
low-power benchmark condition is valid and does not need replacement by faster hardware
or a different power profile.

The current implementation nevertheless fails the requested 30/60 FPS target in the
measured two-policy 5v5 and 50v50 low-power workloads. That is a future optimization
objective, not unfinished validation and not a blocker on the semantic migration.

The temporary fifth neural league generation during episode-stable replacement measured
12.4% additional rollout time and 8.52 MiB allocated memory in the targeted workload.
Stable identity remains the chosen semantic behavior. Long-run drain/cache telemetry is
a useful future optimization aid, but no unvalidated semantic change is pending.

## Completion decision

The investigation, specification, refactor, performance validation, and final validation
phases are complete. Future modes should compose the runtime/controller mechanisms above;
the new source-level guards make an unreviewed independent controller loop a test failure.
