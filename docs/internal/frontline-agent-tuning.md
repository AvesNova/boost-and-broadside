# Frontline scripted-agent tuning, 2026-09-16

The state-derived scripted controller was evaluated against the exact controller
from integration-base commit `d21ff5262b1862c9de564ca1916f9059f78982cb` and
the newest checkpoint from the most recent run:
`checkpoints/misty-energy-739/step_000079626240.pt` (update 161, step
79,626,240). The checkpoint was run through its recorded Frontline
environment, observation contract, policy, and belief tracker.

The benchmark is
[`benchmarks/frontline_agent_head_to_head.py`](../../benchmarks/frontline_agent_head_to_head.py).
It places the candidate on each team in equal numbers of simultaneous games,
uses the authoritative finite-vision mask, pins the legacy source by SHA-256,
and records front, captures, combat deaths, duration, and per-side outcomes.

## Selected defaults

| Parameter | Previous | Selected |
|---|---:|---:|
| `frontline_aggression` | 0.0 | 1.0 |
| `frontline_combat_radius` | 800 px | 600 px |
| `frontline_zone_radius` | derived 660 px | 900 px |
| `frontline_zone_margin` | 1.0 | 1.0 |
| `frontline_separation_radius` | derived 40 px | 120 px |
| `frontline_recovery_health` | 0.5 | 0.5 |

The screen covered aggression from -0.5 to 1.5, combat radii of 600 to 1,200
px, zone support radii of 500 and 900 px, zone margins of 0.5 and 1.5,
separation radii of 40 to 160 px, and recovery scales of 0.3 and 0.5.

The best strategy against the legacy controller was more aggressive still, but
that overcommitted against the learned policy. The selected `a=1.0` setting is
the best common-map checkpoint candidate: lower values ceded pressure and
`a=1.5` abandoned defense. `R_C=600` preserves the old dogfighter through a
larger fraction of an engagement; 900 px zone support credits useful en-route
allies; 120 px separation produced the strongest capture differential without
showing a dense-fleet instability.

## Standard-horizon results

All matches use seed 920000, the standard 300-second Frontline horizon, and
the checkpoint's 4v4, ten-field environment. Wins/losses/draws are from the
candidate's perspective.

| Candidate | Opponent | Games | W-L-D | Score | Captures | Combat deaths | Mean signed front |
|---|---|---:|---:|---:|---:|---:|---:|
| tuned new | legacy | 8 | 8-0-0 | 1.000 | 29-6 | 93-143 | +2.875 |
| tuned new | latest checkpoint | 4 | 3-0-1 | 0.875 | 26-21 | 63-182 | +1.250 |
| legacy | latest checkpoint | 4 | 1-1-2 | 0.500 | 11-11 | 71-181 | 0.000 |

The first row split evenly between team sides: tuned new won all four games as
each team. Against the checkpoint it scored 1.00 as Team 0 and 0.75 as Team 1;
legacy's comparable split was 0.75 and 0.25. These are small samples, so they
are evidence for choosing the new defaults rather than a settled rating claim.

At 16v16, a separate four-game, 120-second validation against legacy was 4-0-0,
with a 6-0 capture count and +1.5 mean signed front. It confirms the rule set
operates beyond training fleet size, but does not establish large-fleet balance
against learned policies.

## Reproduction

```bash
.venv/bin/python benchmarks/frontline_agent_head_to_head.py pair \
  --candidate default --opponent legacy --games 8 --seconds 300 --seed 920000
.venv/bin/python benchmarks/frontline_agent_head_to_head.py pair \
  --candidate default --opponent checkpoint --games 4 --seconds 300 --seed 920000
.venv/bin/python benchmarks/frontline_agent_head_to_head.py pair \
  --candidate legacy --opponent checkpoint --games 4 --seconds 300 --seed 920000
```

CPU evaluation uses one Torch thread per process. CUDA was unavailable during
this experiment, so no GPU timing or allocation claim is made.
