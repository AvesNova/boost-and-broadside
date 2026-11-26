# Scripted look-ahead agent: progress (branch scripted-lookahead)

The design the author approved on 2026-11-26 (brief from the coordinator), built in
`lean/scripted/` (paths.py, opponent.py, parameters.py). Milestone notes: tried / worked /
didn't / next. Numbers only as measured; laptop RTX 4070, performance profile.

## Author additions (2026-11-26, during the build)

1. No enemy known (start, or all dropped): no goal point (no centre, no enemy side: a
   convention between both teams' agents to exploit). Only the own-ship terms: speed, a
   disengaged energy preference (`disengaged_energy_weight`), the boundary, and a gentle seek
   toward the freshest dropped last-seen position (`seek_weight`, eager to patient).
2. Hidden enemies are dead-reckoned from the last sighting by the time since seen (last speed
   and nose rate for up to 2 s, then straight; `opponent.dead_reckon`); their 21 paths start
   from there. The roam discount and the cutoff still apply. No worst case for the hidden
   stretch (beta covers the worst case inside the look-ahead).

## Step 1: paths, pair tables, collapse (2026-11-26)

- Tried: candidate paths from `core.steady_flight` (new: `steady_rates` plus the steady angle)
  stepped over 4 segments (0.25, 0.25, 0.5, 1 s), the nose spring in closed form (offset ODE
  e'' + 2 sigma e' + k omega^2 e = 0, under/critical/over-damped), the path rate linear in the
  angle between now (`core.current_path_rate`: the angle's own rate is not in the observation)
  and the steady state; chord per segment; medium and gradient bend per segment start.
- Worked: `path_check.py` against the real tick holding each of the 21 actions for 2 s, 64
  random 1v1 deals after 10 random ticks: no fields, median / 90th-percentile position error
  4.2 / 17.0 m at 1 s and 10.1 / 27.1 m at 2 s, nose 1.3 / 7.4 deg at 1 s, speed 2.4 / 12.3 m/s
  at 1 s; Snap is the worst level (median 13-14 m, 4-5 deg at 1 s). Fields: 4.7 / 22.9 m at 1 s,
  12.9 / 51.5 m at 2 s. (The 0.25 s point is not a whole tick count: 8 ticks = 0.267 s were
  compared, which accounts for most of its 4.5-4.7 m median.)
- Decided myself: the agent reads the team view state (what `observe` is built from), only
  the columns Obs carries (positions, headings, proper speed = world speed x n, angle of
  attack, nose rate, store, shield, cooldown, pending of own ships, time since seen; ln n and
  its gradient are functions of position and the fields, which Obs carries). Stickiness reads
  the pending action (in Obs), so the agent has no memory and its labels are a function of the
  observation; `fresh_memory` returns an empty record until the callers' plumbing goes.
  Persona drawn per ship at each deal (today's mechanism). Speed kept above 1/4 stall speed in
  the paths (a never-seen enemy's row is all zeros).
- Next: game measurements, then the teacher interface, tests, docs.
