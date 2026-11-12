# Frontline design search — progress

Branch `tune/frontline-design` (from `integrate/next-runs` at 40cbd425), 2026-11-12, box
gpubox4 (RTX 5060 Ti). Bench-only candidates: nothing here changes the game.

## Method

`bench/frontline_design.py`: the candidate rules as per-environment tensors inside the
compiled tick (`frontline.capture_step` and `api.resolve` wrapped, as in
`bench/mode_dynamics.py`), so 4 variants run side by side in one batch and every variant
of a fleet size shares one compile. Games play with the threshold off (10) and a 1200 s
clock; the scripted agent reads neither, so `bench/frontline_design_report.py` replays
any threshold T and clock C from the traced front (a game under (T, C) is the traced game
up to the first capture putting the front at ±T, or C). Settings: real_scale_frontline,
no fields, fog off, 2 Yongma + 3 Jujak a team (slots J, Y, J, Y, J...), scripted mirror
T = 1, zones 330 m x map scale, capture 20 s, respawn 16 s, tree's default cards.
P(same) error clustered by game.

Rules (all bench-only):
- `lock L`: the captured zone's meter stands still L s (no ship counts).
- `early`: the lock ends as soon as the capturing team has no living ship in the zone.
- `stale`: the losing team's ships inside a zone when it falls do not count there until
  they leave it (or die).
- `forward`: respawns at the team's own defense zone (its frontmost owned zone) instead of
  its spawn zone, nose toward the zone it attacks.
- `kappa`: capture time x d / 5.64 km (κ = 3.5 s/km of the adjacent-zone distance d).
- `count c`: meter rate H(c |net|) for H(|net|) (c = 5 / n: a lead counted as its 5v5
  equivalent, the fleet term).
- `prog g`: capture time x (1 + g)^depth, depth = the attacker's lead in captures.

## Results
