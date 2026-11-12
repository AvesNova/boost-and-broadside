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

## 2026-11-12 21:54 KST: soft locks at 10v10 (s10; d = 7.98 km, 192 games; threshold 2 / clock 900, threshold 3 in brackets)

| rule | P(same team next) | ends by the front | median length |
|---|---|---|---|
| extension 45 | 0.229 (0.232) | 0.71 (0.39) | 584 s |
| extension 60 | 0.332 (0.320) | 0.83 (0.48) | 455 s |
| extension 80 | 0.514 (0.519) | 0.89 (0.66) | 378 s |
| hold 60 | 0.153 (0.146) | 0.59 (0.15) | 770 s |
| hold 90 | 0.194 (0.192) | 0.60 (0.22) | 718 s |
| lock 55 | 0.489 (0.450) | 0.88 (0.56) | 361 s |

The seconds that reach P = 0.5 on the frozen cards:

| fleet | d | flat extension | hard lock |
|---|---|---|---|
| 5v5 | 5.64 km | ~38 s (interpolated) | 45 s |
| 10v10 | 7.98 km | 80 s | 55 s |
| 20v20 | 11.28 km | 80 s | 60 s |

The extension is not linear in d: 10v10 needs as much as 20v20, a step from 5v5, so a
fleet term would be needed. The hard lock grows smoothly, roughly as the square root of
d (about 19 s per sqrt(km): 45 / 54 / 64 s). It is the steadier rule across sizes. The
decaying hold is dropped (it is too weak at 10v10 and 20v20). Calibrate again on the final
card.
