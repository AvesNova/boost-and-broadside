"""The frontline geometry exploration (an exploration: rule and radius overrides in the
harness, no game change). On the lift x1.5 + damage x1.5 card with r9m3 on both sides,
Yongma's frontline score at 5v5, 10v10 and 5v5 on 0.5x and 2x arenas, for four geometries:

- A: today's (zone radius 330 m x map_scale, zone ring 0.615 x the arena radius; the arena
  7800 m x (n / 5)^0.5, at least 4200 m);
- B: zones that grow slower than the map: the zone radius and the ring each times
  (3 / map_scale)^0.5, map_scale = arena / 2600 m, so the 5v5 map (map_scale 3) is today's;
- C: the arena grows slower with fleet size, (n / 5)^0.25 instead of ^0.5 (an arena scale of
  (n / 5)^-0.25 on the game's rule; only 10v10 changes);
- D: B and C.

    uv run --no-sync python docs/rewrite/sections/lethality/geometry.py > run.sh

prints the runs (sizes.py, one per cell that differs from A) and a table key per cell.
"""

import json

from bnb.game import api

GAME = api.PRESETS["real_scale_frontline"]
ZONE, RING, SET_SCALE = 330.0, 0.615, 3.0
CELLS = [(5, 1.0, "5v5"), (10, 1.0, "10v10"), (5, 0.5, "0.5x"), (5, 2.0, "2x")]


def cell(variant: str, ships: int, scale: float) -> tuple[float, dict]:
    """(arena scale to pass, settings) for one variant and cell."""
    if variant in ("C", "D"):
        scale = scale * (ships / 5.0) ** -0.25
    settings = {}
    if variant in ("B", "D"):
        radius = api.arena_radius_for(GAME, ships, scale)
        factor = (SET_SCALE / (radius / 2600.0)) ** 0.5
        settings = {"mode.zone_radius": ZONE * factor, "mode.zone_ring_fraction": RING * factor}
    return scale, settings


def main() -> None:
    lines = ["set -e", "export PYTHONPATH=src:.", "S=docs/rewrite/sections/maneuvers-2/sizes.py"]
    plan = {}
    for variant in "ABCD":
        for ships, scale, label in CELLS:
            arena_scale, settings = cell(variant, ships, scale)
            key = (ships, round(arena_scale, 6), json.dumps(settings, sort_keys=True))
            tag = f"{variant}-{label}"
            plan[tag] = dict(
                ships=ships,
                arena=api.arena_radius_for(GAME, ships, arena_scale),
                settings=settings,
            )
            if any(p.get("key") == key for p in plan.values() if "key" in p):
                same = next(t for t, p in plan.items() if p.get("key") == key)
                plan[tag]["same_as"] = same
                continue
            plan[tag]["key"] = key
            flags = " ".join(f"--setting {k}={v:.6g}" for k, v in settings.items())
            lines.append(
                f"uv run --no-sync python $S --mode frontline --seconds 600 "
                f"--candidates ../lethality/geometry/candidates.json "
                f"--tag ../lethality/geometry/{tag} --sizes {ships} --games 256 "
                f"--arena-scale {arena_scale:.6g} {flags} "
                f"> docs/rewrite/sections/lethality/geometry/{tag}.log 2>&1"
            )
    for p in plan.values():
        p.pop("key", None)
    print("\n".join(lines))
    with open("docs/rewrite/sections/lethality/geometry/plan.json", "w") as out:
        json.dump(plan, out, indent=1)


if __name__ == "__main__":
    main()
