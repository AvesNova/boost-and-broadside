"""grid3: vision-free cards around Q4x0.8 (the author's rule: vision equal), a central composite
design on the three raw game knobs on top of confirm1's solved Q4x0.8 handle card (Yongma
sustained turn 1.112, low-speed acceleration held, shield x0.55; Jujak turn 0.95): Yongma store
(max_energy) x1.35 / 1.52 / 1.70, Yongma shot cost x0.75 / 0.64 / 0.55, Jujak shield x1.10 /
1.18 / 1.26 (centre, the six axis points, the eight corners), plus two cards with Yongma's shield
less cut (x0.60, x0.65 of before) and Jujak's at x1.10 (the shields closer: the archetype flag),
and E. Writes candidates/grid3.json."""

import copy
import itertools
import json
import pathlib

HERE = pathlib.Path(__file__).parent
CONFIRM = {e["name"]: e for e in json.loads((HERE / "confirm1.json").read_text())}
LEVELS = {
    "store": (1.35, 1.52, 1.70),
    "shot": (0.75, 0.64, 0.55),
    "jshield": (1.10, 1.18, 1.26),
}
YONGMA_SHIELD = 141.94908  # before's max_shield (x0.55 = 78.07)


def card(store: float, shot: float, jshield: float, yshield: float | None = None) -> dict:
    entry = copy.deepcopy(CONFIRM["g-Q4x0.8h"])
    entry["card"]["energy"]["max_energy"] = store
    entry["card"]["energy"]["bullet_energy_cost"] = shot
    entry["card"]["turn"]["max_shield"] = jshield
    name = f"g3-s{store}-c{shot}-j{jshield}"
    if yshield is not None:
        entry["card"]["energy"]["max_shield"] = {"set": YONGMA_SHIELD * yshield}
        name += f"-y{yshield}"
    entry["name"] = name
    entry["probe"] = {"energy": {"max_energy": store, "bullet_energy_cost": shot},
                      "turn": {"max_shield": jshield}}  # fmt: skip
    return entry


def main() -> None:
    s, c, j = LEVELS["store"], LEVELS["shot"], LEVELS["jshield"]
    points = [(1, 1, 1)]
    for axis in range(3):
        for level in (0, 2):
            point = [1, 1, 1]
            point[axis] = level
            points.append(tuple(point))
    points += list(itertools.product((0, 2), repeat=3))
    out = [copy.deepcopy(CONFIRM["g-E"]) | {"name": "g3-E"}]
    out += [card(s[a], c[b], j[k]) for a, b, k in points]
    out += [card(1.52, 0.64, 1.10, 0.60), card(1.52, 0.64, 1.10, 0.65)]
    (HERE / "grid3.json").write_text(json.dumps(out, indent=0) + "\n")
    print(len(out), "cards")


if __name__ == "__main__":
    main()
