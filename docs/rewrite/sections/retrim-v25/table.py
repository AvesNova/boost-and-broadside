"""The re-trim's table: Yongma's kill / point share per card and condition, the bands, the
conditions in band and the worst miss (kill share). Reads data/rt-<card>-<condition>.json
(`bench.action_mix --entries trims.json --entry <card>`), Y0.46's from
../imugi-initiative/data/bal-after-<condition>.json."""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
DATA = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
BANDS = {
    "deathmatch-2v2": (0.481, 0.521), "deathmatch-5v5": (0.494, 0.507),
    "elimination-5v5": (0.498, 0.545), "frontline-5v5": (0.491, 0.513),
    "deathmatch-10v10": (0.497, 0.504), "elimination-10v10": (0.492, 0.526),
    "frontline-10v10": (0.493, 0.508),
}  # fmt: skip
CARDS = ("Y0.46", "Y0.42", "Y0.38", "Y0.42-J1.22")


def path_of(card: str, condition: str) -> pathlib.Path:
    """Where a card's line for a condition is."""
    if card == "Y0.46" and "10v10" not in condition:
        return HERE.parent / "imugi-initiative" / "data" / f"bal-after-{condition}.json"
    return DATA / f"rt-{card}-{condition}.json"


print("| card | " + " | ".join(BANDS) + " | in band | worst miss |")
print("|---|" + "---|" * (len(BANDS) + 2))
for card in CARDS:
    cells, inside, worst = [], 0, 0.0
    for condition, (low, high) in BANDS.items():
        path = path_of(card, condition)
        if not path.exists():
            cells.append("-")
            continue
        line = json.loads(path.read_text())
        kill, point = line["kill_share_a"][0], line["point_share_a"][0]
        miss = max(low - kill, kill - high, 0.0)
        inside += miss == 0.0
        worst = max(worst, miss)
        cells.append(f"{kill:.3f} / {point:.3f}")
    print(f"| {card} | " + " | ".join(cells) + f" | {inside} | {worst:.3f} |")
