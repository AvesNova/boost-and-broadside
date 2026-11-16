"""Re-trim Bonghwang for a new Yongma-Jujak pair: a grid of its three balance numbers
(sustained turn, damage per bullet, shield) against that pair, and the pick over sizes.

    # 1. the grid: a candidates file for third.py, the pair from two ship files
    PYTHONPATH=$PWD/src:$PWD .venv/bin/python docs/rewrite/sections/third-ship/retrim.py grid \
        TAG YONGMA.toml JUJAK.toml OUT.json
    # 2. play it at 2v2, 5v5 and 10v10 (third.py, one run per size, or the box workers)
    # 3. the pick: every card's worst distance from 0.5 over the sizes and both opponents
    .venv/bin/python docs/rewrite/sections/third-ship/retrim.py pick RESULTS.jsonl ...

The grid's flights are the solved Bonghwang airframes (`candidates/screen10.json`,
`screen12.json`, `screen13.json`: the roll, g 7 or 8, dive 43.6, sustained turn 0.63-0.72);
its gun is the set's; its doctrine Jujak's stored one with the sense off. The first entry is
the pair's reference ("TAG:ref", its "C v J" the pair's Yongma v Jujak).
"""

import argparse
import json
import pathlib
import tomllib

HERE = pathlib.Path(__file__).parent
FLIGHTS = ("screen10.json", "screen12.json", "screen13.json")
JUJAK_DAMAGE = 19.1877
GUN = dict(
    fire_rate=30.0,
    muzzle_speed=1712.0,
    bullet_spread=3.0844823457636035,
    bullet_drag=0.0005,
    gun_range=584.34,
    shots_per_store=865.24,
    explosive_share=0.0,
)
TURNS = {  # (sustained turn, g limit): the solved airframe
    (0.63, 7): "K-roll-g7-dive",
    (0.66, 7): "K-roll-g7-dive-t66",
    (0.68, 7): "K-roll-g7-dive-t68",
    (0.70, 7): "K-roll-g7-dive-t70",
    (0.72, 7): "K-roll-g7-dive-t72",
    (0.66, 8): "K-roll-g8-dive-t66",
    (0.68, 8): "K-roll-g8-dive-t68",
    (0.70, 8): "K-roll-g8-dive-t70",
}
COMBAT = (
    (1.0, 85),
    (1.1, 85),
    (1.15, 85),
    (1.2, 80),
    (1.1, 90),
    (1.15, 90),
    (1.2, 100),
    (1.2, 115),
)


def grid(tag: str, yongma: pathlib.Path, jujak: pathlib.Path, out: pathlib.Path) -> None:
    import cards  # the combat-only recompute, beside this file

    solved = {}
    for name in FLIGHTS:
        solved |= {e["name"]: e for e in json.loads((HERE / "candidates" / name).read_text())}
    y = tomllib.loads(yongma.read_text())["card"]
    j = tomllib.loads(jujak.read_text())["card"]
    pair = dict(
        yongma_card={k: {"set": v} for k, v in y.items()},
        jujak_card={k: {"set": v} for k, v in j.items()},
    )
    entries = [dict(name=f"{tag}:ref", card=y, doctrine_base="yongma", doctrine={}, **pair)]
    for (turn, g), flight in TURNS.items():
        for damage, shield in COMBAT:
            spec = dict(
                name=f"{tag}:Bg{g}t{round(turn * 100)}-d{damage}-s{shield}",
                flight_from=flight,
                handles={k: {"set": v} for k, v in GUN.items()}
                | dict(
                    hit_damage={"set": round(JUJAK_DAMAGE * damage, 4)},
                    max_shield={"set": float(shield)},
                ),
                doctrine=dict(sense=0.0),
            )
            entries.append(cards.combat_only(spec, solved) | pair)
    out.write_text(json.dumps(entries, indent=1))
    print(f"{len(entries)} entries -> {out}")


def pick(paths: list[pathlib.Path]) -> None:
    scores: dict[str, dict[str, float]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            size = f"{row['ships']}v{row['ships']}-{row['mode']}"
            for label in ("C v J", "C v Y"):
                if label in row["jobs"] and not row["name"].endswith(":ref"):
                    scores.setdefault(row["name"], {})[f"{size} {label}"] = row["jobs"][label][
                        "score"
                    ][0]
    ranked = sorted(scores.items(), key=lambda kv: max(abs(v - 0.5) for v in kv[1].values()))
    for name, cells in ranked[:15]:
        worst = max(abs(v - 0.5) for v in cells.values())
        text = ", ".join(f"{k} {v:.2f}" for k, v in sorted(cells.items()))
        print(f"{name}: worst {worst:.3f} ({len(cells)} cells): {text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grid")
    g.add_argument("tag")
    g.add_argument("yongma", type=pathlib.Path)
    g.add_argument("jujak", type=pathlib.Path)
    g.add_argument("out", type=pathlib.Path)
    p = sub.add_parser("pick")
    p.add_argument("results", type=pathlib.Path, nargs="+")
    args = parser.parse_args()
    if args.command == "grid":
        grid(args.tag, args.yongma, args.jujak, args.out)
    else:
        pick(args.results)


if __name__ == "__main__":
    main()
