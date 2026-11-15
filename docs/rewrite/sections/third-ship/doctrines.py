"""Random doctrine candidates for the third ship (the allowed elements only: slash, bleed,
lag, deny and the sense; extend, the crowd extension and the duel extension stay off, the
author's 2026-11-14 decision), each over Jujak's stored doctrine, for `third.py` with a
"versus" (the base doctrine) so its mixed jobs measure the doctrine's strength.

    .venv/bin/python docs/rewrite/sections/third-ship/doctrines.py CARDS.json NAME N OUT.json \
        [--seed 0] [--scale 1.0] [--around DOCTRINE.json]
"""

import argparse
import json
import math
import pathlib
import random


def draw(rng: random.Random, scale: float) -> dict:
    """One random doctrine (named fields)."""
    def weight(p_on: float = 0.5) -> float:
        return round(rng.uniform(0.05, 0.9) * scale, 3) if rng.random() < p_on else 0.0

    def logistic(lo: float, hi: float, s: float = 0.2) -> list[float]:
        return [round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 3), s]

    d = dict(
        strength=round(rng.uniform(0.3, 1.0), 3),
        turn_exponent=round(rng.uniform(-2, 2), 3) if rng.random() < 0.5 else 0.0,
        climb_exponent=round(rng.uniform(-2, 2), 3) if rng.random() < 0.3 else 0.0,
        speed_exponent=round(rng.uniform(-2, 2), 3) if rng.random() < 0.3 else 0.0,
        range_exponent=0.0,
        in_advantage=[round(math.exp(rng.uniform(math.log(0.5), math.log(2.0))), 3), 0.1],
        fight_in_gun_range=logistic(1.5, 4.0, 0.13),
        slash=weight(),
        slash_in_gun_range=logistic(1.5, 4.0),
        bleed=weight(),
        bleed_in_corner_speed=logistic(1.1, 1.8, 0.1),
        bleed_in_stall_angle=logistic(1.2, 3.0),
        lag=weight(0.3) * 0.5,
        lag_in_gun_range=round(rng.uniform(0.1, 0.5), 3),
        deny=weight(),
        deny_in_stall_angle=logistic(0.5, 1.2, 0.18),
        deny_in_gun_range=logistic(2.0, 4.0, 0.2),
        deny_angle=round(rng.uniform(0.2, 0.7), 3),
        sense=1.0,
        sense_one_circle=float(rng.random() < 0.5),
        sense_mirror_only=float(rng.random() < 0.5),
        sense_in_gun_range=logistic(2.0, 6.0),
        sense_in_lift_ratio=[round(rng.choice([0.001, 0.9, 1.0, 1.1]), 3), 0.05],
    )
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cards", type=pathlib.Path)
    parser.add_argument("name")
    parser.add_argument("count", type=int)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    base = next(e for e in json.loads(args.cards.read_text()) if e["name"] == args.name)
    rng = random.Random(args.seed)
    out = [dict(base, name=f"{args.name}+base", doctrine={}, versus={})]
    for number in range(args.count):
        out.append(dict(base, name=f"{args.name}+r{args.seed}-{number:03d}", doctrine=draw(rng, args.scale), versus={}))
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
