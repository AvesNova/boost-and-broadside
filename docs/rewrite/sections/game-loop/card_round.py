"""A card round: card variants (factors on the base card's fields, per type) under one
player's doctrines in self-play, every lineup; per variant the energy v turn score, the
aspect mix (aggregate and per lineup), the pace, passes a kill, the card's head-on TTK in
passes. Writes round<N>/cards.json and cards.md.

    PYTHONPATH=src:. python docs/rewrite/sections/game-loop/card_round.py --round 1 \
        --player r1 --variants round1/variants.json
"""

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import league  # noqa: E402

from bench.game_loop import ENGAGEMENTS, Job, Loop, card_passes, job_summary, run  # noqa: E402

HERE = pathlib.Path(__file__).parent
LINEUPS = ("energy v turn", "turn mirror", "energy mirror", "mixed", "mixed reversed")


def engagement_cells(rows: list[dict]) -> str:
    """The engagement columns from job summaries (games-weighted means)."""
    if not rows or not rows[0].get("engagement"):
        return " | | | "
    total = sum(r["games"] for r in rows)

    def mean(definition: str, key: str, index=None) -> float:
        values = []
        for r in rows:
            value = r["engagement"][definition][key]
            value = value[index] if index is not None else value
            values.append((value if value is not None else float("nan")) * r["games"])
        return sum(values) / total

    merges = "; ".join(
        f"{mean(d, 'merges_per_kill_mean'):.2f} / {mean(d, 'merges_per_kill_median'):.1f} / "
        f"{mean(d, 'first_pass_kill_share'):.2f} / {mean(d, 'merges_per_kill_p90'):.1f}"
        for d in ENGAGEMENTS
    )
    death = "; ".join(f"{mean(d, 'to_death_median'):.1f}" for d in ENGAGEMENTS)
    turned = "; ".join(f"{mean(d, 'circles_turned_mean'):.2f}" for d in ENGAGEMENTS)
    kinds = " / ".join(f"{mean(ENGAGEMENTS[1], 'one_two_scissors', i):.2f}" for i in range(3))
    return f"{merges} | {death} | {turned} | {kinds}"


def mix(shares: list[float]) -> float:
    """The evenness of the offset / flank / tail shares (their entropy over ln 3)."""
    rest = shares[1:]
    total = sum(rest)
    if total <= 0:
        return 0.0
    return -sum(s / total * math.log(s / total) for s in rest if s > 0) / math.log(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--player", required=True, nargs="+")
    parser.add_argument(
        "--versus", default="off", help="the doctrine value: --player against it, mixed"
    )
    parser.add_argument("--variants", required=True)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--environments", type=int, default=8192)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--tag", default="cards")
    args = parser.parse_args()
    started = time.time()
    out_dir = HERE / f"round{args.round}"
    out_dir.mkdir(exist_ok=True)
    variants = json.loads((HERE / args.variants).read_text())
    K = len(variants)
    loop = Loop(args.environments, K, args.seconds)
    loop.set_types([v["energy"] for v in variants], [v["turn"] for v in variants])
    players = [{p.name: p for p in league.load()}[name] for name in args.player]
    versus = {p.name: p for p in league.load()}[args.versus]
    jobs = []
    for player in players:
        jobs += [Job(k, lineup, player, player, args.games) for k in range(K) for lineup in LINEUPS]
        jobs += [
            Job(k, lineup, player, versus, args.games // 2)
            for k in range(K)
            for lineup in ("mixed", "mixed reversed")
        ]
    outs = run(loop, jobs, log=lambda line: print(line, flush=True))
    rows = [job_summary(j, o, args.seconds) for j, o in zip(jobs, outs, strict=True)]
    passes = card_passes(loop)
    table = [
        f"| Variant | Player | v {versus.name}: kill share (score) | E v T (Yongma) "
        "| Kills/min mean (median) | Kills m / o / f / t (all) | "
        "Damage m / o / f / t (all) | Kill mix | Passes a kill | First-merge | Card head-on "
        "passes Y-Y / J-J / Y-J / J-Y | Gun range Y / J | Circles 1 / 2 / sc | Trades "
        "| Life s | Death gap mean (median) s | Kills/min/ship | Spawn to hit s "
        "| First hit / kill s | Merges a kill mean / median / first-pass / p90 (gun range; "
        "2 turn diameters) | Engagement to death median s | Circles turned | 1 / 2 / sc merges |",
        "|---" * 23 + "|",
    ]
    out = []
    for k, variant in enumerate(variants):
        for player in players:
            mine = [
                r
                for r in rows
                if r["variant"] == k and r["a"] == player.name and r["b"] == player.name
            ]
            value = [
                r
                for r in rows
                if r["variant"] == k and r["a"] == player.name and r["b"] == versus.name
            ]
            value_games = sum(r["games"] for r in value)
            doctrine_value = (
                sum(r["kill_score"][0] * r["games"] for r in value) / value_games,
                sum(r["score"][0] * r["games"] for r in value) / value_games,
            )
            evt = next(r for r in mine if r["lineup"] == "energy v turn")
            weights = [r["games"] for r in mine]
            total = sum(weights)

            def mean_of(key):
                return [sum(r[key][a] * r["games"] for r in mine) / total for a in range(4)]

            kills, damage = mean_of("kill_share"), mean_of("damage_share")
            circles = [sum(r["circles"][c] * r["circle_count"] for r in mine) for c in range(3)]
            circles = [c / max(sum(circles), 1) for c in circles]
            trades = sum(r["trade_fraction"] * r["games"] for r in mine) / total
            engaged = engagement_cells(mine)
            paced = {
                key: sum(r[key] * r["games"] for r in mine) / total
                for key in (
                    "life_seconds",
                    "death_gap_mean",
                    "death_gap_median",
                    "kills_per_minute_per_ship",
                    "spawn_to_engagement_median",
                    "first_hit_median",
                    "first_kill_median",
                )
            }
            per_minute = sum(r["kills_per_minute_mean"] * r["games"] for r in mine) / total
            median = sorted(r["kills_per_minute_median"] for r in mine)[len(mine) // 2]
            p = passes[k]
            entry = dict(
                name=variant["name"],
                energy=variant["energy"],
                turn=variant["turn"],
                energy_v_turn=evt["score"],
                kill_share=kills,
                damage_share=damage,
                kill_mix=mix(kills),
                kills_per_minute=per_minute,
                kills_per_minute_median=median,
                passes_per_kill=sum(r["passes_per_kill"] * r["games"] for r in mine) / total,
                first_merge=sum(r["first_merge_kills"][0] * r["games"] for r in mine) / total,
                card=p,
                trade_fraction=trades,
                pace=paced,
                circles=circles,
                rows=mine,
            )
            out.append(entry)
            table.append(
                f"| {variant['name']} | {player.name} "
                f"| {doctrine_value[0]:.3f} ({doctrine_value[1]:.3f}) "
                f"| {evt['score'][0]:.3f} ± {evt['score'][1]:.3f} "
                f"| {per_minute:.2f} ({median:.2f}) | {' / '.join(f'{v:.2f}' for v in kills)} "
                f"| {' / '.join(f'{v:.2f}' for v in damage)} | {entry['kill_mix']:.2f} "
                f"| {entry['passes_per_kill']:.2f} | {entry['first_merge']:.2f} "
                f"| {p['Y on Y']['head_on_passes']:.2f} / {p['J on J']['head_on_passes']:.2f} / "
                f"{p['Y on J']['head_on_passes']:.2f} / {p['J on Y']['head_on_passes']:.2f} "
                f"| {p['gun_range'][0]:.0f} / {p['gun_range'][1]:.0f} "
                f"| {' / '.join(f'{v:.2f}' for v in circles)} | {trades:.2f} "
                f"| {paced['life_seconds']:.0f} | {paced['death_gap_mean']:.1f} "
                f"({paced['death_gap_median']}) | {paced['kills_per_minute_per_ship']:.2f} "
                f"| {paced['spawn_to_engagement_median']} "
                f"| {paced['first_hit_median']:.1f} / {paced['first_kill_median']:.1f} "
                f"| {engaged} |"
            )
    (out_dir / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    (out_dir / f"{args.tag}.md").write_text("\n".join(table) + "\n")
    print("\n".join(table), flush=True)
    print(f"{time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
