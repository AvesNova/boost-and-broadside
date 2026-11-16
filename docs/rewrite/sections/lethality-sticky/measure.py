"""Deathmatch 5v5 behaviour (bench.game_loop's merge definition), Imugi on both sides with
the stored doctrines and real-scale deathmatch's parameters, fields on, the preset's map:
a mirror (mixed lineups) and Yongma against Jujak (both seatings), 512 games each.
Writes one JSON per lineup as it finishes to OUT (default data/<label>.json).

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/lethality-sticky/measure.py LABEL
"""

import datetime
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
SECONDS = 300.0
GAMES = 512
LINEUPS = ("mixed", "energy v turn", "turn v energy")
# the turning-fight probe: 1v1 duels (deathmatch, 120 s, respawns), a mirror and the cross
DUELS = ("turn mirror", "energy v turn")
DUEL_SECONDS = 120.0


def log(line: str) -> None:
    print(f"{datetime.datetime.now():%H:%M:%S} {line}", flush=True)


def row(r: dict) -> dict:
    engagement = r.get("engagement", {})
    first = next(iter(engagement.values()), {}) if engagement else {}
    return dict(
        lineup=r.get("lineup"),
        games=r["games"],
        passes_per_kill=r["passes_per_kill"],
        no_merge_kills=r["no_merge_kills"],
        engagement_length_median=first.get("length_median"),
        kills_per_minute_per_ship=r.get("kills_per_minute_per_ship"),
        kills_per_minute_game=r["kills_per_minute_mean"],
        first_hit_to_kill_median=r["ttk_median"],
        score_a=r["score"],
        kill_share_a=r["kill_score"],
        hit_rate=r.get("hit_rate"),
    )


def main() -> None:
    label = sys.argv[1]
    out_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "data" / f"{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from bench.game_loop import Job, Loop, Player, job_summary, run
    from bnb.controllers.imugi_defaults import MODE_DEFAULTS
    from bnb.controllers.imugi_doctrine import stored
    from bnb.controllers.imugi_parameters import pack

    doctrines = stored()["real-scale"]
    loop = Loop(GAMES, 1, SECONDS, mode="deathmatch", fields=True, fixed_map=True)
    player = Player(
        "imugi",
        doctrines["yongma"],
        doctrines["jujak"],
        parameters=tuple(pack(MODE_DEFAULTS.sets["real_scale_deathmatch"])),
    )
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    started = time.time()
    for number, lineup in enumerate(LINEUPS):
        if lineup in results:
            continue
        began = time.time()
        job = Job(0, lineup, player, player, GAMES)
        (out,) = run(loop, [job], log)
        full = job_summary(job, out, SECONDS)
        results[lineup] = dict(summary=row(full), full=full)
        out_path.write_text(json.dumps(results, indent=1, default=str))
        left = len(LINEUPS) - number - 1
        per = time.time() - began
        log(f"{label} {lineup}: {json.dumps(row(full))}")
        log(
            f"{left} left, done about "
            f"{(datetime.datetime.now() + datetime.timedelta(seconds=per * left)):%H:%M}"
        )
    duel_loop = None
    for lineup in DUELS:
        key = f"1v1 {lineup}"
        if key in results:
            continue
        if duel_loop is None:
            duel_loop = Loop(
                GAMES, 1, DUEL_SECONDS, mode="deathmatch", ships_per_team=1, fields=True,
                fixed_map=True,
            )  # fmt: skip
        job = Job(0, lineup, player, player, GAMES)
        (out,) = run(duel_loop, [job], log)
        full = job_summary(job, out, DUEL_SECONDS)
        results[key] = dict(summary=row(full), full=full)
        out_path.write_text(json.dumps(results, indent=1, default=str))
        log(f"{label} {key}: {json.dumps(row(full))}")
    log(f"all done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
