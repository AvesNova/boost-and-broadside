"""crossover_campaign.py: the crossover fights (the author, 2026-11-10 18:00 KST). For each best
checkpoint (champion, null, nearest), mode and fleet size (5v5; 10v10, 20v20, 40v40 in the training
arena and at the training density), the ship-count handicap at which its score against the scripted
agent crosses 0.5. One `bnb crossover` process compiles one game size and plays every handicap in it
(a per-game bench count):
- n <= 10: the whole integer grid at once (A at n .. 1 against the scripted agent's n).
- n > 10: a coarse grid (A at n x 1.0, 0.9, ..., 0.1), then every integer between the two coarse
  levels the 0.5 crossing falls between (the same game size: the compile cache's graph).
- When A loses even at n:n, the grid goes the other way: A gets more ships (n .. 2n against n, at a
  game size of 2n), and likewise coarse then fine.
Records (each `bnb crossover`'s result.json) under runs/crossover/<agent>/<preset>/; a summary line per
case to runs/crossover/summary.jsonl. A case already in the summary is skipped (restartable).
Usage: crossover_campaign.py [AGENT ...]"""

import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path("/workspace/boost-and-broadside")
AGENTS = {
    "champion": "runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies/step_180355072/uniform_average",
    "null": "runs/2026-11-09-rl-absolute/policies/step_180355072/uniform_average",
    "nearest": "runs/2026-11-10-rl-nearest/policies/step_40894464/uniform_average",
}
PRESETS = [("frontline", 5), ("deathmatch", 5)] + [
    (f"{mode}_{n}v{n}_{variant}", n)
    for n in (10, 20, 40)
    for variant in ("arena", "density")
    for mode in ("frontline", "deathmatch")
]
LAYOUT = {5: (8, 128), 10: (8, 64), 20: (8, 32), 40: (8, 16)}  # blocks, environments per block
OUT = ROOT / "runs" / "crossover"
SUMMARY = OUT / "summary.jsonl"


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M', time.gmtime())} crossover {message}"
    print(line, flush=True)
    with open(ROOT / "runs" / "arms.log", "a") as f:
        f.write(line + "\n")


def run(agent: str, preset: str, size: int, pairs: list[str], games: int, ships: int | None) -> dict:
    out = OUT / agent / preset
    blocks, envs = LAYOUT[min(LAYOUT, key=lambda k: abs(k - (ships or size)))]
    command = [".venv/bin/bnb", "crossover", AGENTS[agent], "--preset", preset, "--pairs", *pairs,
               "--games", str(games), "--blocks", str(blocks), "--block-environments", str(envs),
               "--out", str(out)]
    if ships is not None:
        command += ["--ships-per-team", str(ships)]
    before = set(out.glob("artifacts/crossover/*/result.json"))
    with open(ROOT / "runs" / "crossover.out", "a") as f:
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    (new,) = set(out.glob("artifacts/crossover/*/result.json")) - before
    return json.loads(new.read_text())["payload"]


def crossing(rows: list[dict]) -> tuple[dict, dict] | None:
    """The two neighbouring rows (by A's count) the score crosses 0.5 between, else None."""
    rows = sorted(rows, key=lambda r: r["first_active"])
    for low, high in zip(rows, rows[1:]):
        if (low["score"] - 0.5) * (high["score"] - 0.5) <= 0 and low["score"] != high["score"]:
            return low, high
    return None


def case(agent: str, preset: str, n: int) -> dict:
    rows = []
    if n <= 10:
        payload = run(agent, preset, n, [f"{a}:{n}" for a in range(n, 0, -1)], 256, None)
        rows += payload["rows"]
        ships = None
    else:
        coarse = sorted({max(1, round(n * f / 10)) for f in range(10, 0, -1)}, reverse=True)
        payload = run(agent, preset, n, [f"{a}:{n}" for a in coarse], 128, None)
        rows += payload["rows"]
        ships = None
    if min(r["score"] for r in rows if r["first_active"] == n) < 0.5:  # loses at n:n: add ships
        ships = 2 * n
        grid = range(n, 2 * n + 1) if n <= 10 else sorted({round(n * (1 + f / 10)) for f in range(11)})
        payload = run(agent, preset, n, [f"{a}:{n}" for a in grid], 256 if n <= 10 else 128, ships)
        rows += payload["rows"]
    pair = crossing(rows)
    if pair is not None and n > 10:
        low, high = (int(r["first_active"]) for r in pair)
        between = [a for a in range(low + 1, high)]
        if between:
            payload = run(agent, preset, n, [f"{a}:{n}" for a in between], 256, ships)
            rows += payload["rows"]
            pair = crossing(rows)
    result = dict(agent=agent, preset=preset, size=n, rows=sorted(
        [{k: r[k] for k in ("first_active", "second_active", "games", "wins", "draws", "losses",
                            "score", "score_low", "score_high")} for r in rows],
        key=lambda r: r["first_active"]))
    if pair is None:
        result["crossover"] = None
    else:
        low, high = pair
        t = (0.5 - low["score"]) / (high["score"] - low["score"])
        active = low["first_active"] + t * (high["first_active"] - low["first_active"])
        result["crossover"] = dict(active=active, enemy=n, ships_given=n - active,
                                   fraction_given=(n - active) / n)
    return result


def main() -> None:
    agents = sys.argv[1:] or list(AGENTS)
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if SUMMARY.exists():
        done = {(d["agent"], d["preset"]) for d in map(json.loads, SUMMARY.read_text().splitlines())}
    for preset, n in PRESETS:
        for agent in agents:
            if (agent, preset) in done:
                continue
            log(f"start {agent} {preset}")
            try:
                result = case(agent, preset, n)
            except subprocess.CalledProcessError as error:
                log(f"FAILED {agent} {preset} ({error}; runs/crossover.out)")
                continue
            with open(SUMMARY, "a") as f:
                f.write(json.dumps(result) + "\n")
            c = result["crossover"]
            log(f"done {agent} {preset}: " + ("no crossing in the grid" if c is None else
                f"{c['active']:.2f} vs {n} ({100 * c['fraction_given']:+.0f}% of the fleet given)"))
    log("campaign done")


if __name__ == "__main__":
    main()
