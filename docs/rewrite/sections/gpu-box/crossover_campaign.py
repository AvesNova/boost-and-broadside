"""crossover_campaign.py: the crossover fights (the author, 2026-11-10 18:00 KST). For each best
checkpoint (champion, null, nearest), mode and fleet size (5v5; 10v10, 20v20, 40v40 in the training
arena and at the training density), the ship-count handicap at which its score against the scripted
agent crosses 0.5. One `bnb crossover` process compiles one game size and plays every handicap in it
(a per-game bench count). The author (18:10 KST): measure it upwards, "n network ships beat up to
m scripted ones": the network keeps n ships and the scripted side gets more (`case`).
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
LAYOUT = {5: (8, 128), 10: (8, 64), 20: (8, 32), 40: (8, 16), 80: (8, 8), 120: (4, 8)}  # blocks, envs
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
    """The two neighbouring rows (by the scripted side's count) the score crosses 0.5 between."""
    rows = sorted(rows, key=lambda r: r["second_active"])
    for low, high in zip(rows, rows[1:]):
        if (low["score"] - 0.5) * (high["score"] - 0.5) <= 0 and low["score"] != high["score"]:
            return low, high
    return None


def grid(n: int, top: int) -> list[int]:
    """The scripted side's counts from n to top: every integer while few, else 10 even steps."""
    if top - n <= 10:
        return list(range(n, top + 1))
    return sorted({round(n + (top - n) * f / 10) for f in range(11)})


def case(agent: str, preset: str, n: int) -> dict:
    """A (the network) keeps its n ships; the scripted side gets n .. 2n (3n while A still wins at
    2n), each at a game size of the larger side; then every integer inside the crossing's
    bracket. If A loses at n:n, the scripted side gets fewer: n .. 1 (A then needs more ships than
    the scripted agent to break even)."""
    rows = []
    payload = run(agent, preset, n, [f"{n}:{b}" for b in grid(n, 2 * n)], 256 if n <= 10 else 128,
                  2 * n)
    rows += payload["rows"]
    top = 2 * n
    if all(r["score"] > 0.5 for r in rows):  # still wins at 2n: up to 3n
        payload = run(agent, preset, n, [f"{n}:{b}" for b in grid(2 * n, 3 * n)[1:]],
                      256 if n <= 10 else 128, 3 * n)
        rows += payload["rows"]
        top = 3 * n
    ships = top
    if min(r["score"] for r in rows if r["second_active"] == n) < 0.5:  # loses at n:n: fewer
        down = list(range(n - 1, 0, -1)) if n <= 10 else sorted(
            {max(1, round(n * f / 10)) for f in range(1, 10)}, reverse=True)
        payload = run(agent, preset, n, [f"{n}:{b}" for b in down], 256 if n <= 10 else 128, None)
        rows += payload["rows"]
        ships = None
    pair = crossing(rows)
    if pair is not None:
        low, high = (int(r["second_active"]) for r in pair)
        between = list(range(low + 1, high))
        if between:
            payload = run(agent, preset, n, [f"{n}:{b}" for b in between], 256,
                          ships if high > n else None)
            rows += payload["rows"]
            pair = crossing(rows)
    result = dict(agent=agent, preset=preset, size=n, rows=sorted(
        [{k: r[k] for k in ("first_active", "second_active", "games", "wins", "draws", "losses",
                            "score", "score_low", "score_high")} for r in rows],
        key=lambda r: r["second_active"]))
    if pair is None:
        result["crossover"] = None
    else:
        low, high = pair
        t = (0.5 - low["score"]) / (high["score"] - low["score"])
        enemy = low["second_active"] + t * (high["second_active"] - low["second_active"])
        result["crossover"] = dict(own=n, enemy=enemy, ratio=enemy / n)
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
                f"{n} network ships break even against {c['enemy']:.2f} scripted "
                f"(x{c['ratio']:.2f})"))
    log("campaign done")


if __name__ == "__main__":
    main()
