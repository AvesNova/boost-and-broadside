"""One round of the real-scale balancing (docs/explanation/game/real-scale-balance.md), with
no choices left to make:

    uv run --no-sync python -m bench.balance_round --round 2

Its stages, in order (`--stages` runs some of them; each resumes from what is on disk):

1. ships: rebuild the set from `bnb.game.real_scale` and its balance file
   (`python -m bnb.game.real_scale --resume`: a combat-only change re-arms the cards in
   seconds; a flight change solves those types, minutes each).
2. search: each mode's scripted parameter search (`bench/scripted_search.py`) against its
   current champion (the set in `scripted_real_scale.json`) on mixed lineups, the four
   HELD_OUT types kept out of the draws, at most GENERATIONS generations (the plateau rule:
   the median rating not 10 points up in 10 generations).
3. adopt: each mode's candidate (the best-rated survivor of its last generation) against
   the champion, `bench/scripted_compare.py` at T = 1: 1,024 games on mixed lineups of every
   type, and 512 on mixed lineups of the held-out types. The candidate becomes the champion
   (written to `scripted_real_scale.json`) when its score is more than two standard errors
   above one half on the 1,024; else the champion stays.
4. tournament: the ship-type tournament in real_scale_deathmatch
   (`bench/ship_type_tournament.py play`), A-optimal until every standard error is under
   25 Elo, then a balanced top-up so that every pair has played at least one block of 64
   games (the intransitivity checks need every pair; A-optimal leaves the lopsided ones).
5. analyse: the ratings, pair tables, the correlations and models of the rating, and the
   intransitivity checks (`report`, `analyse`, `cyclic`), each table also in `tables.txt`.
6. propose: the decision rule (`propose`), written to proposal.json and applied to the
   balance file (`bnb.game.real_scale.BALANCE_FILE`) as the next round's moves, and the
   ship files rebuilt with them (so the next round's `ships` stage has nothing left to do);
   or the verdict to stop.

Outputs: the search logs (large) in runs/real-scale-balance/round-N/; everything else in
docs/explanation/game/real-scale-balance/round-N/: agent.json (each search's summary, the
checks and the decision), batches.jsonl (the tournament's record), ratings.csv, pairs.csv,
tournament.json, analysis.json, cyclic.json, tables.txt, proposal.json and round.log (every
stage's output; git ignores it). Its parts hold the benchmark lock.
"""

import argparse
import datetime
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/explanation/game/real-scale-balance"
RUNS = ROOT / "runs/real-scale-balance"
MODES = ("deathmatch", "frontline")
HELD_OUT = "Spitfire XIV,Ki-84,Zerstörer,Lantern"
GENERATIONS = 40
KEEP = 8  # the search's survivors a generation
CHECK_GAMES, HELD_OUT_GAMES = 1024, 512
ADOPT_STANDARD_ERRORS = 2.0
TOP_UP_BLOCK = 64  # games: the least every pair plays
BATCH_BLOCKS = 18
STAGES = ("ships", "search", "adopt", "tournament", "analyse", "propose")
# The decision rule (`propose`): the survivability's shield step, set-wide
SHIELD_CORRELATION = 0.4  # |Spearman| of the shield with the Elo that moves the step
STRONG_CORRELATION = 0.7  # ... that moves it twice as far
SHIELD_STEP_CHANGE = 0.05  # per round (the step starts at 0.2: shields 80 to 160)
SHIELD_STEP_BOUNDS = (0.0, 0.2)
# ... and a loosely known type's own shield
OUTLIER_ELO = 200.0  # Elo from the median, and at least OUTLIER_ERRORS standard errors
OUTLIER_ERRORS = 3.0
SHIELD_MOVE = 20.0  # per round
SHIELD_BOUNDS = (80.0, 160.0)  # the set's range
LAST_ROUND = 4
PYTHON = [sys.executable]


def paths(round_number: int) -> tuple[pathlib.Path, pathlib.Path]:
    """(the round's docs data directory, its runs directory)."""
    return DOCS / f"round-{round_number}", RUNS / f"round-{round_number}"


def run(command: list[str], log: pathlib.Path, capture: pathlib.Path | None = None) -> str:
    """Run a command from the repository root, each line of its output streamed to stdout
    and appended to `log` as it comes (its stdout also to `capture`); returns its stdout,
    and raises on failure."""
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(log, "a") as out:
        out.write(f"\n{stamp} $ {' '.join(command)}\n")
    lines = []
    with (
        subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process,
        open(log, "a") as out,
    ):
        for line in process.stdout:
            out.write(line)
            out.flush()
            print(line, end="", flush=True)
            lines.append(line)
    if process.returncode:
        raise SystemExit(f"failed ({process.returncode}): {' '.join(command)}")
    text = "".join(line for line in lines if not line.startswith("[bench]"))
    if capture is not None:
        with open(capture, "a") as out:
            out.write(text)
    return text


def search_log(runs: pathlib.Path, mode: str) -> pathlib.Path:
    """A mode's search log in the round's runs directory."""
    return runs / f"search-{mode}.jsonl"


def generations_done(log: pathlib.Path) -> int:
    """The generations a search log holds (0 for none)."""
    if not log.exists() or not log.stat().st_size:
        return 0
    return 1 + max(json.loads(line)["generation"] for line in log.read_text().splitlines())


def search(round_number: int, mode: str, log_file: pathlib.Path) -> None:
    """A mode's search, resumed from its log, to at most GENERATIONS generations."""
    _, runs = paths(round_number)
    log = search_log(runs, mode)
    left = GENERATIONS - generations_done(log)
    if left <= 0:
        return
    seed = 10 * round_number + (1 if mode == "deathmatch" else 2)
    run(
        PYTHON
        + ["-m", "bench.scripted_search", "--preset", f"real_scale_{mode}", "--mixed"]
        + ["--hold-out", HELD_OUT, "--champion", "defaults", "--from-defaults"]
        + ["--plateau-on", "median", "--patience", "10", "--min-gain", "10"]
        + ["--generations", str(left), "--population", "32", "--games", "64"]
        + ["--reference-share", "0.5", "--sigma", "0.25", "--decay", "0.93"]
        + ["--seed", str(seed), "--log", str(log)],
        log_file,
    )


def summary_of(log: pathlib.Path) -> dict:
    """A search's summary: per generation its top and median rating and the median's score
    against the champion, the champions it had, and its candidate (the best-rated
    survivor of its last generation)."""
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    last = max(row["generation"] for row in rows)
    per_generation = []
    for generation in range(last + 1):
        one = [row for row in rows if row["generation"] == generation]
        ratings = sorted((row["rating"] for row in one), reverse=True)
        scores = sorted(sum(row["scores"]["champion"]) / len(row["presets"]) for row in one)
        per_generation.append(
            dict(
                generation=generation,
                champion=one[0].get("champion"),
                top=ratings[0],
                median=ratings[len(ratings) // 2],
                median_score=scores[len(scores) // 2],
            )
        )
    survivors = [row for row in rows if row["generation"] == last and row["index"] < KEEP]
    best = max(survivors, key=lambda row: row["rating"])
    return dict(
        generations=last + 1,
        per_generation=per_generation,
        candidate=f"{best['generation']}:{best['index']}",
        candidate_rating=best["rating"],
        candidate_stderr=best["stderr"],
    )


def adopt(round_number: int, mode: str, log_file: pathlib.Path) -> dict:
    """Check a mode's candidate against its champion and adopt it by the rule."""
    from bench.scripted_compare import parameters_of
    from bnb.controllers.scripted_defaults import write_real_scale

    docs, runs = paths(round_number)
    log = search_log(runs, mode)
    summary = summary_of(log)
    preset = f"real_scale_{mode}"
    candidate = f"{log}:{summary['candidate']}"
    checks = {}
    for name, extra, games, seed in (
        ("mixed", [], CHECK_GAMES, 1_000_000),
        ("held_out", ["--types", HELD_OUT, "--no-self-play"], HELD_OUT_GAMES, 2_000_000),
    ):
        out = runs / f"compare-{mode}-{name}.jsonl"
        if not out.exists():
            run(
                PYTHON
                + ["-m", "bench.scripted_compare", "--preset", preset, "--mixed"]
                + ["--first", candidate, "--second", preset, "--games", str(games)]
                + ["--seed", str(seed), "--out", str(out)]
                + extra,
                log_file,
            )
        checks[name] = [json.loads(line) for line in out.read_text().splitlines()]
    head = checks["mixed"][0]
    margin = head["score"] - 0.5 - ADOPT_STANDARD_ERRORS * head["score_stderr"]
    adopted = margin > 0.0
    if adopted:
        write_real_scale(
            preset,
            parameters_of(candidate),
            f"balancing round {round_number}: the search's candidate {summary['candidate']} "
            f"(docs/explanation/game/real-scale-balance.md), scoring {head['score']:.3f} +- "
            f"{head['score_stderr']:.3f} against the champion before it",
        )
    return dict(
        summary, checks=checks, adopted=adopted, score=head["score"], stderr=head["score_stderr"]
    )


def tournament(round_number: int, log_file: pathlib.Path) -> None:
    """The A-optimal tournament, then the top-up to a block on every pair."""
    docs, _ = paths(round_number)
    record = docs / "batches.jsonl"
    common = ["--record", str(record), "--blocks", str(BATCH_BLOCKS), "--block-games", "64"]
    run(
        PYTHON
        + ["-m", "bench.ship_type_tournament", "play", "--allocation", "a_optimal"]
        + ["--target-stderr", "25", "--budget", "100000", "--hours", "1.5"]
        + common,
        log_file,
    )
    short = pairs_short(record)
    if short:
        batch = BATCH_BLOCKS * TOP_UP_BLOCK
        games = math.ceil(short * TOP_UP_BLOCK / batch) * batch
        recorded = json.loads(record.read_text().splitlines()[-1])["games"]
        marker = docs / "top-up.json"
        if not marker.exists():
            marker.write_text(json.dumps(dict(after_games=recorded, pairs_short=short)) + "\n")
        run(
            PYTHON
            + ["-m", "bench.ship_type_tournament", "play", "--top-up", str(games)]
            + ["--hours", "1.0"]
            + common,
            log_file,
        )


def pairs_short(record: pathlib.Path) -> int:
    """The pairs of a record with fewer than TOP_UP_BLOCK games."""
    from bench.ship_type_tournament import _summaries

    _, counts, _, _ = _summaries(record)
    games = counts.sum(-1)
    count = games.shape[0]
    return sum(1 for i in range(count) for j in range(i + 1, count) if games[i, j] < TOP_UP_BLOCK)


def analyse(round_number: int, log_file: pathlib.Path) -> None:
    """The tables, from the record."""
    docs, _ = paths(round_number)
    tables = docs / "tables.txt"
    tables.write_text(f"# Round {round_number} tables\n\n")
    record = ["--record", str(docs / "batches.jsonl")]
    out = ["--out", str(docs)]
    for title, command in (
        ("Ratings", ["report", *record, *out]),
        ("What predicts the rating", ["analyse", *out]),
        ("Intransitivity", ["cyclic", *record, *out]),
    ):
        with open(tables, "a") as file:
            file.write(f"\n## {title}\n\n")
        run(PYTHON + ["-m", "bench.ship_type_tournament", *command], log_file, tables)


def loosely_known(key: str) -> bool:
    """Whether a type has no researched datum (an invented design or a placeholder); False
    for a player's type (not in `real_scale.AIRCRAFT`), which the rounds never move."""
    from bnb.game.real_scale import AIRCRAFT

    aircraft = next((one for one in AIRCRAFT if one.key == key), None)
    if aircraft is None:
        return False
    return all(datum.confidence == "invented" for datum in aircraft.data.values())


def propose(round_number: int, apply: bool = True) -> dict:
    """The next round's ship moves by the decision rule, from this round's tournament:

    - the shield step per survivability rating (set-wide; the ratings 1-5 are estimates and
      the step a proposal): when the Spearman correlation of the shield with the Elo is at
      least SHIELD_CORRELATION, the step falls by SHIELD_STEP_CHANGE (twice that at
      STRONG_CORRELATION or more); when it is at most -SHIELD_CORRELATION, it rises as much;
      always within SHIELD_STEP_BOUNDS;
    - a loosely known type's own shield (an invented design or the placeholder; never a
      researched type): when its Elo is more than OUTLIER_ELO from the median and more than
      OUTLIER_ERRORS standard errors, its shield moves SHIELD_MOVE towards the median's side
      (up when it is below), within SHIELD_BOUNDS;
    - stop when neither fires, or after LAST_ROUND.

    Returns the proposal (also written to proposal.json); `apply` merges its moves into the
    balance file, labelled round + 1. A round proposed already gives its proposal.json again
    (re-applying its moves), so a rerun does not move a value twice."""
    from bnb.game import real_scale, ships

    docs, _ = paths(round_number)
    written = docs / "proposal.json"
    if written.exists():  # proposed already: the same moves again, not a second step
        proposal = json.loads(written.read_text())
        if apply and not proposal["stop"]:
            moves = real_scale.balance_moves()
            for move in proposal["moves"]:
                key = real_scale.RULES if move["type"] == "(every type)" else move["key"]
                fields = {name: move[name] for name in ("value", "round", "reason")}
                moves.setdefault(key, {})[move["handle"]] = fields
            real_scale.BALANCE_FILE.write_text(
                json.dumps(moves, indent=1, ensure_ascii=False) + "\n"
            )
        return proposal
    rated = json.loads((docs / "tournament.json").read_text())["ratings"]
    analysis = json.loads((docs / "analysis.json").read_text())
    rho = analysis["correlations"]["max_shield"]["spearman"]
    nxt = round_number + 1
    moves = real_scale.balance_moves()
    step, _ = real_scale.shield_per_rating()
    proposal = dict(round=round_number, shield_spearman=rho, shield_step=step, moves=[])
    change = 0.0
    if abs(rho) >= SHIELD_CORRELATION:
        change = SHIELD_STEP_CHANGE * (2.0 if abs(rho) >= STRONG_CORRELATION else 1.0)
        change *= -1.0 if rho > 0 else 1.0
    new_step = min(max(step + change, SHIELD_STEP_BOUNDS[0]), SHIELD_STEP_BOUNDS[1])
    if not math.isclose(new_step, step):
        reason = (
            f"the shield's Spearman correlation with the round-{round_number} Elo was "
            f"{rho:+.2f} (the rule moves the step at |rho| >= {SHIELD_CORRELATION:g}); the "
            "survivability ratings are estimates and the step a proposal."
        )
        move = dict(value=round(new_step, 4), round=nxt, reason=reason)
        moves.setdefault(real_scale.RULES, {})["shield_per_rating"] = move
        proposal["moves"].append(dict(type="(every type)", handle="shield_per_rating", **move))
    by_name = {ship.name: ship for ship in ships.load(real_scale.SHIP_SET)}
    elos = sorted(row["elo"] for row in rated)
    median = elos[len(elos) // 2]
    proposal["median_elo"] = median
    for row in rated:
        ship = by_name[row["type"]]
        gap = row["elo"] - median
        if not loosely_known(ship.key):
            continue
        if abs(gap) <= max(OUTLIER_ELO, OUTLIER_ERRORS * row["stderr"]):
            continue
        shield = ship.handles["max_shield"]
        target = shield + (SHIELD_MOVE if gap < 0 else -SHIELD_MOVE)
        target = min(max(target, SHIELD_BOUNDS[0]), SHIELD_BOUNDS[1])
        if math.isclose(target, shield):
            proposal.setdefault("at_bounds", []).append(row["type"])
            continue
        reason = (
            f"an invented or placeholder type rated {row['elo']:.0f} +- {row['stderr']:.0f} in "
            f"round {round_number}, {gap:+.0f} from the median {median:.0f} (the rule moves "
            f"its shield {SHIELD_MOVE:g} towards the median beyond {OUTLIER_ELO:g} Elo)."
        )
        move = dict(value=target, round=nxt, reason=reason)
        moves.setdefault(ship.key, {})["max_shield"] = move
        proposal["moves"].append(dict(type=row["type"], key=ship.key, handle="max_shield", **move))
    proposal["stop"] = not proposal["moves"] or round_number >= LAST_ROUND
    written.write_text(json.dumps(proposal, indent=1, ensure_ascii=False) + "\n")
    if apply and not proposal["stop"]:
        real_scale.BALANCE_FILE.write_text(json.dumps(moves, indent=1, ensure_ascii=False) + "\n")
    return proposal


def main() -> None:
    """The stages the module describes, for one round."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--stages", default=",".join(STAGES), help="comma-separated")
    args = parser.parse_args()
    stages = args.stages.split(",")
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; the stages are {STAGES}")
    docs, runs = paths(args.round)
    docs.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    log_file = docs / "round.log"
    if "ships" in stages:
        run(PYTHON + ["-m", "bnb.game.real_scale", "--resume"], log_file)
    if "search" in stages:
        for mode in MODES:
            search(args.round, mode, log_file)
    if "adopt" in stages:
        agent_file = docs / "agent.json"
        agent = json.loads(agent_file.read_text()) if agent_file.exists() else {}
        for mode in MODES:
            if mode not in agent:
                agent[mode] = adopt(args.round, mode, log_file)
                agent_file.write_text(json.dumps(agent, indent=1, ensure_ascii=False) + "\n")
            print(
                f"{mode}: candidate {agent[mode]['candidate']} scores "
                f"{agent[mode]['score']:.3f} +- {agent[mode]['stderr']:.3f}; adopted "
                f"{agent[mode]['adopted']}",
                flush=True,
            )
    if "tournament" in stages:
        tournament(args.round, log_file)
    if "analyse" in stages:
        analyse(args.round, log_file)
    if "propose" in stages:
        proposal = propose(args.round)
        print(json.dumps(proposal, indent=1, ensure_ascii=False), flush=True)
        if not proposal["stop"]:  # the ship files follow the balance file at once
            run(PYTHON + ["-m", "bnb.game.real_scale", "--resume"], log_file)
    print(f"round {args.round}: stages {', '.join(stages)} done", flush=True)


if __name__ == "__main__":
    main()
