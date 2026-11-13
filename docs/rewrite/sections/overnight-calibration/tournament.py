"""tournament.py: the calibration's ratings rounds and Elo curves.

    round PRESET --target SE [--arms A B ...] [--variant exponential_average]
        one `bnb ratings` round on this box (foreground; run it in tmux): every arm's policy
        files of the variant (BC: the exponential average, RL's hand-off player) found so
        far, plus the scripted agent (1000), adaptive games until every error is under SE.
        Its records go to <ratings>/round-NN-PRESET/, and every earlier round's records are
        reused, so a later round plays only where errors are still above its target.
    curves [--out elo.json]
        every round's newest result per preset, as {preset: {run: [[step, elo, stderr]]}},
        and "mix": the presets' ratings weighted by the training mix's shares (the errors
        combined in quadrature), the curve plateau.py fits to.

Options: --runs DIR (default /workspace/calib/runs), --ratings DIR (default
/workspace/calib/ratings), --bnb PROGRAM (default the repository's .venv/bin/bnb).
"""

import argparse
import json
import math
import os
import pathlib
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
# the training mix's shares (bnb.runtime.train.TRAINING_MIXES["real_scale_all_modes"])
MIX = {
    "real_scale_frontline": 0.5,
    "real_scale_deathmatch": 0.3125,
    "real_scale_elimination": 0.1875,
}


def field(runs: pathlib.Path, arms: list[str], variant: str) -> list[str]:
    """Every arm's policy files of `variant`, in step order."""
    paths = []
    for arm in arms:
        for step in sorted((runs / arm / "policies").glob("step_*"), key=lambda p: int(p.name[5:])):
            if (step / variant / "checkpoint.json").exists():
                paths.append(str(step / variant))
    return paths


def round_(options) -> None:
    launch = tomllib.loads((HERE.parent / "launch" / "launch.toml").read_text())
    runs, ratings = pathlib.Path(options.runs), pathlib.Path(options.ratings)
    players = field(runs, options.arms, options.variant)
    if not players:
        raise SystemExit("no policy files yet")
    ratings.mkdir(parents=True, exist_ok=True)
    earlier = sorted(path for path in ratings.glob("round-*") if path.is_dir())
    out = ratings / f"round-{len(earlier) + 1:02d}-{options.preset}"
    out.mkdir()
    (out / "field.json").write_text(json.dumps(players, indent=1))
    line = [
        options.bnb,
        "ratings",
        arm_and_step(players[0])[0],  # the owner: any run of the field
        "--no-league",
        "--no-ladder",
        "--presets",
        options.preset,
        "--target-stderr",
        str(options.target),
        "--budget",
        str(options.budget),
        "--blocks",
        str(launch["tournament"]["blocks"]),
        "--block-environments",
        str(launch["tournament"]["block_environments"]),
        "--seed",
        str(len(earlier) + 1),
        "--out",
        str(out),
        "--lineup",
        *launch["lineup"],
        "--runs-directory",
        str(runs),
        "--with",
        *players,
    ]
    if earlier:
        line[line.index("--lineup") : line.index("--lineup")] = [
            "--reuse",
            *map(str, earlier),
        ]
    print(" ".join(line), flush=True)
    os.execv(options.bnb, line)


def arm_and_step(path: str) -> tuple[str, int]:
    parts = pathlib.Path(path).parts
    index = parts.index("policies")
    return parts[index - 1], int(parts[index + 1][5:])


def curves(options) -> None:
    ratings = pathlib.Path(options.ratings)
    newest: dict[str, dict] = {}  # preset: the newest result.json
    for result in sorted(ratings.glob("round-*/artifacts/ratings/*/result.json")):
        payload = json.loads(result.read_text())
        rows = payload["payload"]["leaderboard"]
        if rows:
            newest[rows[0]["map"]] = payload
    out: dict[str, dict] = {}
    for preset, payload in newest.items():
        inputs = payload["inputs"]
        table: dict[str, list] = {}
        for row in payload["payload"]["leaderboard"]:
            if row["player"] not in inputs:
                continue
            arm, step = arm_and_step(inputs[row["player"]]["path"])
            table.setdefault(arm, []).append(
                [step, row["calibrated_elo"], row["stderr"], row["games"]]
            )
        out[preset] = {arm: sorted(points) for arm, points in table.items()}
    if out and set(MIX) <= set(out):
        mix: dict[str, list] = {}
        for arm in out["real_scale_frontline"]:
            by_step = [{p[0]: p for p in out[preset].get(arm, [])} for preset in MIX]
            for step in sorted(set.intersection(*(set(table) for table in by_step))):
                elo = sum(MIX[preset] * table[step][1] for preset, table in zip(MIX, by_step))
                error = math.sqrt(
                    sum((MIX[preset] * table[step][2]) ** 2 for preset, table in zip(MIX, by_step))
                )
                mix.setdefault(arm, []).append([step, elo, error])
        out["mix"] = mix
    text = json.dumps(out, indent=1)
    if options.out:
        pathlib.Path(options.out).write_text(text + "\n")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("round", "curves"))
    parser.add_argument("preset", nargs="?")
    parser.add_argument("--target", type=float, default=25.0)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--arms", nargs="+", default=["bc-mb256", "bc-mb128", "bc-mb64", "bc-mb32"])
    parser.add_argument("--variant", default="exponential_average")
    parser.add_argument("--runs", default="/workspace/calib/runs")
    parser.add_argument("--ratings", default="/workspace/calib/ratings")
    parser.add_argument("--bnb", default=str(REPOSITORY / ".venv" / "bin" / "bnb"))
    parser.add_argument("--out", default=None)
    options = parser.parse_args()
    if options.command == "round":
        round_(options)
    else:
        curves(options)


if __name__ == "__main__":
    main()
