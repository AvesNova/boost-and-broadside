"""calib.py COMMAND ARM: one overnight-calibration run as a `bnb train` line, built from the
launch config (../launch/launch.toml) and calib.toml, so every run traces to committed files.

    command ARM    print the `bnb train` arguments (shell-quoted)
    config ARM     print the resolved configuration (`bnb train ... --print-config`)
    run ARM        exec `bnb train` (the run writes its own config.json)

Options: --bnb PROGRAM (default: the repository's .venv/bin/bnb), --runs DIR (default
/workspace/calib/runs). Run with the repository's Python.
"""

import argparse
import json
import os
import pathlib
import shlex
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
LAUNCH = HERE.parent / "launch" / "launch.toml"


def arguments_of(name: str, runs: str) -> list[str]:
    """The `bnb train` arguments of the arm `name`."""
    launch = tomllib.loads(LAUNCH.read_text())
    calib = tomllib.loads((HERE / "calib.toml").read_text())
    arms = {arm["name"]: arm for arm in calib["arm"]}
    if name not in arms:
        raise SystemExit(f"no arm {name!r}; arms: {sorted(arms)}")
    arm = arms[name]
    sets = [*launch["set"], f"lineup={json.dumps(launch['lineup'])}", *calib["set"]]
    line = [
        "--preset",
        launch["preset"],
        "--environments",
        str(calib["environments"]),
        "--seed",
        str(arm.get("seed", 0)),
        "--name",
        name,
        "--runs-directory",
        runs,
        *launch.get("train", []),
    ]
    if arm["stage"] == "bc":
        line += ["--objective", "bc", *calib["bc_train"]]
        sets += [*launch["bc_set"], *calib["bc_set"]]
    else:
        if arm["init_from"] == "PENDING":
            raise SystemExit(f"{name}: init_from is PENDING (set it in calib.toml and commit)")
        policy = pathlib.Path(runs) / arm["init_from"]
        line += ["--objective", "ppo", "--init-from", str(policy), "--updates", str(arm["updates"])]
        sets += [*launch["rl_set"], *calib["rl_set"]]
    sets += arm.get("set", [])
    return line + [part for value in sets for part in ("--set", value)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("command", "config", "run"))
    parser.add_argument("arm")
    parser.add_argument("--bnb", default=str(REPOSITORY / ".venv" / "bin" / "bnb"))
    parser.add_argument("--runs", default="/workspace/calib/runs")
    options = parser.parse_args()
    line = [options.bnb, "train", *arguments_of(options.arm, options.runs)]
    if options.command == "command":
        print(shlex.join(line))
    elif options.command == "config":
        os.execv(options.bnb, [*line, "--print-config"])
    else:
        print(shlex.join(line), flush=True)
        os.execv(options.bnb, line)


if __name__ == "__main__":
    main()
