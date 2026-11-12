"""launch.py COMMAND: the final experiments on one machine of N GPUs (box-agnostic: the
8-GPU Stage A box, or the 1-3 GPU box the winners continue on), over `bnb queue`'s slots.

    check                 the lineup is set (not the placeholder), every arm's resolved
                          configs (`bnb train ... --print-config`), each arm's differences
                          from the control; writes <runs>/launch/configs/<run>.json
    probe [--gpus 0,1,..] the micro-batch rows per distinct model and stage: the largest of
                          launch.toml's [probe] lists whose first update fits, one probe per
                          GPU at a time; writes <runs>/launch/micro_batch.json
    enqueue [--group A]   queue every run of the group (a run already queued is skipped),
                          priority = the arm's cost (longest first), warm key = its model
    status                per job: stage, chain steps, SPS, ETA (KST clock), last KL and
                          the lane scores against the scripted agent; the newest leaderboard
    tournament --target SE [--shards N]   a tournament round over every run's policies at
                          launch.toml's matched steps reached so far (averaged and live),
                          queued as ratings jobs, N per preset (each its own seed and record,
                          at SE x sqrt(N)), reusing every earlier record
    leaderboard           fit every record of the field together (no games: budget 0) and
                          print the ratings by preset, labelled (arm, seed, steps, variant)
    resume-to STEPS [RUN ...]   raise the chains' combined budget (150M -> 500M) and queue
                          their resumes

Options for every command: --config launch.toml, --arms arms.toml, --runs DIR, --queue FILE
(default: launch.toml's [box]), --bnb PROGRAM (default the repository's .venv/bin/bnb).
Run with the repository's Python: `.venv/bin/python docs/rewrite/sections/launch/launch.py`.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
import time
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
PLACEHOLDER = "PLACEHOLDER"
KST = datetime.timezone(datetime.timedelta(hours=9))
# the configuration paths whose values set the compiled shapes and the memory: arms equal
# on all of them share a warm key and a micro-batch probe
MODEL_PATHS = (
    "model",
    "yemong",
    "nearest",
    "relational",
    "environments",
    "length",
    "shards",
    "lookahead",
    "host_store",
    "learner.minibatches",
    "game",
    "lanes",
)
TOURNAMENT_PRIORITY = 100.0  # ratings jobs go before any training job waiting


# ---------------------------------------------------------------- the settings


def load(arguments) -> tuple[dict, list[dict]]:
    """launch.toml and the arms (one entry per arm and seed, its run name)."""
    config = tomllib.loads(pathlib.Path(arguments.config).read_text())
    arms = tomllib.loads(pathlib.Path(arguments.arms).read_text())["arm"]
    runs: dict[str, dict] = {}
    for arm in arms:
        for seed in arm.get("seeds", [0]):
            run = dict(arm, seed=seed, run=f"{arm['name']}-s{seed}")
            run.setdefault("objective", "chain")
            for key in ("set", "bc_set", "rl_set", "train"):
                run.setdefault(key, [])
            same = runs.get(run["run"])
            if same is not None:  # the same run named in two groups: one run
                fields = ("set", "bc_set", "rl_set", "train", "objective")
                if any(same[key] != run[key] for key in fields):
                    raise SystemExit(f"{run['run']} is named twice with other settings")
                continue
            runs[run["run"]] = run
    return config, list(runs.values())


def check_lineup(config: dict) -> list[str]:
    """The lineup's names, or exit while it is the placeholder."""
    lineup = config["lineup"]
    if lineup == PLACEHOLDER or not isinstance(lineup, list):
        raise SystemExit(
            "launch.toml's lineup is the placeholder: set it to the ship lineup (one type name "
            'per slot, such as ["Yongma", "Yongma", "Jujak", "Jujak", "Jujak"]) before launching'
        )
    return lineup


def train_arguments(config: dict, run: dict, runs_directory: str, micro: dict | None) -> list[str]:
    """The `bnb train` arguments of one run: the shared settings, the arm's, the lineup and
    the probed micro-batch rows (per stage)."""
    lineup = check_lineup(config)
    arguments = [
        "--preset",
        config["preset"],
        "--environments",
        str(config["environments"]),
        "--seed",
        str(run["seed"]),
        "--name",
        run["run"],
        "--runs-directory",
        runs_directory,
        *config.get("train", []),
        *run["train"],
    ]
    sets = [*config["set"], *run["set"], f"lineup={json.dumps(lineup)}"]
    if run["objective"] == "chain":
        arguments += [
            "--objective",
            "chain",
            "--chain-budget-steps",
            str(run.get("chain_budget_steps", config["chain_budget_steps"])),
        ]
        bc_sets, rl_sets = [*config["bc_set"], *run["bc_set"]], [*config["rl_set"], *run["rl_set"]]
        if micro is not None:
            bc_sets.append(f"learner.micro_batch_rows={micro['bc']}")
            rl_sets.append(f"learner.micro_batch_rows={micro['rl']}")
        arguments += [part for line in bc_sets for part in ("--bc-set", line)]
        arguments += [part for line in rl_sets for part in ("--rl-set", line)]
    else:  # RL alone: the RL stage's lines are its own
        sets += [*config["rl_set"], *run["rl_set"]]
        if micro is not None:
            arguments += ["--micro-batch-rows", str(micro["rl"])]
    arguments += [part for line in sets for part in ("--set", line)]
    return arguments


def stage_arguments(config: dict, run: dict, stage: str, rows: int) -> list[str]:
    """A run's one stage as a plain `bnb train` (the probe's): BC or PPO, its own lines."""
    lineup = check_lineup(config)
    sets = [*config["set"], *run["set"], f"lineup={json.dumps(lineup)}"]
    sets += [*config[f"{stage}_set"], *run[f"{stage}_set"]]
    sets.append(f"learner.micro_batch_rows={rows}")
    train = list(run["train"])
    if "--updates" in train:  # the run's budget, not the probe's
        index = train.index("--updates")
        del train[index : index + 2]
    arguments = [
        "--preset",
        config["preset"],
        "--environments",
        str(config["environments"]),
        "--objective",
        "bc" if stage == "bc" else "ppo",
        "--probe",
        *config.get("train", []),
        *train,
    ]
    return arguments + [part for line in sets for part in ("--set", line)]


def flat(data, prefix: str = "") -> dict:
    """A nested dict's leaves by dotted path."""
    out = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        out.update(flat(value, path + ".") if isinstance(value, dict) else {path: value})
    return out


def model_key(resolved: dict) -> str:
    """A run's warm key: a digest of the settings that set its compiled shapes and memory
    (both stages' for a chain)."""
    stages = resolved if "bc" in resolved else {"rl": resolved}
    parts = {}
    for stage, config in stages.items():
        values = flat(config)
        parts[stage] = {
            path: value
            for path, value in sorted(values.items())
            if any(path == m or path.startswith(m + ".") for m in MODEL_PATHS)
        }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def resolve(bnb: list[str], arguments: list[str]) -> dict:
    """`bnb train ARGUMENTS --print-config` as JSON (a chain: {"bc": ..., "rl": ...})."""
    done = subprocess.run(
        [*bnb, "train", *arguments, "--print-config"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
    )
    if done.returncode != 0:
        raise SystemExit(
            f"bnb train {shlex.join(arguments)} --print-config failed:\n"
            f"{done.stdout[-2000:]}{done.stderr[-2000:]}"
        )
    return json.loads(done.stdout)


def launch_directory(arguments) -> pathlib.Path:
    path = pathlib.Path(arguments.runs) / "launch"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- check


def check(arguments) -> None:
    """Resolve every arm; print each one's differences from the control's config."""
    config, runs = load(arguments)
    check_lineup(config)
    directory = launch_directory(arguments) / "configs"
    directory.mkdir(exist_ok=True)
    resolved = {}
    for run in runs:
        if run.get("needs"):
            print(f"{run['run']:28s} NOT BUILDABLE: needs {run['needs']}")
            continue
        resolved[run["run"]] = resolve(
            arguments.bnb, train_arguments(config, run, arguments.runs, None)
        )
        (directory / f"{run['run']}.json").write_text(json.dumps(resolved[run["run"]], indent=1))
    control = resolved.get("control-s0")
    for name, value in resolved.items():
        key = model_key(value)
        if control is None or name == "control-s0":
            print(f"{name:28s} key {key}: resolved")
            continue
        differences = []
        stages = ("bc", "rl") if "bc" in value else (None,)
        for stage in stages:
            mine = flat(value[stage] if stage else value)
            theirs = flat(control[stage] if stage else control["rl"])
            for path in sorted(mine.keys() | theirs.keys()):
                if path.startswith("ship_card") or path in ("name",):
                    continue
                if mine.get(path, "<absent>") != theirs.get(path, "<absent>"):
                    where = f"{stage}:" if stage else ""
                    differences.append(f"{where}{path}={json.dumps(mine.get(path, '<absent>'))}")
        print(f"{name:28s} key {key}: " + (", ".join(differences) or "same as control"))
    print(f"configs in {directory}")


# ---------------------------------------------------------------- probe


def probe(arguments) -> None:
    """The largest micro-batch rows that fit, per distinct model and stage."""
    config, runs = load(arguments)
    check_lineup(config)
    out = launch_directory(arguments) / "micro_batch.json"
    found = json.loads(out.read_text()) if out.exists() else {}
    work = {}  # (key, stage) -> a run of that key
    for run in runs:
        if run.get("needs") or (arguments.group and run["group"] != arguments.group):
            continue
        key = model_key(resolve(arguments.bnb, train_arguments(config, run, arguments.runs, None)))
        stages = ("bc", "rl") if run["objective"] == "chain" else ("rl",)
        for stage in stages:
            if stage not in found.get(key, {}):
                work.setdefault((key, stage), run)
    gpus = arguments.gpus.split(",") if arguments.gpus else [str(i) for i in range(gpu_count())]
    logs = launch_directory(arguments) / "probe"
    logs.mkdir(exist_ok=True)
    pending = list(work.items())
    running: dict[str, tuple] = {}  # gpu -> (key, stage, run, candidates left, process)
    print(f"probing {len(pending)} (model, stage) pairs on GPUs {gpus}", flush=True)

    def start(gpu, key, stage, run, candidates):
        rows = candidates[0]
        log = logs / f"{key}-{stage}-{rows}.log"
        line = [*arguments.bnb, "train", *stage_arguments(config, run, stage, rows)]
        process = subprocess.Popen(
            line,
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
            cwd=REPOSITORY,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
        )
        running[gpu] = (key, stage, run, candidates, process, log)
        print(f"{now()} GPU {gpu}: {run['run']} {stage} at {rows} rows ({log.name})", flush=True)

    while pending or running:
        for gpu in gpus:
            if gpu not in running and pending:
                (key, stage), run = pending.pop(0)
                start(gpu, key, stage, run, list(config["probe"][stage]))
        time.sleep(5)
        for gpu, (key, stage, run, candidates, process, log) in list(running.items()):
            if process.poll() is None:
                continue
            del running[gpu]
            peaks = probe_peaks(log.read_text(errors="replace"))
            fits = (
                process.returncode == 0
                and peaks is not None
                and peaks[0] <= config["probe"]["fraction"] * peaks[1]
            )
            rows = candidates[0]
            print(
                f"{now()} GPU {gpu}: {run['run']} {stage} {rows} rows: "
                + (f"reserved {peaks[0]:.1f} of {peaks[1]:.1f} GB" if peaks else "no peaks")
                + (" fits" if fits else f" does not fit (exit {process.returncode})"),
                flush=True,
            )
            if fits:
                found.setdefault(key, {})[stage] = rows
                found[key].setdefault("runs", [])
                if run["run"] not in found[key]["runs"]:
                    found[key]["runs"].append(run["run"])
                out.write_text(json.dumps(found, indent=1))
            elif process.returncode != 0 and not out_of_memory(log.read_text(errors="replace")):
                raise SystemExit(f"{run['run']} {stage}: the probe failed, not for memory ({log})")
            elif len(candidates) > 1:
                start(gpu, key, stage, run, candidates[1:])
            else:
                raise SystemExit(f"{run['run']} {stage}: no micro-batch rows fit ({log})")
    print(f"micro-batch rows in {out}: {json.dumps(found)}")


def probe_peaks(text: str) -> tuple[float, float] | None:
    """(reserved, total) GB from a probe's `memory probe:` line."""
    for line in text.splitlines():
        found = re.match(
            r"memory probe: allocated [\d.]+ GB, reserved ([\d.]+) GB, .* of ([\d.]+) GB", line
        )
        if found:
            return float(found.group(1)), float(found.group(2))
    return None


def out_of_memory(text: str) -> bool:
    """Whether a probe's log ends in an out-of-memory failure (the device's, or the
    probe's budget)."""
    return any(sign in text for sign in ("OutOfMemoryError", "out of memory", "MemoryBudgetError"))


def gpu_count() -> int:
    done = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return max(1, len([line for line in done.stdout.splitlines() if line.startswith("GPU")]))


def now() -> str:
    return datetime.datetime.now(KST).strftime("%H:%M:%S")


# ---------------------------------------------------------------- enqueue


def queue_of(arguments):
    sys.path.insert(0, str(REPOSITORY / "src"))
    from bnb.runs.queue import Queue

    return Queue(pathlib.Path(arguments.queue))


def enqueue(arguments) -> None:
    """Queue every run of the group not yet in the queue."""
    config, runs = load(arguments)
    check_lineup(config)
    micro = (
        json.loads((launch_directory(arguments) / "micro_batch.json").read_text())
        if (launch_directory(arguments) / "micro_batch.json").exists()
        else {}
    )
    store = queue_of(arguments)
    queued = {job.get("run") for job in store.read()["jobs"] if job["state"] != "removed"}
    for run in sorted(runs, key=lambda run: -run["cost"]):
        if run["group"] != arguments.group or run["run"] in queued:
            continue
        if run.get("needs"):
            print(f"skipped {run['run']}: needs {run['needs']}")
            continue
        key = model_key(resolve(arguments.bnb, train_arguments(config, run, arguments.runs, None)))
        rows = micro.get(key)
        if rows is None and not arguments.default_micro_batch:
            raise SystemExit(
                f"{run['run']}: no probed micro-batch rows for key {key} "
                "(launch.py probe, or --default-micro-batch)"
            )
        line = train_arguments(config, run, arguments.runs, rows)
        if arguments.dry_run:
            print(
                f"would queue {run['run']} (priority {run['cost']}, warm {key}): bnb train "
                + shlex.join(line)
            )
            continue
        job = store.add(
            "train",
            line,
            program=arguments.bnb,
            cwd=REPOSITORY,
            runs_directory=pathlib.Path(arguments.runs),
            priority=run["cost"],
            warm=key,
        )
        print(f"job {job['id']}: {run['run']} (priority {run['cost']}, warm {key})")


# ---------------------------------------------------------------- status


def metrics(path: pathlib.Path, last: int = 6) -> list[dict]:
    """The last rows of a metrics.jsonl."""
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    rows = []
    for line in lines[-last:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run_progress(runs_directory: pathlib.Path, name: str) -> dict | None:
    """A run's (or a chain's) stage, chain steps, budget, SPS and last KL and lane scores."""
    chain = runs_directory / name / "chain.json"
    if chain.exists():
        record = json.loads(chain.read_text())
        budget = record["budget_steps"]
        stage = "rl" if record["handover"] else "bc"
        run = record["stages"][stage]
    elif (runs_directory / name / "run.json").exists():
        record, budget, stage, run = None, None, "rl", name
    else:
        return None
    rows = metrics(runs_directory / run / "metrics.jsonl")
    info = dict(stage=stage, run=run, budget=budget, rows=len(rows))
    run_record = (
        json.loads((runs_directory / run / "run.json").read_text())
        if (runs_directory / run / "run.json").exists()
        else {}
    )
    info["status"] = run_record.get("status", "-")
    if budget is None:  # a PPO run: its updates x steps per update
        config = runs_directory / run / "config.json"
        if config.exists():
            values = json.loads(config.read_text())
            per = values["environments"] * values["length"] * values["shards"]
            updates = values["updates"]
            for segment in run_record.get("segments", []):
                updates = segment.get("overrides", {}).get("updates", updates)
            budget = updates * per
            info["budget"] = budget
    if not rows:
        return info
    last = rows[-1]
    steps = last.get("counters/chain_environment_steps", last["counters/environment_steps"])
    sps = statistics.median(row["train/environment_steps_per_second"] for row in rows[-5:])
    info.update(
        steps=steps, sps=sps, updated=(runs_directory / run / "metrics.jsonl").stat().st_mtime
    )
    info["kl"] = last.get("bc/kl", last.get("policy/kl"))
    by_player = {}
    for key, value in last.items():
        if key.startswith("lanes/score/") and key.endswith("_vs_scripted") and "league" not in key:
            by_player.setdefault(key.split("/")[-1].replace("_vs_scripted", ""), []).append(value)
    info["lanes"] = {player: statistics.fmean(values) for player, values in by_player.items()}
    if budget:
        left = max(0, budget - steps) / sps if sps > 0 else math.inf
        info["eta"] = datetime.datetime.fromtimestamp(info["updated"] + left, KST)
    return info


def status(arguments) -> None:
    """The queue's jobs with their runs' progress, then the newest leaderboard."""
    store = queue_of(arguments)
    runs_directory = pathlib.Path(arguments.runs)
    print(f"{datetime.datetime.now(KST):%Y-%m-%d %H:%M KST}  queue {store.path}")
    print(
        f"{'job':>3} {'state':11} {'slot':>4} {'run':28} {'stage':5} {'steps':>12} {'SPS':>7} "
        f"{'ETA (KST)':12} {'KL':>7}  lanes vs scripted (mean over modes)"
    )
    for job in store.read()["jobs"]:
        name = job.get("run") or " ".join(job["arguments"])[:28]
        slot = "-" if job.get("slot") is None else str(job["slot"])
        info = run_progress(runs_directory, job["run"]) if job.get("run") else None
        if not info or "steps" not in info:
            print(
                f"{job['id']:>3} {job['state']:11} {slot:>4} {name[:28]:28} "
                + ("-" if not info else info["stage"])
            )
            continue
        steps = f"{info['steps'] / 1e6:.2f}/{(info['budget'] or 0) / 1e6:g}M"
        eta = info["eta"].strftime("%a %H:%M") if "eta" in info and info["eta"] else "-"
        if job["state"] != "running":
            eta = info["status"]
        stale = time.time() - info["updated"]
        lanes = " ".join(f"{k} {v:.2f}" for k, v in sorted(info["lanes"].items()))
        kl = "-" if info["kl"] is None else f"{info['kl']:.4f}"
        print(
            f"{job['id']:>3} {job['state']:11} {slot:>4} {name[:28]:28} {info['stage']:5} "
            f"{steps:>12} {info['sps']:7,.0f} {eta:12} {kl:>7}  {lanes}"
            + (f"  (last row {stale / 60:.0f} min ago)" if stale > 1800 else "")
        )
    board = sorted((runs_directory / "ratings").glob("leaderboard-*.txt"))
    if board:
        print(f"\nnewest leaderboard {board[-1]}:")
        print(board[-1].read_text())


# ---------------------------------------------------------------- tournament


def policies(arguments, config: dict, runs: list[dict]) -> list[tuple[str, str]]:
    """The field: every run's policy files at the matched chain steps it has reached, each
    variant, as (path, label). A target's file is the snapshot whose chain step is nearest
    the target within one snapshot interval (BC's or RL's, by which stage it fell in)."""
    runs_directory = pathlib.Path(arguments.runs)
    field = []
    for run in runs:
        snapshots = []  # (chain step, stage, path)
        stages = (
            [("bc", f"{run['run']}-bc"), ("rl", f"{run['run']}-rl")]
            if run["objective"] == "chain"
            else [("rl", run["run"])]
        )
        for stage, name in stages:
            record = runs_directory / name / "run.json"
            if not record.exists():
                continue
            start = (json.loads(record.read_text()).get("chain") or {}).get("start_step", 0)
            for path in (runs_directory / name / "policies").glob("step_*"):
                snapshots.append((start + int(path.name[5:]), stage, path))
        for target in config["tournament"]["steps"]:
            near = [s for s in snapshots if abs(s[0] - target) <= 5_300_000]
            if not near:
                continue
            step, stage, path = min(near, key=lambda s: (abs(s[0] - target), s[1] != "rl"))
            for variant in config["tournament"]["variants"]:
                policy = path / variant
                if (policy / "checkpoint.json").exists():
                    label = "averaged" if variant == "uniform_average" else variant
                    field.append(
                        (
                            str(policy),
                            f"{run['run']} {target / 1e6:.0f}M {label} "
                            f"({stage.upper()} {step / 1e6:.1f}M)",
                        )
                    )
    return field


def tournament(arguments) -> None:
    """Queue one round: N ratings jobs per preset over the field, reusing every record."""
    config, runs = load(arguments)
    lineup = check_lineup(config)
    field = policies(arguments, config, runs)
    if not field:
        raise SystemExit("no policies at the matched steps yet")
    ratings = pathlib.Path(arguments.runs) / "ratings"
    rounds = sorted(path for path in ratings.glob("round-*") if path.is_dir())
    number = len(rounds) + 1
    directory = ratings / f"round-{number:02d}"
    directory.mkdir(parents=True)
    (directory / "field.json").write_text(json.dumps(dict(field), indent=1))
    (ratings / "field.txt").write_text("".join(f"{path}\n" for path, _ in field))  # for pull.sh
    owner = next(path.split("/policies/")[0].rsplit("/", 1)[1] for path, _ in field)
    store = queue_of(arguments)
    shards = arguments.shards
    target = arguments.target * math.sqrt(shards)
    tc = config["tournament"]
    for preset in tc["presets"]:
        for shard in range(shards):
            out = directory / f"{preset}-{shard}"
            line = [
                owner,
                "--no-league",
                "--no-ladder",
                "--presets",
                preset,
                "--target-stderr",
                f"{target:.1f}",
                "--budget",
                str(arguments.budget),
                "--blocks",
                str(tc["blocks"]),
                "--block-environments",
                str(tc["block_environments"]),
                "--seed",
                str(1000 * number + shard),
                "--out",
                str(out),
                "--reuse",
                *[str(r) for r in rounds],
                "--lineup",
                *lineup,
                "--runs-directory",
                arguments.runs,
                "--with",
                *[path for path, _ in field],
            ]
            if not rounds:
                line.remove("--reuse")
            job = store.add(
                "ratings", line, program=arguments.bnb, cwd=REPOSITORY, priority=TOURNAMENT_PRIORITY
            )
            print(
                f"job {job['id']}: round {number} {preset} shard {shard} "
                f"(target {target:.0f} per shard, {len(field)} policies + scripted)"
            )


def leaderboard(arguments) -> None:
    """Fit every round's records of the newest field together and print it labelled."""
    config, runs = load(arguments)
    lineup = check_lineup(config)
    ratings = pathlib.Path(arguments.runs) / "ratings"
    rounds = sorted(path for path in ratings.glob("round-*") if path.is_dir())
    if not rounds:
        raise SystemExit("no tournament round yet")
    labels = json.loads((rounds[-1] / "field.json").read_text())
    owner = next(path.split("/policies/")[0].rsplit("/", 1)[1] for path in labels)
    out = ratings / "fits" / datetime.datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    line = [
        *arguments.bnb,
        "ratings",
        owner,
        "--no-league",
        "--no-ladder",
        "--presets",
        *config["tournament"]["presets"],
        "--budget",
        "0",
        "--out",
        str(out),
        "--reuse",
        *[str(r) for r in rounds],
        "--lineup",
        *lineup,
        "--runs-directory",
        arguments.runs,
        "--with",
        *labels,
    ]
    done = subprocess.run(
        line,
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
    if done.returncode != 0:
        raise SystemExit(f"the fit failed:\n{done.stdout[-3000:]}{done.stderr[-3000:]}")
    table = []
    for line in done.stdout.splitlines():
        words = line.split()
        if len(words) >= 6 and words[3] == "+-":
            preset, player, elo, error = words[0], words[1], float(words[2]), float(words[4])
            table.append((preset, -elo, labels.get(player, player), elo, error, words[5]))
    lines = [
        f"leaderboard {datetime.datetime.now(KST):%Y-%m-%d %H:%M KST}, rounds "
        f"{rounds[0].name}..{rounds[-1].name} (scripted = 1000)"
    ]
    for preset, _, label, elo, error, games in sorted(table):
        lines.append(f"{preset:24s} {label:56s} {elo:7.0f} +- {error:4.0f}  {games} games")
    text = "\n".join(lines)
    (ratings / f"leaderboard-{out.name}.txt").write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------- continue


def resume_to(arguments) -> None:
    """Raise each chain's combined budget and queue its resume (150M -> 500M)."""
    store = queue_of(arguments)
    for name in arguments.names:
        line = [
            name,
            "--objective",
            "chain",
            "--chain-budget-steps",
            str(arguments.steps),
            "--runs-directory",
            arguments.runs,
        ]
        job = store.add(
            "resume",
            line,
            program=arguments.bnb,
            cwd=REPOSITORY,
            runs_directory=pathlib.Path(arguments.runs),
        )
        print(f"job {job['id']}: {name} resumed to {arguments.steps:,} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command",
        choices=["check", "probe", "enqueue", "status", "tournament", "leaderboard", "resume-to"],
    )
    parser.add_argument("names", nargs="*", help="resume-to: STEPS RUN ...")
    parser.add_argument("--config", default=str(HERE / "launch.toml"))
    parser.add_argument("--arms", default=str(HERE / "arms.toml"))
    parser.add_argument("--runs", default=None)
    parser.add_argument("--queue", default=None)
    parser.add_argument("--bnb", default=str(REPOSITORY / ".venv" / "bin" / "bnb"))
    parser.add_argument("--group", default="A")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--default-micro-batch", action="store_true")
    parser.add_argument("--target", type=float, default=50.0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--budget", type=int, default=200_000)
    arguments = parser.parse_args()
    box = tomllib.loads(pathlib.Path(arguments.config).read_text())["box"]
    arguments.runs = arguments.runs or box["runs_directory"]
    arguments.queue = arguments.queue or box["queue_file"]
    arguments.bnb = shlex.split(arguments.bnb)
    if arguments.command == "resume-to":
        if not arguments.names:
            raise SystemExit("resume-to STEPS RUN ...")
        arguments.steps, arguments.names = int(arguments.names[0]), arguments.names[1:]
    commands = dict(
        check=check,
        probe=probe,
        enqueue=enqueue,
        status=status,
        tournament=tournament,
        leaderboard=leaderboard,
        **{"resume-to": resume_to},
    )
    commands[arguments.command](arguments)


if __name__ == "__main__":
    main()
