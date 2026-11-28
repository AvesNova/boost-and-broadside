"""W&B logging for lean.train, and for its backfill from a train.log
(docs/rewrite/sections/ops-19/backfill.py).

One W&B run per stage of a run directory, in v2's project (PROJECT), named
`<directory>-<stage>` (`<parent>-<stage>` when the directory is named for the stage, as in
runs/dry-2026-11-28/bc) and grouped by the same base name, so a run's BC and RL sit together.
The W&B run ids are kept in DIR/wandb.json (one per stage), so `--resume DIR` continues the
same W&B run (resume="allow"). Every series' x-axis is STEP, the stage's environment steps.

Modes: "online" (lean.train's default), "offline" (written under DIR/wandb, `wandb sync` later)
and "off". Online without credentials, or when the online start fails (no network), falls back
to offline. W&B never stops training: every failure is a warning and the run goes on without
it. Nothing personal is sent: no git remote, code, host, user name or command line (W&B's
metadata is off); the config carries the run directory's name, not its path.
"""

import dataclasses
import json
import secrets
import subprocess
from pathlib import Path

PROJECT = "boost-and-broadside-v2"  # v2's (src/bnb/runs/upload.py)
STEP = "environment_steps"
IDS = "wandb.json"
MODES = ("online", "offline", "off")
INIT_TIMEOUT = 60  # seconds an online start may take before the offline fallback


def warn(message: str) -> None:
    """A W&B failure, on the run's log; the run goes on."""
    print(f"W&B: {message}", flush=True)


def base_name(directory: Path, stage: str) -> str:
    """The run's name without its stage: the directory's, or its parent's when the directory
    is named for the stage."""
    directory = Path(directory).resolve()
    return directory.parent.name if directory.name == stage else directory.name


def run_name(directory: Path, stage: str) -> str:
    """`<base>-<stage>` (base_name)."""
    return f"{base_name(directory, stage)}-{stage}"


def commit() -> str | None:
    """The source tree's commit hash (with "-dirty" for uncommitted changes), or None."""
    root = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return None
    return head + ("-dirty" if dirty else "")


def as_config(value):
    """A tree of dicts, lists and dataclasses as plain JSON-able values (anything else as
    text)."""

    def plain(item):
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return dataclasses.asdict(item)
        return str(item)

    return json.loads(json.dumps(value, default=plain))


def numbers(record: dict) -> dict:
    """The record's numeric values (W&B series); text and None are left out."""
    return {
        name: float(value) if isinstance(value, bool) else value
        for name, value in record.items()
        if isinstance(value, bool | int | float)
    }


def stored_id(directory: Path | None, stage: str) -> str:
    """The stage's W&B run id from DIR/wandb.json, or a new one written there."""
    if directory is None:
        return secrets.token_hex(4)
    path = Path(directory) / IDS
    try:
        ids = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError) as error:
        warn(f"{path} unreadable ({error}): a new W&B run id")
        ids = {}
    if stage not in ids:
        ids[stage] = secrets.token_hex(4)  # W&B run ids: [a-z0-9]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(ids, indent=1) + "\n")
        except OSError as error:
            warn(f"the W&B run id was not kept in {path} ({error}): a resume starts a new run")
    return ids[stage]


class WandbLog:
    """One stage's W&B run (the module's docstring); `log` and `finish` never raise."""

    def __init__(
        self,
        mode: str,
        directory: Path | None,
        stage: str,
        config: dict,
        wandb_directory: Path | None = None,
        keep_id: bool = True,
    ) -> None:
        self.session, self.imported = None, False
        if mode not in MODES:
            raise ValueError(f"W&B mode {mode!r}: one of {MODES}")
        if mode == "off":
            return
        if directory is None:
            print(
                "W&B: off (no --out: no run directory to name the run and keep its id)",
                flush=True,
            )
            return
        self.name = run_name(directory, stage)
        identifier = stored_id(directory if keep_id else None, stage)
        root = Path(wandb_directory or directory)
        try:
            import wandb

            self.imported = True
            if mode == "online" and not wandb.login(prompt=False, verify=False):
                warn("not logged in: offline (wandb sync later)")
                mode = "offline"
            root.mkdir(parents=True, exist_ok=True)
            arguments = dict(
                project=PROJECT,
                name=self.name,
                group=base_name(directory, stage),
                job_type=stage,
                id=identifier,
                resume="allow",
                dir=str(root),
                config=config,
            )
            try:
                self.session = wandb.init(mode=mode, settings=self.settings(wandb), **arguments)
            except Exception as error:
                if mode != "online":
                    raise
                warn(f"online start failed ({type(error).__name__}: {error}): offline")
                mode = "offline"
                wandb.teardown()  # else the failed start still holds the run id
                self.session = wandb.init(mode=mode, settings=self.settings(wandb), **arguments)
            self.session.define_metric(STEP)
            self.session.define_metric("*", step_metric=STEP)
            print(f"W&B: {mode} run {self.name} (id {identifier}, project {PROJECT})", flush=True)
        except Exception as error:  # training goes on without W&B
            warn(f"off for this stage ({type(error).__name__}: {error})")
            self.session = None

    @staticmethod
    def settings(wandb):
        return wandb.Settings(
            init_timeout=INIT_TIMEOUT,
            disable_git=True,  # the commit is in the config; no remote URL
            save_code=False,
            x_disable_meta=True,  # no host, user name, e-mail, command line or paths
            console="off",  # train.log keeps the console
            silent=True,
        )

    def log(self, record: dict) -> None:
        """Log one update's numbers (STEP among them)."""
        if self.session is None:
            return
        try:
            self.session.log(numbers(record))
        except Exception as error:
            warn(f"a record was not logged ({type(error).__name__}: {error})")

    def finish(self) -> None:
        if self.session is not None:
            try:
                self.session.finish()
            except Exception as error:
                warn(f"finish failed ({type(error).__name__}: {error})")
            self.session = None
        if self.imported:
            shutdown_telemetry()


def shutdown_telemetry() -> None:
    """W&B 0.31's telemetry shut down before exit (src/bnb/runs/sinks.py: else its exit
    handler shuts the meter provider down twice, "shutdown can only be called once")."""
    try:
        from wandb.analytics.opentelemetry import opentelemetry_proxy

        proxy = opentelemetry_proxy.get_open_telemetry_proxy()
        if proxy is not None:
            proxy.shutdown()
    except Exception:
        pass


def update_record(
    stage: str,
    index: int,
    steps: int,
    win_rate: float,
    league,
    rate: float,
    metrics: dict,
    peak_allocated_gib: float | None = None,
) -> dict:
    """One update's W&B record: every number the update's line shows (the league's under the
    same names, lists by position), and every metric (the gradient decomposition's too)."""
    summary = league.summary()
    record = {
        STEP: steps,
        "update": index,
        "scripted_win_rate": win_rate,
        "league_members": league.present_count(),
        "snapshots": league.snapshots,
        "draining": summary["draining"],
        "scripted_slots": summary["scripted_slots"],
        "scripted_dealt": summary["scripted_dealt"],
        "overflow_games": summary["overflow_games"],
        "league_loads": summary["loads"],
        "steps_per_second": rate,
    }
    for kind, count in summary["loaded"].items():
        record[f"league_slots/{kind}"] = count
    rates = league.win_rates()[league.present].tolist()
    for position, value in enumerate(rates):
        record[f"league_win_rates/{position}"] = value
    if peak_allocated_gib is not None:
        record["peak_allocated_gib"] = peak_allocated_gib
    record.update(numbers(metrics))
    return record
