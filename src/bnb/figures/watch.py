"""The watcher (`bnb figures watch RUN`): media of a training run while it trains, made by a
second process that only reads the run.

It polls the run's `checkpoints/` and, for every complete checkpoint without a media
record, writes into the run's `artifacts/figures/step_<steps>/`:

- per mode (`WatchConfig.modes`): a capture on the CPU (`bnb.figures.capture`, a few
  environments, the same seed at every checkpoint, so the games start alike) and the quick
  figure sets drawn from it into `<mode>/`; the capture itself is deleted unless kept;
- per mode: a video of one game, the checkpoint against the scripted agent
  (`bnb.figures.video`), into `videos/`;
- the media record `media.json`, last: what was made, the sets' numbers, the videos'
  tallies, the seconds each part took, any error, and whether it reached W&B.

A part that fails (a figure set, a video) is recorded with its error and the rest goes on;
a checkpoint that the retention deletes before it is read is skipped. With W&B
(`MediaLog`) each checkpoint's media are logged once, at its environment steps, to a
companion run of the training run: never to the training run itself, which its trainer
writes (W&B does not support two writers to one run). The watcher stops once the run is no
longer live (`bnb.runs.records.Run.live`) and every checkpoint has its media.
"""

import dataclasses
import hashlib
import json
import pathlib
import time
import traceback

import torch

from bnb.runs.records import Run, step_of, write_json
from bnb.runs.sinks import STEP, shutdown_telemetry
from bnb.runs.upload import PROJECT

from . import capture as capturing
from . import video
from .command import CAPTURES
from .sets import FIGURE_SETS

QUICK_SETS = ("shots", "belief", "next_state", "density", "enemy_action")
MODES = ("frontline", "deathmatch")  # the training mix's modes (`bnb.runtime.train.TRAINING_MIX`)
RECORD = "media.json"
VIDEOS = "videos"


@dataclasses.dataclass(frozen=True)
class WatchConfig:
    """sets: the figure sets drawn at each checkpoint; modes: the presets captured and
    filmed, one each; environments, ticks and seed: each capture's
    (`bnb.figures.capture.CaptureConfig`); video: the videos'; interval: seconds between
    polls; keep_captures: keep each capture artifact (about 60 MB a mode) beside its
    figures."""

    sets: tuple[str, ...] = QUICK_SETS
    modes: tuple[str, ...] = MODES
    environments: int = 4
    ticks: int = 256
    seed: int = 0
    video: video.VideoConfig = video.VideoConfig()
    interval: float = 60.0
    keep_captures: bool = False


def figures_root(run: Run) -> pathlib.Path:
    """Where the run's figures go: its `artifacts/figures/`."""
    return run.directory / "artifacts" / "figures"


def media_directory(run: Run, checkpoint: pathlib.Path) -> pathlib.Path:
    """The media of one checkpoint: `artifacts/figures/<checkpoint directory>/`."""
    return figures_root(run) / checkpoint.name


def pending(run: Run) -> list[pathlib.Path]:
    """The run's complete checkpoints without a media record, oldest first."""
    found = [run.checkpoint_path(step) for step in run.checkpoint_steps()]
    return [path for path in found if not (media_directory(run, path) / RECORD).is_file()]


def unlogged(run: Run) -> list[pathlib.Path]:
    """The media directories whose record has not reached W&B, oldest first."""
    root = figures_root(run)
    if not root.is_dir():
        return []
    directories = sorted(
        (path for path in root.iterdir() if step_of(path) is not None), key=step_of
    )
    return [
        path
        for path in directories
        if (path / RECORD).is_file() and not read_record(path)["logged"]
    ]


def read_record(directory: pathlib.Path) -> dict:
    """A checkpoint's media record."""
    return json.loads((directory / RECORD).read_text())


def failure(error: BaseException) -> str:
    """An error as a record keeps it: its type, its message and where it was raised."""
    where = traceback.extract_tb(error.__traceback__)[-1]
    return f"{type(error).__name__}: {error} ({pathlib.Path(where.filename).name}:{where.lineno})"


def draw_figures(
    checkpoint: pathlib.Path, mode: str, directory: pathlib.Path, config, reseed
) -> dict:
    """Capture `mode` with the checkpoint, after reseed() (the process generators the
    rollout and the spread draw from, so every checkpoint's capture starts alike), and draw
    the sets into `directory`; returns each set's PNGs (relative to the media directory)
    and numbers, or its error, and the seconds the capture and the drawing took."""
    started = time.perf_counter()
    reseed()
    capture_config = capturing.CaptureConfig(
        preset=mode, environments=config.environments, ticks=config.ticks, seed=config.seed
    )
    artifact = capturing.capture(checkpoint, capture_config, torch.device("cpu"))
    captured = time.perf_counter()
    if config.keep_captures:
        capturing.save(artifact, directory / CAPTURES / f"{mode}_seed{config.seed}.pt")
    sets = {}
    for name in config.sets:
        try:
            written, numbers = FIGURE_SETS[name](artifact, directory)
            sets[name] = dict(
                pngs=[str(path.relative_to(directory.parent)) for path in written],
                numbers=numbers,
            )
        except Exception as error:  # one set's failure leaves the others
            sets[name] = dict(error=failure(error))
    return dict(
        sets=sets,
        capture_seconds=round(captured - started, 1),
        draw_seconds=round(time.perf_counter() - captured, 1),
    )


def attempt(part, checkpoint: pathlib.Path) -> dict:
    """part(), or its error as a record keeps it; FileNotFoundError when it failed because
    the checkpoint is gone (the retention deleted it)."""
    try:
        return part()
    except Exception as error:  # one part's failure leaves the others
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"{checkpoint} is gone") from error
        return dict(error=failure(error))


def make_media(
    run: Run, checkpoint: pathlib.Path, config: WatchConfig, reseed=lambda: None
) -> dict:
    """Every medium of one checkpoint (see the module's docstring); writes and returns its
    record. reseed: seeds the process generators before each capture (the entry point's
    to set). Raises FileNotFoundError when the checkpoint is gone before it is read."""
    directory = media_directory(run, checkpoint)
    started = time.perf_counter()
    figures = {}
    for mode in config.modes:

        def draw() -> dict:
            return draw_figures(checkpoint, mode, directory / mode, config, reseed)

        figures[mode] = attempt(draw, checkpoint)
    videos = {}
    for mode in config.modes:
        path = directory / VIDEOS / video.video_name(mode, config.video)

        def film() -> dict:
            numbers = video.render(checkpoint, mode, config.video, path)
            return dict(path=str(path.relative_to(directory)), **numbers)

        videos[mode] = attempt(film, checkpoint)
    record = dict(
        checkpoint=str(checkpoint),
        environment_steps=step_of(checkpoint),
        figures=figures,
        videos=videos,
        config=dataclasses.asdict(config),
        seconds=round(time.perf_counter() - started, 1),
        logged=False,
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / RECORD, record)
    return record


def scalars(numbers, prefix: str) -> dict[str, float]:
    """The numeric leaves of a set's nested numbers, keyed by their path under prefix
    (lists, such as an error by age, are left to the figures)."""
    if isinstance(numbers, bool | int | float):
        return {prefix: numbers}
    if isinstance(numbers, dict):
        found = {}
        for name, value in numbers.items():
            found.update(scalars(value, f"{prefix}/{name}"))
        return found
    return {}


def companion_id(run: Run) -> str:
    """The companion W&B run's id: the training run's W&B id (or a hash of its name, when
    it has none) followed by `media`, so a restarted watcher resumes the same run."""
    trained = run.record.get("wandb_id")
    if trained is None:
        trained = hashlib.sha1(run.record["name"].encode()).hexdigest()[:8]
    return f"{trained}media"


class MediaLog:
    """The companion W&B run of a training run: `<run>-media` in project PROJECT, grouped
    under the run's name, its notes naming the run it watches. Every series' x-axis is the
    training run's own, `counters/environment_steps` (`bnb.runs.sinks.STEP`). W&B's own
    settings (the entity, `WANDB_MODE=offline`) come from its environment variables; its
    local files go under the run's `artifacts/figures/`."""

    def __init__(self, run: Run, config: WatchConfig) -> None:
        import wandb

        name = run.record["name"]
        trained = run.record.get("wandb_id")
        figures_root(run).mkdir(parents=True, exist_ok=True)
        self.session = wandb.init(
            project=PROJECT,
            name=f"{name}-media",
            id=companion_id(run),
            resume="allow",
            group=name,
            job_type="media",
            notes=(
                f"Videos and head figures of the training run {name} (W&B id {trained}), "
                f"one set per checkpoint, made by `bnb figures watch {name}`."
            ),
            dir=str(figures_root(run)),
            config=dict(run=name, run_wandb_id=trained, watch=dataclasses.asdict(config)),
        )
        self.session.define_metric(STEP)
        self.session.define_metric("*", step_metric=STEP)
        self.wandb = wandb

    def entries(self, record: dict, directory: pathlib.Path) -> dict:
        """What one checkpoint's record logs: each video (`video/<mode>`) and its tally
        (`game/<mode>/...`), each figure (`figures/<mode>/<name>`) and its set's numbers
        (`numbers/<mode>/<set>/...`), and the seconds each part took (`watcher/...`)."""
        wandb = self.wandb
        steps = record["environment_steps"]
        logged = {STEP: steps, "watcher/seconds": record["seconds"]}
        for mode, made in record["videos"].items():
            if "error" in made:
                continue
            caption = (
                f"step {steps:,} against the scripted agent, {mode}: "
                f"{made['game_seconds']:.0f} s of play"
            )
            logged[f"video/{mode}"] = wandb.Video(str(directory / made["path"]), caption=caption)
            numbers = {name: value for name, value in made.items() if name != "path"}
            logged.update(scalars(numbers, f"game/{mode}"))
        for mode, made in record["figures"].items():
            if "error" in made:
                continue
            logged[f"watcher/{mode}/capture_seconds"] = made["capture_seconds"]
            logged[f"watcher/{mode}/draw_seconds"] = made["draw_seconds"]
            for name, drawn in made["sets"].items():
                if "error" in drawn:
                    continue
                for png in drawn["pngs"]:
                    stem = pathlib.Path(png).stem
                    logged[f"figures/{mode}/{stem}"] = wandb.Image(
                        str(directory / png), caption=f"step {steps:,}, {mode}"
                    )
                logged.update(scalars(drawn["numbers"], f"numbers/{mode}/{name}"))
        return logged

    def log(self, directory: pathlib.Path) -> None:
        """Log a media directory's record and mark the record logged."""
        record = read_record(directory)
        self.session.log(self.entries(record, directory))
        write_json(directory / RECORD, {**record, "logged": True})

    def close(self) -> None:
        """Finish the companion run, then W&B's telemetry."""
        self.session.finish()
        shutdown_telemetry()


def watch(
    runs_directory: pathlib.Path,
    name: str,
    config: WatchConfig,
    log: MediaLog | None = None,
    once: bool = False,
    make=make_media,
    sleep=time.sleep,
    say=lambda line: print(line, flush=True),  # a watcher's log is usually a file
) -> None:
    """Make every pending checkpoint's media (`make`), log them with `log`, and poll every
    `config.interval` seconds, until the run is no longer live and nothing is pending (or
    after one pass, with `once`)."""
    while True:
        run = Run.open(runs_directory, name)  # the record again: its status, its checkpoints
        live = run.live()
        made = 0
        for checkpoint in pending(run):
            say(f"media of {checkpoint.name} ...")
            try:
                record = make(run, checkpoint, config)
            except FileNotFoundError as error:
                say(f"skipped {checkpoint.name}: deleted before it was read ({error})")
                continue
            made += 1
            say(f"media of {checkpoint.name}: {summary(record)}")
        if log is not None:
            for directory in unlogged(run):
                log.log(directory)
                say(f"logged {directory.name} to W&B")
        if once or (not live and made == 0):
            return
        sleep(config.interval)


def summary(record: dict) -> str:
    """One line about a media record: the seconds taken and any errors."""
    errors = [f"{mode} figures" for mode, made in record["figures"].items() if "error" in made] + [
        f"{mode} video" for mode, made in record["videos"].items() if "error" in made
    ]
    errors += [
        f"{mode}/{name}"
        for mode, made in record["figures"].items()
        for name, drawn in made.get("sets", {}).items()
        if "error" in drawn
    ]
    tail = f"; failed: {', '.join(errors)}" if errors else ""
    return f"{record['seconds']:.0f} s{tail}"
