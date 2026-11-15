"""Averages of a run's weights built after training from its per-update live policies
(`policies/step_<steps>/live/`, written after every update by a run with
`update_policies`, `bnb.runs.session`), each written as a policy file a tournament rates
like any other (`bnb average RUN --spec SPEC`).

A spec (TOML) names the averages and the evaluation points each is built at:

    [evaluation]
    fractions = [0.5, 1.0]        # of the run's span, its start to its newest update
    # steps = [50_000_000]        # or environment steps

    [[average]]
    family = "exponential"
    halflife_updates = 8          # or halflife_steps; optionally from_fraction / from_step

The families (`AverageSpec`):
- `uniform`: the mean of the updates from `from_fraction` or `from_step` (inclusive) to
  the evaluation point;
- `exponential`: `average = decay * average + (1 - decay) * live` per update, the decay
  from a half-life in updates or in environment steps (`0.5 ** (1 / halflife_updates)`),
  starting from the weights at `from_fraction` or `from_step`, by default the run's
  starting weights;
- `window`: the mean of the last `last_updates` updates, or of the updates in the last
  `last_steps` environment steps or `last_fraction` of the run's span, before the
  evaluation point;
- `training_exponential` and `training_uniform`: the run's own averages
  (`bnb.league.pool.Averages`) rebuilt with its decay and uniform start (`config.json`):
  the exponential one from the starting weights (`policies/step_<start>/live`, written
  before the first update, `updates` 0), the uniform one over the updates at or after
  `TrainConfig.uniform_start`. Where the run saved its own average at the same step, the
  two are compared (`Check`), which is the tool's correctness check;
- `live`: the live weights at the evaluation point (the saved file itself, nothing written).

An evaluation point is the newest saved update at or before the step asked for. The
averages are folded in float64 and written in each tensor's own dtype; integer buffers
take the newest value, as the run's own do. The files are read once each, oldest first,
one at a time: every average of the spec is folded from the same pass, and an average is
held only from its first update to its last evaluation point (the memory is one float64
copy of the weights per average in progress).

Each is written at `<run>/average/<name>/step_<steps>/`, the name encoding the family and
its parameters (`AverageSpec.name`), with the end update's return scale and a record whose
`steps.average` says what it averages.
"""

import dataclasses
import json
import pathlib
import tomllib
from collections.abc import Callable

import safetensors.torch
import torch

from . import provenance
from .checkpoint import RETURN_SCALE, WEIGHTS, read_record, write_checkpoint
from .records import POLICIES, Run, step_of

LIVE = "live"
EXPONENTIAL_AVERAGE = "exponential_average"
UNIFORM_AVERAGE = "uniform_average"
AVERAGES = "average"  # the run's subdirectory the averages are written to
FAMILIES = ("uniform", "exponential", "window", "training_exponential", "training_uniform", "live")
# The check of a rebuilt training average against the run's own: float32 folds on the
# device against float64 ones on the host, so within rounding, not bit for bit.
RELATIVE_TOLERANCE = 1e-4
ABSOLUTE_TOLERANCE = 1e-6


def saved_steps(run: Run) -> list[int]:
    """The environment steps of the run's saved live policies, oldest first."""
    directory = run.directory / POLICIES
    steps = []
    if directory.is_dir():
        for path in directory.iterdir():
            step = step_of(path)
            if step is not None and (path / LIVE).is_dir():
                steps.append(step)
    return sorted(steps)


def load_weights(path: pathlib.Path) -> dict[str, torch.Tensor]:
    """One policy file's weights (a function of its own, so the tests can watch the reads)."""
    return safetensors.torch.load_file(path / WEIGHTS)


@dataclasses.dataclass(frozen=True)
class RunFacts:
    """What a spec's averages are placed by: the saved steps (oldest first), the step the
    run's starting weights are at (None when they were not saved), the environment steps
    per update, and the run's own averages' decay and uniform start."""

    steps: tuple[int, ...]
    start: int | None
    steps_per_update: int
    training_decay: float
    training_uniform_from: int

    @property
    def updates(self) -> tuple[int, ...]:
        """The steps of the updates' files (the starting weights' left out)."""
        return tuple(step for step in self.steps if step != self.start)

    @property
    def span(self) -> tuple[int, int]:
        """The run's first step (its starting weights', or its first saved update's) and its
        newest saved update's."""
        first = self.steps[0] if self.start is None else self.start
        return first, self.steps[-1]

    def at_fraction(self, fraction: float) -> int:
        """The environment step a fraction of the run's span is at."""
        first, last = self.span
        return first + round(fraction * (last - first))


def run_facts(run: Run) -> RunFacts:
    """The run's saved steps, starting step, steps per update and own averages' settings."""
    from bnb.runtime.train import TrainConfig

    from .serialise import from_json

    steps = saved_steps(run)
    if not steps:
        raise FileNotFoundError(
            f"run {run.record['name']} has no saved live policies (run with --update-policies)"
        )
    starts = [
        step
        for step in steps
        if read_record(run.update_policies_path(step) / LIVE)["steps"].get("updates") == 0
    ][:1]
    config, _ = from_json(TrainConfig, json.loads((run.directory / "config.json").read_text()))
    return RunFacts(
        steps=tuple(steps),
        start=starts[0] if starts else None,
        steps_per_update=config.environments * config.length * config.shards,
        training_decay=config.league.exponential_decay,
        training_uniform_from=config.uniform_start(),
    )


def _number(value: float) -> str:
    return f"{value:g}"


def _percent(fraction: float) -> str:
    return f"{_number(100 * fraction)}percent"


@dataclasses.dataclass(frozen=True)
class AverageSpec:
    """One average of a spec: its family (FAMILIES) and parameters, at most one of each
    group: the half-life (`halflife_updates`, `halflife_steps`; exponential), the start
    (`from_fraction`, `from_step`; uniform needs one, exponential may take one), the window
    (`last_updates`, `last_steps`, `last_fraction`; window), and its own evaluation points
    in place of the spec's (`fractions`, `steps`)."""

    family: str
    halflife_updates: float | None = None
    halflife_steps: int | None = None
    from_fraction: float | None = None
    from_step: int | None = None
    last_updates: int | None = None
    last_steps: int | None = None
    last_fraction: float | None = None
    fractions: tuple[float, ...] | None = None
    steps: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"average family {self.family!r}: one of {FAMILIES}")
        groups = dict(
            halflife=("halflife_updates", "halflife_steps"),
            start=("from_fraction", "from_step"),
            window=("last_updates", "last_steps", "last_fraction"),
            evaluation=("fractions", "steps"),
        )
        given = {
            group: [name for name in names if getattr(self, name) is not None]
            for group, names in groups.items()
        }
        for group, names in given.items():
            if len(names) > 1:
                raise ValueError(f"{self.family} average: {' and '.join(names)}; give one")
        allowed = dict(
            uniform={"start"},
            exponential={"halflife", "start"},
            window={"window"},
        ).get(self.family, set())
        required = dict(uniform={"start"}, exponential={"halflife"}, window={"window"})
        for group, names in given.items():
            if names and group != "evaluation" and group not in allowed:
                raise ValueError(f"a {self.family} average takes no {names[0]}")
        for group in required.get(self.family, ()):
            if not given[group]:
                raise ValueError(f"a {self.family} average needs one of {groups[group]}")
        for name in ("from_fraction", "last_fraction"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} {value}: need 0 <= {name} <= 1")
        for name in ("halflife_updates", "halflife_steps", "last_updates", "last_steps"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"{name} {value}: need a positive value")

    def name(self, facts: RunFacts) -> str:
        """The average's directory name, its family and parameters as given."""
        if self.family == "live":
            return LIVE
        if self.family == "training_exponential":
            return f"training_exponential_decay_{_number(facts.training_decay)}"
        if self.family == "training_uniform":
            return f"training_uniform_from_step_{facts.training_uniform_from}"
        start = ""
        if self.from_fraction is not None:
            start = f"from_{_percent(self.from_fraction)}"
        elif self.from_step is not None:
            start = f"from_step_{self.from_step}"
        if self.family == "uniform":
            return f"uniform_{start}"
        if self.family == "exponential":
            if self.halflife_updates is not None:
                halflife = f"{_number(self.halflife_updates)}updates"
            else:
                halflife = f"{self.halflife_steps}steps"
            return f"exponential_halflife_{halflife}" + (f"_{start}" if start else "")
        if self.last_updates is not None:
            return f"window_last_{self.last_updates}updates"
        if self.last_steps is not None:
            return f"window_last_{self.last_steps}steps"
        return f"window_last_{_percent(self.last_fraction)}"

    def decay(self, facts: RunFacts) -> float:
        """The exponential average's decay per update."""
        if self.family == "training_exponential":
            return facts.training_decay
        updates = self.halflife_updates
        if updates is None:
            updates = self.halflife_steps / facts.steps_per_update
        return 0.5 ** (1.0 / updates)

    def start_step(self, facts: RunFacts) -> int | None:
        """The first step the average takes in (None: the run's first file)."""
        if self.family == "training_uniform":
            return facts.training_uniform_from
        if self.from_fraction is not None:
            return facts.at_fraction(self.from_fraction)
        return self.from_step


@dataclasses.dataclass(frozen=True)
class Spec:
    """A spec file's averages and its evaluation points (`fractions` of the run's span, or
    environment `steps`; the newest update by default)."""

    averages: tuple[AverageSpec, ...]
    fractions: tuple[float, ...] | None = None
    steps: tuple[int, ...] | None = None


def read_spec(path: str | pathlib.Path) -> Spec:
    """The spec in the TOML file at `path` (the module's docstring has its form)."""
    data = tomllib.loads(pathlib.Path(path).read_text())
    unknown = set(data) - {"evaluation", "average"}
    if unknown:
        raise ValueError(f"{path}: unknown tables {sorted(unknown)} (evaluation, average)")
    evaluation = data.get("evaluation", {})
    unknown = set(evaluation) - {"fractions", "steps"}
    if unknown or ("fractions" in evaluation and "steps" in evaluation):
        raise ValueError(f"{path}: [evaluation] takes fractions or steps")
    fields = {field.name for field in dataclasses.fields(AverageSpec)}
    averages = []
    for entry in data.get("average", []):
        entry = {key: value for key, value in entry.items() if key != "why"}  # the comment
        unknown = set(entry) - fields
        if unknown:
            raise ValueError(f"{path}: an average has unknown keys {sorted(unknown)}")
        for key in ("fractions", "steps"):
            if key in entry:
                entry[key] = tuple(entry[key])
        averages.append(AverageSpec(**entry))
    if not averages:
        raise ValueError(f"{path}: no [[average]] entries")

    def points(key: str):
        return tuple(evaluation[key]) if key in evaluation else None

    return Spec(tuple(averages), points("fractions"), points("steps"))


@dataclasses.dataclass(frozen=True)
class Output:
    """One average at one evaluation point: its name, the step it ends at, and the steps of
    the files it folds, oldest first (the first is the exponential average's seed)."""

    name: str
    end: int
    folded: tuple[int, ...]


@dataclasses.dataclass
class Accumulator:
    """An average being folded: its kind ("uniform" or "exponential"), decay, the steps it
    folds and the outputs it is written for, by end step."""

    kind: str
    decay: float
    folded: tuple[int, ...]
    outputs: dict[int, list[Output]]
    family: AverageSpec
    weights: dict[str, torch.Tensor] | None = None
    count: int = 0

    def fold(self, live: dict[str, torch.Tensor]) -> None:
        if self.weights is None:
            self.weights = {
                name: value.to(torch.float64) if value.is_floating_point() else value.clone()
                for name, value in live.items()
            }
            self.count = 1
            return
        self.count += 1
        for name, value in live.items():
            if not value.is_floating_point():
                self.weights[name] = value.clone()
            elif self.kind == "uniform":
                self.weights[name].add_(
                    value.to(torch.float64) - self.weights[name], alpha=1.0 / self.count
                )
            else:
                self.weights[name].mul_(self.decay).add_(
                    value.to(torch.float64), alpha=1.0 - self.decay
                )

    def snapshot(self, like: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The average now, in each tensor's dtype in `like` (a saved live file's)."""
        return {
            name: value.to(like[name].dtype).contiguous() for name, value in self.weights.items()
        }


def _evaluation_steps(average: AverageSpec, spec: Spec, facts: RunFacts) -> list[int]:
    """The saved updates an average is built at: the newest at or before each point."""
    fractions = average.fractions if average.steps is None else None
    steps = average.steps
    if fractions is None and steps is None:
        fractions, steps = spec.fractions, spec.steps
    if fractions is None and steps is None:
        fractions = (1.0,)
    asked = [facts.at_fraction(fraction) for fraction in fractions] if fractions else list(steps)
    updates = facts.updates
    found = []
    for step in asked:
        earlier = [update for update in updates if update <= step]
        if not earlier:
            raise ValueError(f"no saved update at or before step {step}")
        found.append(earlier[-1])
    return sorted(set(found))


def plan(spec: Spec, facts: RunFacts) -> tuple[list[Accumulator], list[Output]]:
    """The accumulators that fold the spec's averages (one per average and start, shared by
    its evaluation points where the folded updates are a prefix of each other's) and the
    live outputs, which need no folding. An evaluation point before an average's start has
    no output."""
    accumulators: list[Accumulator] = []
    lives: list[Output] = []
    updates = facts.updates
    for average in spec.averages:
        name = average.name(facts)
        ends = _evaluation_steps(average, spec, facts)
        if average.family == "live":
            lives.extend(Output(name, end, (end,)) for end in ends)
            continue
        if average.family in ("exponential", "training_exponential"):
            start = average.start_step(facts)
            if start is None:
                start = facts.span[0]
            folded = tuple(step for step in facts.steps if step >= start)
            if not folded:
                raise ValueError(f"{name}: no saved file at or after step {start}")
            if average.family == "training_exponential" and facts.start is None:
                raise ValueError(
                    f"{name}: the run's starting weights were not saved (a run before "
                    "2026-11-15, or update_policies turned on at a resume): its own "
                    "exponential average cannot be rebuilt"
                )
            groups = [(folded, ends)]
            kind, decay = "exponential", average.decay(facts)
        elif average.family in ("uniform", "training_uniform"):
            start = average.start_step(facts)
            folded = tuple(step for step in updates if step >= start)
            groups = [(folded, ends)]
            kind, decay = "uniform", 0.0
        else:  # window: its own updates before each evaluation point
            groups = []
            for end in ends:
                before = [step for step in updates if step <= end]
                if average.last_updates is not None:
                    window = before[-average.last_updates :]
                else:
                    length = average.last_steps
                    if length is None:
                        first, last = facts.span
                        length = round(average.last_fraction * (last - first))
                    window = [step for step in before if step > end - length]
                groups.append((tuple(window), [end]))
            kind, decay = "uniform", 0.0
        for folded, group_ends in groups:
            outputs: dict[int, list[Output]] = {}
            for end in group_ends:
                taken = tuple(step for step in folded if step <= end)
                if taken:  # else the point is before the average's start: none there
                    outputs.setdefault(end, []).append(Output(name, end, taken))
            if outputs:
                accumulators.append(Accumulator(kind, decay, folded, outputs, average))
    return accumulators, lives


@dataclasses.dataclass(frozen=True)
class Check:
    """A rebuilt training average against the run's own at the same step: the largest
    absolute difference and whether every tensor is within the tolerances."""

    name: str
    step: int
    largest_difference: float
    within_rounding: bool


def _compare(name: str, step: int, built: dict, saved: dict) -> Check:
    largest, within = 0.0, built.keys() == saved.keys()
    for key, value in saved.items():
        if key not in built:
            continue
        other = built[key]
        if value.is_floating_point():
            difference = (other.double() - value.double()).abs()
            largest = max(largest, float(difference.max()) if difference.numel() else 0.0)
            within &= torch.allclose(
                other.double(),
                value.double(),
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_TOLERANCE,
            )
        else:
            within &= torch.equal(other, value)
    return Check(name, step, largest, bool(within))


def _write(
    run: Run, output: Output, weights: dict, average: AverageSpec, decay: float
) -> pathlib.Path:
    """The policy file of one output, beside the end update's record and return scale."""
    last = run.update_policies_path(output.end) / LIVE
    record = dict(read_record(last))
    record.pop("files")
    description = dict(
        name=output.name,
        family=average.family,
        files=len(output.folded),
        first_step=output.folded[0],
        last_step=output.folded[-1],
        commit=provenance.commit()[0],
    )
    if average.family in ("exponential", "training_exponential"):
        description["decay"] = decay
    record.update(steps=dict(record["steps"], average=description), time=provenance.now())
    files = {WEIGHTS: weights}
    if record["return_scale"]:
        files[RETURN_SCALE] = safetensors.torch.load_file(last / RETURN_SCALE)
    path = run.directory / AVERAGES / output.name / f"step_{output.end}"
    return write_checkpoint(path, record, files)


def player_name(run: Run, path: pathlib.Path) -> str:
    """A policy file's name for `bnb ratings --with`: its path inside the runs directory."""
    return str(path.relative_to(run.directory.parent))


def output_path(run: Run, output: Output) -> pathlib.Path:
    """Where an average's output is written: `<run>/average/<name>/step_<end>/`."""
    return run.directory / AVERAGES / output.name / f"step_{output.end}"


def build(
    run: Run, spec: Spec, report: Callable[[str], None] = print
) -> tuple[list[str], list[Check]]:
    """Build the averages of `spec` not yet written from the run's saved live policies, in
    one pass over the files, and write them; returns (every output's player name, in the
    spec's order, live files included; the checks of the rebuilt training averages
    against the run's own). `report` gets a line per output written and per check. An
    average whose outputs were all written before is not rebuilt (nor checked again)."""
    facts = run_facts(run)
    accumulators, lives = plan(spec, facts)
    paths = {}
    for output in lives:
        paths[(output.name, output.end)] = run.update_policies_path(output.end) / LIVE
    pending = []
    for accumulator in accumulators:
        outputs = [output for group in accumulator.outputs.values() for output in group]
        for output in outputs:
            paths[(output.name, output.end)] = output_path(run, output)
        if any(not output_path(run, output).exists() for output in outputs):
            pending.append(accumulator)
    checks: list[Check] = []
    for step in sorted({step for accumulator in pending for step in accumulator.folded}):
        folding = [
            accumulator
            for accumulator in pending
            if step in accumulator.folded and step <= max(accumulator.outputs)
        ]
        if not folding:
            continue
        live = load_weights(run.update_policies_path(step) / LIVE)
        for accumulator in folding:
            accumulator.fold(live)
            for output in accumulator.outputs.get(step, []):
                checks.extend(_finish(run, accumulator, output, live, report))
            if step == max(accumulator.outputs):
                accumulator.weights = None  # done: its memory goes
        del live
    names = []
    for average in spec.averages:
        name = average.name(facts)
        for end in _evaluation_steps(average, spec, facts):
            if (name, end) not in paths:  # before the average's start
                continue
            player = player_name(run, paths[(name, end)])
            if player not in names:
                names.append(player)
    return names, checks


def _finish(
    run: Run,
    accumulator: Accumulator,
    output: Output,
    live: dict[str, torch.Tensor],
    report: Callable[[str], None],
) -> list[Check]:
    """Write one output (unless written before) and, for a training average, check it
    against the run's own at the same step."""
    weights = accumulator.snapshot(live)
    average = accumulator.family
    if not output_path(run, output).exists():
        _write(run, output, weights, average, accumulator.decay)
        report(
            f"{output.name} at step {output.end}: {len(output.folded)} files from step "
            f"{output.folded[0]}"
        )
    saved_name = dict(
        training_exponential=EXPONENTIAL_AVERAGE, training_uniform=UNIFORM_AVERAGE
    ).get(average.family)
    if saved_name is None:
        return []
    saved = run.update_policies_path(output.end) / saved_name
    if not saved.is_dir():
        return []
    check = _compare(output.name, output.end, weights, load_weights(saved))
    verdict = "within rounding" if check.within_rounding else "DIFFERS"
    report(
        f"check {output.name} at step {output.end} against the run's {saved_name}: "
        f"{verdict} (largest difference {check.largest_difference:.3g})"
    )
    return [check]
