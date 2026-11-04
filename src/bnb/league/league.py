"""The league during training (league spec 1-5 and 7): the pool, the slots' draws and
drains, the gauges, the averages and the snapshots, attached to a trainer as its hooks.

Per shard (`after_shard`): read the shard's results by entry and mode; end a finished
drain; begin the next one (one slot at a time, round robin), drawing its entry from the
sampling weights. Before learning (`collected`): when a snapshot is due, keep the weights
that played this update. After learning (`updated`): each mode's gauge, the new snapshot and
the retention, the averages, the next sampling weights, and the metrics.

A slot's entry is drawn from the modes' sampling weights mixed by the modes' shares of the
opponent rows (a slot plays every mode); a ladder row draws its rung in the graph, from its
own mode's weights over the ladder. The host's draws use the same counter-based uniforms as
the graph (`bnb.runtime.opponents.counter_uniform`), keyed by the run's seed.
"""

import copy
import dataclasses
import math
import pathlib
import shutil
import time

import torch

from bnb.contracts.series import Table
from bnb.runs import checkpoint
from bnb.runs.records import Run
from bnb.runtime.lanes import LIVE
from bnb.runtime.opponents import (
    LADDER_KIND,
    NETWORK,
    OPPONENTS,
    OUTCOMES,
    counter_uniform,
    draw,
    stream_key,
)
from bnb.runtime.train import Trainer

from .gauge import Gauge, Record, gauge_rating
from .lanes import AVERAGE, LaneRatings
from .pool import (
    AVERAGES,
    EXPONENTIAL_AVERAGE,
    UNIFORM_AVERAGE,
    Entry,
    average_into,
    ladder_entries,
    retained,
    sampling_weights,
    snapshot_name,
)

# The opponent kinds the league's scores and game shares are sliced by (diagnostics audit,
# group 20): the scripted agent, the ladder's other rungs (random among them), the averages
# and the checkpoints.
ENTRY_KINDS = ("scripted", "rung", "average", "checkpoint")
SNAPSHOT_SECONDS = 30 * 60  # the measured snapshot cadence's target (league spec 5)
SNAPSHOT_DIRECTORY = "step_{steps}"
CHECKPOINT_POLICY = "checkpoint"  # a snapshot directory's policy files: the snapshot, and
# the averages frozen at the same step (the offline tournament rates those)


class League:
    """The league of one run (see the module docstring). directory: where the snapshots'
    policy files go (the run's `league/`); run: the run's record, which notes a measured
    snapshot cadence; state: a training checkpoint's league state
    (`bnb.runs.checkpoint.league_state`) to continue from; ratings: the live policy's
    starting rating per mode (a warm start's parent's; 0 where not given). Attaching it sets
    the trainer's hooks; without a state it fills every slot with a draw. With rating lanes
    (`bnb.league.lanes`) the live and averaged ratings and the ladder's are the lanes'."""

    def __init__(
        self,
        trainer: Trainer,
        directory: str | pathlib.Path,
        run: Run | None = None,
        state: tuple[dict, dict[str, torch.Tensor]] | None = None,
        ratings: dict[str, float] | None = None,
    ) -> None:
        if trainer.opponents is None:
            raise ValueError("the trainer's lineup has no league seat (`bnb.runtime.opponents`)")
        self.config = trainer.config.league
        self.opponents = trainer.opponents
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.modes = trainer.modes
        rows = torch.tensor(trainer.mode_rows, dtype=torch.float64)
        self.mode_shares = rows / rows.sum()
        self.ladder = ladder_entries(self.config.ladder)
        self.swap_key = stream_key(trainer.config.seed, "league.swap")
        self.scratch = copy.deepcopy(trainer.model).cpu()  # writes the snapshots' files
        self.device = trainer.rollout.device
        self.pending: dict[str, torch.Tensor] | None = None
        self.started: float | None = None
        self.record = self._empty_record()
        self.lane_seat = trainer.lane_seat
        self.lanes = None
        if trainer.lanes:
            self.lanes = LaneRatings(trainer, None if state is None else state[0]["lanes"])
        if state is None:
            self._start(trainer, ratings or {})
        else:
            self._restore(trainer, *state)
        self._load_lanes()
        trainer.league = self

    # ---- starting and restoring

    def _empty_record(self) -> dict[str, dict[str, Record]]:
        return {mode: {} for mode in self.modes}

    def _start(self, trainer: Trainer, ratings: dict[str, float]) -> None:
        weights = trainer.model.state_dict()
        self.entries: dict[str, Entry] = {entry.name: entry for entry in self.ladder}
        self.gauges = {mode: Gauge(ratings.get(mode, 0.0)) for mode in self.modes}
        for gauge in self.gauges.values():
            for entry in self.ladder:
                gauge.add(entry.name, "ladder", gauge_rating(entry.scripted_probability))
        self.averages = {
            name: {key: value.detach().clone() for key, value in weights.items()}
            for name in AVERAGES
        }
        for name in AVERAGES:
            self.entries[name] = Entry(name, "average")
            for gauge in self.gauges.values():
                gauge.add(name, "average")
        self.averaged = 1
        self.snapshots: dict[str, dict[str, torch.Tensor]] = {}
        self.snapshot_every = self.config.snapshot_every
        self.ordinal = 0
        self.swaps = 0
        self.next_slot = 0
        self._weigh()
        self.held: list[str | None] = [None] * (self.config.slots + 1)
        carry = trainer.rollout.carry["controllers"][OPPONENTS]
        for slot in range(self.config.slots):
            entry = self._draw()
            self._load(slot, entry)
            self.held[slot] = entry.name
            if entry.kind == "ladder":
                uniform = counter_uniform(
                    self.swap_key, torch.tensor(-1 - slot), self.opponents.rows.cpu()
                )
                self.opponents.draw_ladder(carry, slot, uniform)

    def _restore(self, trainer: Trainer, record: dict, tensors: dict[str, torch.Tensor]) -> None:
        self.entries = {item["name"]: Entry(**item) for item in record["entries"]}
        self.gauges = {mode: Gauge.from_state(record["gauges"][mode]) for mode in self.modes}
        self.averages = {
            name: {
                key: tensors[f"{name}.{key}"].to(value.device)
                for key, value in trainer.model.state_dict().items()
            }
            for name in AVERAGES
        }
        self.averaged = record["averaged"]
        self.snapshots = {
            name: {key: tensors[f"{name}.{key}"].clone() for key in trainer.model.state_dict()}
            for name, entry in self.entries.items()
            if entry.kind == "checkpoint"
        }
        self.snapshot_every = record["snapshot_every"]
        self.ordinal, self.swaps, self.next_slot = (
            record["ordinal"],
            record["swaps"],
            record["next_slot"],
        )
        self.held = record["held"]
        self.mixed = torch.tensor(record["mixed"], dtype=torch.float64)
        self.weights = {
            mode: torch.tensor(values, dtype=torch.float64)
            for mode, values in record["weights"].items()
        }
        steps = trainer.ppo.environment_steps
        for path in self.directory.glob("step_*"):  # a rewound timeline's snapshots
            digits = path.name[len("step_") :]
            if digits.isdigit() and int(digits) > steps:
                shutil.rmtree(path)

    # ---- sampling

    def _weigh(self) -> None:
        """Each mode's sampling weights over the entries (in `entries` order), the slots'
        mix of them, and the ladder rows' weights on the device."""
        names = list(self.entries)
        self.weights = {}
        for mode, gauge in self.gauges.items():
            assert gauge.names() == names, (gauge.names(), names)
            self.weights[mode] = sampling_weights(gauge.win_probabilities(), self.config.floor)
        self.mixed = sum(
            share * self.weights[mode] for share, mode in zip(self.mode_shares.tolist(), self.modes)
        )
        ladder = [names.index(entry.name) for entry in self.ladder]
        self.opponents.set_ladder_weights(
            torch.stack([self.weights[mode][ladder] for mode in self.modes]).float()
        )

    def _draw(self) -> Entry:
        """The next slot's entry, from the mixed weights (the swap counter's uniform)."""
        uniform = counter_uniform(
            self.swap_key, torch.tensor(self.swaps), torch.zeros(1, dtype=torch.int64)
        )
        self.swaps += 1
        index = int(draw(self.mixed, uniform.double())[0])
        return self.entries[list(self.entries)[index]]

    def _weights_of(self, entry: Entry) -> dict[str, torch.Tensor]:
        if entry.kind == "average":
            return self.averages[entry.name]
        return self.snapshots[entry.name]

    def _load_lanes(self) -> None:
        """The averages' weights into the lanes that play them."""
        if self.lane_seat is not None:
            for name, weights in self.averages.items():
                self.lane_seat.load(name, weights)

    def _load(self, set_index: int, entry: Entry) -> None:
        if entry.network:
            self.opponents.load(set_index, self._weights_of(entry))
        self.opponents.set_kind(set_index, NETWORK if entry.network else LADDER_KIND)

    # ---- the hooks

    def after_shard(self, trainer: Trainer, shard: int) -> None:
        """Read the shard's results; end a finished drain and begin the next."""
        slots = self.config.slots
        tally = trainer.rollout.take_tally(OPPONENTS).reshape(-1, len(self.modes), len(OUTCOMES))
        ladder_names = [entry.name for entry in self.ladder]
        for column, row in enumerate(tally.tolist()):
            name = self.held[column] if column <= slots else ladder_names[column - slots - 1]
            for mode, (wins, ties, losses) in zip(self.modes, row):
                if wins + ties + losses == 0 or name not in self.entries:
                    continue  # no games, or against an entry retention has dropped
                games = self.record[mode].setdefault(name, Record())
                games.wins += wins
                games.ties += ties
                games.losses += losses
        carry = trainer.rollout.carry["controllers"][OPPONENTS]
        draining = self.opponents.draining_slot
        if draining is not None and self.opponents.drained(carry):
            self.opponents.finish_drain(carry)
            self.held[draining], self.held[slots] = self.held[slots], None
            draining = None
        if draining is None:
            slot = self.next_slot
            self.next_slot = (slot + 1) % slots
            entry = self._draw()
            self._load(slots, entry)
            self.held[slots] = entry.name
            self.opponents.begin_drain(slot)

    def _steps_after_update(self, trainer: Trainer) -> tuple[int, int]:
        before = trainer.ppo.environment_steps
        return before, before + trainer.steps_per_update

    def collected(self, trainer: Trainer) -> None:
        """When a snapshot is due at this update, keep the weights that played it."""
        before, after = self._steps_after_update(trainer)
        every = self.snapshot_every
        if every is not None and after // every > before // every:
            self.pending = {
                key: value.detach().to("cpu", copy=True)
                for key, value in trainer.model.state_dict().items()
            }

    @torch.no_grad()
    def updated(self, trainer: Trainer) -> dict:
        """The gauges, the snapshot and retention, the averages and the next weights, after
        the update's learning. Returns the league's metrics."""
        steps = trainer.ppo.environment_steps
        if self.lanes is not None:
            self.lanes.update(trainer.lane_record)
        for mode, gauge in self.gauges.items():
            if self.lanes is None:
                gauge.update(self.record[mode])
                continue
            rated = self.lanes.ratings.get(mode, {})
            gauge.calibrate(
                {entry.name: rated[entry.name] for entry in self.ladder if entry.name in rated}
            )
            gauge.update(
                self.record[mode],
                live=rated.get(LIVE),
                rated={name: rated[name] for name in AVERAGES if name in rated},
            )
        if self.pending is not None:
            self._snapshot(trainer, steps, self.pending)
            self.pending = None
        self.averaged += 1
        uniform, exponential = self.averages[UNIFORM_AVERAGE], self.averages[EXPONENTIAL_AVERAGE]
        average_into(
            uniform,
            exponential,
            trainer.model.state_dict(),
            self.averaged,
            self.config.exponential_decay,
        )
        self._load_lanes()
        self._weigh()
        metrics = self.metrics()
        self.record = self._empty_record()
        self._measure_cadence(trainer)
        return metrics

    def _measure_cadence(self, trainer: Trainer) -> None:
        """Without a set cadence: about SNAPSHOT_SECONDS at the second update's pace (the
        first compiles), in whole updates, recorded in the run."""
        if self.snapshot_every is not None:
            return
        now = time.perf_counter()
        if self.started is not None:
            updates = max(1, round(SNAPSHOT_SECONDS / (now - self.started)))
            self.snapshot_every = updates * trainer.steps_per_update
            if self.run is not None:
                self.run.update(snapshot_every=self.snapshot_every)
        self.started = now

    def _snapshot(self, trainer: Trainer, steps: int, weights: dict[str, torch.Tensor]) -> None:
        name = snapshot_name(steps)
        self.entries[name] = Entry(name, "checkpoint", steps=steps, ordinal=self.ordinal)
        self.ordinal += 1
        self.snapshots[name] = weights
        for mode, gauge in self.gauges.items():
            gauge.add(name, "checkpoint")  # at this update's live rating
            gauge.credit(name, self.record[mode])
        self._write(trainer, steps, weights)
        ordinals = {e.ordinal: e.name for e in self.entries.values() if e.kind == "checkpoint"}
        for ordinal in set(ordinals) - retained(set(ordinals)):
            dropped = self.entries.pop(ordinals[ordinal])
            del self.snapshots[dropped.name]
            for gauge in self.gauges.values():
                gauge.remove(dropped.name)
            path = self.directory / SNAPSHOT_DIRECTORY.format(steps=dropped.steps)
            if path.exists():
                shutil.rmtree(path)

    def _write(self, trainer: Trainer, steps: int, weights: dict[str, torch.Tensor]) -> None:
        """The snapshot and the averages at this step as policy files under
        `<directory>/step_<steps>/`."""
        path = self.directory / SNAPSHOT_DIRECTORY.format(steps=steps)
        if path.exists():  # a rewound timeline's
            shutil.rmtree(path)
        path.mkdir()
        for policy, state in ((CHECKPOINT_POLICY, weights), *self.averages.items()):
            self.scratch.load_state_dict(state)
            checkpoint.save_policy(
                path / policy,
                self.scratch,
                steps=dict(environment_steps=steps),
                game=trainer.config.game,
                types=trainer.ppo.types,
            )

    # ---- metrics and state

    def metrics(self) -> dict:
        """The league's series of this update (diagnostics audit, group 20), on the host.
        Per mode: the live and averaged ratings (`rating/live`, `rating/average` and their
        standard errors: the lanes', `bnb.league.lanes.LaneRatings.series`, or without lanes
        the gauge's), the training games that the gauge read and the sampling distribution's
        normalised entropy. By
        opponent kind (`entry_kind`): the live policy's score and its share of the games.
        The pool's size, its snapshots and whether a slot is draining; and `pool/entries`,
        a table of every entry per mode (rating, error, sampling share, games, score, and
        the slot holding it: the network slots in order, then the incoming one), as
        JSON (`bnb.contracts.series.Table.as_json`)."""
        out: dict = {}
        games_by_kind = dict.fromkeys(ENTRY_KINDS, 0.0)
        points_by_kind = dict.fromkeys(ENTRY_KINDS, 0.0)
        rows = []
        if self.lanes is not None:
            out.update(self.lanes.series())
        for mode, gauge in self.gauges.items():
            if self.lanes is None:  # without lanes, the gauge's own estimates
                out[f"rating/live/{mode}"] = gauge.live.rating  # the last finite value
                out[f"rating/live_stderr/{mode}"] = gauge.live.stderr
                average = gauge.entries[AVERAGE]
                out[f"rating/average/{mode}"] = average.reported
                out[f"rating/average_stderr/{mode}"] = average.stderr
            weights = self.weights[mode].tolist()
            out[f"pool/sampling_entropy/{mode}"] = normalised_entropy(weights)
            window = 0.0
            for (name, rated), share in zip(gauge.entries.items(), weights, strict=True):
                games = self.record[mode].get(name)
                played = games.games if games is not None else 0.0
                score = (games.wins + games.ties / 2) / played if played else None
                kind = entry_kind(name, rated.kind)
                if played:
                    games_by_kind[kind] += played
                    points_by_kind[kind] += games.wins + games.ties / 2
                    window += played
                slot = self.held.index(name) if name in self.held else None
                rows.append(
                    (mode, name, kind, rated.reported, rated.stderr, share, played, score, slot)
                )
            out[f"rating/window_games/{mode}"] = window
        total = sum(games_by_kind.values())
        for kind in ENTRY_KINDS:
            if games_by_kind[kind]:
                out[f"pool/score/{kind}"] = points_by_kind[kind] / games_by_kind[kind]
            if total:
                out[f"pool/game_share/{kind}"] = games_by_kind[kind] / total
        out["pool/size"] = len(self.entries)
        out["pool/snapshots"] = len(self.snapshots)
        out["pool/draining"] = int(self.opponents.draining_slot is not None)
        out["pool/entries"] = Table(
            (
                "mode",
                "entry",
                "kind",
                "rating",
                "stderr",
                "sampling_share",
                "games",
                "score",
                "slot",
            ),
            tuple(rows),
        ).as_json()
        return out

    def state(self) -> tuple[dict, dict[str, torch.Tensor]]:
        """What a resume needs: the entries, the gauges, the slots' entries, the sampling
        state and the averages' count as a record; the averages' and snapshots' weights as
        tensors."""
        record = dict(
            entries=[dataclasses.asdict(entry) for entry in self.entries.values()],
            gauges={mode: gauge.state() for mode, gauge in self.gauges.items()},
            averaged=self.averaged,
            snapshot_every=self.snapshot_every,
            ordinal=self.ordinal,
            swaps=self.swaps,
            next_slot=self.next_slot,
            held=list(self.held),
            mixed=self.mixed.tolist(),
            weights={mode: weights.tolist() for mode, weights in self.weights.items()},
            lanes=None if self.lanes is None else self.lanes.state(),
        )
        tensors = {}
        for name, weights in (*self.averages.items(), *self.snapshots.items()):
            for key, value in weights.items():
                tensors[f"{name}.{key}"] = value
        return record, tensors


def entry_kind(name: str, kind: str) -> str:
    """An entry's kind for the series (ENTRY_KINDS): a ladder entry is "scripted" or "rung"."""
    if kind == "ladder":
        return "scripted" if name == "scripted" else "rung"
    return kind


def normalised_entropy(weights: list[float]) -> float:
    """The entropy of a sampling distribution over its maximum, log of its size: 1 when
    uniform, 0 on one entry (and 0 for a single entry)."""
    if len(weights) < 2:
        return 0.0
    total = sum(weights)
    entropy = -sum(w / total * math.log(w / total) for w in weights if w > 0)
    return entropy / math.log(len(weights))
