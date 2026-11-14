"""The league during training (league spec 1-5 and 7): the pool, the slots' draws and
cuts, the gauges, the averages and the snapshots, attached to a trainer as its hooks.

Per shard (`after_shard`): read the shard's results by entry and mode. Before learning
(`collected`): when a snapshot is due, keep the weights that played this update. After
learning (`updated`): the new snapshot and the retention, the run's ratings from the
update's games (the lanes' and these, `bnb.league.lanes.RunRatings`) and each mode's gauge
from them, the averages (which the slots holding them follow), the next sampling weights,
every `cut_every` updates the clean cut of `cut_slots` slots (round robin, each drawing its
entry from the sampling weights), and the metrics.

A slot's entry is drawn from the modes' sampling weights mixed by the modes' shares of the
opponent rows (a slot plays every mode), by counter-based uniforms
(`bnb.streams.keyed_uniform` of the swap count) keyed by the run's seed. The pool's one scripted
entry is the full-strength scripted agent: no weaker rung and no random player.
"""

import copy
import dataclasses
import math
import pathlib
import shutil
import time

import torch

from bnb.contracts.series import Table
from bnb.controllers.scripted_cheese import agent_of, player_name
from bnb.runs import checkpoint
from bnb.runs.records import Run
from bnb.runs.session import CADENCE_UPDATES
from bnb.runtime.lanes import UNIFORM_AVERAGE, league_slot
from bnb.runtime.opponents import (
    NETWORK,
    OPPONENTS,
    OUTCOMES,
    SCRIPTED_KIND,
    SCRIPTED_PLAYER,
)
from bnb.runtime.train import Trainer
from bnb.streams import inverse_cdf, keyed_uniform, stream_key

from .gauge import GAUGE_TOP, Gauge, Record
from .lanes import RunRatings
from .pool import (
    AVERAGES,
    SCRIPTED_ENTRY,
    Averages,
    Entry,
    retained,
    sampling_weights,
    snapshot_name,
)

# The opponent kinds the league's scores and game shares are sliced by (diagnostics audit,
# group 20): the scripted agent, the averages and the checkpoints (and the cheese agents,
# when the pool holds any: `League.entry_kinds`).
ENTRY_KINDS = ("scripted", "average", "checkpoint")
SNAPSHOT_SECONDS = 30 * 60  # the measured snapshot cadence's target (league spec 5)
SNAPSHOT_DIRECTORY = "step_{steps}"
# an entry rated further than this from the live policy after it has played is an anomaly
GAUGE_DISTANCE = 1500.0
# the cut's series of an update (`pool/cut/<name>`)
CUT_SERIES = ("draws", "cuts", "cut_rows", "uncredited_games", "carries_reset")
CHECKPOINT_POLICY = "checkpoint"  # a snapshot directory's policy files: the snapshot, and
# the averages frozen at the same step (the offline tournament rates those)


class League:
    """The league of one run (see the module docstring). directory: where the snapshots'
    policy files go (the run's `league/`); run: the run's record, which notes a measured
    snapshot cadence; state: a training checkpoint's league state
    (`bnb.runs.checkpoint.league_state`) to continue from; ratings: the live policy's
    starting rating per mode (a warm start's parent's; 0 where not given). Attaching it sets
    the trainer's hooks; without a state it fills every slot with a draw. Every rating is
    the run's one fit (`ratings`, `bnb.league.lanes.RunRatings`) of the rating lanes' games,
    where the run has lanes, and the league's own."""

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
        self.entry_kinds = ENTRY_KINDS + (("cheese",) if self.config.cheese else ())
        self.opponents = trainer.opponents
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.modes = trainer.modes
        rows = torch.tensor(trainer.mode_rows, dtype=torch.float64)
        self.mode_shares = rows / rows.sum()
        self.swap_key = stream_key(trainer.config.seed, "league.swap")
        self.scratch = copy.deepcopy(trainer.model).cpu()  # writes the snapshots' files
        self.device = trainer.rollout.device
        self.pending: dict[str, torch.Tensor] | None = None
        self.started: float | None = None
        self.durations: list[float] = []  # the updates after the first, in seconds
        self.record = self._empty_record()
        self.lane_seat = trainer.lane_seat
        self.ratings = RunRatings(
            trainer, None if state is None else state[0]["ratings"], modes=self.modes
        )
        if state is None:
            self._start(trainer, ratings)
        else:
            self._restore(trainer, *state)
        self._load_lanes(fresh=False)  # no lane game has begun with these weights yet
        trainer.league = self

    # ---- starting and restoring

    def _empty_record(self) -> dict[str, dict[str, Record]]:
        return {mode: {} for mode in self.modes}

    def _start(self, trainer: Trainer, ratings: dict[str, float] | None) -> None:
        """ratings: a warm start's parent's, per mode; a mode it has none for (a clone
        trained without rating lanes) starts at the scripted agent's rating, which the clone
        imitates. A run from scratch (None) starts at 0."""
        weights = trainer.model.state_dict()
        self.entries: dict[str, Entry] = {SCRIPTED_PLAYER: SCRIPTED_ENTRY}
        unrated = 0.0 if ratings is None else GAUGE_TOP
        self.gauges = {mode: Gauge((ratings or {}).get(mode, unrated)) for mode in self.modes}
        for gauge in self.gauges.values():
            gauge.add(SCRIPTED_PLAYER, "scripted", GAUGE_TOP)
        self.averages = Averages(weights, self.config, trainer.config.uniform_start())
        for name in AVERAGES:
            self.entries[name] = Entry(name, "average")
            for gauge in self.gauges.values():
                gauge.add(name, "average")
        for agent in self.config.cheese:  # unrated by the lanes: the games' fit rates them
            name = player_name(agent)
            self.entries[name] = Entry(name, "cheese")
            for gauge in self.gauges.values():
                gauge.add(name, "cheese")
        self.snapshots: dict[str, dict[str, torch.Tensor]] = {}
        self.snapshot_every = self.config.snapshot_every
        self.ordinal = 0
        self.swaps = 0
        self.next_slot = 0
        self.updates = 0  # the cuts' clock
        self.played: set[str] = set()  # the entries with credited training games
        self.last_cut = dict.fromkeys(CUT_SERIES, 0)
        self._weigh()
        self.held: list[str | None] = [None] * self.config.slots
        for slot in range(self.config.slots):
            entry = self._draw()
            self._load(slot, entry)
            self.held[slot] = entry.name

    def _restore(self, trainer: Trainer, record: dict, tensors: dict[str, torch.Tensor]) -> None:
        self.entries = {item["name"]: Entry(**item) for item in record["entries"]}
        self.gauges = {mode: Gauge.from_state(record["gauges"][mode]) for mode in self.modes}
        self.averages = Averages(
            trainer.model.state_dict(),
            self.config,
            trainer.config.uniform_start(),
            record["averaged"],
            tensors,
        )
        self.snapshots = {
            name: {key: tensors[f"{name}.{key}"].clone() for key in trainer.model.state_dict()}
            for name, entry in self.entries.items()
            if entry.kind == "checkpoint"
        }
        self.snapshot_every = record["snapshot_every"]
        self.ordinal, self.swaps, self.next_slot, self.updates = (
            record["ordinal"],
            record["swaps"],
            record["next_slot"],
            record["updates"],
        )
        self.last_cut = dict(record["last_cut"])
        self.played = set(record["played"])
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
        """Each mode's sampling weights over the entries (in `entries` order) and the
        slots' mix of them."""
        names = list(self.entries)
        self.weights = {}
        for mode, gauge in self.gauges.items():
            assert gauge.names() == names, (gauge.names(), names)
            self.weights[mode] = sampling_weights(gauge.win_probabilities(), self.config.floor)
        self.mixed = sum(
            share * self.weights[mode] for share, mode in zip(self.mode_shares.tolist(), self.modes)
        )

    def _draw(self) -> Entry:
        """The next slot's entry, from the mixed weights (the swap counter's uniform)."""
        uniform = keyed_uniform(self.swap_key, torch.tensor([self.swaps]))
        self.swaps += 1
        index = int(inverse_cdf(self.mixed, uniform.double())[0])
        return self.entries[list(self.entries)[index]]

    def _weights_of(self, entry: Entry) -> dict[str, torch.Tensor]:
        if entry.kind == "average":
            return self.averages.weights[entry.name]
        return self.snapshots[entry.name]

    def _distinct(self, entry: Entry) -> bool:
        """Whether `entry` plays other than the live weights: not the uniform average before
        it has started (until then it is the live weights, and its games self-play)."""
        return entry.name != UNIFORM_AVERAGE or self.averages.uniform_started

    def _load_lanes(self, fresh: bool = True) -> None:
        """The averages' weights into the lanes that play them, and each network slot's
        entry into its lane player (`league_<k>`): a snapshot or an average, which the lanes
        rate as that entry (an average's weights moving on with it); the scripted agent,
        which they leave unrated. fresh: games in progress were played by the previous entry
        (`LaneSeat.load`)."""
        if self.lane_seat is None:
            return
        for name, weights in self.averages.weights.items():
            self.lane_seat.load(name, weights)
        for slot in range(self.config.slots):
            name = self.held[slot]
            entry = self.entries.get(name) if name is not None else None
            if entry is not None and entry.network and self._distinct(entry):
                self.ratings.assign(league_slot(slot), name, self._weights_of(entry), fresh)
            else:
                self.ratings.assign(league_slot(slot), None, None)

    def _follow_averages(self) -> None:
        """Each slot holding an average takes its new weights, as the lanes' averages do: an
        average is one moving player."""
        for set_index, name in enumerate(self.held):
            entry = self.entries.get(name) if name is not None else None
            if entry is not None and entry.kind == "average":
                self.opponents.load(set_index, self.averages.weights[name])

    def _load(self, slot: int, entry: Entry) -> None:
        if entry.network:
            self.opponents.load(slot, self._weights_of(entry))
        if entry.kind == "cheese":
            self.opponents.set_cheese(slot, agent_of(entry.name))
        else:
            self.opponents.set_kind(slot, NETWORK if entry.network else SCRIPTED_KIND)

    def _cut(self, trainer: Trainer) -> None:
        """The clean cut, when due: `cut_slots` slots (round robin) each draw an entry; a
        slot whose draw differs from what it holds takes it in place (`OpponentSlots.cut`):
        its games in progress play on uncredited, and the opponent's model carry on its rows
        is reset, or kept (`carry_at_cut="keep"`) when the old and the new entry are both
        networks (`same_build`)."""
        self.last_cut = dict.fromkeys(CUT_SERIES, 0)
        self.updates += 1
        if self.updates % self.config.cut_every:
            return
        carry = trainer.rollout.carry["controllers"][OPPONENTS]
        rows_per_slot = int(self.opponents.rows.numel()) // self.config.slots
        for _ in range(self.config.cut_slots):
            slot = self.next_slot
            self.next_slot = (slot + 1) % self.config.slots
            entry = self._draw()
            self.last_cut["draws"] += 1
            if entry.name == self.held[slot]:
                continue  # the same entry plays on, its games credited
            old = self.entries.get(self.held[slot]) if self.held[slot] is not None else None
            self._load(slot, entry)
            self.held[slot] = entry.name
            keep = self.config.carry_at_cut == "keep" and same_build(old, entry)
            self.last_cut["uncredited_games"] += self.opponents.cut(carry, slot, keep)
            self.last_cut["cuts"] += 1
            self.last_cut["cut_rows"] += rows_per_slot
            self.last_cut["carries_reset"] += 0 if keep else rows_per_slot

    # ---- the hooks

    def after_shard(self, trainer: Trainer, shard: int) -> None:
        """Read the shard's results."""
        slots = self.config.slots
        tally = trainer.rollout.take_tally(OPPONENTS).reshape(-1, len(self.modes), len(OUTCOMES))
        for column, row in enumerate(tally.tolist()):
            name = self.held[column] if column < slots else SCRIPTED_PLAYER
            for mode, (wins, ties, losses) in zip(self.modes, row):
                if wins + ties + losses == 0 or name not in self.entries:
                    continue  # no games, or against an entry retention has dropped
                games = self.record[mode].setdefault(name, Record())
                games.wins += wins
                games.ties += ties
                games.losses += losses

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
        """The snapshot and retention, the ratings and the gauges, the averages and the next
        weights, after the update's learning. Returns the league's metrics."""
        steps = trainer.ppo.environment_steps
        snapshot = None
        if self.pending is not None:
            snapshot = self._snapshot(trainer, steps, self.pending)
            self.pending = None
        games = {
            mode: {
                name: record
                for name, record in records.items()
                if name not in self.entries or self._distinct(self.entries[name])
            }
            for mode, records in self.record.items()
        }
        self.ratings.update(trainer.lane_record, games, snapshot)
        for records in games.values():
            self.played.update(name for name, record in records.items() if record.games)
        for mode, gauge in self.gauges.items():
            gauge.update(self.ratings.ratings.get(mode, {}))
        self.averages.fold(trainer.model.state_dict(), steps)
        self._follow_averages()
        self._weigh()
        self._cut(trainer)
        self._load_lanes()
        metrics = self.metrics()
        self.record = self._empty_record()
        self._measure_cadence(trainer)
        return metrics

    def _measure_cadence(self, trainer: Trainer) -> None:
        """Without a set cadence: about SNAPSHOT_SECONDS at the pace of the slowest of the
        second to fourth updates (the first compiles, and a PPO update that stops early on its
        KL is several times shorter), in whole updates, recorded in the run."""
        if self.snapshot_every is not None:
            return
        now = time.perf_counter()
        if self.started is not None:
            self.durations.append(now - self.started)
            if len(self.durations) == CADENCE_UPDATES:
                updates = max(1, round(SNAPSHOT_SECONDS / max(self.durations)))
                self.snapshot_every = updates * trainer.steps_per_update
                if self.run is not None:
                    self.run.update(snapshot_every=self.snapshot_every)
        self.started = now

    def _snapshot(self, trainer: Trainer, steps: int, weights: dict[str, torch.Tensor]) -> str:
        """A snapshot of `weights` (the ones that played this update) into the pool, and the
        retention after it; returns its name."""
        name = snapshot_name(steps)
        self.entries[name] = Entry(name, "checkpoint", steps=steps, ordinal=self.ordinal)
        self.ordinal += 1
        self.snapshots[name] = weights
        for gauge in self.gauges.values():
            gauge.add(name, "checkpoint")  # at the live rating, until the fit rates it
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
        return name

    def _write(self, trainer: Trainer, steps: int, weights: dict[str, torch.Tensor]) -> None:
        """The snapshot and the averages at this step as policy files under
        `<directory>/step_<steps>/`."""
        path = self.directory / SNAPSHOT_DIRECTORY.format(steps=steps)
        if path.exists():  # a rewound timeline's
            shutil.rmtree(path)
        path.mkdir()
        for policy, state in ((CHECKPOINT_POLICY, weights), *self.averages.weights.items()):
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
        Per mode: the run's ratings (`rating/live`, `rating/average` and every rated player's,
        with their standard errors: `bnb.league.lanes.RunRatings.series`), the update's
        training games against the entries and the sampling distribution's normalised
        entropy. By opponent kind (`entry_kinds`): the live policy's score and its share of
        the games. The pool's size, its snapshots; the update's cut (`pool/cut/*`: the slots
        drawn, the slots that took another entry, their rows, the games left uncredited and
        the rows whose opponent carry was reset) and the slots' mix after it
        (`pool/slot_share/<kind>`, `pool/distinct_held`); and
        `pool/entries`, a table of every entry per mode (rating, error, sampling share,
        games, score, and the slot holding it after the cut), as JSON
        (`bnb.contracts.series.Table.as_json`)."""
        out: dict = {}
        games_by_kind = dict.fromkeys(self.entry_kinds, 0.0)
        points_by_kind = dict.fromkeys(self.entry_kinds, 0.0)
        rows = []
        out.update(self.ratings.series())
        for mode, gauge in self.gauges.items():
            weights = self.weights[mode].tolist()
            out[f"pool/sampling_entropy/{mode}"] = normalised_entropy(weights)
            window = 0.0
            for (name, rated), share in zip(gauge.entries.items(), weights, strict=True):
                games = self.record[mode].get(name)
                played = games.games if games is not None else 0.0
                score = (games.wins + games.ties / 2) / played if played else None
                kind = rated.kind
                if played:
                    games_by_kind[kind] += played
                    points_by_kind[kind] += games.wins + games.ties / 2
                    window += played
                slot = self.held.index(name) if name in self.held else None
                rows.append(
                    (mode, name, kind, rated.rating, rated.stderr, share, played, score, slot)
                )
            out[f"rating/window_games/{mode}"] = window
            out[f"pool/gauge_anomalies/{mode}"] = len(self.gauge_anomalies(mode))
        total = sum(games_by_kind.values())
        for kind in self.entry_kinds:
            if games_by_kind[kind]:
                out[f"pool/score/{kind}"] = points_by_kind[kind] / games_by_kind[kind]
            if total:
                out[f"pool/game_share/{kind}"] = games_by_kind[kind] / total
        out["pool/size"] = len(self.entries)
        out["pool/snapshots"] = len(self.snapshots)
        for name, value in self.last_cut.items():
            out[f"pool/cut/{name}"] = value
        held = [self.entries[name] for name in self.held if name in self.entries]
        for kind in self.entry_kinds:
            out[f"pool/slot_share/{kind}"] = sum(entry.kind == kind for entry in held) / len(
                self.held
            )
        out["pool/distinct_held"] = len({entry.name for entry in held})
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

    def gauge_anomalies(self, mode: str) -> list[str]:
        """The entries of `mode`'s gauge that have played (credited training games) but whose
        rating is more than GAUGE_DISTANCE from the live policy's, or whose error is still
        infinite: a fit that ran away (league-sampling-sim, round 1). The trigger rule
        `gauge_runaway` alerts on their count, `pool/gauge_anomalies/<mode>`."""
        gauge = self.gauges[mode]
        return [
            name
            for name, rated in gauge.entries.items()
            if name in self.played
            and (
                abs(rated.rating - gauge.live.rating) > GAUGE_DISTANCE
                or not math.isfinite(rated.stderr)
            )
        ]

    def state(self) -> tuple[dict, dict[str, torch.Tensor]]:
        """What a resume needs: the entries, the gauges, the slots' entries, the sampling
        state and the averages' count as a record; the averages' and snapshots' weights as
        tensors."""
        record = dict(
            entries=[dataclasses.asdict(entry) for entry in self.entries.values()],
            gauges={mode: gauge.state() for mode, gauge in self.gauges.items()},
            averaged=self.averages.count,
            snapshot_every=self.snapshot_every,
            ordinal=self.ordinal,
            swaps=self.swaps,
            next_slot=self.next_slot,
            updates=self.updates,
            last_cut=dict(self.last_cut),
            played=sorted(self.played),
            held=list(self.held),
            mixed=self.mixed.tolist(),
            weights={mode: weights.tolist() for mode, weights in self.weights.items()},
            ratings=self.ratings.state(),
        )
        tensors = self.averages.tensors()
        for name, weights in self.snapshots.items():
            for key, value in weights.items():
                tensors[f"{name}.{key}"] = value
        return record, tensors


def same_build(old: Entry | None, new: Entry) -> bool:
    """Whether a cut from `old` to `new` may keep the opponent's model carry: both are
    networks, and every network of the training league (snapshots and averages of the live
    policy, in its one stacked template) has the live policy's build. The scripted
    agent's carry is its own, so a cut to or from it resets."""
    return old is not None and old.network and new.network


def normalised_entropy(weights: list[float]) -> float:
    """The entropy of a sampling distribution over its maximum, log of its size: 1 when
    uniform, 0 on one entry (and 0 for a single entry)."""
    if len(weights) < 2:
        return 0.0
    total = sum(weights)
    entropy = -sum(w / total * math.log(w / total) for w in weights if w > 0)
    return min(1.0, entropy / math.log(len(weights)))  # a uniform one rounds just past 1
