"""Elo-rated league roster and measurement ladder for mixed-opponent training.

Maintains a pool of rated agents (past checkpoints, avg policy, scripted agent)
and supports Elo-proximity-weighted sampling for league play.

The checkpoint entries double as the Elo measurement ladder:

    Ladder invariant: at most one checkpoint entry is floating (``fixed=False``)
    — the newest milestone snapshot, whose rating is still settling. All older
    checkpoints and the random anchor are frozen (``fixed=True``); their ratings
    are permanent calibration references and are never modified again.

Entry kinds:
    "checkpoint"  — a past training-policy snapshot loaded from a .pt file.
    "avg"         — the live running-average policy (weights accessed externally).
    "scripted"    — the StochasticScriptedAgent (no weights to load).
    "semi_random" — a scripted/uniform blend at a fixed p_scripted, rated
                    offline. Interior rungs between random and scripted, so the
                    ladder has a well-matched reference at every height of the
                    climb instead of only two saturated ones.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from boost_and_broadside.agents.semi_random_scripted import semi_random_label
from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.train.rl.policy_io import PolicyBundle, load_policy_bundle

_DEFAULT_ELO = 0.0


@dataclass
class RosterEntry:
    """A single rated agent in the league roster."""

    kind: str  # "random" | "checkpoint" | "avg" | "scripted"
    label: str  # W&B key suffix (e.g. "random", "avg", "scripted", "ckpt_1024000")
    elo: float  # Current Elo rating
    global_step: int  # Training step when this agent was snapshotted
    update: int  # PPO update index when snapshotted
    path: str | None = None  # .pt file path; None for all non-checkpoint kinds
    fixed: bool = False  # If True, the rating is frozen forever (ladder anchor)
    # Cleared by retire() when this run turns out not to be able to host the
    # entry as an opponent (see EloRoster.retire). Its rating stays on the
    # ladder; only opponent sampling skips it.
    usable: bool = True
    # Scripted-action probability for "semi_random" entries; None otherwise.
    p_scripted: float | None = None
    policy: object = field(default=None, repr=False)  # Loaded YemongPolicy; None if unloaded
    # The configs this entry's weights were trained under. Held because a roster
    # spans a run's history: an entry need not share the live policy's architecture
    # or physics, and each one plays as whatever it was.
    bundle: PolicyBundle | None = field(default=None, repr=False)

    @property
    def is_stationary(self) -> bool:
        """Whether this is a fixed player whose rating is a measured constant.

        Stationary references never age out of the measurement ladder — unlike
        checkpoints, which rotate — because their strength does not change.
        """
        return self.kind in ("random", "semi_random", "scripted")


class EloRoster:
    """Elo-rated pool of league opponents with proximity-weighted sampling.

    Entries:
        "random"     — always present; Elo fixed at 0 as the absolute ladder anchor.
        "avg"        — added when the avg model first becomes ready.
        "checkpoint" — added at Elo milestones; frozen at the following milestone.

    Entries are never removed: the frozen checkpoints are the measurement
    ladder, and the full set (with ratings) is kept for post-hoc analysis.
    ``max_size`` bounds only how many checkpoint policies stay loaded on the
    device at once (least recently used are unloaded first).

    Sampling is weighted by Elo proximity so the training policy tends to face
    near-equal opponents:

        w_i = exp( -|elo_i - training_elo| / elo_temperature )

    The "random" entry is excluded from sampling (only used as an eval anchor).

    Args:
        max_size:         Maximum number of checkpoint policies kept loaded.
        elo_temperature:  Elo bandwidth for proximity sampling (in Elo points).
                          Higher → more uniform; lower → tighter focus on peers.
        uniform_sampling: If True, sample opponents uniformly at random instead
                          of Elo-proximity weighting.
    """

    def __init__(
        self,
        max_size: int = 20,
        elo_temperature: float = 200.0,
        uniform_sampling: bool = False,
    ) -> None:
        self.max_size = max_size
        self.elo_temperature = elo_temperature
        self.uniform_sampling = uniform_sampling
        self.entries: list[RosterEntry] = []
        self._load_order: list[RosterEntry] = []  # loaded checkpoints, oldest use first
        # Random agent entry: the absolute ladder anchor, frozen at Elo 0.
        self.entries.append(
            RosterEntry(
                kind="random",
                label="random",
                elo=_DEFAULT_ELO,
                global_step=0,
                update=0,
                fixed=True,
            )
        )

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_special(
        self,
        kind: str,
        global_step: int = 0,
        update: int = 0,
        initial_elo: float = _DEFAULT_ELO,
    ) -> RosterEntry:
        """Add or return the existing entry for a special agent ("avg" or "scripted").

        Idempotent: if an entry of this kind already exists it is returned unchanged.

        Args:
            kind:        "avg" or "scripted".
            global_step: Training step when this agent became available.
            update:      PPO update index when it became available.
            initial_elo: Starting Elo.  Pass the current training Elo so the new
                         entry begins calibrated rather than at an arbitrary default.
        """
        assert kind in ("avg", "scripted"), f"add_special: invalid kind {kind!r}"
        for entry in self.entries:
            if entry.kind == kind:
                return entry
        entry = RosterEntry(
            kind=kind,
            label=kind,
            elo=initial_elo,
            global_step=global_step,
            update=update,
        )
        self.entries.append(entry)
        return entry

    def add_reference(self, p_scripted: float, elo: float) -> RosterEntry:
        """Add or return a semi-random reference rung at a fixed rating.

        Interior rungs between random and scripted. Without them the ladder has
        exactly two stationary references and the live policy saturates both —
        winning ~100% against random and losing ~100% against scripted — leaving
        its rating barely identified for the whole early climb.

        The rating is a measured property of a stationary player, fitted offline
        by ``--mode semi_random_tournament`` under the same env config, tick rate
        and fleet size the run uses, so it is ``fixed`` from the start.

        Args:
            p_scripted: Probability the rung takes the scripted action; the rest
                        of the time it acts uniformly at random.
            elo:        Fitted rating on this run's gauge.
        """
        if not 0.0 < p_scripted < 1.0:
            raise ValueError(
                f"p_scripted must lie strictly in (0, 1) — 0 is the random agent "
                f"and 1 is the scripted agent, got {p_scripted}"
            )
        label = semi_random_label(p_scripted)
        for entry in self.entries:
            if entry.label == label:
                return entry
        entry = RosterEntry(
            kind="semi_random",
            label=label,
            elo=elo,
            global_step=0,
            update=0,
            fixed=True,
            p_scripted=p_scripted,
        )
        self.entries.append(entry)
        return entry

    def add_checkpoint(
        self,
        path: str,
        global_step: int,
        update: int,
        initial_elo: float = _DEFAULT_ELO,
    ) -> RosterEntry:
        """Add a floating checkpoint entry (the new milestone snapshot).

        The previous floating checkpoint must be frozen first — the ladder
        allows exactly one settling rating at a time.

        Weights are NOT loaded here; call ``load_policy()`` when needed.

        Args:
            path:        Absolute path to the saved .pt file.
            global_step: Training step at which the snapshot was taken.
            update:      PPO update index at which it was saved.
            initial_elo: Starting Elo.  Pass the current training Elo so the new
                         entry begins calibrated rather than at an arbitrary default.

        Returns:
            The newly created RosterEntry.
        """
        assert self.floating_checkpoint() is None, (
            "add_checkpoint: freeze the current floating checkpoint before adding a new one"
        )
        entry = RosterEntry(
            kind="checkpoint",
            label=f"ckpt_{global_step}",
            elo=initial_elo,
            global_step=global_step,
            update=update,
            path=path,
        )
        self.entries.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Measurement ladder
    # ------------------------------------------------------------------

    def floating_checkpoint(self) -> RosterEntry | None:
        """Return the single floating (non-fixed) checkpoint entry, if any."""
        floating = [e for e in self.entries if e.kind == "checkpoint" and not e.fixed]
        assert len(floating) <= 1, "ladder invariant violated: multiple floating checkpoints"
        return floating[0] if floating else None

    def set_floating_elo(self, elo: float) -> None:
        """Sync the floating checkpoint's rating from the continuous evaluator."""
        entry = self.floating_checkpoint()
        if entry is not None:
            entry.elo = elo

    def set_special_elo(self, kind: str, elo: float) -> None:
        """Sync a special entry's rating from the continuous evaluator.

        No-op when the entry does not exist yet — "avg" only joins the roster
        once the running average starts accumulating.

        Proximity sampling reads these ratings, so a stale one misdirects the
        draw: the average policy tracks the live policy closely and should be
        drawn often, while the scripted agent should fade as the live rating
        outruns it. Neither happens if their entries keep the rating they were
        created with.
        """
        for entry in self.entries:
            if entry.kind == kind:
                entry.elo = elo
                return

    def retire(self, entry: RosterEntry) -> None:
        """Drop an entry from opponent sampling, keeping its rating on the ladder.

        For entries this run cannot host. The rollout observation's shape is
        fixed when the wrapper is built, so — unlike the eval battery, which
        widens to suit — a bullet-reading opponent in a bullet-free run would
        play blind and be rated as a weaker agent than it is. Retiring beats
        raising: the roster spans a run's history and a single incompatible
        entry should not end training hours in.
        """
        entry.usable = False
        self._unload(entry)
        if entry in self._load_order:
            self._load_order.remove(entry)

    def freeze_floating(self) -> RosterEntry | None:
        """Permanently freeze the floating checkpoint's rating at its current value.

        Returns:
            The newly frozen entry, or None if no checkpoint was floating.
        """
        entry = self.floating_checkpoint()
        if entry is not None:
            entry.fixed = True
        return entry

    def ladder_anchors(self, checkpoint_count: int) -> list[RosterEntry]:
        """Return the measurement ladder: stationary references, then checkpoints.

        Stationary references (random, the semi-random rungs, scripted) come
        first and are *all* returned, every time. They are fixed players whose
        ratings are measured constants, so they stay useful for as long as the
        live policy is near them, and dropping one would throw away a calibration
        point that cannot be regenerated in-run. Frozen checkpoints follow,
        oldest-first, truncated to the newest ``checkpoint_count`` — those do age
        out, because the live policy leaves them behind.

        The evaluator relies on the stationary block being a contiguous prefix.
        """
        stationary = sorted(
            (e for e in self.entries if e.is_stationary and e.usable),
            key=lambda e: e.elo,
        )
        frozen = sorted(
            (e for e in self.entries if e.kind == "checkpoint" and e.fixed),
            key=lambda e: e.global_step,
        )
        return stationary + (frozen[-checkpoint_count:] if checkpoint_count > 0 else [])

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, training_elo: float) -> RosterEntry | None:
        """Sample one entry, either uniformly or weighted by Elo proximity.

        Frozen checkpoints, the running-average policy and the scripted agent
        are all ordinary candidates — proximity weighting is what decides the
        opponent mix. Retired entries are skipped.

        The random anchor is excluded, and that exclusion is load-bearing rather
        than incidental: the live rating starts at 0 and so does random's, so
        including it would make the early league almost entirely random play
        instead of the scripted agent that is the only useful opponent then.

        Returns None if no candidates exist.
        """
        candidates = [e for e in self.entries if e.kind != "random" and e.usable]
        if not candidates:
            return None

        if self.uniform_sampling:
            idx = int(torch.randint(len(candidates), (1,)).item())
            return candidates[idx]

        weights = [math.exp(-abs(e.elo - training_elo) / self.elo_temperature) for e in candidates]
        total = sum(weights)
        r = torch.rand(1).item() * total
        cumulative = 0.0
        for entry, weight in zip(candidates, weights):
            cumulative += weight
            if r <= cumulative:
                return entry
        return candidates[-1]  # floating-point edge case

    # ------------------------------------------------------------------
    # Policy loading / eviction
    # ------------------------------------------------------------------

    def load_policy(
        self,
        entry: RosterEntry,
        ship_config: ShipConfig,
        num_ships: int,
        device: str | torch.device,
        *,
        model_config: ModelConfig | None = None,
        compile_mode: str | None = None,
        team_pma_k: tuple[int, ...] = (),
    ) -> None:
        """Load checkpoint weights into entry.policy (no-op if already loaded).

        The entry is rebuilt from the configs its own file records; ``model_config``
        and ``ship_config`` are fallbacks for snapshots written before checkpoints
        carried provenance. ``num_ships`` is the live environment's, since that is
        the game the opponent will actually play.

        At most ``max_size`` checkpoint policies stay loaded; the least
        recently used beyond that are unloaded to reclaim device memory.
        """
        if entry.kind != "checkpoint":
            return
        if entry.policy is None:
            entry.bundle = load_policy_bundle(
                entry.path,
                device=device,
                num_ships=num_ships,
                ship_config=ship_config,
                model_config=model_config,
                team_pma_k=team_pma_k,
                compile_mode=compile_mode,
            )
            entry.policy = entry.bundle.policy
        if entry in self._load_order:
            self._load_order.remove(entry)
        self._load_order.append(entry)
        while len(self._load_order) > self.max_size:
            self._unload(self._load_order.pop(0))

    @staticmethod
    def _unload(entry: RosterEntry) -> None:
        entry.policy = None
        entry.bundle = None

    def evict_all_checkpoint_policies(self) -> None:
        """Free loaded weights from all checkpoint entries to reclaim GPU memory."""
        for entry in self.entries:
            if entry.kind == "checkpoint":
                self._unload(entry)
        self._load_order.clear()

    # ------------------------------------------------------------------
    # Checkpoint file paths referenced by the roster (must not be pruned)
    # ------------------------------------------------------------------

    def kept_paths(self) -> set[str]:
        """Return the set of .pt paths that are currently roster entries."""
        return {e.path for e in self.entries if e.kind == "checkpoint" and e.path}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_json(self, path: str | Path) -> None:
        """Persist roster metadata (Elo ratings, file paths) to JSON."""
        data = {
            "entries": [
                {
                    "kind": e.kind,
                    "label": e.label,
                    "elo": e.elo,
                    "global_step": e.global_step,
                    "update": e.update,
                    "path": e.path,
                    "fixed": e.fixed,
                }
                for e in self.entries
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_json(self, path: str | Path) -> None:
        """Restore roster metadata from JSON (replaces current entries; no weights loaded)."""
        data = json.loads(Path(path).read_text())
        self.entries = [
            RosterEntry(
                kind=d["kind"],
                label=d["label"],
                elo=d["elo"],
                global_step=d["global_step"],
                update=d["update"],
                path=d.get("path"),
                fixed=d.get("fixed", False),
            )
            for d in data["entries"]
        ]
        self._load_order.clear()
