"""ELO-rated league roster and measurement ladder for mixed-opponent training.

Maintains a pool of rated agents (past checkpoints, avg policy, scripted agent)
and supports ELO-proximity-weighted sampling for league play.

The checkpoint entries double as the ELO measurement ladder:

    Ladder invariant: at most one checkpoint entry is floating (``fixed=False``)
    — the newest milestone snapshot, whose rating is still settling. All older
    checkpoints and the random anchor are frozen (``fixed=True``); their ratings
    are permanent calibration references and are never modified again.

Entry kinds:
    "checkpoint" — a past training-policy snapshot loaded from a .pt file.
    "avg"        — the live running-average policy (weights accessed externally).
    "scripted"   — the StochasticScriptedAgent (no weights to load).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.train.rl.features import FeatureCoordinator

_DEFAULT_ELO = 0.0


def load_checkpoint_policy(
    path: str,
    model_config: ModelConfig,
    coordinator: FeatureCoordinator,
    num_value_components: int,
    num_ships: int,
    device: str | torch.device,
    compile_mode: str | None = None,
    team_pma_k: tuple[int, ...] = (),
):
    """Construct an eval-mode MVPPolicy from a saved checkpoint file.

    Args:
        path:                 Path to a .pt file containing "policy_state_dict".
        model_config:         Policy architecture config.
        coordinator:          Feature coordinator shared with the trainer.
        num_value_components: Number of critic output components.
        num_ships:            Ship count (N) the policy heads are sized for.
        device:               Target device.
        compile_mode:         Optional torch.compile mode; None loads uncompiled.
        team_pma_k:           Fallback win-component indices for checkpoints
                              saved before "team_pma_k" was stored.

    Returns:
        The loaded policy in eval mode (compiled when compile_mode is set).
    """
    from boost_and_broadside.models.mvp.policy import MVPPolicy

    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_team_pma_k = tuple(ckpt.get("team_pma_k", team_pma_k))
    policy = MVPPolicy(
        model_config,
        coordinator,
        num_value_components=num_value_components,
        num_ships=num_ships,
        team_pma_k=ckpt_team_pma_k,
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    policy.to(device)
    return torch.compile(policy, mode=compile_mode) if compile_mode is not None else policy


@dataclass
class RosterEntry:
    """A single rated agent in the league roster."""

    kind: str  # "random" | "checkpoint" | "avg" | "scripted"
    label: str  # W&B key suffix (e.g. "random", "avg", "scripted", "ckpt_1024000")
    elo: float  # Current ELO rating
    global_step: int  # Training step when this agent was snapshotted
    update: int  # PPO update index when snapshotted
    path: str | None = None  # .pt file path; None for all non-checkpoint kinds
    fixed: bool = False  # If True, the rating is frozen forever (ladder anchor)
    policy: object = field(default=None, repr=False)  # Loaded MVPPolicy; None if unloaded


class EloRoster:
    """ELO-rated pool of league opponents with proximity-weighted sampling.

    Entries:
        "random"     — always present; ELO fixed at 0 as the absolute ladder anchor.
        "avg"        — added when the avg model first becomes ready.
        "checkpoint" — added at ELO milestones; frozen at the following milestone.

    Entries are never removed: the frozen checkpoints are the measurement
    ladder, and the full set (with ratings) is kept for post-hoc analysis.
    ``max_size`` bounds only how many checkpoint policies stay loaded on the
    device at once (least recently used are unloaded first).

    Sampling is weighted by ELO proximity so the training policy tends to face
    near-equal opponents:

        w_i = exp( -|elo_i - training_elo| / elo_temperature )

    The "random" entry is excluded from sampling (only used as an eval anchor).

    Args:
        max_size:         Maximum number of checkpoint policies kept loaded.
        elo_temperature:  ELO bandwidth for proximity sampling (in ELO points).
                          Higher → more uniform; lower → tighter focus on peers.
        uniform_sampling: If True, sample opponents uniformly at random instead
                          of ELO-proximity weighting.
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
        # Random agent entry: the absolute ladder anchor, frozen at ELO 0.
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
            initial_elo: Starting ELO.  Pass the current training ELO so the new
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
            initial_elo: Starting ELO.  Pass the current training ELO so the new
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

    def freeze_floating(self) -> RosterEntry | None:
        """Permanently freeze the floating checkpoint's rating at its current value.

        Returns:
            The newly frozen entry, or None if no checkpoint was floating.
        """
        entry = self.floating_checkpoint()
        if entry is not None:
            entry.fixed = True
        return entry

    def ladder_anchors(self, count: int) -> list[RosterEntry]:
        """Return the newest ``count`` frozen ladder entries, oldest first.

        The ladder is the random anchor followed by every frozen checkpoint in
        snapshot order; its tail holds the calibration anchors the continuous
        evaluator rates the live policy against.
        """
        random_entry = next(e for e in self.entries if e.kind == "random")
        frozen = sorted(
            (e for e in self.entries if e.kind == "checkpoint" and e.fixed),
            key=lambda e: e.global_step,
        )
        ladder = [random_entry] + frozen
        return ladder[-count:]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, training_elo: float) -> RosterEntry | None:
        """Sample one entry, either uniformly or weighted by ELO proximity.

        The random anchor is excluded (eval-only reference); frozen checkpoints
        remain valid league opponents. Returns None if no candidates exist.
        """
        candidates = [e for e in self.entries if e.kind != "random"]
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
        model_config: ModelConfig,
        coordinator: FeatureCoordinator,
        num_value_components: int,
        num_ships: int,
        device: str | torch.device,
        compile_mode: str | None = None,
        team_pma_k: tuple[int, ...] = (),
    ) -> None:
        """Load checkpoint weights into entry.policy (no-op if already loaded).

        At most ``max_size`` checkpoint policies stay loaded; the least
        recently used beyond that are unloaded to reclaim device memory.
        """
        if entry.kind != "checkpoint":
            return
        if entry.policy is None:
            entry.policy = load_checkpoint_policy(
                entry.path,
                model_config,
                coordinator,
                num_value_components,
                num_ships,
                device,
                compile_mode,
                team_pma_k,
            )
        if entry in self._load_order:
            self._load_order.remove(entry)
        self._load_order.append(entry)
        while len(self._load_order) > self.max_size:
            self._load_order.pop(0).policy = None

    def evict_all_checkpoint_policies(self) -> None:
        """Free loaded weights from all checkpoint entries to reclaim GPU memory."""
        for entry in self.entries:
            if entry.kind == "checkpoint":
                entry.policy = None
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
        """Persist roster metadata (ELO ratings, file paths) to JSON."""
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
