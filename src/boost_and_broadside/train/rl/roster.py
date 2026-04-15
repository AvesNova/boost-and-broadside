"""ELO-rated league roster for mixed-opponent training.

Maintains a pool of rated agents (past checkpoints, avg policy, scripted agent)
and supports ELO-proximity-weighted sampling for league play.

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


_DEFAULT_ELO = 0.0


@dataclass
class RosterEntry:
    """A single rated agent in the league roster."""

    kind: str  # "random" | "checkpoint" | "avg" | "scripted"
    label: str  # W&B key suffix (e.g. "random", "avg", "scripted", "ckpt_1024000")
    elo: float  # Current ELO rating
    global_step: int  # Training step when this agent was snapshotted
    update: int  # PPO update index when snapshotted
    path: str | None = None  # .pt file path; None for all non-checkpoint kinds
    fixed: bool = False  # If True, ELO is never modified (e.g. random anchor at 0)
    _policy: object = field(
        default=None, repr=False
    )  # Loaded MVPPolicy; None if evicted


class EloRoster:
    """ELO-rated pool of league opponents with proximity-weighted sampling.

    Entries:
        "random"     — always present; ELO fixed at 0 as an absolute anchor.
        "avg"        — added when the avg model first becomes ready.
        "scripted"   — added only after ``scripted_roster_min_steps`` global steps
                       to avoid inflating its ELO before the policy is meaningful.
        "checkpoint" — added at ELO milestones; weakest pruned when over capacity.

    Sampling is weighted by ELO proximity so the training policy tends to face
    near-equal opponents:

        w_i = exp( -|elo_i - training_elo| / elo_temperature )

    The "random" entry is excluded from sampling (only used as an eval anchor).

    Args:
        max_size:        Maximum number of "checkpoint" entries.  Special entries
                         are not counted toward this cap.
        k_factor:         ELO K-factor — how many points change per match.
        elo_temperature:  ELO bandwidth for proximity sampling (in ELO points).
                          Higher → more uniform; lower → tighter focus on peers.
        uniform_sampling: If True, sample opponents uniformly at random instead
                          of ELO-proximity weighting.
    """

    def __init__(
        self,
        max_size: int = 20,
        k_factor: float = 32.0,
        elo_temperature: float = 200.0,
        uniform_sampling: bool = False,
    ) -> None:
        self.max_size = max_size
        self.k_factor = k_factor
        self.elo_temperature = elo_temperature
        self.uniform_sampling = uniform_sampling
        self.entries: list[RosterEntry] = []
        # Random agent entry: ELO starts at 0 and participates in zero-sum updates.
        self.entries.append(
            RosterEntry(
                kind="random",
                label="random",
                elo=_DEFAULT_ELO,
                global_step=0,
                update=0,
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
        for e in self.entries:
            if e.kind == kind:
                return e
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
        """Add a checkpoint entry, evicting the lowest-ELO checkpoint if at capacity.

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
        entry = RosterEntry(
            kind="checkpoint",
            label=f"ckpt_{global_step}",
            elo=initial_elo,
            global_step=global_step,
            update=update,
            path=path,
        )
        self.entries.append(entry)
        self._evict_excess_checkpoints()
        return entry

    def _evict_excess_checkpoints(self) -> None:
        """Remove the lowest-ELO checkpoint entry when over max_size."""
        ckpts = [e for e in self.entries if e.kind == "checkpoint"]
        while len(ckpts) > self.max_size:
            worst = min(ckpts, key=lambda e: e.elo)
            self.entries.remove(worst)
            ckpts.remove(worst)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, training_elo: float) -> RosterEntry | None:
        """Sample one entry, either uniformly or weighted by ELO proximity.

        Fixed entries (e.g. the random anchor) are excluded from sampling.
        Returns None if no non-fixed entries exist.
        """
        candidates = [e for e in self.entries if not e.fixed]
        if not candidates:
            return None

        if self.uniform_sampling:
            idx = int(torch.randint(len(candidates), (1,)).item())
            return candidates[idx]

        weights = [
            math.exp(-abs(e.elo - training_elo) / self.elo_temperature)
            for e in candidates
        ]
        total = sum(weights)
        r = torch.rand(1).item() * total
        cumulative = 0.0
        for entry, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return entry
        return candidates[-1]  # floating-point edge case

    # ------------------------------------------------------------------
    # ELO update
    # ------------------------------------------------------------------

    def update_elo(
        self,
        training_elo: float,
        entry: RosterEntry,
        win_rate: float,
    ) -> float:
        """Apply a zero-sum ELO update for one matchup.

        Args:
            training_elo: Current ELO of the training policy.
            entry:        The opponent roster entry (modified in-place).
            win_rate:     Empirical score for the training policy
                          (1.0 = all wins, 0.5 = all ties, 0.0 = all losses).

        Returns:
            Updated training ELO.
        """
        expected = 1.0 / (1.0 + 10.0 ** ((entry.elo - training_elo) / 400.0))
        delta = self.k_factor * (win_rate - expected)
        if not entry.fixed:
            entry.elo -= delta  # zero-sum; fixed entries (e.g. random) stay put
        return training_elo + delta

    # ------------------------------------------------------------------
    # Policy loading / eviction
    # ------------------------------------------------------------------

    def load_policy(
        self,
        entry: RosterEntry,
        model_config,
        ship_config,
        num_value_components: int,
        device,
        compile_mode: str | None = None,
    ) -> None:
        """Load checkpoint weights into entry._policy (no-op if already loaded)."""
        if entry._policy is not None or entry.kind != "checkpoint":
            return
        from boost_and_broadside.models.mvp.policy import MVPPolicy

        ckpt = torch.load(entry.path, map_location=device, weights_only=False)
        policy = MVPPolicy(
            model_config, ship_config, num_value_components=num_value_components
        )
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        policy.to(device)
        entry._policy = (
            torch.compile(policy, mode=compile_mode)
            if compile_mode is not None
            else policy
        )

    def evict_all_checkpoint_policies(self) -> None:
        """Free loaded weights from all checkpoint entries to reclaim GPU memory."""
        for e in self.entries:
            if e.kind == "checkpoint":
                e._policy = None

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
