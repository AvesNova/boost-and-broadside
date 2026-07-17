"""Persistent league roster with count-based ratings and PFSP sampling."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.train.rl.features import FeatureCoordinator
from boost_and_broadside.train.rl.rating import MatchCounts, expected_score, solve_league

_ROSTER_VERSION = 3
_LIVE_ID = "live"
_RANDOM_ID = "random"
_SCRIPTED_ID = "scripted"
_AVG_ID = "avg"
# Probe kinds are rated one-directionally against the pool: their games never
# affect pool (live/checkpoint/avg) ratings, so a stomp-heavy probe matchup
# cannot distort the ladder. Scripted's flat derived rating doubles as the
# pool-scale drift canary.
_PROBE_KINDS = frozenset({"scripted", "random"})
# Initial rating guess for the scripted probe, kept until it has games against
# identified pool members.
_SCRIPTED_INITIAL_ELO = 1000.0


@dataclass
class RosterEntry:
    """One first-class league member and its lazily loaded policy."""

    agent_id: str
    kind: str
    label: str
    elo: float
    global_step: int
    update: int
    stationary: bool
    protected: bool = False
    path: str | None = None
    policy: object = field(default=None, repr=False)

    @property
    def is_probe(self) -> bool:
        """True for one-directionally rated members (scripted, random)."""
        return self.kind in _PROBE_KINDS


class LeagueRoster:
    """League members, persistent match evidence, and PFSP curriculum sampling."""

    def __init__(
        self,
        league_size: int,
        pfsp_mode: str,
        pfsp_exponent: float,
        admission_prior_games: float,
        *,
        device: str | torch.device = "cpu",
        count_dtype: torch.dtype = torch.float64,
    ) -> None:
        if league_size < 1:
            raise ValueError(f"league_size must be positive, got {league_size}")
        if pfsp_mode not in ("hard", "variance"):
            raise ValueError(f"pfsp_mode must be 'hard' or 'variance', got {pfsp_mode!r}")
        if pfsp_exponent <= 0.0:
            raise ValueError(f"pfsp_exponent must be positive, got {pfsp_exponent}")
        if admission_prior_games < 0.0:
            raise ValueError(
                f"admission_prior_games must be non-negative, got {admission_prior_games}"
            )
        self.league_size = league_size
        self.pfsp_mode = pfsp_mode
        self.pfsp_exponent = pfsp_exponent
        self.admission_prior_games = admission_prior_games
        self.entries = [
            RosterEntry(_RANDOM_ID, "random", "random", 0.0, 0, 0, True),
            RosterEntry(_SCRIPTED_ID, "scripted", "scripted", _SCRIPTED_INITIAL_ELO, 0, 0, True),
        ]
        self.counts = MatchCounts(
            (_LIVE_ID, _RANDOM_ID, _SCRIPTED_ID), device=device, dtype=count_dtype
        )
        # Per-update outcome tally: same agent layout as counts, but zeroed after
        # every flush and never decayed — the honest source for matchup metrics.
        self.update_tally = MatchCounts(
            (_LIVE_ID, _RANDOM_ID, _SCRIPTED_ID), device=device, dtype=count_dtype
        )
        # Bumped whenever the count-matrix agent layout changes so cached dense
        # indices held by the evaluator and opponent slots can be invalidated.
        self.layout_version = 0
        self.ratings = {_LIVE_ID: 0.0, _RANDOM_ID: 0.0, _SCRIPTED_ID: _SCRIPTED_INITIAL_ELO}
        self.standard_errors = {
            _LIVE_ID: float("inf"),
            _RANDOM_ID: 0.0,
            _SCRIPTED_ID: float("inf"),
        }

    def entry(self, agent_id: str) -> RosterEntry:
        """Return one roster entry by stable ID."""
        for roster_entry in self.entries:
            if roster_entry.agent_id == agent_id:
                return roster_entry
        raise KeyError(f"unknown roster agent_id {agent_id!r}")

    def add_avg(self, global_step: int, update: int) -> RosterEntry:
        """Add the nonstationary average policy on its first active update."""
        for roster_entry in self.entries:
            if roster_entry.kind == "avg":
                return roster_entry
        live_elo = self.ratings[_LIVE_ID]
        roster_entry = RosterEntry(
            _AVG_ID,
            "avg",
            "avg",
            live_elo,
            global_step,
            update,
            False,
        )
        self.entries.append(roster_entry)
        self.counts.add_agent(_AVG_ID)
        self.update_tally.add_agent(_AVG_ID)
        self.layout_version += 1
        self.ratings[_AVG_ID] = live_elo
        self.standard_errors[_AVG_ID] = float("inf")
        return roster_entry

    def add_checkpoint(
        self,
        path: str,
        global_step: int,
        update: int,
        initial_elo: float | None = None,
        *,
        protected: bool = False,
    ) -> RosterEntry:
        """Admit a frozen live snapshot with a draw prior against live.

        A ``protected`` checkpoint is exempt from eviction; the first one
        admitted (ckpt_0, the initial weights) is the pool's rating anchor.
        """
        agent_id = f"ckpt_{global_step}"
        if agent_id in self.counts.agent_ids:
            raise ValueError(f"checkpoint agent_id already exists: {agent_id!r}")
        elo = self.ratings[_LIVE_ID] if initial_elo is None else initial_elo
        roster_entry = RosterEntry(
            agent_id,
            "checkpoint",
            agent_id,
            elo,
            global_step,
            update,
            True,
            protected=protected,
            path=path,
        )
        self.entries.append(roster_entry)
        self.counts.add_agent(agent_id)
        self.update_tally.add_agent(agent_id)
        self.layout_version += 1
        if self.admission_prior_games > 0.0:
            self.counts.add_pair(agent_id, _LIVE_ID, draws=self.admission_prior_games)
        self.ratings[agent_id] = elo
        self.standard_errors[agent_id] = float("inf")
        self._evict_excess_checkpoints()
        return roster_entry

    def add_special(
        self,
        kind: str,
        global_step: int = 0,
        update: int = 0,
        initial_elo: float = 0.0,
    ) -> RosterEntry:
        """Return a built-in member or activate avg during trainer transition."""
        if kind == "avg":
            roster_entry = self.add_avg(global_step, update)
            roster_entry.elo = initial_elo
            self.ratings[_AVG_ID] = initial_elo
            return roster_entry
        if kind == "scripted":
            return self.entry(_SCRIPTED_ID)
        raise ValueError(f"invalid special roster kind {kind!r}")

    def record_outcomes(
        self,
        agent_indices: torch.Tensor,
        opponent_indices: torch.Tensor,
        outcomes: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        """Accumulate directed outcomes into both lifetime and per-update counts."""
        self.counts.record(agent_indices, opponent_indices, outcomes, weights)
        self.update_tally.record(agent_indices, opponent_indices, outcomes, weights)

    def reset_update_tally(self) -> None:
        """Zero the per-update outcome tally after it has been flushed."""
        self.update_tally.tensor.zero_()

    def refresh_ratings(
        self,
        ratings: dict[str, float],
        standard_errors: dict[str, float] | None = None,
    ) -> None:
        """Refresh derived rating caches after a Bradley-Terry solve."""
        missing = set(self.counts.agent_ids) - ratings.keys()
        if missing:
            raise ValueError(f"ratings missing agent IDs: {sorted(missing)}")
        self.ratings = dict(ratings)
        if standard_errors is not None:
            self.standard_errors = dict(standard_errors)
        for roster_entry in self.entries:
            roster_entry.elo = ratings[roster_entry.agent_id]

    def pool_agent_ids(self) -> list[str]:
        """Return the pool members (live plus every non-probe entry)."""
        return [_LIVE_ID] + [entry.agent_id for entry in self.entries if not entry.is_probe]

    def probe_agent_ids(self) -> list[str]:
        """Return the one-directionally rated members."""
        return [entry.agent_id for entry in self.entries if entry.is_probe]

    def anchor_id(self) -> str | None:
        """Return the pool anchor (the protected initial checkpoint), if admitted."""
        for entry in self.entries:
            if entry.kind == "checkpoint" and entry.protected:
                return entry.agent_id
        return None

    def newest_checkpoint(self) -> RosterEntry | None:
        """Return the most recently admitted checkpoint entry."""
        checkpoints = [entry for entry in self.entries if entry.kind == "checkpoint"]
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda entry: (entry.update, entry.global_step))

    def solve_ratings(
        self, *, prior_draws: float = 1.0, prior_frac: float = 0.0
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Solve and cache ratings from the current persistent count matrix."""
        ratings, standard_errors = solve_league(
            self.counts,
            self.anchor_id(),
            self.pool_agent_ids(),
            self.probe_agent_ids(),
            self.ratings,
            prior_draws=prior_draws,
            prior_frac=prior_frac,
        )
        self.refresh_ratings(ratings, standard_errors)
        return ratings, standard_errors

    def sample_opponents(
        self,
        k: int,
        ratings: dict[str, float] | None = None,
    ) -> list[RosterEntry]:
        """Sample up to ``k`` distinct opponents using configured PFSP weights."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        active_ratings = self.ratings if ratings is None else ratings
        candidates = [entry for entry in self.entries if entry.agent_id in active_ratings]
        sample_count = min(k, len(candidates))
        if sample_count == 0:
            return []
        probabilities = torch.tensor(
            [
                expected_score(active_ratings[_LIVE_ID], active_ratings[entry.agent_id])
                for entry in candidates
            ],
            dtype=torch.float64,
        )  # (C,)
        if self.pfsp_mode == "hard":
            weights = (1.0 - probabilities).pow(self.pfsp_exponent)  # (C,)
        else:
            weights = probabilities * (1.0 - probabilities)  # (C,)
        if not torch.isfinite(weights).all() or weights.sum() <= 0.0:
            weights.fill_(1.0)
        sampled_indices = torch.multinomial(weights, sample_count, replacement=False)
        return [candidates[index] for index in sampled_indices.tolist()]

    def _evict_excess_checkpoints(self) -> None:
        """Evict redundant old checkpoints while protecting ladder landmarks.

        Protected entries (the ckpt_0 anchor) never count against the league
        size and are never removable — evicting the anchor would sever the
        pool's rating scale.
        """
        checkpoints = [
            entry for entry in self.entries if entry.kind == "checkpoint" and not entry.protected
        ]
        while len(checkpoints) > self.league_size:
            newest = sorted(
                checkpoints, key=lambda entry: (entry.update, entry.global_step), reverse=True
            )[:4]
            top_rated = sorted(checkpoints, key=lambda entry: entry.elo, reverse=True)[:2]
            protected_ids = {entry.agent_id for entry in newest + top_rated}
            removable = [entry for entry in checkpoints if entry.agent_id not in protected_ids]
            if not removable:
                removable = sorted(
                    checkpoints, key=lambda entry: (entry.update, entry.global_step)
                )[:1]
            crowded = min(
                removable,
                key=lambda entry: (
                    self._nearest_rating_gap(entry, checkpoints),
                    entry.update,
                    entry.global_step,
                ),
            )
            self.entries.remove(crowded)
            checkpoints.remove(crowded)
            self.counts.remove_agent(crowded.agent_id)
            self.update_tally.remove_agent(crowded.agent_id)
            self.layout_version += 1
            self.ratings.pop(crowded.agent_id, None)
            self.standard_errors.pop(crowded.agent_id, None)

    @staticmethod
    def _nearest_rating_gap(entry: RosterEntry, checkpoints: list[RosterEntry]) -> float:
        """Return distance to the nearest other frozen checkpoint rating."""
        gaps = [abs(entry.elo - other.elo) for other in checkpoints if other is not entry]
        return min(gaps, default=float("inf"))

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
        """Lazily load checkpoint weights into a roster entry."""
        if entry.policy is not None or entry.kind != "checkpoint":
            return
        if entry.path is None:
            raise ValueError(f"checkpoint entry {entry.agent_id!r} has no path")
        import torch.nn as nn

        from boost_and_broadside.models.mvp.policy import MVPPolicy

        checkpoint = torch.load(entry.path, map_location=device, weights_only=False)
        checkpoint_team_pma_k = tuple(checkpoint.get("team_pma_k", team_pma_k))
        policy = MVPPolicy(
            model_config,
            coordinator,
            num_value_components=num_value_components,
            num_ships=num_ships,
            team_pma_k=checkpoint_team_pma_k,
        )
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        policy.to(device)
        for module in policy.modules():
            if isinstance(module, nn.RMSNorm) and module.weight.is_cuda:
                module.weight.data = module.weight.data.bfloat16()
        entry.policy = (
            torch.compile(policy, mode=compile_mode) if compile_mode is not None else policy
        )

    def evict_all_checkpoint_policies(self) -> None:
        """Release lazily loaded checkpoint policies."""
        for roster_entry in self.entries:
            if roster_entry.kind == "checkpoint":
                roster_entry.policy = None

    def kept_paths(self) -> set[str]:
        """Return checkpoint files still referenced by the retained roster."""
        return {
            entry.path
            for entry in self.entries
            if entry.kind == "checkpoint" and entry.path is not None
        }

    def save_json(self, path: str | Path, counts: MatchCounts | None = None) -> None:
        """Persist v3 roster metadata and the complete pairwise count matrix."""
        persisted_counts = self.counts if counts is None else counts
        data = {
            "version": _ROSTER_VERSION,
            "entries": [
                {
                    "agent_id": entry.agent_id,
                    "kind": entry.kind,
                    "label": entry.label,
                    "elo": entry.elo,
                    "global_step": entry.global_step,
                    "update": entry.update,
                    "stationary": entry.stationary,
                    "protected": entry.protected,
                    "path": entry.path,
                }
                for entry in self.entries
            ],
            "counts": persisted_counts.to_dict(),
            "ratings": self.ratings,
            "standard_errors": {
                key: value if math.isfinite(value) else None
                for key, value in self.standard_errors.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_json(self, path: str | Path) -> None:
        """Replace roster state from v3 JSON, rejecting all older formats."""
        data = json.loads(Path(path).read_text())
        if data.get("version") != _ROSTER_VERSION:
            raise ValueError("roster.json must use version 3; old runs cannot be resumed")
        self.entries = [
            RosterEntry(
                agent_id=entry["agent_id"],
                kind=entry["kind"],
                label=entry["label"],
                elo=entry["elo"],
                global_step=entry["global_step"],
                update=entry["update"],
                stationary=entry["stationary"],
                protected=entry["protected"],
                path=entry.get("path"),
            )
            for entry in data["entries"]
        ]
        self.counts = MatchCounts.from_dict(
            data["counts"], device=self.counts.device, dtype=self.counts.tensor.dtype
        )
        expected_ids = {_LIVE_ID, *(entry.agent_id for entry in self.entries)}
        if set(self.counts.agent_ids) != expected_ids:
            raise ValueError("roster entries and count-matrix agent IDs do not match")
        self.update_tally = MatchCounts(
            self.counts.agent_ids, device=self.counts.device, dtype=self.counts.tensor.dtype
        )
        self.layout_version += 1
        self.ratings = {key: float(value) for key, value in data["ratings"].items()}
        self.standard_errors = {
            key: float(value) if value is not None else float("inf")
            for key, value in data["standard_errors"].items()
        }
