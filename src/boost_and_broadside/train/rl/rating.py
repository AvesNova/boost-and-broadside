"""Sampling-invariant league ratings from persistent pairwise match counts."""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

WIN = 0
LOSS = 1
DRAW = 2
ELO_SCALE = 400.0 / math.log(10.0)


class MatchCounts:
    """Dense directed win, loss, and draw counts keyed by stable agent IDs."""

    def __init__(
        self,
        agent_ids: Sequence[str] = (),
        counts: torch.Tensor | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_ids must be unique")
        self.agent_ids = list(agent_ids)
        size = len(self.agent_ids)
        if counts is None:
            self.tensor = torch.zeros(size, size, 3, device=device, dtype=dtype)  # (K, K, 3)
        else:
            if counts.shape != (size, size, 3):
                raise ValueError(
                    f"counts must have shape {(size, size, 3)}, got {tuple(counts.shape)}"
                )
            self.tensor = counts.to(device=device, dtype=dtype)
        self._indices = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}

    def __len__(self) -> int:
        return len(self.agent_ids)

    @property
    def device(self) -> torch.device:
        """Return the device holding the count tensor."""
        return self.tensor.device

    def index(self, agent_id: str) -> int:
        """Return the dense tensor index for an agent ID."""
        try:
            return self._indices[agent_id]
        except KeyError as error:
            raise KeyError(f"unknown agent_id {agent_id!r}") from error

    def add_agent(self, agent_id: str) -> int:
        """Add an empty row and column, returning the new stable dense index."""
        if agent_id in self._indices:
            raise ValueError(f"duplicate agent_id {agent_id!r}")
        old_size = len(self)
        grown = torch.zeros(
            old_size + 1,
            old_size + 1,
            3,
            device=self.tensor.device,
            dtype=self.tensor.dtype,
        )  # (K+1, K+1, 3)
        grown[:old_size, :old_size].copy_(self.tensor)
        self.tensor = grown
        self.agent_ids.append(agent_id)
        self._indices[agent_id] = old_size
        return old_size

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent and every count involving it."""
        index = self.index(agent_id)
        keep = torch.ones(len(self), dtype=torch.bool, device=self.tensor.device)  # (K,)
        keep[index] = False
        self.tensor = self.tensor[keep][:, keep]
        self.agent_ids.pop(index)
        self._indices = {
            current_id: current_index for current_index, current_id in enumerate(self.agent_ids)
        }

    def record(
        self,
        agent_indices: torch.Tensor,
        opponent_indices: torch.Tensor,
        outcomes: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        """Accumulate directed outcomes without synchronizing the tensor device.

        Outcomes use the module constants ``WIN``, ``LOSS``, and ``DRAW`` from
        the perspective of ``agent_indices``. All inputs must be on the count
        tensor's device and share a shape.
        """
        if agent_indices.shape != opponent_indices.shape or agent_indices.shape != outcomes.shape:
            raise ValueError("agent_indices, opponent_indices, and outcomes must share a shape")
        if weights is not None and weights.shape != outcomes.shape:
            raise ValueError("weights must share the outcome shape")
        flat_indices = (
            agent_indices.to(torch.long) * len(self) + opponent_indices.to(torch.long)
        ) * 3 + outcomes.to(torch.long)
        if weights is None:
            weights = torch.ones_like(outcomes, dtype=self.tensor.dtype)
        self.tensor.view(-1).scatter_add_(0, flat_indices.reshape(-1), weights.reshape(-1))

    def add_pair(
        self,
        agent_id: str,
        opponent_id: str,
        *,
        wins: float = 0.0,
        losses: float = 0.0,
        draws: float = 0.0,
    ) -> None:
        """Add CPU-side aggregate results for one directed matchup."""
        index = self.index(agent_id)
        opponent_index = self.index(opponent_id)
        additions = torch.tensor(
            (wins, losses, draws), device=self.tensor.device, dtype=self.tensor.dtype
        )
        self.tensor[index, opponent_index].add_(additions)

    def decay(self, agent_id: str, gamma: float) -> None:
        """Decay all observations involving one nonstationary agent."""
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        index = self.index(agent_id)
        self.tensor[index].mul_(gamma)
        self.tensor[:, index].mul_(gamma)
        # Self-play observations involve the agent once, not twice.
        if gamma > 0.0:
            self.tensor[index, index].div_(gamma)

    def to_cpu(self) -> "MatchCounts":
        """Return an independent CPU snapshot suitable for rating solves."""
        return MatchCounts(self.agent_ids, self.tensor.detach().to("cpu").clone())

    def select_agents(self, agent_ids: Sequence[str]) -> "MatchCounts":
        """Return an independent sub-matrix restricted to ``agent_ids`` (in order)."""
        indices = torch.tensor(
            [self.index(agent_id) for agent_id in agent_ids],
            dtype=torch.long,
            device=self.tensor.device,
        )
        selected = self.tensor.index_select(0, indices).index_select(1, indices)
        return MatchCounts(list(agent_ids), selected.clone())

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable count data."""
        snapshot = self.tensor.detach().to("cpu")
        return {"agent_ids": self.agent_ids, "counts": snapshot.tolist()}

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "MatchCounts":
        """Restore count data produced by :meth:`to_dict`."""
        agent_ids = data.get("agent_ids")
        raw_counts = data.get("counts")
        valid_ids = isinstance(agent_ids, list) and all(
            isinstance(value, str) for value in agent_ids
        )
        if not valid_ids:
            raise ValueError("match counts require a string agent_ids list")
        counts = torch.tensor(raw_counts, device=device, dtype=dtype)
        return cls(agent_ids, counts, device=device, dtype=dtype)

    def save_json(self, path: str | Path) -> None:
        """Persist IDs and counts to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "MatchCounts":
        """Load IDs and counts from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text()), device=device, dtype=dtype)


def expected_score(
    rating: float | torch.Tensor, opponent_rating: float | torch.Tensor
) -> float | torch.Tensor:
    """Return the Bradley-Terry win probability on the ELO scale."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def _pair_results(
    counts: MatchCounts, prior_draws: float, prior_frac: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse directed observations into fractional wins and totals.

    The virtual-draw prior on each played pair is ``prior_draws + prior_frac *
    games``. The proportional part keeps a fully saturated pair's implied gap
    bounded (~``400*log10(2/prior_frac)``) no matter how many games it logs, so
    lopsided matchups cannot buy rating spread with volume.
    """
    raw = counts.tensor.to(dtype=torch.float64, device="cpu")  # (K, K, 3)
    wins = raw[:, :, WIN] + raw[:, :, DRAW] * 0.5
    totals = raw.sum(dim=2)
    pair_wins = wins + raw[:, :, LOSS].T + raw[:, :, DRAW].T * 0.5  # (K, K)
    pair_totals = totals + totals.T  # (K, K)
    played = pair_totals > 0.0
    pair_prior = played * (prior_draws + prior_frac * pair_totals)  # (K, K)
    pair_wins = pair_wins + pair_prior * 0.5
    pair_totals = pair_totals + pair_prior
    pair_wins.fill_diagonal_(0.0)
    pair_totals.fill_diagonal_(0.0)
    return pair_wins, pair_totals


def _anchor_component(pair_totals: torch.Tensor, anchor_index: int) -> torch.Tensor:
    """Return the boolean mask of agents connected to the anchor by played pairs."""
    adjacency = pair_totals > 0.0  # (K, K)
    reached = torch.zeros(pair_totals.shape[0], dtype=torch.bool)  # (K,)
    reached[anchor_index] = True
    while True:
        expanded = reached | (adjacency & reached[None, :]).any(dim=1)
        if bool((expanded == reached).all()):
            return reached
        reached = expanded


def solve_bt(
    counts: MatchCounts,
    anchor_id: str,
    warm_start: Mapping[str, float] | None = None,
    *,
    prior_draws: float = 1.0,
    prior_frac: float = 0.0,
    max_iterations: int = 1_000,
    tolerance: float = 1e-10,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit an anchored Bradley-Terry model by Zermelo MM iterations.

    Ties count as half a win for each player. Virtual draws regularize only
    observed pairs, preserving the match graph rather than inventing evidence.
    Agents without any observed games keep their ``warm_start`` rating exactly
    — the warm start doubles as the prior guess until evidence arrives. The
    same holds for any agent whose played pairs have no path to the anchor:
    a disconnected component's offset is not identified by the data, so its
    members are frozen at their warm start (SE = inf) rather than positioned
    by solver dynamics.
    """
    if prior_draws < 0.0:
        raise ValueError(f"prior_draws must be non-negative, got {prior_draws}")
    if prior_frac < 0.0:
        raise ValueError(f"prior_frac must be non-negative, got {prior_frac}")
    anchor_index = counts.index(anchor_id)
    pair_wins, pair_totals = _pair_results(counts, prior_draws, prior_frac)
    size = len(counts)
    if size == 0:
        raise ValueError("cannot solve an empty rating table")

    ratings_start = warm_start or {}
    theta = torch.tensor(
        [float(ratings_start.get(agent_id, 0.0)) / ELO_SCALE for agent_id in counts.agent_ids],
        dtype=torch.float64,
    )  # (K,)
    theta = theta - theta[anchor_index]
    anchored = _anchor_component(pair_totals, anchor_index)  # (K,)
    active = (pair_totals.sum(dim=1) > 0.0) & anchored  # (K,)

    for _ in range(max_iterations):
        strengths = theta.exp()  # (K,)
        denominators = (
            pair_totals / (strengths[:, None] + strengths[None, :]).clamp_min(1e-300)
        ).sum(dim=1)
        fractional_wins = pair_wins.sum(dim=1)
        updated_strengths = torch.where(
            active,
            fractional_wins / denominators.clamp_min(1e-300),
            strengths,
        )
        updated_theta = updated_strengths.clamp_min(1e-300).log()
        updated_theta = updated_theta - updated_theta[anchor_index]
        if torch.max(torch.abs(updated_theta - theta)) <= tolerance:
            theta = updated_theta
            break
        theta = updated_theta

    ratings = {
        agent_id: float(theta[index] * ELO_SCALE) for index, agent_id in enumerate(counts.agent_ids)
    }
    # Game-less agents would otherwise drift with the anchor's raw MM scale;
    # pin them to their prior guess instead.
    for index, agent_id in enumerate(counts.agent_ids):
        if not bool(active[index]):
            ratings[agent_id] = float(ratings_start.get(agent_id, 0.0))
    ratings[anchor_id] = 0.0
    standard_errors = _standard_errors(theta, pair_totals, anchor_index, counts.agent_ids, active)
    return ratings, standard_errors


def _standard_errors(
    theta: torch.Tensor,
    pair_totals: torch.Tensor,
    anchor_index: int,
    agent_ids: Sequence[str],
    identified: torch.Tensor,
) -> dict[str, float]:
    """Compute anchored ELO standard errors from observed Fisher information.

    Agents outside ``identified`` (game-less or disconnected from the anchor)
    report infinite error — their position was pinned, not measured.
    """
    probabilities = torch.sigmoid(theta[:, None] - theta[None, :])  # (K, K)
    edge_information = pair_totals * probabilities * (1.0 - probabilities)  # (K, K)
    information = torch.diag(edge_information.sum(dim=1)) - edge_information  # (K, K)
    free_indices = [
        index
        for index in range(len(agent_ids))
        if index != anchor_index and bool(identified[index])
    ]
    errors = {agent_ids[anchor_index]: 0.0}
    for index in range(len(agent_ids)):
        if index != anchor_index and not bool(identified[index]):
            errors[agent_ids[index]] = math.inf
    if not free_indices:
        return errors
    reduced = information[free_indices][:, free_indices]
    try:
        covariance = torch.linalg.inv(reduced)
    except torch.linalg.LinAlgError:
        covariance = torch.linalg.pinv(reduced, hermitian=True)
    diagonal = covariance.diag()
    connected = reduced.diag() > 0.0
    for position, index in enumerate(free_indices):
        variance = diagonal[position]
        errors[agent_ids[index]] = (
            float(torch.sqrt(variance.clamp_min(0.0)) * ELO_SCALE)
            if connected[position] and torch.isfinite(variance)
            else math.inf
        )
    return errors


def solve_probe(
    counts: MatchCounts,
    pool_ratings: Mapping[str, float],
    probe_id: str,
    *,
    warm_start: float = 0.0,
    prior_draws: float = 1.0,
    prior_frac: float = 0.0,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    """Rate one probe agent against frozen pool ratings (one-dimensional MLE).

    Probe games inform only the probe: the pool ratings are constants here, so
    a probe's saturated or lopsided matchups can never distort pool members.
    Returns ``(rating, standard_error)``; with no games against any rated pool
    member the probe keeps ``warm_start`` with infinite error.
    """
    pair_wins, pair_totals = _pair_results(counts, prior_draws, prior_frac)
    probe_index = counts.index(probe_id)
    opponents = [
        (counts.index(agent_id), float(rating) / ELO_SCALE)
        for agent_id, rating in pool_ratings.items()
        if agent_id != probe_id and float(pair_totals[probe_index, counts.index(agent_id)]) > 0.0
    ]
    if not opponents:
        return warm_start, math.inf

    def score(theta: float) -> float:
        total = 0.0
        for opponent_index, opponent_theta in opponents:
            games = float(pair_totals[probe_index, opponent_index])
            wins = float(pair_wins[probe_index, opponent_index])
            probability = 1.0 / (1.0 + math.exp(opponent_theta - theta))
            total += wins - games * probability
        return total

    low, high = -40.0, 40.0
    for _ in range(200):
        midpoint = (low + high) / 2.0
        if score(midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
        if high - low <= tolerance:
            break
    theta = (low + high) / 2.0
    information = 0.0
    for opponent_index, opponent_theta in opponents:
        games = float(pair_totals[probe_index, opponent_index])
        probability = 1.0 / (1.0 + math.exp(opponent_theta - theta))
        information += games * probability * (1.0 - probability)
    standard_error = ELO_SCALE / math.sqrt(information) if information > 0.0 else math.inf
    return theta * ELO_SCALE, standard_error


def solve_league(
    counts: MatchCounts,
    anchor_id: str | None,
    pool_ids: Sequence[str],
    probe_ids: Sequence[str],
    warm_start: Mapping[str, float] | None = None,
    *,
    prior_draws: float = 1.0,
    prior_frac: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Two-stage league solve: anchored pool fit, then one-directional probes.

    Pool agents are rated only from pool-vs-pool games, anchored at
    ``anchor_id`` (rating 0). Probe agents are then rated one-directionally
    against the identified pool members — probe games never move pool ratings.
    Before the anchor exists (or for agents the data cannot place), warm-start
    ratings are kept with infinite standard error.
    """
    ratings_start = dict(warm_start or {})
    ratings = {agent_id: float(ratings_start.get(agent_id, 0.0)) for agent_id in counts.agent_ids}
    standard_errors = {agent_id: math.inf for agent_id in counts.agent_ids}

    pool_present = [agent_id for agent_id in pool_ids if agent_id in counts.agent_ids]
    if anchor_id is not None and anchor_id in pool_present:
        pool_ratings, pool_errors = solve_bt(
            counts.select_agents(pool_present),
            anchor_id,
            ratings_start,
            prior_draws=prior_draws,
            prior_frac=prior_frac,
        )
        ratings.update(pool_ratings)
        standard_errors.update(pool_errors)

    identified_pool = {
        agent_id: ratings[agent_id]
        for agent_id in pool_present
        if math.isfinite(standard_errors[agent_id])
    }
    for probe_id in probe_ids:
        if probe_id not in counts.agent_ids:
            continue
        ratings[probe_id], standard_errors[probe_id] = solve_probe(
            counts,
            identified_pool,
            probe_id,
            warm_start=float(ratings_start.get(probe_id, 0.0)),
            prior_draws=prior_draws,
            prior_frac=prior_frac,
        )
    return ratings, standard_errors


# Finite stand-in for unmeasured agents when weighting candidate pairs, so one
# infinite standard error cannot capture the entire scheduling distribution.
_SE_WEIGHT_CAP = 500.0


def information_pair_weights(
    ratings: Mapping[str, float],
    standard_errors: Mapping[str, float],
    candidate_pairs: Sequence[tuple[str, str]],
) -> torch.Tensor:
    """Return normalized information-gain weights for candidate match pairs."""
    weights = []
    for first_id, second_id in candidate_pairs:
        probability = float(expected_score(ratings[first_id], ratings[second_id]))
        uncertainty = (
            min(standard_errors[first_id], _SE_WEIGHT_CAP) ** 2
            + min(standard_errors[second_id], _SE_WEIGHT_CAP) ** 2
        )
        weights.append(probability * (1.0 - probability) * uncertainty)
    result = torch.tensor(weights, dtype=torch.float64)  # (P,)
    total = result.sum()
    if not torch.isfinite(total) or total <= 0.0:
        return torch.full_like(result, 1.0 / len(result))
    return result / total
