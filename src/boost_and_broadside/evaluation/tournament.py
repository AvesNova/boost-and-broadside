"""Shared stationary-player tournament engine and adaptive allocation."""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from boost_and_broadside.agents.semi_random_scripted import (
    SemiRandomScriptedAgent,
    semi_random_label,
)
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EloCalibrateConfig, EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.field_cache import FieldMapCache
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.environment import create_evaluation_env
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.evaluation.run_catalog import (
    CheckpointNotFoundError,
    select_final_training_checkpoint,
    select_named_best_policy,
    select_tournament_ladder_policies,
)
from boost_and_broadside.train.rl.bradley_terry import (
    RatingFit,
    allocate_games,
    fisher_information,
    fit_bradley_terry,
    rating_covariance,
    rating_stderr,
)
from boost_and_broadside.train.rl.checkpoint_schema import require_observation_schema
from boost_and_broadside.train.rl.policy_io import load_policy_bundle

TIE_MODES = ("half_win", "decisive")

_COLLISION_BUDGET = 4_000_000


def parallel_envs_for(total_ships: int, maximum: int) -> int:
    """Largest parallel batch under the shared collision-memory budget."""
    if total_ships <= 0 or maximum <= 0:
        raise ValueError("ship and environment counts must be positive")
    return max(1, min(maximum, _COLLISION_BUDGET // (total_ships * total_ships)))


def rating_views(
    ratings: np.ndarray, pair_games: np.ndarray, labels: list[str]
) -> dict[str, dict[str, list[float]]]:
    """Transform one fitted vector into the three established reporting gauges."""
    random_index = labels.index("random")
    scripted_index = labels.index("scripted")
    random_zero = ratings - ratings[random_index]
    random_error = rating_stderr(pair_games, ratings, anchor=random_index)
    scripted_1000 = ratings - ratings[scripted_index] + 1000.0
    scripted_error = rating_stderr(pair_games, ratings, anchor=scripted_index)

    gap = float(random_zero[scripted_index])
    if abs(gap) < 1e-9:
        dual = np.full_like(ratings, np.nan)
        dual_error = np.full_like(ratings, np.inf)
    else:
        dual = 1000.0 * random_zero / gap
        covariance = rating_covariance(pair_games, ratings, anchor=random_index)
        dual_error = np.zeros_like(ratings)
        for index in range(ratings.size):
            gradient = np.zeros_like(ratings)
            gradient[index] += 1000.0 / gap
            gradient[scripted_index] -= 1000.0 * random_zero[index] / gap**2
            variance = float(gradient @ covariance @ gradient)
            dual_error[index] = np.sqrt(max(variance, 0.0))

    return {
        "random_zero": {"ratings": random_zero.tolist(), "stderr": random_error.tolist()},
        "scripted_1000": {
            "ratings": scripted_1000.tolist(),
            "stderr": scripted_error.tolist(),
        },
        "random_zero_scripted_1000": {
            "ratings": dual.tolist(),
            "stderr": dual_error.tolist(),
        },
    }


def effective_wins(wins: np.ndarray, ties: np.ndarray, tie_mode: str) -> np.ndarray:
    """Win counts as the chosen draw convention sees them."""
    assert tie_mode in TIE_MODES, f"unknown tie_mode {tie_mode!r}"
    if tie_mode == "decisive":
        return wins
    # A draw is half a win for each side. Bradley-Terry's MM iteration is
    # weight-based, so fractional counts need no special handling.
    return wins + 0.5 * (ties + ties.T)


@dataclass
class Player:
    """One stationary agent in the calibration tournament."""

    label: str
    agent: ResolvedAgent
    training_elo: float | None  # rating the run itself assigned, for comparison
    global_step: int | None  # None for agents with no place on the timeline


class Progress:
    """Single-line terminal progress for a long tournament.

    Batches take minutes and episodes finish at wildly different times, so a
    silent run gives no way to tell slow progress from a hang. Redraws in place
    on a terminal; falls back to one line per milestone when redirected, so log
    files do not fill with carriage returns.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.live = enabled and sys.stdout.isatty()
        self.enabled = enabled
        self._width = 0

    def stage(self, message: str) -> None:
        """Announce a phase change; always its own line."""
        if not self.enabled:
            return
        self._clear()
        print(f"  {message}", flush=True)

    def bar(self, done: int, total: int, prefix: str, suffix: str = "") -> None:
        """Draw an in-place progress bar. No-op when output is redirected."""
        if not self.live:
            return
        filled = int(18 * done / max(total, 1))
        text = f"  {prefix} [{'#' * filled}{'.' * (18 - filled)}] {done}/{total} {suffix}"
        self._width = max(self._width, len(text))
        print(f"\r{text.ljust(self._width)}", end="", flush=True)

    def _clear(self) -> None:
        if self.live and self._width:
            print(f"\r{' ' * self._width}\r", end="", flush=True)
            self._width = 0

    def done(self, message: str) -> None:
        """Replace the current bar with a permanent line."""
        self._clear()
        if self.enabled:
            print(f"  {message}", flush=True)


@dataclass
class BatchStat:
    """Convergence record for one adaptive batch."""

    batch: int
    games: int
    cumulative_games: int
    max_stderr: float
    mean_stderr: float
    seconds: float
    ratings: list[float] = field(default_factory=list)


def load_run_config(run_dir: Path) -> tuple[EnvConfig, ModelConfig, str]:
    """Recover the environment, model, and paradigm the run actually trained under.

    Ladder snapshots are policy-only, so this reads the resumable checkpoint.
    Calibrating under a different ship count or paradigm than the run used would
    measure a different game than the one the counts came from.
    """
    try:
        selected = select_final_training_checkpoint(run_dir)
    except CheckpointNotFoundError:
        try:
            selected = select_named_best_policy(run_dir, "training")
        except CheckpointNotFoundError:
            selected = select_named_best_policy(run_dir, "avg")
    checkpoint = torch.load(str(selected.path), map_location="cpu", weights_only=False)
    require_observation_schema(checkpoint, str(selected.path))
    env_config = EnvConfig(**checkpoint["env_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    paradigm = checkpoint.get("train_config", {}).get("paradigm", "ego_pass")
    return env_config, model_config, paradigm


def load_ladder_policy(
    path: Path, model_config: ModelConfig, ship_config: ShipConfig, num_ships: int, device: str
):
    """Build an eval-mode policy from a ladder snapshot.

    Snapshots span a run's history, so each is rebuilt from the configs it
    recorded; the run's own configs are the fallback for snapshots written before
    checkpoints carried provenance.
    """
    return load_policy_bundle(
        str(path),
        device=device,
        num_ships=num_ships,
        ship_config=ship_config,
        model_config=model_config,
    ).policy


def build_players(
    run_dir: Path,
    roster: dict,
    model_config: ModelConfig,
    ship_config: ShipConfig,
    num_ships: int,
    device: str,
    reference_probabilities: tuple[float, ...] = (),
) -> list[Player]:
    """Assemble the tournament field.

    The field is random, the scripted controller, optional semi-random reference
    rungs between them, every ladder snapshot, and the run's final checkpoint.
    The random anchor comes first so it can serve as the fallback rating gauge.
    The rungs cost batch budget but repair the field's weakest link: without
    them, random connects to everything else only through near-certain games.
    The final checkpoint is included so the endpoint of the calibrated curve is
    pinned by a full tournament rating rather than only by the last update's
    online record.
    """
    players = [Player("random", ResolvedAgent("random", None), 0.0, 0)]
    scripted = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    players.append(Player("scripted", ResolvedAgent("scripted", scripted), None, None))

    # One shared scripted instance lets play_batch compute the scripted action
    # once per step for every rung.
    for probability in sorted(reference_probabilities):
        assert 0.0 < probability < 1.0, (
            f"reference probability {probability} duplicates the random/scripted endpoints"
        )
        agent = SemiRandomScriptedAgent(ship_config, probability, scripted_agent=scripted)
        players.append(
            Player(semi_random_label(probability), ResolvedAgent("semi_random", agent), None, None)
        )

    declared = [entry for entry in roster["entries"] if entry["kind"] == "checkpoint"]
    policies = select_tournament_ladder_policies(run_dir, roster)
    selected_labels = {policy.label for policy in policies}
    for entry in declared:
        if entry["label"] not in selected_labels:
            print(f"  [warn] missing ladder snapshot for {entry['label']}, skipping")

    ladder_steps = set()
    for policy_ref in policies:
        policy = load_ladder_policy(
            policy_ref.checkpoint.path, model_config, ship_config, num_ships, device
        )
        ladder_steps.add(policy_ref.global_step)
        players.append(
            Player(
                policy_ref.label,
                ResolvedAgent("policy", policy),
                policy_ref.training_elo,
                policy_ref.global_step,
            )
        )

    try:
        final_checkpoint = select_final_training_checkpoint(run_dir)
    except CheckpointNotFoundError:
        final_checkpoint = None
    if final_checkpoint is not None:
        final_path = final_checkpoint.path
        final_step = final_checkpoint.step
        assert final_step is not None
        if final_step not in ladder_steps:
            policy = load_ladder_policy(final_path, model_config, ship_config, num_ships, device)
            players.append(
                Player(f"ckpt_{final_step}", ResolvedAgent("policy", policy), None, final_step)
            )
    return players


class Tournament:
    """Plays batches of matches between stationary players and tallies outcomes."""

    def __init__(
        self,
        players: list[Player],
        ship_config: ShipConfig,
        env_config: EnvConfig,
        paradigm: str,
        num_envs: int,
        device: str,
        include_bullets: bool = False,
        # Required when env_config has fields: TensorEnv refuses to build without
        # one, and the rungs must play the same map distribution the run trains on.
        field_map: FieldMapCache | None = None,
    ) -> None:
        self.players = players
        self.size = len(players)
        self.ship_config = ship_config
        # Whether the field's policies read bullets; ratings measured without an
        # input the policies trained on would describe a different agent.
        self.include_bullets = include_bullets
        self.env_config = env_config
        self.ego_pass = paradigm == "ego_pass"
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_ships = env_config.num_ships
        self.num_tokens = env_config.num_ships + env_config.num_fields
        self.max_steps = env_config.max_episode_steps
        self.team_sizes = (self.num_ships // 2, self.num_ships - self.num_ships // 2)

        self.env = create_evaluation_env(
            num_envs, ship_config, env_config, self.device, field_map=field_map
        )
        self.wins = np.zeros((self.size, self.size), dtype=np.float64)
        self.ties = np.zeros((self.size, self.size), dtype=np.float64)
        # [team-0 player, team-1 player, team0 win/team1 win/tie]. Unlike the
        # rating matrices, this preserves side assignment for later bias checks.
        self.directed_outcomes = np.zeros((self.size, self.size, 3), dtype=np.float64)

    def scored_wins(self, tie_mode: str) -> np.ndarray:
        """Win counts as the given draw convention scores them."""
        return effective_wins(self.wins, self.ties, tie_mode)

    def pair_games(self, tie_mode: str = "half_win") -> np.ndarray:
        """Symmetric per-pair game counts under the given convention.

        Under half-win scoring a drawn game still counts as a game played, so it
        contributes information; under decisive-only it does not exist at all.
        """
        scored = self.scored_wins(tie_mode)
        return scored + scored.T

    def _assign(self, allocation: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand a per-pair game allocation into per-env team assignments.

        Each pair's games are split evenly between the two team roles, so any
        systematic advantage to playing team 0 cancels out instead of being
        absorbed into one player's rating.
        """
        team0: list[int] = []
        team1: list[int] = []
        for i in range(self.size):
            for j in range(i + 1, self.size):
                total = int(allocation[i, j])
                if total <= 0:
                    continue
                forward = total // 2
                team0.extend([i] * forward + [j] * (total - forward))
                team1.extend([j] * forward + [i] * (total - forward))
        if not team0:  # allocation underflowed to nothing — fall back to one pair
            team0, team1 = [0], [1]
        team0 = team0[: self.num_envs]
        team1 = team1[: self.num_envs]
        if len(team0) < self.num_envs:  # rounding shortfall — recycle assignments
            base0, base1 = list(team0), list(team1)
            for offset in range(self.num_envs - len(base0)):
                team0.append(base0[offset % len(base0)])
                team1.append(base1[offset % len(base1)])
        return (
            torch.tensor(team0, device=self.device, dtype=torch.long),
            torch.tensor(team1, device=self.device, dtype=torch.long),
        )

    def play_batch(self, allocation: np.ndarray, progress: "Progress | None" = None) -> int:
        """Play one episode in every env under the given allocation; tally results.

        Every env is run to completion rather than cutting the batch off at a
        step limit. Episodes that reach the horizon are draws, and draws are
        exactly the slow ones, so stopping early would silently bias the outcome
        mix toward decisive games.
        """
        env_team0, env_team1 = self._assign(allocation)
        runner = MatchRunner(
            self.env,
            [player.agent for player in self.players],
            team0_index=env_team0,
            team1_index=env_team1,
            ship_config=self.ship_config,
            num_ships=self.num_ships,
        )
        runner.init_hidden()

        self.env.reset(options={"team_sizes": self.team_sizes})
        finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        games = 0
        step = 0

        while not finished.all():
            step += 1
            # Polling the count costs a device sync, so do it sparsely — the
            # step loop already runs several policy forwards per iteration.
            if progress is not None and step % 16 == 0:
                progress.bar(
                    games, self.num_envs, "playing", f"episodes   step {step}/{self.max_steps}"
                )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                dones, truncated = runner.step()

            done_any = dones | truncated
            newly_done = done_any & ~finished
            if newly_done.any():
                games += int(newly_done.sum().item())
                self._tally(newly_done, env_team0, env_team1)
                finished |= newly_done
            runner.reset_finished(done_any, options={"team_sizes": self.team_sizes})
        return games

    def _tally(
        self, newly_done: torch.Tensor, env_team0: torch.Tensor, env_team1: torch.Tensor
    ) -> None:
        """Accumulate wins and ties for the episodes that just ended."""
        alive = self.env.state.ship_alive
        team = self.env.state.ship_team_id
        team0_alive = (alive & (team == 0)).any(dim=1)
        team1_alive = (alive & (team == 1)).any(dim=1)
        team0_won = newly_done & team0_alive & ~team1_alive
        team1_won = newly_done & team1_alive & ~team0_alive
        tied = newly_done & ~team0_won & ~team1_won

        flat = self.size * env_team0 + env_team1
        for outcome_index, outcome in enumerate((team0_won, team1_won, tied)):
            if not outcome.any():
                continue
            counts = torch.bincount(flat[outcome], minlength=self.size**2)
            self.directed_outcomes[..., outcome_index] += (
                counts.reshape(self.size, self.size).cpu().numpy().astype(np.float64)
            )
        if team0_won.any():
            counts = torch.bincount(flat[team0_won], minlength=self.size**2)
            self.wins += counts.reshape(self.size, self.size).cpu().numpy().astype(np.float64)
        if tied.any():
            counts = torch.bincount(flat[tied], minlength=self.size**2)
            self.ties += counts.reshape(self.size, self.size).cpu().numpy().astype(np.float64)
        if team1_won.any():
            flipped = self.size * env_team1 + env_team0
            counts = torch.bincount(flipped[team1_won], minlength=self.size**2)
            self.wins += counts.reshape(self.size, self.size).cpu().numpy().astype(np.float64)


def choose_reference(pair_games: np.ndarray, ratings: np.ndarray, random_index: int) -> int:
    """Pick the player to measure everything else against.

    Standard errors are only meaningful relative to some player, and picking a
    badly-connected one loads its own uncertainty onto every other rating at
    once. The most information-dense player is chosen so the reported errors
    describe what the tournament actually determines.

    Under half-win scoring the random anchor is usually a perfectly serviceable
    gauge, since draws against it are informative. Under decisive-only it is not
    — its games are then either dropped as draws or foregone conclusions, and
    pinning anything through that link would take on the order of 80,000 games.
    The anchor is excluded from the choice either way: it is the reported zero,
    so measuring it against itself would say nothing.
    """
    information = np.diag(fisher_information(pair_games, ratings)).copy()
    information[random_index] = -1.0
    return int(np.argmax(information))


def run_tournament(
    tournament: Tournament,
    random_index: int,
    config: EloCalibrateConfig,
    progress: Progress | None = None,
    initial_stats: list[BatchStat] | None = None,
    on_batch: Callable[[Tournament, RatingFit, list[BatchStat], int], None] | None = None,
    seed_base: int | None = None,
) -> tuple[RatingFit, list[BatchStat], int]:
    """Play adaptive batches until every rating is pinned or the budget runs out.

    Allocation, standard errors, and the convergence test all use the reference
    gauge rather than the random anchor. Optimizing against random would sink the
    whole budget into the one link that cannot be improved at any realistic cost.
    """
    stats = list(initial_stats or [])
    size = tournament.size
    prior_games = config.prior_games
    tie_mode = config.tie_mode
    reference = random_index
    scored = tournament.scored_wins(tie_mode)
    if scored.sum() > 0:
        provisional = fit_bradley_terry(scored, anchor=random_index, prior_games=prior_games)
        reference = choose_reference(
            tournament.pair_games(tie_mode), provisional.ratings, random_index
        )
        fit = fit_bradley_terry(scored, anchor=reference, prior_games=prior_games)
    else:
        fit = fit_bradley_terry(scored, anchor=reference, prior_games=prior_games)
    cumulative = stats[-1].cumulative_games if stats else int(tournament.pair_games().sum() / 2)
    for batch in range(len(stats) + 1, config.max_batches + 1):
        if seed_base is not None:
            torch.manual_seed(seed_base + batch)
        if progress is not None:
            worst = f"{stats[-1].max_stderr:.1f}" if stats else "—"
            progress.bar(
                0,
                tournament.num_envs,
                f"batch {batch}/{config.max_batches}",
                f"episodes   worst SE so far {worst}",
            )
        allocation = allocate_games(
            tournament.pair_games(tie_mode),
            fit.ratings,
            anchor=reference,
            budget=tournament.num_envs,
        )
        started = time.perf_counter()
        games = tournament.play_batch(allocation, progress)
        elapsed = time.perf_counter() - started
        cumulative += games
        scored = tournament.scored_wins(tie_mode)
        if batch == 1:
            provisional = fit_bradley_terry(scored, anchor=random_index, prior_games=prior_games)
            reference = choose_reference(
                tournament.pair_games(tie_mode), provisional.ratings, random_index
            )
            if progress is not None:
                progress.done(f"reference gauge: {tournament.players[reference].label}")
        fit = fit_bradley_terry(scored, anchor=reference, prior_games=prior_games)

        # Random's own error is the unresolvable anchor link, reported separately.
        rated = (fit.games > 0) & (np.arange(size) != random_index)
        worst = float(fit.stderr[rated].max()) if rated.any() else float("inf")
        stat = BatchStat(
            batch=batch,
            games=games,
            cumulative_games=cumulative,
            max_stderr=worst,
            mean_stderr=float(np.mean(fit.stderr[rated])) if rated.any() else float("inf"),
            seconds=elapsed,
            ratings=[float(r) for r in fit.ratings],
        )
        stats.append(stat)
        line = (
            f"batch {batch:2d}/{config.max_batches}  games={games:6d}  "
            f"total={cumulative:7d}  worst SE={stat.max_stderr:6.2f}  "
            f"mean SE={stat.mean_stderr:5.2f}  ({elapsed:.0f}s)"
        )
        if progress is not None:
            progress.done(line)
        else:
            print(f"  {line}", flush=True)
        if on_batch is not None:
            on_batch(tournament, fit, stats, reference)
        if worst <= config.target_stderr:
            message = f"converged: every rating within +/-{config.target_stderr:.1f} of reference"
            progress.done(message) if progress else print(f"  {message}")
            break
    else:
        message = (
            f"budget exhausted at {config.max_batches} batches; worst rating is still "
            f"+/-{stats[-1].max_stderr:.1f} against a +/-{config.target_stderr:.1f} target"
        )
        progress.done(message) if progress else print(f"  {message}")
    return fit, stats, reference
