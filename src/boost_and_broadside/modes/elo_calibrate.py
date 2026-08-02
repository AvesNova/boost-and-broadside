"""elo_calibrate mode: re-rate a finished training run from raw match counts.

A training run leaves behind two kinds of evidence. Its frozen ladder
checkpoints are files on disk that can be replayed against each other as many
times as wanted, so their ratings are limited only by how much compute is spent.
Its live and averaged policies are gone — each existed in one form for a single
update — so all that survives of them is the win/loss/tie record captured at the
time.

This mode uses the first to calibrate the second. It plays an adaptive
tournament among the stationary players until every rating is pinned to a target
standard error, then replays each update's stored record against those now-known
opponents to recover what the live policy was actually worth at that moment.
The result is a training curve that no longer depends on where the in-training
K-factor filter happened to drift.

Every rating is fit twice, under both draw conventions (see TIE_MODES). The two
differ only where draws are common, which in this game means the weak end of the
ladder, and the half-win fit is the one directly comparable to the in-training
curve. Both come from the same match counts, so carrying both is free.

Reported ratings are shifted so the scripted controller reads
SCRIPTED_ANCHOR_ELO (see that constant for why). Because the raw win/tie
matrices are persisted, ``refit=True`` reruns every fit and artifact from the
stored counts without playing a single game.

Writes to the run's checkpoint directory:
    elo_calibrated.json      ratings, per-update curve, batch stats, and the raw
                             win/tie matrices, so any later refit needs no replay
    elo_calibration/*.png    live curve vs in-training, the two draw conventions,
                             per-checkpoint ratings, convergence, and draw rates
    elo_calibration/         the calibrated ratings in the W&B export format
      history.jsonl,           (see modes/elo_calibrate_history.py), so one loader
      summary.json             reads them alongside the run's in-training history
"""

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from boost_and_broadside.agents.semi_random_scripted import SemiRandomScriptedAgent
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EloCalibrateConfig, EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
    infer_num_value_components,
    infer_team_pma_k,
    init_hidden,
    reset_done_envs,
)
from boost_and_broadside.train.rl.bradley_terry import (
    RatingFit,
    allocate_games,
    fisher_information,
    fit_bradley_terry,
    fit_single_rating,
)
from boost_and_broadside.train.rl.checkpoint_schema import require_observation_schema

# Opponents that are not stationary and so cannot be tournament players. "avg"
# changes every update, exactly like the live policy it is measured against.
_NON_STATIONARY = frozenset({"avg"})

# How draws enter the likelihood.
#
#   "half_win" — a draw is half a win to each side. The default, and the
#                convention Elo has always used. It treats drawing as evidence
#                of parity, which is the whole content of a draw: tying every
#                game against a 3600-rated opponent implies a rating of 3600.
#   "decisive" — draws dropped; the rating answers the narrower question "who
#                wins, given that somebody wins".
#
# Note that neither introduces a draw *parameter*. Davidson and Rao-Kupper do,
# and both are scale-invariant in the strengths, so they can only express draw
# frequency as a function of the rating gap — which measurement here contradicts,
# since draws track the absolute level of a matchup instead. That rules those two
# models out; it says nothing against half-win scoring, which models no draw
# process at all and simply fits the expected score.
#
# Dropping draws is the costly option in this game. Against the random anchor the
# live policy's whole-run record is 2794W/10L/1120T: decisive-only extracts a
# Fisher information of 10 from it, half-win extracts 487 — a 49x difference,
# because a 99.6% win rate sits where p(1-p) has almost nothing left to give.
# "decisive" is kept as a diagnostic rather than a default: it is the check for a
# policy farming draws, which would earn parity under half-win scoring without
# ever having to win. A large disagreement between the two is that signature.
#
# Both are fit from the same match counts, so carrying both costs no extra play.
TIE_MODES = ("half_win", "decisive")

# Reporting anchor: every rating is shifted so the scripted controller reads
# 1000. Scripted is the one opponent shared across runs and fleet scales, so
# pinning it makes ratings comparable between them, and its link to the field
# is far tighter than random's — every trained agent beats random so decisively
# that those games barely constrain the scale. Random still plays in the
# tournament and is reported like any other player; it lands below zero here.
SCRIPTED_ANCHOR_ELO = 1000.0


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


def _load_run_config(run_dir: Path) -> tuple[EnvConfig, ModelConfig, str]:
    """Recover the environment, model, and paradigm the run actually trained under.

    Ladder snapshots are policy-only, so this reads the resumable checkpoint.
    Calibrating under a different ship count or paradigm than the run used would
    measure a different game than the one the counts came from.
    """
    candidates = sorted(run_dir.glob("step_*.pt")) + sorted(run_dir.glob("best_*.pt"))
    if not candidates:
        sys.exit(f"Error: no step_*.pt or best_*.pt in '{run_dir}' to read the run config from.")
    checkpoint = torch.load(str(candidates[-1]), map_location="cpu", weights_only=False)
    require_observation_schema(checkpoint, str(candidates[-1]))
    env_config = EnvConfig(**checkpoint["env_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    paradigm = checkpoint.get("train_config", {}).get("paradigm", "ego_pass")
    return env_config, model_config, paradigm


def _load_ladder_policy(
    path: Path, model_config: ModelConfig, ship_config: ShipConfig, num_ships: int, device: str
):
    """Build an eval-mode policy from a ladder snapshot."""
    from boost_and_broadside.models.yemong.policy import YemongPolicy
    from boost_and_broadside.train.rl.features import build_standard_coordinator

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    require_observation_schema(checkpoint, str(path))
    coordinator = build_standard_coordinator(ship_config)
    num_components = infer_num_value_components(checkpoint)
    policy = YemongPolicy(
        model_config,
        coordinator,
        num_value_components=num_components,
        num_ships=num_ships,
        team_pma_k=infer_team_pma_k(checkpoint),
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
    policy.eval()
    policy.requires_grad_(False)
    return policy


def semi_random_label(probability: float) -> str:
    """Canonical player label for a scripted-action mixture probability."""
    if probability == 0.0:
        return "random"
    if probability == 1.0:
        return "scripted"
    digits = f"{probability:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"semi_scripted_{digits}"


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

    entries = [e for e in roster["entries"] if e["kind"] == "checkpoint"]
    ladder_steps = set()
    for entry in sorted(entries, key=lambda e: e["global_step"]):
        path = Path(entry["path"])
        if not path.exists():  # roster may outlive a pruned file
            path = run_dir / path.name
        if not path.exists():
            print(f"  [warn] missing ladder snapshot for {entry['label']}, skipping")
            continue
        policy = _load_ladder_policy(path, model_config, ship_config, num_ships, device)
        ladder_steps.add(int(entry["global_step"]))
        players.append(
            Player(
                entry["label"], ResolvedAgent("policy", policy), entry["elo"], entry["global_step"]
            )
        )

    final_checkpoints = sorted(run_dir.glob("step_*.pt"))
    if final_checkpoints:
        final_path = final_checkpoints[-1]
        final_step = int(final_path.stem.removeprefix("step_"))
        if final_step not in ladder_steps:
            policy = _load_ladder_policy(final_path, model_config, ship_config, num_ships, device)
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
    ) -> None:
        self.players = players
        self.size = len(players)
        self.ship_config = ship_config
        self.env_config = env_config
        self.ego_pass = paradigm == "ego_pass"
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_ships = env_config.num_ships
        self.num_tokens = env_config.num_ships + env_config.num_fields
        self.max_steps = env_config.max_episode_steps
        self.team_sizes = (self.num_ships // 2, self.num_ships - self.num_ships // 2)

        self.env = TensorEnv(num_envs, ship_config, env_config, self.device)
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
        active: list[torch.Tensor] = []
        for index, player in enumerate(self.players):
            mask = (env_team0 == index) | (env_team1 == index)
            indices = mask.nonzero(as_tuple=True)[0]
            active.append(indices)
            if player.agent.kind == "policy":
                init_hidden(player.agent, int(indices.numel()), self.num_tokens, self.device)

        self.env.reset(options={"team_sizes": self.team_sizes})
        finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        actions = torch.zeros(
            self.size, self.num_envs, self.num_ships, 3, dtype=torch.long, device=self.device
        )
        arange = torch.arange(self.num_envs, device=self.device)
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
            state = self.env.state
            obs = observation_from_state(state, self.ship_config)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                semi_scripted_cache: dict[int, torch.Tensor] = {}
                semi_random_action: torch.Tensor | None = None
                for index, player in enumerate(self.players):
                    indices = active[index]
                    if indices.numel() == 0:
                        continue
                    agent = player.agent
                    if agent.kind == "policy":
                        view = self._perspective_obs(
                            obs.slice_envs(indices), env_team1[indices] == index
                        )
                        actions[index, indices] = get_actions(
                            agent,
                            view,
                            state,
                            int(indices.numel()),
                            self.num_ships,
                            self.device,
                        ).long()
                    elif agent.kind == "semi_random":
                        cache_key = id(agent.agent.scripted_agent)
                        if cache_key not in semi_scripted_cache:
                            semi_scripted_cache[cache_key] = agent.agent.scripted_agent.get_actions(
                                state
                            )
                        scripted_action = semi_scripted_cache[cache_key]
                        if semi_random_action is None:
                            semi_random_action = agent.agent.random_actions_like(scripted_action)
                        actions[index] = agent.agent.mix_actions(
                            scripted_action, semi_random_action
                        ).long()
                    else:
                        actions[index] = get_actions(
                            agent, None, state, self.num_envs, self.num_ships, self.device
                        ).long()

                team0_actions = actions[env_team0, arange]
                team1_actions = actions[env_team1, arange]
                action = torch.where(
                    (state.ship_team_id == 0).unsqueeze(-1), team0_actions, team1_actions
                )
                dones, truncated = self.env.step(action)

            done_any = dones | truncated
            newly_done = done_any & ~finished
            if newly_done.any():
                games += int(newly_done.sum().item())
                self._tally(newly_done, env_team0, env_team1)
                finished |= newly_done
            if done_any.any():
                self.env.reset_envs(done_any, options={"team_sizes": self.team_sizes})
                for index, player in enumerate(self.players):
                    if player.agent.kind == "policy" and active[index].numel() > 0:
                        reset_done_envs(player.agent, done_any[active[index]], self.num_tokens)
        return games

    def _perspective_obs(self, sliced, as_team1: torch.Tensor):
        """Return each env's observation from the acting agent's own side.

        An ego_pass policy only ever learned to act as team 0, so in envs where
        it is playing team 1 it must see mirrored team IDs. The selection is
        done inside a single observation rather than by running the policy twice
        because the agent carries one recurrent state: a second forward pass per
        step would advance that state twice and corrupt it.
        """
        if not self.ego_pass:
            return sliced
        team_id = sliced[ObsKey.TEAM_ID]
        ships = team_id[..., : self.num_ships]
        mirrored = torch.where(ships == 0, 1, torch.where(ships == 1, 0, ships))
        merged = team_id.clone()
        merged[..., : self.num_ships] = torch.where(as_team1.view(-1, 1), mirrored, ships)
        return sliced.update(ObsKey.TEAM_ID, merged)

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


def calibrate_live_curve(
    history: list[dict], ratings: dict[str, float], tie_mode: str = "decisive"
) -> list[dict]:
    """Recover the live and averaged policies' ratings, update by update.

    Each update's record is refit against opponents whose ratings are now known,
    which is what makes this independent of the in-training filter. The averaged
    policy is a second stage: its only opponent is the live policy, so it can
    only be placed once the live rating for that same update is known.
    """
    curve = []
    for record in history:
        counts = record.get("counts") or {}
        opponents, wins, losses = [], [], []
        for label, (win, loss, tie) in counts.items():
            if label in _NON_STATIONARY or label not in ratings:
                continue
            share = 0.0 if tie_mode == "decisive" else 0.5 * tie
            opponents.append(ratings[label])
            wins.append(win + share)
            losses.append(loss + share)
        if not opponents:
            continue
        live, live_stderr = fit_single_rating(
            np.array(opponents), np.array(wins, dtype=float), np.array(losses, dtype=float)
        )
        point = {
            "update": record["update"],
            "global_step": record["global_step"],
            "live_training": record["live"],
            "live_calibrated": live,
            "live_stderr": live_stderr,
            "games": int(sum(wins) + sum(losses)),
        }
        avg_counts = counts.get("avg")
        if avg_counts is not None and np.isfinite(live):
            # Stored from the live policy's perspective, so invert for the avg.
            live_win, live_loss, live_tie = avg_counts
            share = 0.0 if tie_mode == "decisive" else 0.5 * live_tie
            avg, avg_stderr = fit_single_rating(
                np.array([live]),
                np.array([float(live_loss) + share]),
                np.array([float(live_win) + share]),
            )
            point["avg_training"] = record["avg"]
            point["avg_calibrated"] = avg
            point["avg_stderr"] = avg_stderr
        curve.append(point)
    return curve


def training_tie_rates(
    history: list[dict], curve: list[dict], ratings: dict[str, float]
) -> list[dict]:
    """Per-update draw rates from the training record, placed by rating level.

    The tournament only ever pits trained agents against each other, so its
    draw rates all come from the strong end of the scale. The training record is
    the only evidence covering the weak end — where the live policy was still
    near-random and stalemates were common — and that is exactly the range that
    decides whether draws track a matchup's level or its gap.
    """
    live_by_update = {
        point["update"]: point["live_calibrated"]
        for point in curve
        if np.isfinite(point["live_calibrated"])
    }
    rows = []
    for record in history:
        live = live_by_update.get(record["update"])
        if live is None:
            continue
        for label, (win, loss, tie) in (record.get("counts") or {}).items():
            total = win + loss + tie
            if label not in ratings or total < 20:  # too few games to be a rate
                continue
            rows.append(
                {
                    "a": "live",
                    "b": label,
                    "games": int(total),
                    "tie_rate": float(tie / total),
                    "mean_rating": float((live + ratings[label]) / 2.0),
                    "rating_gap": float(abs(live - ratings[label])),
                }
            )
    return rows


def tie_rate_table(
    labels: list[str], wins: np.ndarray, ties: np.ndarray, ratings: np.ndarray
) -> list[dict]:
    """Per-pair tie rates against the pair's rating level, for the draw model."""
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            decisive = wins[i, j] + wins[j, i]
            tied = ties[i, j] + ties[j, i]
            total = decisive + tied
            if total <= 0:
                continue
            rows.append(
                {
                    "a": labels[i],
                    "b": labels[j],
                    "games": int(total),
                    "tie_rate": float(tied / total),
                    "mean_rating": float((ratings[i] + ratings[j]) / 2.0),
                    "rating_gap": float(abs(ratings[i] - ratings[j])),
                }
            )
    return rows


def _load_stored_result(run_dir: Path) -> dict:
    """Read a previous calibration's persisted result, or exit with guidance."""
    path = run_dir / "elo_calibrated.json"
    if not path.exists():
        sys.exit(f"Error: no elo_calibrated.json in '{run_dir}'; run without --refit first.")
    return json.loads(path.read_text())


def run_elo_calibrate_mode(
    run_spec: str,
    ship_config: ShipConfig,
    device: str,
    config: EloCalibrateConfig,
    checkpoint_dir: str = "checkpoints",
    plot: bool = True,
    refit: bool = False,
) -> dict:
    """Re-rate a finished run and write calibrated ratings, curve, and plots.

    With ``refit=True`` no game is played: the raw win/tie matrices persisted by
    a previous calibration are loaded and refit under the current reporting
    conventions. Refitting reuses the stored reference gauge, so the underlying
    fit reproduces the original; only downstream reporting can differ. This is
    the cheap path for a change of anchor or draw convention.
    """
    from boost_and_broadside.modes.elo_stats import find_run_dir

    progress = Progress()
    prior_games = config.prior_games
    run_dir = find_run_dir(run_spec, checkpoint_dir)
    history_path = run_dir / "elo_history.jsonl"

    if refit:
        stored = _load_stored_result(run_dir)
        print(f"\n=== Elo refit from stored matrices (no play): {run_dir.name} ===")
        players = [
            Player(p["label"], ResolvedAgent("stored", None), p["training_elo"], p["global_step"])
            for p in stored["players"]
        ]
        wins = np.asarray(stored["wins_matrix"], dtype=np.float64)
        ties = np.asarray(stored["ties_matrix"], dtype=np.float64)
        directed_outcomes = stored.get("directed_outcomes")
        stats = [BatchStat(**batch) for batch in stored["batches"]]
        target_stderr = float(stored["target_stderr"])
        reference = stored["player_labels"].index(stored["reference"])
        fit = fit_bradley_terry(
            effective_wins(wins, ties, config.tie_mode),
            anchor=reference,
            prior_games=prior_games,
        )
        progress.done(f"refit {len(players)} players from stored matrices")
    else:
        num_envs = config.num_envs
        roster_path = run_dir / "roster.json"
        if not roster_path.exists():
            sys.exit(f"Error: no roster.json in '{run_dir}'; nothing to calibrate.")

        roster = json.loads(roster_path.read_text())
        env_config, model_config, paradigm = _load_run_config(run_dir)
        print(f"\n=== Elo calibration: {run_dir.name} ===")
        print(
            f"  {env_config.num_ships} ships, {paradigm}, "
            f"max_episode_steps={env_config.max_episode_steps}"
        )

        ladder_count = sum(1 for e in roster["entries"] if e["kind"] == "checkpoint")
        progress.stage(f"loading {ladder_count} ladder snapshots...")
        players = build_players(
            run_dir,
            roster,
            model_config,
            ship_config,
            env_config.num_ships,
            device,
            reference_probabilities=config.reference_probabilities,
        )
        progress.done(f"field ({len(players)}): {', '.join(p.label for p in players)}")
        anchor = next(i for i, p in enumerate(players) if p.label == "random")

        tournament = Tournament(players, ship_config, env_config, paradigm, num_envs, device)
        pairs = len(players) * (len(players) - 1) // 2
        progress.stage(
            f"{num_envs} games/batch over {pairs} pairs, target +/-{config.target_stderr:.0f} "
            f"Elo, max {config.max_batches} batches"
        )
        fit, stats, reference = run_tournament(tournament, anchor, config, progress)
        wins, ties = tournament.wins, tournament.ties
        directed_outcomes = tournament.directed_outcomes.tolist()
        target_stderr = config.target_stderr

    # Report on the scripted-anchored scale (scripted = 1000) while keeping the
    # errors from the gauge that is actually resolved. The shift is a constant:
    # it moves every rating together and cancels in any comparison between two
    # of them.
    scripted_index = next(i for i, p in enumerate(players) if p.label == "scripted")
    shifted = fit.ratings - fit.ratings[scripted_index] + SCRIPTED_ANCHOR_ELO
    ratings = {player.label: float(shifted[i]) for i, player in enumerate(players)}
    stderrs = {player.label: float(fit.stderr[i]) for i, player in enumerate(players)}

    # Refit the same match counts under the other convention. No extra play is
    # needed, and the comparison is the diagnostic for a policy that farms draws:
    # half-win scoring grants it parity it never had to win, decisive-only does
    # not, so a large disagreement for one agent is that signature.
    alt_mode = next(mode for mode in TIE_MODES if mode != config.tie_mode)
    alt_fit = fit_bradley_terry(
        effective_wins(wins, ties, alt_mode), anchor=reference, prior_games=prior_games
    )
    alt_shifted = alt_fit.ratings - alt_fit.ratings[scripted_index] + SCRIPTED_ANCHOR_ELO
    alt_ratings = {player.label: float(alt_shifted[i]) for i, player in enumerate(players)}

    history = []
    if history_path.exists():
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    progress.stage(f"refitting {len(history)} update records under both draw conventions...")
    curve = calibrate_live_curve(history, ratings, config.tie_mode)
    alt_curve = {
        point["update"]: point for point in calibrate_live_curve(history, alt_ratings, alt_mode)
    }
    for point in curve:
        other = alt_curve.get(point["update"])
        if other is not None:
            point["live_calibrated_alt"] = other["live_calibrated"]
            point["live_stderr_alt"] = other["live_stderr"]
    progress.done(f"calibrated {len(curve)} live-curve points")

    result = {
        "run": run_dir.name,
        "players": [
            {
                "label": player.label,
                "training_elo": player.training_elo,
                "calibrated_elo": ratings[player.label],
                "calibrated_elo_alt": alt_ratings[player.label],
                "stderr": stderrs[player.label],
                "stderr_alt": float(alt_fit.stderr[index]),
                "global_step": player.global_step,
                "games": float(fit.games[index]),
            }
            for index, player in enumerate(players)
        ],
        # The raw tally is the expensive artifact — everything else is a refit of
        # it. Persisting it means a new draw convention or estimator can be tried
        # without replaying a single match (see ``refit``).
        "player_labels": [player.label for player in players],
        "wins_matrix": np.asarray(wins).tolist(),
        "ties_matrix": np.asarray(ties).tolist(),
        "curve": curve,
        "batches": [vars(stat) for stat in stats],
        "tie_rates": tie_rate_table([p.label for p in players], wins, ties, shifted),
        "training_tie_rates": training_tie_rates(history, curve, ratings),
        "converged": bool(stats[-1].max_stderr <= target_stderr) if stats else False,
        "target_stderr": target_stderr,
        "reference": players[reference].label,
        "tie_mode": config.tie_mode,
        "tie_mode_alt": alt_mode,
        # Every reported rating is measured relative to the reference and then
        # shifted so scripted reads SCRIPTED_ANCHOR_ELO. That shift is itself
        # uncertain by this much, in common across all of them; it cancels
        # between any two ratings.
        "anchor": "scripted",
        "anchor_elo": SCRIPTED_ANCHOR_ELO,
        "anchor_offset_stderr": float(fit.stderr[scripted_index]),
    }
    if directed_outcomes is not None:  # absent from results stored before it existed
        result["directed_outcomes"] = directed_outcomes
    output = run_dir / "elo_calibrated.json"
    output.write_text(json.dumps(result, indent=2))
    progress.done(f"wrote {output}")

    # Also leave the calibrated ratings in the W&B export format, so the same
    # loader and chart system can read them alongside the run's in-training
    # history without reshaping either.
    from boost_and_broadside.modes.elo_calibrate_history import write_chart_data

    for path in write_chart_data(result, run_dir):
        progress.done(f"wrote {path}")

    _print_summary(result)
    if plot:
        from boost_and_broadside.modes.elo_calibrate_plots import write_plots

        progress.stage("rendering plots...")
        written = write_plots(result, run_dir, plot_decisive=config.plot_decisive)
        progress.done(f"wrote {len(written)} plots to {run_dir / 'elo_calibration'}")
        for path in written:
            print(f"    {path.name}")
    return result


def _print_summary(result: dict) -> None:
    """Print the calibrated-vs-training comparison for every ladder player."""
    print(f"\n  {'agent':<20} {'training':>10} {'calibrated':>12} {'+/-':>7} {'drift':>9}")
    print(f"  {'-' * 62}")
    for player in result["players"]:
        training = player["training_elo"]
        calibrated = player["calibrated_elo"]
        drift = f"{calibrated - training:+9.1f}" if training is not None else f"{'—':>9}"
        training_text = f"{training:10.1f}" if training is not None else f"{'—':>10}"
        print(
            f"  {player['label']:<20} {training_text} {calibrated:12.1f} "
            f"{player['stderr']:7.1f} {drift}"
        )
    primary, alt = result.get("tie_mode", "half_win"), result.get("tie_mode_alt", "decisive")
    print(f"\n  {'agent':<20} {primary:>14} {alt:>14} {'difference':>12}")
    print(f"  {'-' * 62}")
    for player in result["players"]:
        other = player.get("calibrated_elo_alt")
        if other is None:
            continue
        print(
            f"  {player['label']:<20} {player['calibrated_elo']:14.1f} {other:14.1f} "
            f"{player['calibrated_elo'] - other:+12.1f}"
        )
    random_rating = next(
        (p["calibrated_elo"] for p in result["players"] if p["label"] == "random"), None
    )
    random_text = (
        f" Random reads {random_rating:.0f} on this scale." if random_rating is not None else ""
    )
    print(
        f"\n  Errors are relative to '{result['reference']}'. Ratings are shifted so the "
        f"scripted\n  controller reads {result['anchor_elo']:.0f}; that shift is uncertain by "
        f"+/-{result['anchor_offset_stderr']:.0f} in common across\n  every rating above and "
        f"cancels whenever two are compared.{random_text}\n  Random's own link to the field is "
        "coarse because every trained agent beats it\n  decisively — nothing plays near its "
        "level, so those games say little however\n  they are scored. The live curve does not "
        "have this problem: early in training\n  it drew against random constantly, and draws "
        "are informative."
    )
    # Spacing is the part a shared offset cannot flatter, so report it directly.
    # Random is excluded: the step from it to the first rung is the coarse anchor
    # link, not a rung-to-rung gap, and listing it here would read as one.
    ladder = [
        p for p in result["players"] if p["training_elo"] is not None and p["label"] != "random"
    ]
    ladder.sort(key=lambda p: p["global_step"] or 0)
    if len(ladder) > 1:
        print(f"\n  {'rung-to-rung gap':<20} {'training':>10} {'calibrated':>12} {'delta':>9}")
        print(f"  {'-' * 54}")
        for previous, current in zip(ladder, ladder[1:]):
            training_gap = current["training_elo"] - previous["training_elo"]
            calibrated_gap = current["calibrated_elo"] - previous["calibrated_elo"]
            print(
                f"  {current['label']:<20} {training_gap:10.1f} {calibrated_gap:12.1f} "
                f"{calibrated_gap - training_gap:+9.1f}"
            )
