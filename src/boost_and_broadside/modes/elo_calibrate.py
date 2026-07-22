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
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
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

# Opponents that are not stationary and so cannot be tournament players. "avg"
# changes every update, exactly like the live policy it is measured against.
_NON_STATIONARY = frozenset({"avg"})


@dataclass
class Player:
    """One stationary agent in the calibration tournament."""

    label: str
    agent: ResolvedAgent
    training_elo: float | None  # rating the run itself assigned, for comparison
    global_step: int | None  # None for agents with no place on the timeline


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
    env_config = EnvConfig(**checkpoint["env_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    paradigm = checkpoint.get("train_config", {}).get("paradigm", "ego_pass")
    return env_config, model_config, paradigm


def _load_ladder_policy(
    path: Path, model_config: ModelConfig, ship_config: ShipConfig, num_ships: int, device: str
):
    """Build an eval-mode policy from a ladder snapshot."""
    from boost_and_broadside.models.mvp.policy import MVPPolicy
    from boost_and_broadside.train.rl.features import build_standard_coordinator

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    coordinator = build_standard_coordinator(ship_config)
    num_components = checkpoint["policy_state_dict"]["value_head_local.3.weight"].shape[0]
    policy = MVPPolicy(
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


def build_players(
    run_dir: Path,
    roster: dict,
    model_config: ModelConfig,
    ship_config: ShipConfig,
    num_ships: int,
    device: str,
) -> list[Player]:
    """Assemble the tournament field: random, the scripted agent, every ladder snapshot.

    The random anchor comes first so it can serve as the rating gauge. Only
    agents the live policy actually played are included — every extra player
    dilutes the batch budget without informing the curve being calibrated.
    """
    players = [Player("random", ResolvedAgent("random", None), 0.0, 0)]
    scripted = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    players.append(Player("scripted", ResolvedAgent("scripted", scripted), None, None))

    entries = [e for e in roster["entries"] if e["kind"] == "checkpoint"]
    for entry in sorted(entries, key=lambda e: e["global_step"]):
        path = Path(entry["path"])
        if not path.exists():  # roster may outlive a pruned file
            path = run_dir / path.name
        if not path.exists():
            print(f"  [warn] missing ladder snapshot for {entry['label']}, skipping")
            continue
        policy = _load_ladder_policy(path, model_config, ship_config, num_ships, device)
        players.append(
            Player(entry["label"], ResolvedAgent("policy", policy), entry["elo"], entry["global_step"])
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
        self.num_tokens = env_config.num_ships + env_config.num_obstacles
        self.team_sizes = (self.num_ships // 2, self.num_ships - self.num_ships // 2)

        self.env = TensorEnv(num_envs, ship_config, env_config, self.device)
        self.wins = np.zeros((self.size, self.size), dtype=np.float64)
        self.ties = np.zeros((self.size, self.size), dtype=np.float64)

    def pair_games(self) -> np.ndarray:
        """Symmetric decisive-game counts per pair."""
        return self.wins + self.wins.T

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

    def play_batch(self, allocation: np.ndarray) -> int:
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

        while not finished.all():
            state = self.env.state
            obs = observation_from_state(state, self.ship_config)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
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
                            agent, view, state, int(indices.numel()),
                            self.num_ships, self.device,
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
        for outcome, target in ((team0_won, self.wins), (tied, self.ties)):
            if not outcome.any():
                continue
            counts = torch.bincount(flat[outcome], minlength=self.size**2)
            target += counts.reshape(self.size, self.size).cpu().numpy().astype(np.float64)
        if team1_won.any():
            flipped = self.size * env_team1 + env_team0
            counts = torch.bincount(flipped[team1_won], minlength=self.size**2)
            self.wins += counts.reshape(self.size, self.size).cpu().numpy().astype(np.float64)


def choose_reference(tournament: Tournament, ratings: np.ndarray, random_index: int) -> int:
    """Pick the player to measure everything else against.

    Not the random anchor. Random's games are all either draws (28% of them,
    against similarly hopeless opponents) or foregone conclusions (99.6% decisive
    losses to anything trained), and both carry almost no information about where
    it sits: pinning a rating to +/-10 through that link would take on the order
    of 80,000 games. Measuring in a gauge the tournament cannot resolve would put
    that irreducible uncertainty into every rating at once, hiding the fact that
    the ladder's internal spacing is known far better than its absolute height.

    The most information-dense player is chosen instead, so reported standard
    errors describe what the tournament actually determines.
    """
    information = np.diag(fisher_information(tournament.pair_games(), ratings)).copy()
    information[random_index] = -1.0
    return int(np.argmax(information))


def run_tournament(
    tournament: Tournament,
    random_index: int,
    target_stderr: float,
    max_batches: int,
    prior_games: float,
) -> tuple[RatingFit, list[BatchStat], int]:
    """Play adaptive batches until every rating is pinned or the budget runs out.

    Allocation, standard errors, and the convergence test all use the reference
    gauge rather than the random anchor. Optimizing against random would sink the
    whole budget into the one link that cannot be improved at any realistic cost.
    """
    stats: list[BatchStat] = []
    size = tournament.size
    reference = random_index
    fit = fit_bradley_terry(np.zeros((size, size)), anchor=reference, prior_games=prior_games)
    cumulative = 0
    for batch in range(1, max_batches + 1):
        allocation = allocate_games(
            tournament.pair_games(), fit.ratings, anchor=reference, budget=tournament.num_envs
        )
        started = time.perf_counter()
        games = tournament.play_batch(allocation)
        elapsed = time.perf_counter() - started
        cumulative += games
        if batch == 1:
            provisional = fit_bradley_terry(
                tournament.wins, anchor=random_index, prior_games=prior_games
            )
            reference = choose_reference(tournament, provisional.ratings, random_index)
            print(f"  reference gauge: {tournament.players[reference].label}")
        fit = fit_bradley_terry(tournament.wins, anchor=reference, prior_games=prior_games)

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
        print(
            f"  batch {batch:2d}  games={games:6d}  cumulative={cumulative:7d}  "
            f"max SE={stat.max_stderr:6.2f}  mean SE={stat.mean_stderr:5.2f}  "
            f"({elapsed:.1f}s)"
        )
        if worst <= target_stderr:
            print(f"  converged: every rating within +/-{target_stderr:.1f} of the reference")
            break
    else:
        print(f"  budget exhausted at {max_batches} batches before reaching the target")
    return fit, stats, reference


def calibrate_live_curve(
    history: list[dict], ratings: dict[str, float]
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
        for label, (win, loss, _tie) in counts.items():
            if label in _NON_STATIONARY or label not in ratings:
                continue
            opponents.append(ratings[label])
            wins.append(win)
            losses.append(loss)
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
            live_win, live_loss, _ = avg_counts
            avg, avg_stderr = fit_single_rating(
                np.array([live]), np.array([float(live_loss)]), np.array([float(live_win)])
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


def tie_rate_table(tournament: Tournament, ratings: np.ndarray) -> list[dict]:
    """Per-pair tie rates against the pair's rating level, for the draw model."""
    rows = []
    for i in range(tournament.size):
        for j in range(i + 1, tournament.size):
            decisive = tournament.wins[i, j] + tournament.wins[j, i]
            ties = tournament.ties[i, j] + tournament.ties[j, i]
            total = decisive + ties
            if total <= 0:
                continue
            rows.append(
                {
                    "a": tournament.players[i].label,
                    "b": tournament.players[j].label,
                    "games": int(total),
                    "tie_rate": float(ties / total),
                    "mean_rating": float((ratings[i] + ratings[j]) / 2.0),
                    "rating_gap": float(abs(ratings[i] - ratings[j])),
                }
            )
    return rows


def run_elo_calibrate_mode(
    run_spec: str,
    ship_config: ShipConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    num_envs: int = 4096,
    target_stderr: float = 10.0,
    max_batches: int = 12,
    prior_games: float = 1.0,
    plot: bool = True,
) -> dict:
    """Re-rate a finished run and write calibrated ratings, curve, and plots."""
    from boost_and_broadside.modes.elo_stats import find_run_dir

    run_dir = find_run_dir(run_spec, checkpoint_dir)
    roster_path = run_dir / "roster.json"
    history_path = run_dir / "elo_history.jsonl"
    if not roster_path.exists():
        sys.exit(f"Error: no roster.json in '{run_dir}'; nothing to calibrate.")

    roster = json.loads(roster_path.read_text())
    env_config, model_config, paradigm = _load_run_config(run_dir)
    print(f"\n=== ELO calibration: {run_dir.name} ===")
    print(
        f"  {env_config.num_ships} ships, {paradigm}, "
        f"max_episode_steps={env_config.max_episode_steps}"
    )

    players = build_players(
        run_dir, roster, model_config, ship_config, env_config.num_ships, device
    )
    print(f"  field: {', '.join(p.label for p in players)}")
    anchor = next(i for i, p in enumerate(players) if p.label == "random")

    tournament = Tournament(
        players, ship_config, env_config, paradigm, num_envs, device
    )
    print(f"  {num_envs} envs/batch, target +/-{target_stderr:.0f} ELO, max {max_batches} batches")
    fit, stats, reference = run_tournament(
        tournament, anchor, target_stderr, max_batches, prior_games
    )

    # Report on the familiar scale (random = 0) while keeping the errors from the
    # gauge that is actually resolved. The shift is a constant: it moves every
    # rating together and cancels in any comparison between two of them.
    anchor_offset = float(fit.ratings[anchor])
    shifted = fit.ratings - anchor_offset
    ratings = {player.label: float(shifted[i]) for i, player in enumerate(players)}
    stderrs = {player.label: float(fit.stderr[i]) for i, player in enumerate(players)}

    history = []
    if history_path.exists():
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    curve = calibrate_live_curve(history, ratings)
    print(f"  calibrated {len(curve)} live-curve points from {len(history)} update records")

    result = {
        "run": run_dir.name,
        "players": [
            {
                "label": player.label,
                "training_elo": player.training_elo,
                "calibrated_elo": ratings[player.label],
                "stderr": stderrs[player.label],
                "global_step": player.global_step,
                "games": float(fit.games[index]),
            }
            for index, player in enumerate(players)
        ],
        "curve": curve,
        "batches": [vars(stat) for stat in stats],
        "tie_rates": tie_rate_table(tournament, shifted),
        "training_tie_rates": training_tie_rates(history, curve, ratings),
        "converged": bool(stats[-1].max_stderr <= target_stderr) if stats else False,
        "target_stderr": target_stderr,
        "reference": players[reference].label,
        # Every reported rating is measured relative to the reference and then
        # shifted so random reads 0. That shift is itself uncertain by this much,
        # in common across all of them; it cancels between any two ratings.
        "anchor_offset_stderr": float(fit.stderr[anchor]),
    }
    output = run_dir / "elo_calibrated.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"  wrote {output}")

    _print_summary(result)
    if plot:
        from boost_and_broadside.modes.elo_calibrate_plots import write_plots

        for path in write_plots(result, run_dir):
            print(f"  wrote {path}")
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
    print(
        f"\n  Errors are relative to '{result['reference']}'. Shifting the scale so random "
        f"reads 0\n  adds +/-{result['anchor_offset_stderr']:.0f} in common to every rating "
        "above — that link runs through\n  games random cannot meaningfully contest, so it "
        "stays coarse at any budget.\n  It cancels whenever two ratings are compared."
    )
    # Spacing is the part a shared offset cannot flatter, so report it directly.
    # Random is excluded: the step from it to the first rung is the coarse anchor
    # link, not a rung-to-rung gap, and listing it here would read as one.
    ladder = [
        p
        for p in result["players"]
        if p["training_elo"] is not None and p["label"] != "random"
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
