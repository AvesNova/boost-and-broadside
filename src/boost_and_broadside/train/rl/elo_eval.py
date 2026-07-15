"""Continuous in-training ELO evaluation and shared rating math."""

from collections import deque

import torch

from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EloEvalConfig, EnvConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import observation_from_state
from boost_and_broadside.env.obstacle_cache import ObstacleCache
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
    init_hidden,
    reset_done_envs,
)

_ELO_RATING_SCALE = 400.0


def expected_score(
    rating: float | torch.Tensor, opponent_rating: float | torch.Tensor
) -> float | torch.Tensor:
    """Return the standard logistic ELO expected score."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / _ELO_RATING_SCALE))


def optimal_eval_ratio(
    training_elo: float | torch.Tensor, scripted_elo: float
) -> float | torch.Tensor:
    """Return the information-proportional fraction of games routed to random."""
    random_win_rate = expected_score(training_elo, 0.0)
    scripted_win_rate = expected_score(training_elo, scripted_elo)
    random_variance = random_win_rate * (1.0 - random_win_rate)
    scripted_variance = scripted_win_rate * (1.0 - scripted_win_rate)
    total = random_variance + scripted_variance
    if isinstance(total, torch.Tensor):
        return torch.where(
            total <= 1e-8,
            torch.full_like(total, 0.5),
            random_variance / total.clamp(min=1e-8),
        )
    return 0.5 if total <= 1e-8 else random_variance / total


class EloEvaluator:
    """Run live-vs-anchor, live-vs-average, and average-vs-anchor matches."""

    def __init__(
        self,
        config: EloEvalConfig,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        device: torch.device,
        obstacle_cache: ObstacleCache | None,
        live_policy: MVPPolicy,
        avg_policy: MVPPolicy,
        scripted_agent: StochasticScriptedAgent | None,
        num_ships: int,
        num_tokens: int,
        live_elo: float,
        avg_elo: float,
        random_window: deque[float],
        scripted_window: deque[float],
        avg_vs_scripted_window: deque[float],
        live_vs_avg_window: deque[float],
    ) -> None:
        self.config = config
        self.device = device
        self.ship_config = ship_config
        self.num_ships = num_ships
        self.num_tokens = num_tokens
        self.matchup_size = config.envs_per_matchup
        self.batch_size = 3 * self.matchup_size
        self.live_anchor_slice = slice(0, self.matchup_size)
        self.live_avg_slice = slice(self.matchup_size, 2 * self.matchup_size)
        self.avg_anchor_slice = slice(2 * self.matchup_size, 3 * self.matchup_size)

        self.env = TensorEnv(
            self.batch_size,
            ship_config,
            env_config,
            device,
            obstacle_cache,
        )
        self.env.reset()
        self.env.state.step_count.random_(0, env_config.max_episode_steps)

        self.live_agent = ResolvedAgent("policy", live_policy)
        self.avg_agent = ResolvedAgent("policy", avg_policy)
        self.scripted_agent = (
            ResolvedAgent("scripted", scripted_agent) if scripted_agent is not None else None
        )
        self.random_agent = ResolvedAgent("random", None)
        init_hidden(self.live_agent, 2 * self.matchup_size, num_tokens, device)
        init_hidden(self.avg_agent, 2 * self.matchup_size, num_tokens, device)

        random_ratio = optimal_eval_ratio(live_elo, config.scripted_elo)
        self.anchor_is_scripted = torch.rand(self.batch_size, device=device) > random_ratio
        self.live_elo = torch.tensor(float(live_elo), device=device, dtype=torch.float64)
        self.avg_elo = torch.tensor(float(avg_elo), device=device, dtype=torch.float64)
        self._win_history: list[torch.Tensor] = []
        self._done_history: list[torch.Tensor] = []
        self._anchor_history: list[torch.Tensor] = []
        self.random_window = random_window
        self.scripted_window = scripted_window
        self.avg_vs_scripted_window = avg_vs_scripted_window
        self.live_vs_avg_window = live_vs_avg_window

    def step(self, rollout_step: int, avg_active: bool) -> None:
        """Advance evaluation slots on the configured rollout cadence."""
        if rollout_step % self.config.step_interval != 0:
            return

        size = self.matchup_size
        live_anchor = self.live_anchor_slice
        avg_anchor = self.avg_anchor_slice

        with torch.no_grad():
            state = self.env.state
            obs = observation_from_state(state, self.ship_config)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action_live = get_actions(
                    self.live_agent,
                    obs.slice_envs(slice(0, 2 * size)),
                    state,
                    2 * size,
                    self.num_ships,
                    self.device,
                ).long()
                action_avg = get_actions(
                    self.avg_agent,
                    obs.slice_envs(slice(size, 3 * size)),
                    state,
                    2 * size,
                    self.num_ships,
                    self.device,
                ).long()

                if self.scripted_agent is not None:
                    action_scripted = torch.cat(
                        [
                            get_actions(
                                self.scripted_agent,
                                None,
                                state.slice_envs(slice(0, size)),
                                size,
                                self.num_ships,
                                self.device,
                            ),
                            get_actions(
                                self.scripted_agent,
                                None,
                                state.slice_envs(slice(2 * size, 3 * size)),
                                size,
                                self.num_ships,
                                self.device,
                            ),
                        ],
                        dim=0,
                    ).long()
                else:
                    action_scripted = torch.zeros(
                        2 * size,
                        self.num_ships,
                        3,
                        dtype=torch.long,
                        device=self.device,
                    )
                action_random = get_actions(
                    self.random_agent,
                    None,
                    state,
                    2 * size,
                    self.num_ships,
                    self.device,
                ).long()
                anchor_flags = torch.cat(
                    [self.anchor_is_scripted[live_anchor], self.anchor_is_scripted[avg_anchor]]
                )
                action_anchor = torch.where(
                    anchor_flags.view(-1, 1, 1), action_scripted, action_random
                )

                action_team0 = torch.cat([action_live, action_avg[size:]], dim=0)
                action_team1 = torch.cat(
                    [action_anchor[:size], action_avg[:size], action_anchor[size:]], dim=0
                )
                action = torch.where(
                    (state.ship_team_id == 0).unsqueeze(-1), action_team0, action_team1
                )
                dones, truncated = self.env.step(action)
                done_any = dones | truncated

            alive = self.env.state.ship_alive
            team = self.env.state.ship_team_id
            team0_alive = (alive & (team == 0)).any(dim=1)
            team1_alive = (alive & (team == 1)).any(dim=1)
            team0_won = done_any & team0_alive & ~team1_alive
            team1_won = done_any & team1_alive & ~team0_alive
            tied = done_any & ~team0_won & ~team1_won
            score = team0_won.float() + 0.5 * tied.float()
            done_float = done_any.float()
            anchor_elo = self.anchor_is_scripted.float() * self.config.scripted_elo

            expected_live = expected_score(self.live_elo, anchor_elo[live_anchor])
            self.live_elo = (
                self.live_elo
                + (
                    self.config.k_factor
                    * (score[live_anchor] - expected_live)
                    * done_float[live_anchor]
                ).sum()
            )

            if avg_active:
                expected_avg = expected_score(self.avg_elo, anchor_elo[avg_anchor])
                self.avg_elo = (
                    self.avg_elo
                    + (
                        self.config.k_factor
                        * (score[avg_anchor] - expected_avg)
                        * done_float[avg_anchor]
                    ).sum()
                )

            self._win_history.append(team0_won.float())
            self._done_history.append(done_any)
            self._anchor_history.append(self.anchor_is_scripted.clone())

            random_ratio = optimal_eval_ratio(self.live_elo, self.config.scripted_elo)
            self.anchor_is_scripted = torch.where(
                done_any,
                torch.rand(self.batch_size, device=self.device) > random_ratio,
                self.anchor_is_scripted,
            )
            self.env.reset_envs(done_any)
            reset_done_envs(self.live_agent, done_any[: 2 * size], self.num_tokens)
            reset_done_envs(self.avg_agent, done_any[size:], self.num_tokens)

    def flush(self, avg_active: bool) -> tuple[float, float]:
        """Flush GPU ratings and outcome history to CPU once per PPO update."""
        live_elo = float(self.live_elo.item())
        avg_elo = float(self.avg_elo.item())
        if not self._win_history:
            return live_elo, avg_elo

        wins = torch.stack(self._win_history).cpu()
        dones = torch.stack(self._done_history).cpu()
        anchors = torch.stack(self._anchor_history).cpu()
        self._win_history.clear()
        self._done_history.clear()
        self._anchor_history.clear()

        for index in range(wins.shape[0]):
            win, done, anchor = wins[index], dones[index], anchors[index]
            live_done = done[self.live_anchor_slice]
            live_anchor = anchor[self.live_anchor_slice]
            live_win = win[self.live_anchor_slice]
            self.scripted_window.extend(live_win[live_done & live_anchor].tolist())
            self.random_window.extend(live_win[live_done & ~live_anchor].tolist())
            if avg_active:
                avg_done = done[self.avg_anchor_slice]
                avg_anchor = anchor[self.avg_anchor_slice]
                self.avg_vs_scripted_window.extend(
                    win[self.avg_anchor_slice][avg_done & avg_anchor].tolist()
                )
                self.live_vs_avg_window.extend(
                    win[self.live_avg_slice][done[self.live_avg_slice]].tolist()
                )
        return live_elo, avg_elo

    def seed_avg_elo_from_live(self) -> None:
        """Seed the first averaged-policy rating from the identical live snapshot."""
        self.avg_elo = self.live_elo.clone()
