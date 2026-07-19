"""Continuous in-training ELO ladder evaluation and shared rating math.

The live policy's rating is measured against a ladder of frozen references
rather than the scripted agent, so it stays informative after the live policy
saturates the scripted matchup. Matchup slots (each ``envs_per_matchup`` wide,
in batch order):

    0  live vs anchor      — updates live only (anchor ratings are frozen)
    1  live vs floating    — zero-sum between live and the floating checkpoint
    2  live vs scripted    — win-rate window + one-way scripted rating update
    3  live vs avg         — win-rate window + one-way avg rating update
    4  floating vs anchor  — one-way floating-checkpoint rating update

The live policy plays team 0 in slots 0-3; the floating checkpoint plays team 0
in slot 4. Anchors are the newest frozen ladder entries (just the random agent
until checkpoints freeze). Per-episode anchor assignment is sampled proportional
to the Bernoulli variance of the expected score, concentrating eval games where
they carry the most rating information.

Before the first milestone there is no floating checkpoint: slot 1 falls back to
extra live-vs-random games and slot 4 idles (random-vs-random play, ignored).
"""

from collections import deque
from dataclasses import dataclass

import torch

from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EloEvalConfig, EnvConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import MVPObservation, observation_from_state
from boost_and_broadside.env.obstacle_cache import ObstacleCache
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
    init_hidden,
    reset_done_envs,
)

_ELO_RATING_SCALE = 400.0

# Frozen ladder entries the live and floating ratings are measured against.
# Two anchors keep each new frozen rating tied to more than one chain link,
# damping the random-walk error a single-link ladder accumulates.
MAX_ANCHORS = 2


def expected_score(
    rating: float | torch.Tensor, opponent_rating: float | torch.Tensor
) -> float | torch.Tensor:
    """Return the standard logistic ELO expected score."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / _ELO_RATING_SCALE))


def information_weights(
    rating: float | torch.Tensor, opponent_ratings: torch.Tensor
) -> torch.Tensor:
    """Return per-opponent eval-game weights ∝ the variance of the expected score.

    Games against near-equal opponents carry the most rating information
    (score variance p·(1-p) peaks at p=0.5); lopsided matchups get fewer games.

    Args:
        rating:           Rating of the player being measured.
        opponent_ratings: (A,) candidate opponent ratings.

    Returns:
        (A,) weights summing to 1 (uniform when all matchups are saturated).
    """
    win_prob = expected_score(rating, opponent_ratings)  # (A,)
    variance = win_prob * (1.0 - win_prob)  # (A,)
    total = variance.sum()
    uniform = torch.full_like(variance, 1.0 / variance.numel())
    return torch.where(total <= 1e-8, uniform, variance / total.clamp(min=1e-8))


@dataclass(frozen=True)
class EloSnapshot:
    """CPU copy of the evaluator's ratings, flushed once per PPO update."""

    live_elo: float
    avg_elo: float
    scripted_elo: float
    floating_elo: float | None  # None before the first milestone snapshot
    floating_games: int  # rated games the floating checkpoint has accumulated


class EloEvaluator:
    """Run the five ladder matchup slots on the configured rollout cadence."""

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
        ego_pass: bool,
        live_elo: float,
        avg_elo: float,
        scripted_elo: float,
        anchors: list[tuple[MVPPolicy | None, float]],
        floating: tuple[MVPPolicy, float] | None,
        floating_games: int,
        random_window: deque[float],
        ladder_window: deque[float],
        floating_window: deque[float],
        scripted_window: deque[float],
        live_vs_avg_window: deque[float],
    ) -> None:
        """Build the eval battery from the current ladder state.

        Args:
            anchors:  (policy, frozen_elo) per anchor, oldest first; a None
                      policy is the random agent. At most MAX_ANCHORS entries.
            floating: (policy, elo) of the floating checkpoint, or None before
                      the first milestone (then anchors must be just random).
            floating_games: Rated games already accumulated by the floating
                      checkpoint (restored on resume).
        """
        assert 1 <= len(anchors) <= MAX_ANCHORS, f"expected 1-{MAX_ANCHORS} anchors, got {anchors}"
        assert floating is not None or (len(anchors) == 1 and anchors[0][0] is None), (
            "without a floating checkpoint the ladder must consist of only the random anchor"
        )
        self.config = config
        self.device = device
        self.ship_config = ship_config
        self.num_ships = num_ships
        self.num_tokens = num_tokens
        self.ego_pass = ego_pass
        self.max_episode_steps = env_config.max_episode_steps
        self.matchup_size = config.envs_per_matchup
        self.batch_size = 5 * self.matchup_size

        self.env = TensorEnv(
            self.batch_size,
            ship_config,
            env_config,
            device,
            obstacle_cache,
        )
        self.env.reset()
        self.env.state.step_count.random_(0, env_config.max_episode_steps)

        size = self.matchup_size
        self.live_agent = ResolvedAgent("policy", live_policy)
        self.avg_agent = ResolvedAgent("policy", avg_policy)
        self.scripted_agent = (
            ResolvedAgent("scripted", scripted_agent) if scripted_agent is not None else None
        )
        self.random_agent = ResolvedAgent("random", None)
        init_hidden(self.live_agent, 4 * size, num_tokens, device)
        init_hidden(self.avg_agent, size, num_tokens, device)

        self.live_elo = torch.tensor(float(live_elo), device=device, dtype=torch.float64)
        self.avg_elo = torch.tensor(float(avg_elo), device=device, dtype=torch.float64)
        self.scripted_elo = torch.tensor(float(scripted_elo), device=device, dtype=torch.float64)
        self._anchor_specs = list(anchors)
        self._floating_policy = floating[0] if floating is not None else None
        floating_elo = floating[1] if floating is not None else 0.0
        self.floating_elo = torch.tensor(float(floating_elo), device=device, dtype=torch.float64)
        self.floating_games = torch.tensor(
            float(floating_games), device=device, dtype=torch.float64
        )
        self._anchor_idx_live = torch.zeros(size, dtype=torch.long, device=device)
        self._anchor_idx_float = torch.zeros(size, dtype=torch.long, device=device)
        self._build_ladder_agents()

        self._win_history: list[torch.Tensor] = []
        self._done_history: list[torch.Tensor] = []
        self._anchor_idx_history: list[torch.Tensor] = []
        self.random_window = random_window
        self.ladder_window = ladder_window
        self.floating_window = floating_window
        self.scripted_window = scripted_window
        self.live_vs_avg_window = live_vs_avg_window

    # ------------------------------------------------------------------
    # Ladder state
    # ------------------------------------------------------------------

    def _build_ladder_agents(self) -> None:
        """(Re)build anchor/floating agents and rating tensors from the specs."""
        size = self.matchup_size
        self._anchor_elos = torch.tensor(
            [elo for _, elo in self._anchor_specs], device=self.device, dtype=torch.float64
        )  # (A,)
        self._anchor_agents_live: list[ResolvedAgent | None] = []
        self._anchor_agents_float: list[ResolvedAgent | None] = []
        for policy, _ in self._anchor_specs:
            if policy is None:
                self._anchor_agents_live.append(None)
                self._anchor_agents_float.append(None)
                continue
            agent_live = ResolvedAgent("policy", policy)
            agent_float = ResolvedAgent("policy", policy)
            init_hidden(agent_live, size, self.num_tokens, self.device)
            init_hidden(agent_float, size, self.num_tokens, self.device)
            self._anchor_agents_live.append(agent_live)
            self._anchor_agents_float.append(agent_float)

        if self._floating_policy is not None:
            self.float_opp_agent = ResolvedAgent("policy", self._floating_policy)
            self.float_pro_agent = ResolvedAgent("policy", self._floating_policy)
            init_hidden(self.float_opp_agent, size, self.num_tokens, self.device)
            init_hidden(self.float_pro_agent, size, self.num_tokens, self.device)
        else:
            self.float_opp_agent = None
            self.float_pro_agent = None

    def promote_floating(self, snapshot_policy: MVPPolicy) -> None:
        """Freeze the floating checkpoint into the anchor set and start a new one.

        The caller freezes the matching roster entry; here the current floating
        policy becomes the newest anchor at its settled rating (dropping the
        oldest anchor beyond MAX_ANCHORS) and the fresh snapshot starts floating
        at the live policy's current rating.
        """
        if self._floating_policy is not None:
            self._anchor_specs.append((self._floating_policy, float(self.floating_elo.item())))
            del self._anchor_specs[:-MAX_ANCHORS]
        self._floating_policy = snapshot_policy
        self.floating_elo = self.live_elo.clone()
        self.floating_games = torch.zeros((), device=self.device, dtype=torch.float64)
        self._build_ladder_agents()
        self._reset_ladder_slots()

    def _reset_ladder_slots(self) -> None:
        """Hard-reset the episodes of every slot whose participants just changed."""
        size = self.matchup_size
        mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        mask[0 : 2 * size] = True  # slots 0-1: anchor set / floating opponent changed
        mask[4 * size :] = True  # slot 4: floating protagonist changed
        self.env.reset_envs(mask)
        # Stagger episode ends so rating updates arrive continuously.
        staggered = torch.randint_like(self.env.state.step_count, 0, self.max_episode_steps)
        self.env.state.step_count[mask] = staggered[mask]
        reset_done_envs(self.live_agent, mask[: 4 * size], self.num_tokens)
        self._anchor_idx_live.zero_()
        self._anchor_idx_float.zero_()
        self._resample_anchor_assignments(mask)

    def seed_avg_elo_from_live(self) -> None:
        """Seed the first averaged-policy rating from the identical live snapshot."""
        self.avg_elo = self.live_elo.clone()

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _opponent_obs(self, obs: MVPObservation, lo: int, hi: int) -> MVPObservation:
        """Return the team-1 perspective for policy opponents in envs [lo, hi)."""
        sliced = obs.slice_envs(slice(lo, hi))
        return sliced.flip_team(self.num_ships) if self.ego_pass else sliced

    def _anchor_actions(
        self,
        obs: MVPObservation,
        lo: int,
        hi: int,
        agents: list[ResolvedAgent | None],
    ) -> torch.Tensor:
        """Anchor-side actions for envs [lo, hi): every anchor acts, the
        per-episode assignment selects. Non-acting anchors still observe so
        their hidden states stay valid when the assignment changes."""
        size = hi - lo
        state = self.env.state.slice_envs(slice(lo, hi))
        per_anchor = []
        for agent in agents:
            if agent is None:
                actions = get_actions(
                    self.random_agent, None, state, size, self.num_ships, self.device
                )
            else:
                actions = get_actions(
                    agent,
                    self._opponent_obs(obs, lo, hi),
                    state,
                    size,
                    self.num_ships,
                    self.device,
                )
            per_anchor.append(actions.long())  # (B_slot, N, 3)
        if len(per_anchor) == 1:
            return per_anchor[0]
        idx = self._anchor_idx_live if lo == 0 else self._anchor_idx_float  # (B_slot,)
        return torch.where(idx.view(-1, 1, 1) == 1, per_anchor[1], per_anchor[0])

    def _compute_team_actions(self, obs: MVPObservation) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (team0, team1) actions, each (5·size, N, 3), for one eval step."""
        size = self.matchup_size
        state = self.env.state

        action_live = get_actions(
            self.live_agent,
            obs.slice_envs(slice(0, 4 * size)),
            state,
            4 * size,
            self.num_ships,
            self.device,
        ).long()  # (4·size, N, 3)
        action_avg = get_actions(
            self.avg_agent,
            self._opponent_obs(obs, 3 * size, 4 * size),
            state,
            size,
            self.num_ships,
            self.device,
        ).long()

        if self.scripted_agent is not None:
            action_scripted = get_actions(
                self.scripted_agent,
                None,
                state.slice_envs(slice(2 * size, 3 * size)),
                size,
                self.num_ships,
                self.device,
            ).long()
        else:  # idle slot — outcomes are never scored
            action_scripted = self._random_actions(size)

        action_anchor_live = self._anchor_actions(obs, 0, size, self._anchor_agents_live)
        if self.float_pro_agent is not None:
            action_float_opp = get_actions(
                self.float_opp_agent,
                self._opponent_obs(obs, size, 2 * size),
                state,
                size,
                self.num_ships,
                self.device,
            ).long()
            action_float_pro = get_actions(
                self.float_pro_agent,
                obs.slice_envs(slice(4 * size, 5 * size)),
                state,
                size,
                self.num_ships,
                self.device,
            ).long()
            action_anchor_float = self._anchor_actions(
                obs, 4 * size, 5 * size, self._anchor_agents_float
            )
        else:
            # Fallback: slot 1 plays the random anchor (extra live rating games),
            # slot 4 idles as unscored random-vs-random play.
            action_float_opp = self._random_actions(size)
            action_float_pro = self._random_actions(size)
            action_anchor_float = self._random_actions(size)

        action_team0 = torch.cat([action_live, action_float_pro], dim=0)  # (5·size, N, 3)
        action_team1 = torch.cat(
            [
                action_anchor_live,
                action_float_opp,
                action_scripted,
                action_avg,
                action_anchor_float,
            ],
            dim=0,
        )  # (5·size, N, 3)
        return action_team0, action_team1

    def _random_actions(self, num_envs: int) -> torch.Tensor:
        """Uniform random actions for idle or random-anchor slots."""
        return get_actions(
            self.random_agent, None, self.env.state, num_envs, self.num_ships, self.device
        ).long()

    def step(self, rollout_step: int, avg_active: bool) -> None:
        """Advance evaluation slots on the configured rollout cadence."""
        if rollout_step % self.config.step_interval != 0:
            return

        with torch.no_grad():
            state = self.env.state
            obs = observation_from_state(state, self.ship_config)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action_team0, action_team1 = self._compute_team_actions(obs)
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
            score = team0_won.float() + 0.5 * tied.float()  # (5·size,)
            self._apply_rating_updates(score, done_any.float(), avg_active)

            self._win_history.append(team0_won.float())
            self._done_history.append(done_any)
            self._anchor_idx_history.append(self._anchor_idx_live.clone())

            self._resample_anchor_assignments(done_any)
            self.env.reset_envs(done_any)
            self._reset_agent_hiddens(done_any)

    def _apply_rating_updates(
        self, score: torch.Tensor, done_float: torch.Tensor, avg_active: bool
    ) -> None:
        """Apply per-slot ELO updates for episodes that just finished."""
        size = self.matchup_size
        k = self.config.k_factor
        live_before = self.live_elo

        anchor_elo_live = self._anchor_elos[self._anchor_idx_live]  # (size,)
        slot0 = slice(0, size)
        delta_live = (
            k * (score[slot0] - expected_score(live_before, anchor_elo_live)) * done_float[slot0]
        ).sum()

        slot1 = slice(size, 2 * size)
        if self.float_pro_agent is None:
            # Slot 1 fallback: extra games against the (sole, random) anchor.
            delta_live = (
                delta_live
                + (
                    k
                    * (score[slot1] - expected_score(live_before, self._anchor_elos[0]))
                    * done_float[slot1]
                ).sum()
            )
        else:
            zero_sum = (
                k
                * (score[slot1] - expected_score(live_before, self.floating_elo))
                * done_float[slot1]
            ).sum()
            delta_live = delta_live + zero_sum
            slot4 = slice(4 * size, 5 * size)
            anchor_elo_float = self._anchor_elos[self._anchor_idx_float]  # (size,)
            self.floating_elo = (
                self.floating_elo
                - zero_sum
                + (
                    k
                    * (score[slot4] - expected_score(self.floating_elo, anchor_elo_float))
                    * done_float[slot4]
                ).sum()
            )
            self.floating_games = (
                self.floating_games + done_float[slot1].sum() + done_float[slot4].sum()
            )
        self.live_elo = live_before + delta_live

        if self.scripted_agent is not None:
            slot2 = slice(2 * size, 3 * size)
            self.scripted_elo = (
                self.scripted_elo
                + (
                    k
                    * ((1.0 - score[slot2]) - expected_score(self.scripted_elo, live_before))
                    * done_float[slot2]
                ).sum()
            )
        if avg_active:
            slot3 = slice(3 * size, 4 * size)
            self.avg_elo = (
                self.avg_elo
                + (
                    k
                    * ((1.0 - score[slot3]) - expected_score(self.avg_elo, live_before))
                    * done_float[slot3]
                ).sum()
            )

    def _resample_anchor_assignments(self, done_any: torch.Tensor) -> None:
        """Redraw anchor assignments for finished ladder-slot episodes."""
        if self._anchor_elos.numel() < 2:
            return
        size = self.matchup_size
        weights_live = information_weights(self.live_elo, self._anchor_elos)  # (A,)
        draw_live = torch.rand(size, device=self.device)
        self._anchor_idx_live = torch.where(
            done_any[:size], (draw_live > weights_live[0]).long(), self._anchor_idx_live
        )
        if self.float_pro_agent is not None:
            weights_float = information_weights(self.floating_elo, self._anchor_elos)
            draw_float = torch.rand(size, device=self.device)
            self._anchor_idx_float = torch.where(
                done_any[4 * size :], (draw_float > weights_float[0]).long(), self._anchor_idx_float
            )

    def _reset_agent_hiddens(self, done_any: torch.Tensor) -> None:
        """Reset recurrent state for every policy agent's finished envs."""
        size = self.matchup_size
        reset_done_envs(self.live_agent, done_any[: 4 * size], self.num_tokens)
        reset_done_envs(self.avg_agent, done_any[3 * size : 4 * size], self.num_tokens)
        if self.float_opp_agent is not None:
            reset_done_envs(self.float_opp_agent, done_any[size : 2 * size], self.num_tokens)
            reset_done_envs(self.float_pro_agent, done_any[4 * size :], self.num_tokens)
        for agent in self._anchor_agents_live:
            if agent is not None:
                reset_done_envs(agent, done_any[:size], self.num_tokens)
        for agent in self._anchor_agents_float:
            if agent is not None:
                reset_done_envs(agent, done_any[4 * size :], self.num_tokens)

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def flush(self, avg_active: bool) -> EloSnapshot:
        """Flush GPU ratings and outcome history to CPU once per PPO update."""
        floating_active = self.float_pro_agent is not None
        snapshot = EloSnapshot(
            live_elo=float(self.live_elo.item()),
            avg_elo=float(self.avg_elo.item()),
            scripted_elo=float(self.scripted_elo.item()),
            floating_elo=float(self.floating_elo.item()) if floating_active else None,
            floating_games=int(self.floating_games.item()) if floating_active else 0,
        )
        if not self._win_history:
            return snapshot

        size = self.matchup_size
        wins = torch.stack(self._win_history).cpu()  # (T_eval, 5·size)
        dones = torch.stack(self._done_history).cpu()
        anchor_idx = torch.stack(self._anchor_idx_history).cpu()  # (T_eval, size)
        self._win_history.clear()
        self._done_history.clear()
        self._anchor_idx_history.clear()
        # Anchor identities are constant within an update (promotion happens
        # between updates), so classify slot-0 games with the current mapping.
        anchor_is_random = torch.tensor(
            [agent is None for agent in self._anchor_agents_live], dtype=torch.bool
        )  # (A,)

        for index in range(wins.shape[0]):
            win, done, idx = wins[index], dones[index], anchor_idx[index]
            done0, win0 = done[:size], win[:size]
            versus_random = anchor_is_random[idx]  # (size,)
            self.random_window.extend(win0[done0 & versus_random].tolist())
            self.ladder_window.extend(win0[done0 & ~versus_random].tolist())
            done1, win1 = done[size : 2 * size], win[size : 2 * size]
            if floating_active:
                self.floating_window.extend(win1[done1].tolist())
            else:
                self.random_window.extend(win1[done1].tolist())
            if self.scripted_agent is not None:
                done2 = done[2 * size : 3 * size]
                self.scripted_window.extend(win[2 * size : 3 * size][done2].tolist())
            if avg_active:
                done3 = done[3 * size : 4 * size]
                self.live_vs_avg_window.extend(win[3 * size : 4 * size][done3].tolist())
        return snapshot
