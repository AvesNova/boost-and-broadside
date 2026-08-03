"""Opponent perspectives, action overrides, league sampling, and policy averaging."""

from typing import NamedTuple

import torch

from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.env.state import TensorState


class RolloutNetworkOutput(NamedTuple):
    """Policy outputs computed alongside one environment step."""

    action_t0: torch.Tensor
    action_t1: torch.Tensor | None
    logprob: torch.Tensor
    value_norm: torch.Tensor
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
    action_avg: torch.Tensor | None
    avg_hidden: torch.Tensor | None
    action_league: torch.Tensor | None
    league_hidden: torch.Tensor | None


class ScriptedStepOutput(NamedTuple):
    """Scripted-agent outputs and active opponent flags for one step."""

    use_avg: bool
    use_league: bool
    use_scripted_opponent: bool
    expert_probs: torch.Tensor | None
    scripted_action: torch.Tensor | None
    league_action: torch.Tensor | None


class PrimaryStepOutput(NamedTuple):
    """Mutable rollout state returned after one primary-scale step."""

    obs: YemongObservation
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
    avg_hidden: torch.Tensor | None
    league_hidden: torch.Tensor | None
    action_buffer: torch.Tensor
    dones: torch.Tensor


class EnvironmentStepOutput(NamedTuple):
    """Environment and policy outputs computed concurrently when CUDA is available."""

    obs: YemongObservation
    reward: torch.Tensor
    dones: torch.Tensor
    truncated: torch.Tensor
    network: RolloutNetworkOutput


def slice_obs(obs: YemongObservation, start: int, end: int) -> YemongObservation:
    """Return a view of observation tensors for environments ``[start, end)``."""
    return obs.slice_envs(slice(start, end))


def slice_state(state: TensorState, start: int, end: int) -> TensorState:
    """Return a view-backed state for environments ``[start, end)``."""
    return state.slice_envs(slice(start, end))


def flip_team_obs(obs: YemongObservation, num_ships: int) -> YemongObservation:
    """Flip ship team IDs while leaving field team IDs unchanged."""
    return obs.flip_team(num_ships)


class OpponentMixin:
    """Opponent-management behavior mixed into PPOTrainer."""

    def _update_avg_model(self) -> None:
        """Add the current training policy snapshot to the uniform running average."""
        first_update = self._avg_update_count == 0
        self._avg_update_count += 1
        for cumulative, parameter in zip(self._avg_param_cumsum, self._policy_module.parameters()):
            cumulative.add_(parameter.detach().float())
        for avg_parameter, cumulative in zip(
            self._avg_policy_module.parameters(), self._avg_param_cumsum
        ):
            avg_parameter.data.copy_(cumulative / self._avg_update_count)
        if first_update:
            self.roster.add_special("avg", self._global_step, 0, initial_elo=self._training_elo)

    def _opponent_obs(self, obs_slice: YemongObservation, num_ships: int) -> YemongObservation:
        """Return the observation perspective used by policy opponents."""
        return flip_team_obs(obs_slice, num_ships) if self._ego_pass else obs_slice

    def _combine_actions(
        self,
        action_t0: torch.Tensor,
        action_t1: torch.Tensor | None,
        team_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge perspective actions and build the actor-loss mask."""
        if self._ego_pass:
            team0_mask = team_id == 0
            action = torch.where(team0_mask.unsqueeze(-1), action_t0, action_t1)
            return action, team0_mask
        return action_t0.clone(), torch.ones_like(team_id, dtype=torch.bool)

    def _apply_opponent_override(
        self,
        action: torch.Tensor,
        actor_mask: torch.Tensor,
        team_id: torch.Tensor,
        start: int,
        end: int,
        opp_action: torch.Tensor,
    ) -> None:
        """Replace opponent-controlled actions in environments ``[start, end)``."""
        if self._ego_pass:
            opp_mask = team_id[start:end] == 1
        else:
            flags = self._opp_team_flag[start - self.B_self : end - self.B_self]
            opp_mask = team_id[start:end] == flags.unsqueeze(1)
        action[start:end] = torch.where(opp_mask.unsqueeze(-1), opp_action, action[start:end])
        actor_mask[start:end] &= ~opp_mask

    def _prepare_league_opponent(self, num_recurrent: int) -> torch.Tensor | None:
        """Sample and prepare the league opponent for one rollout."""
        league_active = self.B_league > 0 and self._schedule_state.league_fraction > 0.0
        if not league_active:
            self.roster.evict_all_checkpoint_policies()
            self._current_league_entry = None
            self._current_league_policy = None
            return None

        entry = self.roster.sample(self._training_elo)
        self._current_league_entry = entry
        if entry is None or (entry.kind == "avg" and self._avg_update_count == 0):
            self._current_league_entry = None
            self._current_league_policy = None
            return None
        if entry.kind == "checkpoint":
            self.roster.load_policy(
                entry,
                self.ship_config,
                self.wrapper.num_ships,
                self.device,
                model_config=self.model_config,
                compile_mode=self._compile_mode,
                team_pma_k=self._win_k,
            )
            self._current_league_policy = entry.policy
        elif entry.kind == "avg":
            self._current_league_policy = self.avg_policy
        else:
            self._current_league_policy = None
            return None
        return self._current_league_policy.initial_hidden(self.B_league, num_recurrent, self.device)

    def _rollout_network_forwards(
        self,
        obs: YemongObservation,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        num_ships: int,
        num_recurrent: int,
        use_avg: bool,
        avg_start: int,
        avg_end: int,
        avg_hidden: torch.Tensor | None,
        use_league: bool,
        league_start: int,
        league_end: int,
        league_hidden: torch.Tensor | None,
    ) -> RolloutNetworkOutput:
        """Run the live, average, and league policy forwards for one rollout step."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (
                action_t0,
                action_t1,
                logprob,
                value_norm,
                _,
                hidden,
                hidden_t1,
            ) = self._rollout_policy_pass(obs, hidden, hidden_t1, num_ships, num_recurrent)

        action_avg = None
        if use_avg:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                obs_avg = self._opponent_obs(slice_obs(obs, avg_start, avg_end), num_ships)
                action_avg, _, _, _, avg_hidden = self.avg_policy.get_action_and_value(
                    obs_avg, avg_hidden
                )

        action_league = None
        if use_league and self._current_league_policy is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                obs_league = self._opponent_obs(slice_obs(obs, league_start, league_end), num_ships)
                action_league, _, _, _, league_hidden = (
                    self._current_league_policy.get_action_and_value(obs_league, league_hidden)
                )

        return RolloutNetworkOutput(
            action_t0=action_t0,
            action_t1=action_t1,
            logprob=logprob,
            value_norm=value_norm,
            hidden=hidden,
            hidden_t1=hidden_t1,
            action_avg=action_avg,
            avg_hidden=avg_hidden,
            action_league=action_league,
            league_hidden=league_hidden,
        )

    def _scripted_step_outputs(
        self,
        scripted_start: int,
        scripted_end: int,
        league_start: int,
        league_end: int,
    ) -> ScriptedStepOutput:
        """Compute scripted BC targets and opponent actions before stream launch."""
        if self._policy_gradient_coef == 0.0:
            with torch.no_grad():
                _, expert_probs = self.scripted_agent.get_actions_and_probs(self.wrapper.env.state)
            return ScriptedStepOutput(False, False, False, expert_probs, None, None)

        use_avg = (
            self._avg_update_count > 0
            and self.B_avg > 0
            and self._schedule_state.avg_model_fraction > 0.0
        )
        use_league = (
            self.B_league > 0
            and self._current_league_entry is not None
            and self._schedule_state.league_fraction > 0.0
        )
        use_scripted_slots = self.B_sc > 0
        use_scripted_opponent = use_scripted_slots and self._schedule_state.scripted_fraction > 0.0
        expert_probs = None
        scripted_action = None
        league_action = None

        if self._behavior_cloning_coef > 0.0 and self.scripted_agent is not None:
            with torch.no_grad():
                all_scripted_actions, expert_probs = self.scripted_agent.get_actions_and_probs(
                    self.wrapper.env.state
                )
            if use_scripted_opponent:
                scripted_action = all_scripted_actions[scripted_start:scripted_end]
        elif use_scripted_slots:
            with torch.no_grad():
                scripted_state = slice_state(self.wrapper.env.state, scripted_start, scripted_end)
                scripted_action, _ = self.scripted_agent.get_actions_and_probs(scripted_state)

        if use_league and self._current_league_policy is None:
            with torch.no_grad():
                league_state = slice_state(self.wrapper.env.state, league_start, league_end)
                league_action = self.scripted_agent.get_actions(league_state)

        return ScriptedStepOutput(
            use_avg,
            use_league,
            use_scripted_opponent,
            expert_probs,
            scripted_action,
            league_action,
        )

    def _step_environment_and_network(
        self,
        action_buffer: torch.Tensor,
        network_args: tuple,
        env_stream: torch.cuda.Stream | None,
        net_stream: torch.cuda.Stream | None,
    ) -> EnvironmentStepOutput:
        """Advance the environment and policy, overlapping them on CUDA streams."""
        if env_stream is None:
            next_obs, reward, dones, truncated, _ = self.wrapper.step(action_buffer)
            network = self._rollout_network_forwards(*network_args)
            return EnvironmentStepOutput(next_obs, reward, dones, truncated, network)

        env_stream.wait_stream(torch.cuda.current_stream())
        net_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(env_stream):
            next_obs, reward, dones, truncated, _ = self.wrapper.step(action_buffer)
        with torch.cuda.stream(net_stream):
            network = self._rollout_network_forwards(*network_args)
        torch.cuda.current_stream().wait_stream(env_stream)
        torch.cuda.current_stream().wait_stream(net_stream)
        return EnvironmentStepOutput(next_obs, reward, dones, truncated, network)

    def _select_primary_actions(
        self,
        network: RolloutNetworkOutput,
        scripted: ScriptedStepOutput,
        team_id: torch.Tensor,
        scripted_start: int,
        scripted_end: int,
        avg_start: int,
        avg_end: int,
        league_start: int,
        league_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine live-policy actions and apply active opponent overrides."""
        action, actor_mask = self._combine_actions(network.action_t0, network.action_t1, team_id)
        if self._policy_gradient_coef == 0.0:
            return action, actor_mask

        league_action = (
            scripted.league_action if scripted.league_action is not None else network.action_league
        )
        if scripted.use_scripted_opponent:
            self._apply_opponent_override(
                action, actor_mask, team_id, scripted_start, scripted_end, scripted.scripted_action
            )
        if scripted.use_avg:
            self._apply_opponent_override(
                action, actor_mask, team_id, avg_start, avg_end, network.action_avg
            )
        if scripted.use_league:
            self._apply_opponent_override(
                action, actor_mask, team_id, league_start, league_end, league_action
            )
        return action, actor_mask

    def _reset_primary_hidden(
        self,
        network: RolloutNetworkOutput,
        done_any: torch.Tensor,
        num_recurrent: int,
        avg_start: int,
        avg_end: int,
        league_start: int,
        league_end: int,
        use_avg: bool,
        use_league: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Reset recurrent states for completed primary-scale environments."""
        hidden = self.policy.reset_hidden_for_envs(network.hidden, done_any, num_recurrent)
        hidden_t1 = network.hidden_t1
        if self._ego_pass:
            hidden_t1 = self.policy.reset_hidden_for_envs(hidden_t1, done_any, num_recurrent)

        avg_hidden = network.avg_hidden
        if use_avg and self.B_avg > 0:
            avg_hidden = self.avg_policy.reset_hidden_for_envs(
                avg_hidden, done_any[avg_start:avg_end], num_recurrent
            )

        league_hidden = network.league_hidden
        if use_league and self._current_league_policy is not None:
            league_hidden = self._current_league_policy.reset_hidden_for_envs(
                league_hidden, done_any[league_start:league_end], num_recurrent
            )
        return hidden, hidden_t1, avg_hidden, league_hidden

    def _refresh_opponent_team_flags(self, done_any: torch.Tensor) -> None:
        """Resample shared-pass opponent team assignments for completed environments."""
        if self._ego_pass or self._opp_team_flag.numel() == 0:
            return
        new_flags = torch.randint(
            0,
            2,
            self._opp_team_flag.shape,
            device=self.device,
            dtype=torch.int32,
        )
        self._opp_team_flag = torch.where(done_any[self.B_self :], new_flags, self._opp_team_flag)

    def _collect_primary_step(
        self,
        obs: YemongObservation,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        avg_hidden: torch.Tensor | None,
        league_hidden: torch.Tensor | None,
        action_buffer: torch.Tensor,
        num_envs: int,
        num_ships: int,
        num_recurrent: int,
        scripted_start: int,
        scripted_end: int,
        avg_start: int,
        avg_end: int,
        league_start: int,
        league_end: int,
        env_stream: torch.cuda.Stream | None,
        net_stream: torch.cuda.Stream | None,
    ) -> PrimaryStepOutput:
        """Collect one primary-scale transition and update recurrent rollout state."""
        team_id = obs["team_id"][:, :num_ships]
        scripted = self._scripted_step_outputs(
            scripted_start, scripted_end, league_start, league_end
        )
        network_args = (
            obs,
            hidden,
            hidden_t1,
            num_ships,
            num_recurrent,
            scripted.use_avg,
            avg_start,
            avg_end,
            avg_hidden,
            scripted.use_league,
            league_start,
            league_end,
            league_hidden,
        )
        step = self._step_environment_and_network(
            action_buffer, network_args, env_stream, net_stream
        )
        action, actor_mask = self._select_primary_actions(
            step.network,
            scripted,
            team_id,
            scripted_start,
            scripted_end,
            avg_start,
            avg_end,
            league_start,
            league_end,
        )
        step.obs["previous_action"][:, :num_ships] = action
        done_any = step.dones | step.truncated
        self.buffer.add(
            obs=obs,
            action=action,
            logprob=step.network.logprob,
            reward=step.reward,
            done=step.dones.float(),
            value=self.scaler.denormalize(step.network.value_norm),
            alive=obs["alive"][:, :num_ships].bool(),
            actor_mask=actor_mask,
            expert_probs=scripted.expert_probs,
            terminated=done_any,
        )

        hidden, hidden_t1, avg_hidden, league_hidden = self._reset_primary_hidden(
            step.network,
            done_any,
            num_recurrent,
            avg_start,
            avg_end,
            league_start,
            league_end,
            scripted.use_avg,
            scripted.use_league,
        )
        action_buffer = action.detach().clone()
        action_buffer[done_any] = 0
        self._refresh_opponent_team_flags(done_any)
        self._global_step += num_envs
        return PrimaryStepOutput(
            step.obs,
            hidden,
            hidden_t1,
            avg_hidden,
            league_hidden,
            action_buffer,
            step.dones,
        )
