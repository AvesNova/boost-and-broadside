"""Unified PFSP opponent execution, perspectives, and policy averaging."""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

from boost_and_broadside.env.observation import MVPObservation
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.modes.agent_factory import ResolvedAgent, get_actions
from boost_and_broadside.train.rl.rating import DRAW, LOSS, WIN
from boost_and_broadside.train.rl.roster import RosterEntry


@dataclass
class OpponentSlot:
    """One contiguous training slice assigned to a sampled league member."""

    entry: RosterEntry
    start: int
    end: int
    policy: MVPPolicy | None
    hidden: torch.Tensor | None


class RolloutNetworkOutput(NamedTuple):
    """Policy outputs computed alongside one environment step."""

    action_t0: torch.Tensor
    action_t1: torch.Tensor | None
    logprob: torch.Tensor
    value_norm: torch.Tensor
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
    opponent_actions: list[torch.Tensor | None]


class ScriptedStepOutput(NamedTuple):
    """Non-policy opponent actions and optional full-batch BC targets."""

    expert_probs: torch.Tensor | None
    opponent_actions: list[torch.Tensor | None]


class PrimaryStepOutput(NamedTuple):
    """Mutable rollout state returned after one primary-scale step."""

    obs: MVPObservation
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
    action_buffer: torch.Tensor
    dones: torch.Tensor


class EnvironmentStepOutput(NamedTuple):
    """Environment and policy outputs computed concurrently when CUDA is available."""

    obs: MVPObservation
    reward: torch.Tensor
    dones: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, torch.Tensor]
    network: RolloutNetworkOutput


def slice_obs(obs: MVPObservation, start: int, end: int) -> MVPObservation:
    """Return a view of observation tensors for environments ``[start, end)``."""
    return obs.slice_envs(slice(start, end))


def slice_state(state: TensorState, start: int, end: int) -> TensorState:
    """Return a view-backed state for environments ``[start, end)``."""
    return state.slice_envs(slice(start, end))


def flip_team_obs(obs: MVPObservation, num_ships: int) -> MVPObservation:
    """Flip ship team IDs while leaving obstacle team IDs unchanged."""
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
            entry = self.roster.add_avg(self._global_step, self._current_update)
            if entry.agent_id not in self._latest_counts_snapshot.agent_ids:
                self._latest_counts_snapshot.add_agent(entry.agent_id)

    def _opponent_obs(self, obs_slice: MVPObservation, num_ships: int) -> MVPObservation:
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
        slot: OpponentSlot,
        opponent_action: torch.Tensor,
    ) -> None:
        """Replace the sampled opponent's team actions in one contiguous slot."""
        if self._ego_pass:
            opponent_mask = team_id[slot.start : slot.end] == 1
        else:
            flags = self._opp_team_flag[slot.start - self.B_self : slot.end - self.B_self]
            opponent_mask = team_id[slot.start : slot.end] == flags.unsqueeze(1)
        action[slot.start : slot.end] = torch.where(
            opponent_mask.unsqueeze(-1),
            opponent_action,
            action[slot.start : slot.end],
        )
        actor_mask[slot.start : slot.end] &= ~opponent_mask

    def _clear_opponents(self) -> None:
        """Release held opponents and disable outcome attribution."""
        self.roster.evict_all_checkpoint_policies()
        self._held_opponents = []
        self._held_hiddens = []
        self._held_ages = []
        self._opponent_slots = []
        self._opp_partition = ()
        if self._opp_valid.numel() > 0:
            self._opp_valid.fill_(False)
            self._opp_active_mask.fill_(False)

    def _prepare_opponents(self, num_tokens: int) -> list[OpponentSlot]:
        """Refresh held PFSP opponents and rebuild their slot views for one rollout.

        Opponents are held for ``opponent_hold_rollouts`` rollouts with at most
        one replacement per rollout, so episodes usually complete against a
        single opponent. Per-env attribution is invalidated wherever the
        assignment changes; ``_record_training_outcomes`` re-validates each env
        at its first episode reset under the new assignment.
        """
        active_count = round(self._schedule_state.opponent_fraction * self.cfg.scales[0].num_envs)
        active_count = min(self.B_opp, max(0, active_count))
        if active_count == 0 or self._policy_gradient_coef == 0.0:
            self._clear_opponents()
            return []

        # Carry recurrent state forward from the previous rollout's slot views.
        for index, slot in enumerate(self._opponent_slots):
            if index < len(self._held_hiddens):
                self._held_hiddens[index] = slot.hidden

        layout_changed = self.roster.layout_version != self._opp_layout_version
        self._opp_layout_version = self.roster.layout_version

        available_ratings = dict(self.roster.ratings)
        if self.scripted_agent is None:
            available_ratings.pop("scripted", None)
        if self._avg_update_count == 0:
            available_ratings.pop("avg", None)
        for entry in self.roster.entries:
            if entry.kind == "checkpoint" and (entry.path is None or not Path(entry.path).exists()):
                available_ratings.pop(entry.agent_id, None)

        self._refresh_held_opponents(min(self.cfg.league_k, active_count), available_ratings)
        if not self._held_opponents:
            self._clear_opponents()
            return []
        held_ids = {entry.agent_id for entry in self._held_opponents}
        for entry in self.roster.entries:
            if entry.kind == "checkpoint" and entry.agent_id not in held_ids:
                entry.policy = None
        return self._build_opponent_slots(active_count, num_tokens, layout_changed)

    def _refresh_held_opponents(self, k: int, available_ratings: dict[str, float]) -> None:
        """Age out, trim, and top up the held opponent set with PFSP samples."""
        held = self._held_opponents
        hiddens = self._held_hiddens
        ages = self._held_ages
        for index in range(len(ages)):
            ages[index] += 1
        # Drop members evicted from the roster or no longer sampleable.
        for index in reversed(range(len(held))):
            if held[index].agent_id not in available_ratings:
                del held[index], hiddens[index], ages[index]
        while len(held) > k:
            oldest = max(range(len(held)), key=lambda position: ages[position])
            del held[oldest], hiddens[oldest], ages[oldest]
        # Replace at most one expired member per rollout (staggered turnover).
        replace_index = None
        if held:
            oldest = max(range(len(held)), key=lambda position: ages[position])
            if ages[oldest] >= self.cfg.opponent_hold_rollouts:
                replace_index = oldest
        need = k - len(held) + (1 if replace_index is not None else 0)
        if need <= 0:
            return
        held_ids = {entry.agent_id for entry in held}
        fresh_ratings = {
            agent_id: rating
            for agent_id, rating in available_ratings.items()
            if agent_id == "live" or agent_id not in held_ids
        }
        sampled = self.roster.sample_opponents(need, fresh_ratings)
        if not sampled:
            sampled = self.roster.sample_opponents(need, available_ratings)
        for entry in sampled:
            if replace_index is not None:
                held[replace_index] = entry
                hiddens[replace_index] = None
                ages[replace_index] = 0
                replace_index = None
            else:
                held.append(entry)
                hiddens.append(None)
                ages.append(0)

    def _build_opponent_slots(
        self, active_count: int, num_tokens: int, layout_changed: bool
    ) -> list[OpponentSlot]:
        """Partition the active region and refresh per-env attribution state."""
        entries = self._held_opponents
        base, remainder = divmod(active_count, len(entries))
        start = self.cfg.scales[0].num_envs - active_count
        new_partition: list[tuple[str, int, int]] = []
        new_count_index = torch.zeros_like(self._opp_count_index)
        new_active = torch.zeros_like(self._opp_active_mask)
        slots: list[OpponentSlot] = []
        for index, entry in enumerate(entries):
            size = base + (1 if index < remainder else 0)
            end = start + size
            segment = (entry.agent_id, start, end)
            if index >= len(self._opp_partition) or self._opp_partition[index] != segment:
                self._held_hiddens[index] = None
            policy = self._resolve_opponent_policy(entry)
            hidden = self._held_hiddens[index] if policy is not None else None
            if policy is not None and hidden is None:
                hidden = policy.initial_hidden(size, num_tokens, self.device)
            self._held_hiddens[index] = hidden
            relative = slice(start - self.B_self, end - self.B_self)
            new_count_index[relative] = self.roster.counts.index(entry.agent_id)
            new_active[relative] = True
            new_partition.append(segment)
            slots.append(OpponentSlot(entry, start, end, policy, hidden))
            start = end
        changed = (new_count_index != self._opp_count_index) | (new_active != self._opp_active_mask)
        if layout_changed:
            self._opp_valid.fill_(False)
        else:
            self._opp_valid &= ~changed
        self._opp_valid &= new_active
        self._opp_count_index = new_count_index
        self._opp_active_mask = new_active
        self._opp_partition = tuple(new_partition)
        self._opponent_slots = slots
        return slots

    def _resolve_opponent_policy(self, entry: RosterEntry) -> MVPPolicy | None:
        """Return the policy implementation for one sampled roster entry."""
        if entry.kind == "avg":
            return self.avg_policy
        if entry.kind != "checkpoint":
            return None
        self.roster.load_policy(
            entry,
            self.model_config,
            self.coordinator,
            self.wrapper.num_active_components,
            self.wrapper.num_ships,
            self.device,
            self._compile_mode,
            team_pma_k=self._win_k,
        )
        return entry.policy

    def _rollout_network_forwards(
        self,
        obs: MVPObservation,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        num_ships: int,
        num_tokens: int,
        opponent_slots: list[OpponentSlot],
    ) -> RolloutNetworkOutput:
        """Run live and policy-opponent forwards for one rollout step."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (
                action_t0,
                action_t1,
                logprob,
                value_norm,
                _,
                hidden,
                hidden_t1,
            ) = self._rollout_policy_pass(obs, hidden, hidden_t1, num_ships, num_tokens)

        opponent_actions: list[torch.Tensor | None] = []
        for slot in opponent_slots:
            if slot.policy is None:
                opponent_actions.append(None)
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                opponent_obs = self._opponent_obs(slice_obs(obs, slot.start, slot.end), num_ships)
                action, _, _, _, slot.hidden = slot.policy.get_action_and_value(
                    opponent_obs, slot.hidden
                )
            opponent_actions.append(action)
        return RolloutNetworkOutput(
            action_t0,
            action_t1,
            logprob,
            value_norm,
            hidden,
            hidden_t1,
            opponent_actions,
        )

    def _scripted_step_outputs(
        self,
        opponent_slots: list[OpponentSlot],
    ) -> ScriptedStepOutput:
        """Compute BC targets and non-policy opponent actions before stream launch."""
        if self._policy_gradient_coef == 0.0:
            with torch.no_grad():
                _, expert_probs = self.scripted_agent.get_actions_and_probs(self.wrapper.env.state)
            return ScriptedStepOutput(expert_probs, [None] * len(opponent_slots))

        all_scripted_actions = None
        expert_probs = None
        if self._behavior_cloning_coef > 0.0 and self.scripted_agent is not None:
            with torch.no_grad():
                all_scripted_actions, expert_probs = self.scripted_agent.get_actions_and_probs(
                    self.wrapper.env.state
                )

        opponent_actions: list[torch.Tensor | None] = []
        random_agent = ResolvedAgent("random", None)
        for slot in opponent_slots:
            if slot.entry.kind == "random":
                opponent_actions.append(
                    get_actions(
                        random_agent,
                        None,
                        self.wrapper.env.state,
                        slot.end - slot.start,
                        self.wrapper.num_ships,
                        self.device,
                    )
                )
            elif slot.entry.kind == "scripted":
                if all_scripted_actions is not None:
                    opponent_actions.append(all_scripted_actions[slot.start : slot.end])
                else:
                    scripted_state = slice_state(self.wrapper.env.state, slot.start, slot.end)
                    with torch.no_grad():
                        opponent_actions.append(self.scripted_agent.get_actions(scripted_state))
            else:
                opponent_actions.append(None)
        return ScriptedStepOutput(expert_probs, opponent_actions)

    def _step_environment_and_network(
        self,
        action_buffer: torch.Tensor,
        network_args: tuple,
        env_stream: torch.cuda.Stream | None,
        net_stream: torch.cuda.Stream | None,
    ) -> EnvironmentStepOutput:
        """Advance environment and policy, overlapping them on CUDA streams."""
        if env_stream is None:
            next_obs, reward, dones, truncated, info = self.wrapper.step(action_buffer)
            network = self._rollout_network_forwards(*network_args)
            return EnvironmentStepOutput(next_obs, reward, dones, truncated, info, network)

        env_stream.wait_stream(torch.cuda.current_stream())
        net_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(env_stream):
            next_obs, reward, dones, truncated, info = self.wrapper.step(action_buffer)
        with torch.cuda.stream(net_stream):
            network = self._rollout_network_forwards(*network_args)
        torch.cuda.current_stream().wait_stream(env_stream)
        torch.cuda.current_stream().wait_stream(net_stream)
        return EnvironmentStepOutput(next_obs, reward, dones, truncated, info, network)

    def _select_primary_actions(
        self,
        network: RolloutNetworkOutput,
        scripted: ScriptedStepOutput,
        team_id: torch.Tensor,
        opponent_slots: list[OpponentSlot],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine live actions and apply every active opponent override."""
        action, actor_mask = self._combine_actions(network.action_t0, network.action_t1, team_id)
        if self._policy_gradient_coef == 0.0:
            return action, actor_mask
        for index, slot in enumerate(opponent_slots):
            opponent_action = scripted.opponent_actions[index]
            if opponent_action is None:
                opponent_action = network.opponent_actions[index]
            self._apply_opponent_override(action, actor_mask, team_id, slot, opponent_action)
        return action, actor_mask

    def _reset_primary_hidden(
        self,
        network: RolloutNetworkOutput,
        done_any: torch.Tensor,
        num_tokens: int,
        opponent_slots: list[OpponentSlot],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reset recurrent states for completed primary environments."""
        hidden = self.policy.reset_hidden_for_envs(network.hidden, done_any, num_tokens)
        hidden_t1 = network.hidden_t1
        if self._ego_pass:
            hidden_t1 = self.policy.reset_hidden_for_envs(hidden_t1, done_any, num_tokens)
        for slot in opponent_slots:
            if slot.policy is not None:
                slot.hidden = slot.policy.reset_hidden_for_envs(
                    slot.hidden, done_any[slot.start : slot.end], num_tokens
                )
        return hidden, hidden_t1

    def _record_training_outcomes(
        self,
        done_any: torch.Tensor,
        opponent_slots: list[OpponentSlot],
        info: dict[str, torch.Tensor],
    ) -> None:
        """Accumulate live-vs-opponent outcomes for validly attributed episodes.

        An env only counts once it has reset under its current opponent
        assignment, so the first (possibly mixed-opponent) episode after any
        reassignment is discarded rather than misattributed.
        """
        if not opponent_slots or self._opp_valid.numel() == 0:
            return
        region = slice(self.B_self, self.B_self + self.B_opp)
        team0_won = info["team0_won"][region]
        team1_won = info["team1_won"][region]
        if self._ego_pass:
            opponent_team = torch.ones_like(self._opp_count_index)
        else:
            opponent_team = self._opp_team_flag.long()
        opponent_won = torch.where(opponent_team == 0, team0_won, team1_won)
        live_won = torch.where(opponent_team == 0, team1_won, team0_won)
        outcomes = torch.where(
            live_won,
            torch.full_like(self._opp_count_index, WIN),
            torch.where(
                opponent_won,
                torch.full_like(self._opp_count_index, LOSS),
                torch.full_like(self._opp_count_index, DRAW),
            ),
        )
        done_region = done_any[region]
        recordable = done_region & self._opp_valid
        live_index = self.roster.counts.index("live")
        self.roster.record_outcomes(
            torch.full_like(self._opp_count_index, live_index),
            self._opp_count_index,
            outcomes,
            recordable.to(self.roster.counts.tensor.dtype),
        )
        self._opp_valid |= done_region & self._opp_active_mask

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
        obs: MVPObservation,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        action_buffer: torch.Tensor,
        num_envs: int,
        num_ships: int,
        num_tokens: int,
        opponent_slots: list[OpponentSlot],
        env_stream: torch.cuda.Stream | None,
        net_stream: torch.cuda.Stream | None,
    ) -> PrimaryStepOutput:
        """Collect one primary transition and update recurrent rollout state."""
        team_id = obs["team_id"][:, :num_ships]
        scripted = self._scripted_step_outputs(opponent_slots)
        network_args = (obs, hidden, hidden_t1, num_ships, num_tokens, opponent_slots)
        step = self._step_environment_and_network(
            action_buffer, network_args, env_stream, net_stream
        )
        action, actor_mask = self._select_primary_actions(
            step.network, scripted, team_id, opponent_slots
        )
        step.obs["previous_action"][:, :num_ships] = action
        done_any = step.dones | step.truncated
        self._record_training_outcomes(done_any, opponent_slots, step.info)
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

        hidden, hidden_t1 = self._reset_primary_hidden(
            step.network, done_any, num_tokens, opponent_slots
        )
        action_buffer = action.detach().clone()
        action_buffer[done_any] = 0
        self._refresh_opponent_team_flags(done_any)
        self._global_step += num_envs
        return PrimaryStepOutput(step.obs, hidden, hidden_t1, action_buffer, step.dones)
