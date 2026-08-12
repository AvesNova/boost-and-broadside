"""Opponent perspectives, action overrides, league sampling, and policy averaging.

Every opponent is a league entry. The scripted agent, the running-average
policy and each frozen checkpoint sit on one Elo scale in the roster, and the
league half of the batch is divided into slots that each draw from it by rating
proximity. There is no per-opponent-type environment group and no per-type
schedule: the opponent curriculum is whatever the ratings imply, which early on
is the scripted agent (the only thing to draw) and later is a spread of
checkpoints and the average policy near the live rating.
"""

import dataclasses
from typing import NamedTuple

import torch

from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.models.yemong.policy import YemongPolicy
from boost_and_broadside.train.rl.roster import RosterEntry


@dataclasses.dataclass
class LeagueSlot:
    """One contiguous block of league envs and the opponent currently in it.

    Resampled at each rollout boundary, so an episode never changes opponent
    mid-flight except where a scheduled league fraction moves the block edge.

    A ``policy`` of None means the scripted agent acts for this slot: it has no
    weights to load and no recurrent state to carry, which is why it is the one
    entry kind that costs no extra forward pass.
    """

    start: int
    end: int
    entry: RosterEntry
    policy: YemongPolicy | None
    hidden: torch.Tensor | None


class RolloutNetworkOutput(NamedTuple):
    """Policy outputs computed alongside one environment step."""

    action_t0: torch.Tensor
    action_t1: torch.Tensor | None
    logprob: torch.Tensor
    value_norm: torch.Tensor
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
    # Per-slot opponent actions, aligned with the slot list. None where the slot
    # is scripted — ScriptedStepOutput supplies those.
    slot_actions: list[torch.Tensor | None]


class ScriptedStepOutput(NamedTuple):
    """Scripted-agent outputs for one step."""

    expert_probs: torch.Tensor | None
    # Slot index → scripted action, for slots the scripted agent is playing.
    slot_actions: dict[int, torch.Tensor]


class PrimaryStepOutput(NamedTuple):
    """Mutable rollout state returned after one primary-scale step."""

    obs: YemongObservation
    hidden: torch.Tensor
    hidden_t1: torch.Tensor | None
    action_buffer: torch.Tensor
    # done | truncated — the GAE boundary, not physics termination alone.
    terminated: torch.Tensor


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
            self.roster.add_special("avg", self._global_step, 0, initial_elo=self._live_elo)

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

    def _active_league_width(self) -> int:
        """Envs the league plays this rollout, from the current scheduled fraction.

        The block is allocated once at the run's peak fraction; this narrows the
        active window inside it, so a fraction that steps down actually returns
        envs to self-play rather than only flipping the group off entirely.
        """
        total_envs = self.cfg.scales[0].num_envs
        width = round(self._schedule_state.league_fraction * total_envs)
        return max(0, min(width, self.B_league))

    def _sample_league_entry(self) -> RosterEntry | None:
        """Draw one league opponent, retiring any entry this run cannot host."""
        while (entry := self.roster.sample(self._live_elo)) is not None:
            if entry.kind != "checkpoint":
                return entry
            self.roster.load_policy(
                entry,
                self.ship_config,
                self.wrapper.num_ships,
                self.device,
                model_config=self.model_config,
                compile_mode=self._compile_mode,
                team_pma_k=self._win_k,
            )
            if entry.bundle.reads_bullets and not self.model_config.reads_bullets:
                print(
                    f"[PPOTrainer] retiring league entry {entry.label!r}: it reads "
                    "bullets and this run's observation carries none, so it would "
                    "play blind. Train with n_bullet_cross_per_block > 0 to face it."
                )
                self.roster.retire(entry)
                continue
            return entry
        return None

    def _league_policy(self, entry: RosterEntry) -> YemongPolicy | None:
        """Resolve a drawn entry to the policy that acts for it.

        None means the scripted agent plays this slot.
        """
        if entry.kind == "avg":
            return self.avg_policy
        if entry.kind == "checkpoint":
            return entry.policy  # already loaded by _sample_league_entry
        return None

    def _prepare_league_slots(self, num_recurrent: int) -> list[LeagueSlot]:
        """Draw this rollout's league opponents and lay them out over the block."""
        width = self._active_league_width()
        if width == 0:
            self.roster.evict_all_checkpoint_policies()
            return []

        start = self.cfg.scales[0].num_envs - width
        n_slots = min(self.cfg.league_slots, width)
        base, remainder = divmod(width, n_slots)

        slots: list[LeagueSlot] = []
        offset = start
        for index in range(n_slots):
            slot_width = base + (1 if index < remainder else 0)
            entry = self._sample_league_entry()
            if entry is None:
                break  # empty roster — the whole block falls back to self-play
            policy = self._league_policy(entry)
            hidden = (
                policy.initial_hidden(slot_width, num_recurrent, self.device)
                if policy is not None
                else None
            )
            slots.append(
                LeagueSlot(
                    start=offset,
                    end=offset + slot_width,
                    entry=entry,
                    policy=policy,
                    hidden=hidden,
                )
            )
            offset += slot_width
        return slots

    def _rollout_network_forwards(
        self,
        obs: YemongObservation,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        num_ships: int,
        num_recurrent: int,
        slots: list[LeagueSlot],
    ) -> RolloutNetworkOutput:
        """Run the live pass and one forward per policy-backed league slot.

        Slot hidden states are advanced in place; scripted slots yield None and
        are filled from ScriptedStepOutput.
        """
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

        slot_actions: list[torch.Tensor | None] = []
        for slot in slots:
            if slot.policy is None:
                slot_actions.append(None)
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                obs_slot = self._opponent_obs(slice_obs(obs, slot.start, slot.end), num_ships)
                action, _, _, _, slot.hidden = slot.policy.get_action_and_value(
                    obs_slot, slot.hidden
                )
            slot_actions.append(action)

        return RolloutNetworkOutput(
            action_t0=action_t0,
            action_t1=action_t1,
            logprob=logprob,
            value_norm=value_norm,
            hidden=hidden,
            hidden_t1=hidden_t1,
            slot_actions=slot_actions,
        )

    def _scripted_step_outputs(self, slots: list[LeagueSlot]) -> ScriptedStepOutput:
        """Compute scripted BC targets and scripted-slot actions before stream launch."""
        if self._policy_gradient_coef == 0.0:
            with torch.no_grad():
                _, expert_probs = self.scripted_agent.get_actions_and_probs(self.wrapper.env.state)
            return ScriptedStepOutput(expert_probs, {})

        scripted_slots = [index for index, slot in enumerate(slots) if slot.policy is None]
        if self._behavior_cloning_coef > 0.0 and self.scripted_agent is not None:
            # BC needs targets for every env anyway, so slot actions come free.
            with torch.no_grad():
                actions, expert_probs = self.scripted_agent.get_actions_and_probs(
                    self.wrapper.env.state
                )
            return ScriptedStepOutput(
                expert_probs,
                {index: actions[slots[index].start : slots[index].end] for index in scripted_slots},
            )

        if not scripted_slots:
            return ScriptedStepOutput(None, {})

        # One pass over the envs the scripted slots span, then slice per slot.
        # Non-scripted slots between two scripted ones ride along in the span;
        # their actions are simply discarded, which is cheaper than the bookkeeping
        # to avoid computing them.
        low = min(slots[index].start for index in scripted_slots)
        high = max(slots[index].end for index in scripted_slots)
        with torch.no_grad():
            actions = self.scripted_agent.get_actions(
                slice_state(self.wrapper.env.state, low, high)
            )
        return ScriptedStepOutput(
            None,
            {
                index: actions[slots[index].start - low : slots[index].end - low]
                for index in scripted_slots
            },
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
        slots: list[LeagueSlot],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine live-policy actions and hand each league slot its own side."""
        action, actor_mask = self._combine_actions(network.action_t0, network.action_t1, team_id)
        if self._policy_gradient_coef == 0.0:
            return action, actor_mask

        for index, slot in enumerate(slots):
            opponent_action = network.slot_actions[index]
            if opponent_action is None:
                opponent_action = scripted.slot_actions[index]
            self._apply_opponent_override(
                action, actor_mask, team_id, slot.start, slot.end, opponent_action
            )
        return action, actor_mask

    def _reset_primary_hidden(
        self,
        network: RolloutNetworkOutput,
        done_any: torch.Tensor,
        num_recurrent: int,
        slots: list[LeagueSlot],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reset recurrent states for completed primary-scale environments."""
        hidden = self.policy.reset_hidden_for_envs(network.hidden, done_any, num_recurrent)
        hidden_t1 = network.hidden_t1
        if self._ego_pass:
            hidden_t1 = self.policy.reset_hidden_for_envs(hidden_t1, done_any, num_recurrent)

        for slot in slots:
            if slot.policy is not None:
                slot.hidden = slot.policy.reset_hidden_for_envs(
                    slot.hidden, done_any[slot.start : slot.end], num_recurrent
                )
        return hidden, hidden_t1

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
        action_buffer: torch.Tensor,
        num_envs: int,
        num_ships: int,
        num_recurrent: int,
        slots: list[LeagueSlot],
        env_stream: torch.cuda.Stream | None,
        net_stream: torch.cuda.Stream | None,
    ) -> PrimaryStepOutput:
        """Collect one primary-scale transition and update recurrent rollout state."""
        team_id = obs["team_id"][:, :num_ships]
        scripted = self._scripted_step_outputs(slots)
        network_args = (obs, hidden, hidden_t1, num_ships, num_recurrent, slots)
        step = self._step_environment_and_network(
            action_buffer, network_args, env_stream, net_stream
        )
        action, actor_mask = self._select_primary_actions(step.network, scripted, team_id, slots)
        step.obs["previous_action"][:, :num_ships] = action
        done_any = step.dones | step.truncated
        self.buffer.add(
            obs=obs,
            action=action,
            logprob=step.network.logprob,
            reward=step.reward,
            value=self.scaler.denormalize(step.network.value_norm),
            alive=obs["alive"][:, :num_ships].bool(),
            actor_mask=actor_mask,
            expert_probs=scripted.expert_probs,
            terminated=done_any,
        )

        hidden, hidden_t1 = self._reset_primary_hidden(step.network, done_any, num_recurrent, slots)
        action_buffer = action.detach().clone()
        action_buffer[done_any] = 0
        self._refresh_opponent_team_flags(done_any)
        self._global_step += num_envs
        return PrimaryStepOutput(
            step.obs,
            hidden,
            hidden_t1,
            action_buffer,
            done_any,
        )
