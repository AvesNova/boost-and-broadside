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

from boost_and_broadside.env.observation import (
    ObsKey,
    YemongObservation,
    write_pending_action_view,
)
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.models.yemong.policy import YemongPolicy
from boost_and_broadside.runtime.actions import PendingActionState
from boost_and_broadside.train.rl.belief import BeliefTracker, DualBeliefTracker
from boost_and_broadside.train.rl.physical_belief import physical_means_from_state
from boost_and_broadside.train.rl.roster import RosterEntry


@dataclasses.dataclass
class LeagueSlot:
    """One fixed-shape generation of a logical league slot.

    ``active`` selects the environments whose current episode belongs to this
    generation. During replacement, the old and new generations share the same
    contiguous block; completed episodes move to the new generation while
    unfinished episodes keep their old controller, recurrent state, and belief.

    A ``policy`` of None means the scripted agent acts for this slot: it has no
    weights to load and no recurrent state to carry, which is why it is the one
    entry kind that costs no extra forward pass.
    """

    start: int
    end: int
    entry: RosterEntry
    policy: YemongPolicy | None
    hidden: torch.Tensor | None
    belief: BeliefTracker | None = None
    #: This slot's own legal team-1 view, composed at the end of the previous
    #: decision. Only slots with a belief carry one.
    obs: YemongObservation | None = None
    active: torch.Tensor | None = None
    target_active: torch.Tensor | None = None
    retiring: bool = False
    replacement_for: "LeagueSlot | None" = dataclasses.field(default=None, repr=False)


class RolloutNetworkOutput(NamedTuple):
    """Policy outputs computed alongside one environment step."""

    action_t0: torch.Tensor
    action_t1: torch.Tensor | None
    logprob: torch.Tensor
    value_norm: torch.Tensor
    pred_next_t0: torch.Tensor
    pred_next_t1: torch.Tensor | None
    enemy_action_logits_t0: torch.Tensor
    enemy_action_logits_t1: torch.Tensor | None
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
    action_state: PendingActionState
    # done | truncated — the GAE boundary, not physics termination alone.
    terminated: torch.Tensor


class EnvironmentStepOutput(NamedTuple):
    """Environment and policy outputs computed concurrently when CUDA is available.

    No observation: composing one needs the belief's forecast, which needs the
    policy output this step produced, so the view is built after the two streams
    join rather than inside the environment's half.
    """

    reward: torch.Tensor
    dones: torch.Tensor
    truncated: torch.Tensor
    transition_contiguous: torch.Tensor
    actuator_contiguous: torch.Tensor
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
        team1 = obs_slice.for_team(1)
        return flip_team_obs(team1, num_ships) if self._ego_pass else team1

    def _compose_league_views(
        self,
        slots: list[LeagueSlot],
        action_state: PendingActionState,
        num_ships: int,
        *,
        only_new: bool = False,
    ) -> None:
        """Build each belief-bearing league slot's own legal team-1 view.

        A slot with its own belief cannot reuse the live policy's team-1 view:
        that view was composed against the *trainee's* memory of the
        battlefield, and this opponent is a different policy whose forecasts and
        sightings are its own. Both views are team-1 legal, so sharing one would
        not leak anything -- it would attribute the wrong memory to the wrong
        player, which makes the opponent's behaviour unreproducible from its own
        weights.

        Slots without a belief are the shared-pass configuration, which has no
        fog and therefore nothing observer-specific to compose; they keep
        slicing the shared view.

        ``only_new`` composes for slots that do not have a view yet, which is
        what a rollout-shard boundary wants: nothing stepped between the last
        decision of the previous shard and the first of this one, so recomposing
        a surviving slot would run ``observe`` twice for one decision and age its
        hidden ships at double rate.
        """

        for slot in slots:
            if slot.policy is None or slot.belief is None:
                continue
            if only_new and slot.obs is not None:
                continue
            state = slice_state(self.wrapper.env.state, slot.start, slot.end)
            visibility = self.wrapper.last_visibility.ship[slot.start : slot.end, 1]
            bullets = self.wrapper.last_visibility.bullet
            source = slot.belief.observe(state, visibility)
            view = self._build_opponent_view(
                state,
                self.ship_config,
                include_bullets=self.wrapper.include_bullets,
                ship_visibility=visibility,
                bullet_visibility=(None if bullets is None else bullets[slot.start : slot.end, 1]),
                perspective_team=1,
                belief=source,
            )
            # The command physics will consume next, rather than the one already
            # spent, exactly as the shared view carries it.
            write_pending_action_view(
                view.data[ObsKey.PREVIOUS_ACTION][:, :num_ships],
                action_state.pending[slot.start : slot.end],
                state.ship_team_id[:, :num_ships],
                observer_team=1,
                spawn_revealed=state.ship_spawned[:, :num_ships],
                belief_action=source.action,
            )
            slot.obs = flip_team_obs(view, num_ships) if self._ego_pass else view

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
        slot: LeagueSlot,
        opp_action: torch.Tensor,
    ) -> None:
        """Replace the opponent side only where this generation is active."""
        start, end = slot.start, slot.end
        if self._ego_pass:
            opp_mask = team_id[start:end] == 1
        else:
            flags = self._opp_team_flag[start - self.B_self : end - self.B_self]
            opp_mask = team_id[start:end] == flags.unsqueeze(1)
        if slot.active is not None:
            opp_mask &= slot.active.unsqueeze(-1)
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

    def _prepare_league_slots(self, num_ships: int) -> list[LeagueSlot]:
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
                policy.initial_hidden(
                    slot_width, policy.recurrent_token_count(num_ships), self.device
                )
                if policy is not None
                else None
            )
            belief = (
                BeliefTracker(
                    slot_width,
                    num_ships,
                    self.ship_config.dt * self.env_config.action_repeat,
                    self.ship_config,
                    self.device,
                )
                if policy is not None and self._ego_pass
                else None
            )
            slots.append(
                LeagueSlot(
                    start=offset,
                    end=offset + slot_width,
                    entry=entry,
                    policy=policy,
                    hidden=hidden,
                    belief=belief,
                    active=torch.ones(slot_width, dtype=torch.bool, device=self.device),
                    target_active=torch.ones(slot_width, dtype=torch.bool, device=self.device),
                )
            )
            offset += slot_width
        return slots

    def _begin_league_replacement(
        self,
        slots: list[LeagueSlot],
        num_ships: int,
    ) -> None:
        """Retire drained generations and, when possible, rotate one logical slot.

        Called once per rollout shard. The only device-to-host checks are here,
        never in the per-step path. At most one replacement drains at a time,
        bounding fixed-shape policy forwards to ``K + 1`` generations. Checkpoint
        weight residency is governed separately by the roster's LRU cache.
        """
        desired_start = self.cfg.scales[0].num_envs - self._active_league_width()
        layout_changed = False
        for slot in slots:
            indices = torch.arange(slot.start, slot.end, device=self.device)
            desired = indices >= desired_start
            if slot.target_active is None or not torch.equal(slot.target_active, desired):
                slot.target_active = desired
                layout_changed = True

        if slots and desired_start < min(slot.start for slot in slots):
            start = desired_start
            end = min(slot.start for slot in slots)
            entry = self._sample_league_entry()
            if entry is not None:
                policy = self._league_policy(entry)
                width = end - start
                slots.append(
                    LeagueSlot(
                        start=start,
                        end=end,
                        entry=entry,
                        policy=policy,
                        hidden=(
                            policy.initial_hidden(
                                width, policy.recurrent_token_count(num_ships), self.device
                            )
                            if policy is not None
                            else None
                        ),
                        belief=(
                            BeliefTracker(
                                width,
                                num_ships,
                                self.ship_config.dt * self.env_config.action_repeat,
                                self.ship_config,
                                self.device,
                            )
                            if policy is not None and self._ego_pass
                            else None
                        ),
                        active=torch.zeros(width, dtype=torch.bool, device=self.device),
                        target_active=torch.ones(width, dtype=torch.bool, device=self.device),
                    )
                )
                layout_changed = True

        for old in [slot for slot in slots if slot.retiring]:
            if bool(old.active.any()):
                return
            replacement = next(slot for slot in slots if slot.replacement_for is old)
            replacement.replacement_for = None
            slots.remove(old)

        slots[:] = [
            slot
            for slot in slots
            if slot.replacement_for is not None
            or slot.retiring
            or bool(slot.active.any())
            or bool(slot.target_active.any())
        ]

        if layout_changed or any(slot.retiring for slot in slots) or not slots:
            return

        logical = [
            slot for slot in slots if slot.replacement_for is None and bool(slot.active.any())
        ]
        if not logical:
            return
        rotation = getattr(self, "_league_rotation_index", 0) % len(logical)
        old = logical[rotation]
        self._league_rotation_index = rotation + 1
        entry = self._sample_league_entry()
        if entry is None:
            return
        policy = self._league_policy(entry)
        width = old.end - old.start
        replacement = LeagueSlot(
            start=old.start,
            end=old.end,
            entry=entry,
            policy=policy,
            hidden=(
                policy.initial_hidden(width, policy.recurrent_token_count(num_ships), self.device)
                if policy is not None
                else None
            ),
            belief=(
                BeliefTracker(
                    width,
                    num_ships,
                    self.ship_config.dt * self.env_config.action_repeat,
                    self.ship_config,
                    self.device,
                )
                if policy is not None and self._ego_pass
                else None
            ),
            active=torch.zeros(width, dtype=torch.bool, device=self.device),
            target_active=old.target_active.clone(),
            replacement_for=old,
        )
        old.retiring = True
        slots.append(replacement)

    @staticmethod
    def _advance_league_replacements(
        slots: list[LeagueSlot],
        done_any: torch.Tensor,
    ) -> None:
        """Move newly reset episodes from a retiring generation to its replacement."""
        for replacement in [slot for slot in slots if slot.replacement_for is not None]:
            old = replacement.replacement_for
            completed = done_any[old.start : old.end] & old.active
            old.active &= ~completed
            replacement.active |= completed & old.target_active

        for slot in [slot for slot in slots if slot.replacement_for is None and not slot.retiring]:
            completed = done_any[slot.start : slot.end]
            slot.active |= completed & slot.target_active
            slot.active &= ~(completed & ~slot.target_active)

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
                pred_next_t0,
                pred_next_t1,
                hidden,
                enemy_action_logits_t0,
                enemy_action_logits_t1,
                hidden_t1,
            ) = self._rollout_policy_pass(obs, hidden, hidden_t1, num_ships, num_recurrent)

        slot_actions: list[torch.Tensor | None] = []
        for slot in slots:
            if slot.policy is None:
                slot_actions.append(None)
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # A slot with a belief reads the view composed for it at the end
                # of the previous decision. It cannot be composed here: the
                # environment step is running concurrently on another stream, so
                # ``env.state`` is mid-flight.
                obs_slot = (
                    slot.obs
                    if slot.belief is not None
                    else self._opponent_obs(slice_obs(obs, slot.start, slot.end), num_ships)
                )
                action, _, _, prediction, enemy_logits, slot.hidden = (
                    slot.policy.get_action_and_value(
                        obs_slot, slot.hidden, return_enemy_action=True
                    )
                )
                if slot.belief is not None:
                    slot.belief.advance(prediction, enemy_logits)
            slot_actions.append(action)

        return RolloutNetworkOutput(
            action_t0=action_t0,
            action_t1=action_t1,
            logprob=logprob,
            value_norm=value_norm,
            pred_next_t0=pred_next_t0,
            pred_next_t1=pred_next_t1,
            enemy_action_logits_t0=enemy_action_logits_t0,
            enemy_action_logits_t1=enemy_action_logits_t1,
            hidden=hidden,
            hidden_t1=hidden_t1,
            slot_actions=slot_actions,
        )

    def _scripted_step_outputs(self, slots: list[LeagueSlot]) -> ScriptedStepOutput:
        """Compute scripted BC targets and scripted-slot actions before stream launch."""
        if self._policy_gradient_coef == 0.0:
            with torch.no_grad():
                _, expert_probs = self.scripted_agent.get_actions_and_probs(
                    self.wrapper.env.state, self.wrapper.last_visibility.ship
                )
            return ScriptedStepOutput(expert_probs, {})

        scripted_slots = [index for index, slot in enumerate(slots) if slot.policy is None]
        if self._behavior_cloning_coef > 0.0 and self.scripted_agent is not None:
            # BC needs targets for every env anyway, so slot actions come free.
            with torch.no_grad():
                actions, expert_probs = self.scripted_agent.get_actions_and_probs(
                    self.wrapper.env.state, self.wrapper.last_visibility.ship
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
                slice_state(self.wrapper.env.state, low, high),
                self.wrapper.last_visibility.ship[low:high],
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
        action_state: PendingActionState,
        network_args: tuple,
        env_stream: torch.cuda.Stream | None,
        net_stream: torch.cuda.Stream | None,
    ) -> EnvironmentStepOutput:
        """Advance the environment and policy, overlapping them on CUDA streams."""
        if env_stream is None:
            _, reward, dones, truncated, info = self.wrapper.step(
                action_state.applied_action(), observe=False
            )
            network = self._rollout_network_forwards(*network_args)
            return EnvironmentStepOutput(
                reward,
                dones,
                truncated,
                info["transition_contiguous"],
                info["actuator_contiguous"],
                network,
            )

        env_stream.wait_stream(torch.cuda.current_stream())
        net_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(env_stream):
            _, reward, dones, truncated, info = self.wrapper.step(
                action_state.applied_action(), observe=False
            )
        with torch.cuda.stream(net_stream):
            network = self._rollout_network_forwards(*network_args)
        torch.cuda.current_stream().wait_stream(env_stream)
        torch.cuda.current_stream().wait_stream(net_stream)
        return EnvironmentStepOutput(
            reward,
            dones,
            truncated,
            info["transition_contiguous"],
            info["actuator_contiguous"],
            network,
        )

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
            self._apply_opponent_override(action, actor_mask, team_id, slot, opponent_action)
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
                # A league policy owns its recurrent stride, which need not be
                # the live policy's.
                slot.hidden = slot.policy.reset_hidden_for_envs(
                    slot.hidden,
                    done_any[slot.start : slot.end],
                    slot.hidden.shape[1] // (slot.end - slot.start),
                )
                if slot.belief is not None:
                    slot.belief.reset(done_any[slot.start : slot.end])
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
        beliefs: DualBeliefTracker | None,
        hidden: torch.Tensor,
        hidden_t1: torch.Tensor | None,
        action_state: PendingActionState,
        num_envs: int,
        num_ships: int,
        num_recurrent: int,
        slots: list[LeagueSlot],
        env_stream: torch.cuda.Stream | None,
        net_stream: torch.cuda.Stream | None,
    ) -> PrimaryStepOutput:
        """Collect one primary-scale transition and update recurrent rollout state."""
        # Ground truth, never the observation's copy. Hidden ships are zeroed in
        # ``ObsKey.TEAM_ID`` and zero is also Team 0's real id, so that channel
        # cannot distinguish "on team 0" from "not currently seen". Read before
        # the step: ``reset_envs`` reshuffles slot-to-team assignment, and these
        # actions were selected under the pre-step one.
        team_id = self.wrapper.env.state.ship_team_id[:, :num_ships]
        privileged_means = physical_means_from_state(self.wrapper.env.state, self.ship_config)[
            :, :num_ships
        ]
        scripted = self._scripted_step_outputs(slots)
        network_args = (obs, hidden, hidden_t1, num_ships, num_recurrent, slots)
        step = self._step_environment_and_network(
            action_state, network_args, env_stream, net_stream
        )
        action, actor_mask = self._select_primary_actions(step.network, scripted, team_id, slots)
        done_any = step.dones | step.truncated
        decision_committed = action_state.commit(
            action,
            step.actuator_contiguous,
            done_any,
        )
        self.buffer.add(
            obs=obs,
            action=action,
            logprob=step.network.logprob,
            reward=step.reward,
            value=self.scaler.denormalize(step.network.value_norm),
            alive=obs["alive"][:, :num_ships].bool(),
            actor_mask=actor_mask,
            decision_committed=decision_committed,
            expert_probs=scripted.expert_probs,
            terminated=done_any,
            transition_contiguous=step.transition_contiguous,
            privileged_means=privileged_means,
            scaled_predictions=step.network.pred_next_t0,
        )

        hidden, hidden_t1 = self._reset_primary_hidden(step.network, done_any, num_recurrent, slots)
        self._advance_league_replacements(slots, done_any)
        # Belief before observation, in that order and only now: the forecast is
        # this decision's policy output, which the environment step ran
        # concurrently with, so the composition cannot be queued alongside the
        # physics. ``reset`` precedes assimilation so a finished episode's memory
        # never reaches the view built from its successor's opening state.
        if beliefs is not None:
            beliefs.advance(
                step.network.pred_next_t0,
                step.network.pred_next_t1,
                step.network.enemy_action_logits_t0,
                step.network.enemy_action_logits_t1,
            )
            beliefs.reset(done_any)
            sources = beliefs.observe(self.wrapper.env.state, self.wrapper.last_visibility.ship)
        else:
            sources = None
        next_obs = self.wrapper.observe(sources)
        self._compose_league_views(slots, action_state, num_ships)
        # Re-read after the step: ``next_obs`` describes the post-reset state, and
        # a reset reshuffles team assignment, so the privacy mask has to match the
        # observation it is being written into rather than the pre-step layout.
        action_state.write_observation(
            next_obs,
            self.wrapper.env.state.ship_team_id[:, :num_ships],
            self.wrapper.env.state.ship_spawned[:, :num_ships],
            num_ships,
            belief_action=None if sources is None else (sources[0].action, sources[1].action),
        )
        self._refresh_opponent_team_flags(done_any)
        self._global_step += num_envs
        return PrimaryStepOutput(
            next_obs,
            hidden,
            hidden_t1,
            action_state,
            done_any,
        )
