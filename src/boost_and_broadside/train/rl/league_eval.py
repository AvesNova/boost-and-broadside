"""Continuous information-scheduled evaluation across the complete league."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig, LeagueEvalConfig, ModelConfig, ShipConfig
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
from boost_and_broadside.train.rl.features import FeatureCoordinator
from boost_and_broadside.train.rl.rating import (
    DRAW,
    LOSS,
    WIN,
    MatchCounts,
    information_pair_weights,
)
from boost_and_broadside.train.rl.roster import LeagueRoster


def schedule_eval_pairs(
    agent_ids: list[str],
    ratings: dict[str, float],
    standard_errors: dict[str, float],
    num_pairs: int,
    *,
    avg_active: bool,
    newest_checkpoint_id: str | None = None,
    generator: torch.Generator | None = None,
) -> list[tuple[str, str]]:
    """Select distinct directed pairs by expected information gain.

    Three pairs are guaranteed regardless of information weight: live vs
    scripted feeds the win-rate schedule gates, live vs the newest checkpoint
    keeps the pool's freshest rung measured (and drives gap-triggered
    admission), and one random pair keeps the canary metrics alive.
    """
    candidates = [(first, second) for first in agent_ids for second in agent_ids if first != second]
    if not candidates or num_pairs <= 0:
        return []
    weights = information_pair_weights(ratings, standard_errors, candidates)
    selected: list[tuple[str, str]] = []

    def add_required(predicate) -> None:
        eligible = [index for index, pair in enumerate(candidates) if predicate(pair)]
        eligible = [index for index in eligible if candidates[index] not in selected]
        if not eligible:
            return
        eligible_weights = weights[eligible]
        chosen = eligible[int(torch.argmax(eligible_weights))]
        selected.append(candidates[chosen])

    if "scripted" in agent_ids:
        add_required(lambda pair: "live" in pair and "scripted" in pair)
    if newest_checkpoint_id is not None and newest_checkpoint_id in agent_ids:
        add_required(lambda pair: "live" in pair and newest_checkpoint_id in pair)
    add_required(lambda pair: "live" in pair)
    if avg_active and "avg" in agent_ids:
        add_required(lambda pair: "avg" in pair)
    add_required(lambda pair: "random" in pair)

    target = min(num_pairs, len(candidates))
    if len(selected) >= target:
        return selected[:target]
    remaining = [index for index, pair in enumerate(candidates) if pair not in selected]
    remaining_weights = weights[remaining]
    fill_count = min(target - len(selected), len(remaining))
    if fill_count > 0:
        sampled = torch.multinomial(
            remaining_weights,
            fill_count,
            replacement=False,
            generator=generator,
        )
        selected.extend(candidates[remaining[index]] for index in sampled.tolist())
    return selected


@dataclass
class _PolicySlot:
    """One persistent compiled policy whose weights may be swapped per block."""

    module: MVPPolicy
    policy: MVPPolicy
    agent_id: str | None = None


class LeagueEvaluator:
    """Run scheduled league pairs and accumulate outcomes into shared GPU counts."""

    def __init__(
        self,
        config: LeagueEvalConfig,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        model_config: ModelConfig,
        coordinator: FeatureCoordinator,
        num_value_components: int,
        team_pma_k: tuple[int, ...],
        device: torch.device,
        obstacle_cache: ObstacleCache | None,
        roster: LeagueRoster,
        live_policy: MVPPolicy,
        avg_policy: MVPPolicy,
        scripted_agent: StochasticScriptedAgent | None,
        num_ships: int,
        num_tokens: int,
        compile_mode: str | None,
    ) -> None:
        self.config = config
        self.ship_config = ship_config
        self.device = device
        self.roster = roster
        self.live_policy = live_policy
        self.avg_policy = avg_policy
        self.scripted_agent = scripted_agent
        self.num_ships = num_ships
        self.num_tokens = num_tokens
        eval_env_config = dataclasses.replace(
            env_config,
            single_team=False,
            max_episode_steps=config.max_episode_steps,
        )
        self.env = TensorEnv(
            config.eval_num_envs,
            ship_config,
            eval_env_config,
            device,
            obstacle_cache,
        )
        # Envs are hard-reset again at every block assignment so first episodes
        # are attributable; no step-count randomization — blocks align to
        # fresh, honest-length episodes.
        self.env.reset()
        self._slots = [
            self._build_slot(
                model_config,
                coordinator,
                num_value_components,
                team_pma_k,
                compile_mode,
            )
            for _ in range(config.eval_slots)
        ]
        self._agents: list[ResolvedAgent] = []
        self._active_envs: list[torch.Tensor] = []
        self._env_agent0 = torch.zeros(config.eval_num_envs, dtype=torch.long, device=device)
        self._env_agent1 = torch.zeros(config.eval_num_envs, dtype=torch.long, device=device)
        self._env_count0 = torch.zeros(config.eval_num_envs, dtype=torch.long, device=device)
        self._env_count1 = torch.zeros(config.eval_num_envs, dtype=torch.long, device=device)
        # Attribution validity: hard resets at block assignment mean every env
        # is valid from its first episode; the mask exists so recording can be
        # suspended if an assignment ever changes without a reset.
        self._attribution_valid = torch.zeros(config.eval_num_envs, dtype=torch.bool, device=device)
        self._blocks_remaining = 0
        self._scheduled_layout_version = -1
        self._current_pairs: list[tuple[str, str]] = []
        self._arange_envs = torch.arange(config.eval_num_envs, device=device)
        self._all_actions = torch.empty(
            0,
            config.eval_num_envs,
            num_ships,
            3,
            dtype=torch.int32,
            device=device,
        )  # (A, B_eval, N, 3)
        self._stream = torch.cuda.Stream() if device.type == "cuda" else None

    def _build_slot(
        self,
        model_config: ModelConfig,
        coordinator: FeatureCoordinator,
        num_value_components: int,
        team_pma_k: tuple[int, ...],
        compile_mode: str | None,
    ) -> _PolicySlot:
        """Construct one checkpoint policy slot, compiling it exactly once."""
        module = MVPPolicy(
            model_config,
            coordinator,
            num_value_components=num_value_components,
            num_ships=self.num_ships,
            team_pma_k=team_pma_k,
        ).to(self.device)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        policy = (
            torch.compile(module, mode=compile_mode, dynamic=True)
            if compile_mode is not None
            else module
        )
        return _PolicySlot(module, policy)

    def _load_checkpoint_slots(self) -> dict[str, MVPPolicy]:
        """Assign the most uncertain frozen checkpoints to slots, loading only changes."""
        checkpoints = [entry for entry in self.roster.entries if entry.kind == "checkpoint"]
        checkpoints.sort(
            key=lambda entry: (
                self.roster.standard_errors.get(entry.agent_id, float("inf")),
                entry.update,
            ),
            reverse=True,
        )
        desired = [
            entry for entry in checkpoints if entry.path is not None and Path(entry.path).exists()
        ][: len(self._slots)]
        desired_ids = {entry.agent_id for entry in desired}
        held = {slot.agent_id: slot for slot in self._slots if slot.agent_id is not None}
        free_slots = [slot for slot in self._slots if slot.agent_id not in desired_ids]
        loaded: dict[str, MVPPolicy] = {}
        for entry in desired:
            slot = held.get(entry.agent_id)
            if slot is None:
                slot = free_slots.pop()
                checkpoint = torch.load(entry.path, map_location=self.device, weights_only=False)
                slot.module.load_state_dict(checkpoint["policy_state_dict"])
                for module in slot.module.modules():
                    if isinstance(module, nn.RMSNorm) and module.weight.is_cuda:
                        module.weight.data = module.weight.data.bfloat16()
                slot.agent_id = entry.agent_id
            loaded[entry.agent_id] = slot.policy
        return loaded

    def begin_rollout(self, avg_active: bool) -> list[tuple[str, str]]:
        """Schedule a new evaluation block when the current one expires.

        Every (re)schedule hard-resets all eval environments, so first
        episodes start fresh under the new assignment and are immediately
        attributable — no discarded watermark episode. A roster layout change
        (admission or eviction shifts dense count indices) forces an immediate
        reschedule so stale indices are never recorded into.
        """
        roster_changed = self.roster.layout_version != self._scheduled_layout_version
        if self._blocks_remaining > 0 and not roster_changed:
            self._blocks_remaining -= 1
            return self._current_pairs
        checkpoint_policies = self._load_checkpoint_slots()
        available_ids = ["live", "random"]
        if self.scripted_agent is not None:
            available_ids.append("scripted")
        if avg_active and "avg" in self.roster.counts.agent_ids:
            available_ids.append("avg")
        available_ids.extend(checkpoint_policies)
        newest = self.roster.newest_checkpoint()
        pairs = schedule_eval_pairs(
            available_ids,
            self.roster.ratings,
            self.roster.standard_errors,
            self.config.eval_pairs,
            avg_active=avg_active,
            newest_checkpoint_id=newest.agent_id if newest is not None else None,
        )
        if not pairs:
            raise RuntimeError("league evaluation requires at least two executable agents")
        participating_ids = {agent_id for pair in pairs for agent_id in pair}
        scheduled_ids = [agent_id for agent_id in available_ids if agent_id in participating_ids]
        self._assign_agents(scheduled_ids, checkpoint_policies)
        self._assign_env_pairs(pairs)
        # Hard reset: fresh episodes under the new assignment record from their
        # first completion. flush() synchronized the eval stream last update,
        # so mutating env state on the default stream here is race-free.
        self.env.reset_envs(torch.ones_like(self._attribution_valid))
        self._attribution_valid.fill_(True)
        self._scheduled_layout_version = self.roster.layout_version
        self._blocks_remaining = self.config.eval_block_rollouts - 1
        self._current_pairs = pairs
        return pairs

    def _assign_agents(
        self,
        agent_ids: list[str],
        checkpoint_policies: dict[str, MVPPolicy],
    ) -> None:
        """Resolve active evaluator agents for one scheduled block."""
        agents = {
            "live": ResolvedAgent("policy", self.live_policy),
            "random": ResolvedAgent("random", None),
        }
        if self.scripted_agent is not None:
            agents["scripted"] = ResolvedAgent("scripted", self.scripted_agent)
        if "avg" in agent_ids:
            agents["avg"] = ResolvedAgent("policy", self.avg_policy)
        agents.update(
            {
                agent_id: ResolvedAgent("policy", policy)
                for agent_id, policy in checkpoint_policies.items()
            }
        )
        self._agents = [agents[agent_id] for agent_id in agent_ids]
        self._agent_ids = agent_ids
        agent_count = len(agent_ids)
        if self._all_actions.shape[0] != agent_count:
            self._all_actions = torch.zeros(
                agent_count,
                self.config.eval_num_envs,
                self.num_ships,
                3,
                dtype=torch.int32,
                device=self.device,
            )  # (A, B_eval, N, 3)

    def _assign_env_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Partition evaluator environments evenly across directed pairs."""
        pair_count = min(len(pairs), self.config.eval_num_envs)
        pairs = pairs[:pair_count]
        base, remainder = divmod(self.config.eval_num_envs, pair_count)
        agent_indices = {agent_id: index for index, agent_id in enumerate(self._agent_ids)}
        offset = 0
        for pair_index, (first_id, second_id) in enumerate(pairs):
            size = base + (1 if pair_index < remainder else 0)
            env_slice = slice(offset, offset + size)
            self._env_agent0[env_slice] = agent_indices[first_id]
            self._env_agent1[env_slice] = agent_indices[second_id]
            self._env_count0[env_slice] = self.roster.counts.index(first_id)
            self._env_count1[env_slice] = self.roster.counts.index(second_id)
            offset += size
        self._active_envs = []
        for agent_index, agent in enumerate(self._agents):
            active = (
                (self._env_agent0 == agent_index) | (self._env_agent1 == agent_index)
            ).nonzero(as_tuple=True)[0]
            self._active_envs.append(active)
            init_hidden(agent, active.shape[0], self.num_tokens, self.device)

    def step(self, rollout_step: int) -> None:
        """Advance evaluation on its configured rollout cadence."""
        if rollout_step % self.config.step_interval != 0:
            return
        if self._stream is None:
            self._step_evaluation()
            return
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self._step_evaluation()

    def _step_evaluation(self) -> None:
        """Run one evaluator transition on the active stream."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.env.state
            obs = observation_from_state(state, self.ship_config)
            for agent_index, (agent, active) in enumerate(zip(self._agents, self._active_envs)):
                if agent.kind == "scripted":
                    self._all_actions[agent_index] = get_actions(
                        agent,
                        None,
                        state,
                        self.config.eval_num_envs,
                        self.num_ships,
                        self.device,
                    )
                    continue
                active_obs = MVPObservation(data={key: value[active] for key, value in obs.items()})
                actions = get_actions(
                    agent,
                    active_obs if agent.kind == "policy" else None,
                    state,
                    active.shape[0],
                    self.num_ships,
                    self.device,
                )
                self._all_actions[agent_index, active] = actions.int()

            team0_actions = self._all_actions[self._env_agent0, self._arange_envs]
            team1_actions = self._all_actions[self._env_agent1, self._arange_envs]
            actions = torch.where(
                (state.ship_team_id == 0).unsqueeze(-1), team0_actions, team1_actions
            )
            dones, truncated = self.env.step(actions)
            done_any = dones | truncated

        alive = self.env.state.ship_alive
        team = self.env.state.ship_team_id
        team0_alive = (alive & (team == 0)).any(dim=1)
        team1_alive = (alive & (team == 1)).any(dim=1)
        team0_won = team0_alive & ~team1_alive
        team1_won = team1_alive & ~team0_alive
        outcomes = torch.where(
            team0_won,
            torch.full_like(self._env_count0, WIN),
            torch.where(
                team1_won,
                torch.full_like(self._env_count0, LOSS),
                torch.full_like(self._env_count0, DRAW),
            ),
        )
        recordable = done_any & self._attribution_valid
        self.roster.record_outcomes(
            self._env_count0,
            self._env_count1,
            outcomes,
            recordable.to(self.roster.counts.tensor.dtype),
        )
        self.env.reset_envs(done_any)
        for agent, active in zip(self._agents, self._active_envs):
            reset_done_envs(agent, done_any[active], self.num_tokens)

    def flush(self) -> tuple[MatchCounts, MatchCounts]:
        """Snapshot lifetime counts and drain the per-update outcome tally."""
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        counts = self.roster.counts.to_cpu()
        tally = self.roster.update_tally.to_cpu()
        self.roster.reset_update_tally()
        return counts, tally
