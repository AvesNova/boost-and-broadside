from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from boost_and_broadside.agents.scripted_utils import (
    compute_team_target_bearings,
    predict_interception,
    select_targets,
)
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.config import ShipConfig, ZoneRole
from boost_and_broadside.env.frontline import (
    toroidal_displacement,
    wrap_positions,
    zone_membership,
)
from boost_and_broadside.env.state import TensorState


class FrontlineTendency(IntEnum):
    """Episode-long strategic preference for a scripted ship."""

    OFFENSIVE = 0
    DEFENSIVE = 1
    TIMID = 2


@dataclass
class _FrontlineMemory:
    """Mutable decisions that must survive controller calls within an episode."""

    tendencies: torch.Tensor
    healing: torch.Tensor
    tie_attack: torch.Tensor
    last_step_count: torch.Tensor


class StochasticScriptedAgent:
    """
    A stochastic scripted agent that uses configurable probability ramps
    instead of hard boundaries.

    Outputs probability distributions over the three action heads
    (power × turn × shoot) using physics-grounded heuristics:
    aim at nearest enemy, boost when far, reverse when close, shoot when aligned.
    """

    def __init__(self, ship_config: ShipConfig, agent_config: StochasticAgentConfig):
        self.ship_config = ship_config
        self.config = agent_config
        self._frontline_state: TensorState | None = None
        self._frontline_memory: _FrontlineMemory | None = None

    def _new_frontline_memory(self, state: TensorState) -> _FrontlineMemory:
        batch_size, num_ships = state.ship_pos.shape
        identity_draw = torch.rand((batch_size, num_ships), device=state.device)
        tendencies = torch.where(
            identity_draw < 0.5,
            int(FrontlineTendency.OFFENSIVE),
            torch.where(
                identity_draw < 0.75,
                int(FrontlineTendency.DEFENSIVE),
                int(FrontlineTendency.TIMID),
            ),
        )
        return _FrontlineMemory(
            tendencies=tendencies,
            healing=torch.zeros((batch_size, num_ships), dtype=torch.bool, device=state.device),
            # Timid ships follow the non-timid majority. A stable per-team coin
            # breaks exact offensive/defensive ties without favoring either role.
            tie_attack=torch.rand((batch_size, 2), device=state.device) < 0.5,
            last_step_count=state.step_count.clone(),
        )

    def _frontline_episode_memory(self, state: TensorState) -> _FrontlineMemory:
        """Return episode memory, rerolling only environments that reset.

        A scripted controller is called with the authoritative state, sometimes
        more than once for the same frame (for example when it controls both
        teams). Tracking the last observed counter makes the reset operation
        idempotent while preserving tendencies through same-slot respawns.
        """

        if self._frontline_state is not state or self._frontline_memory is None:
            self._frontline_state = state
            self._frontline_memory = self._new_frontline_memory(state)
            return self._frontline_memory

        memory = self._frontline_memory
        if memory.tendencies.shape != state.ship_pos.shape:
            memory = self._new_frontline_memory(state)
            self._frontline_memory = memory
            return memory

        reset = (state.step_count == 0) & (memory.last_step_count != 0)
        if reset.any():
            fresh = self._new_frontline_memory(state)
            ship_reset = reset.unsqueeze(1)
            team_reset = reset.unsqueeze(1)
            memory.tendencies = torch.where(ship_reset, fresh.tendencies, memory.tendencies)
            memory.healing = memory.healing & ~ship_reset
            memory.tie_attack = torch.where(team_reset, fresh.tie_attack, memory.tie_attack)
        memory.last_step_count = state.step_count.clone()
        return memory

    def _linear_ramp(
        self, x: torch.Tensor, low: float, high: float, prob_lo: float, prob_hi: float
    ) -> torch.Tensor:
        """
        Maps x linearly from [low, high] to [prob_lo, prob_hi], clamped.
        prob_lo is the output probability when x <= low.
        prob_hi is the output probability when x >= high.
        For an inverted ramp, pass prob_lo=1.0, prob_hi=0.0.
        """
        if high == low:
            return torch.full_like(x, prob_lo)
        t = torch.clamp((x - low) / (high - low), 0.0, 1.0)
        return prob_lo + t * (prob_hi - prob_lo)

    def _prob_or(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        """Independent OR logic: P(A or B) = P(A) + P(B) - P(A)*P(B)"""
        return p_a + p_b - (p_a * p_b)

    def _prob_and(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        """Independent AND logic: P(A and B) = P(A) * P(B)"""
        return p_a * p_b

    def _compute_action_probs(
        self,
        state: TensorState,
        closest_dist: torch.Tensor,
        dir_turn: torch.Tensor,
        dir_shoot: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute marginal probability distributions for Power(3), Turn(7), Shoot(2).

        dir_turn:  direction used for turn decisions (may be team-target-blended).
        dir_shoot: direction used for shoot alignment (always personal intercept).

        Returns unmasked distributions of shape (B, N, C).
        """
        batch_size, num_ships = state.ship_pos.shape

        att = state.ship_attitude
        rel_angle = torch.angle(dir_turn * torch.conj(att))  # range (-pi, pi)
        abs_angle = torch.abs(rel_angle)

        # --- 1. Turn Probabilities (7 options) ---
        p_needs_turn = self._linear_ramp(
            abs_angle,
            self.config.turn_angle_ramp[0],
            self.config.turn_angle_ramp[1],
            *self.config.turn_angle_prob,
        )
        p_is_sharp = self._linear_ramp(
            abs_angle,
            self.config.sharp_turn_angle_ramp[0],
            self.config.sharp_turn_angle_ramp[1],
            *self.config.sharp_turn_angle_prob,
        )

        p_dir_right = torch.where(
            rel_angle > 0, torch.ones_like(rel_angle), torch.zeros_like(rel_angle)
        )
        p_dir_left = torch.where(
            rel_angle < 0, torch.ones_like(rel_angle), torch.zeros_like(rel_angle)
        )

        p_turn_left_base = self._prob_and(p_needs_turn, p_dir_left)
        p_turn_right_base = self._prob_and(p_needs_turn, p_dir_right)

        p_sharp_left = self._prob_and(p_turn_left_base, p_is_sharp)
        p_sharp_right = self._prob_and(p_turn_right_base, p_is_sharp)
        p_normal_left = self._prob_and(p_turn_left_base, 1.0 - p_is_sharp)
        p_normal_right = self._prob_and(p_turn_right_base, 1.0 - p_is_sharp)
        p_straight = 1.0 - p_needs_turn

        p_air_brake = torch.zeros_like(p_straight)
        p_sharp_air_brake = torch.zeros_like(p_straight)

        turn_probs = torch.stack(
            [
                p_straight,
                p_normal_left,
                p_normal_right,
                p_sharp_left,
                p_sharp_right,
                p_air_brake,
                p_sharp_air_brake,
            ],
            dim=-1,
        )
        turn_probs = turn_probs / (turn_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # --- 2. Power Probabilities (3 options) ---
        speed = state.ship_vel.abs()
        power_ratio = state.ship_power / self.ship_config.max_power

        p_is_close = self._linear_ramp(
            closest_dist,
            self.config.close_range_ramp[0],
            self.config.close_range_ramp[1],
            *self.config.close_range_prob,
        )
        p_reverse = p_is_close

        p_slow = self._linear_ramp(
            speed,
            self.config.boost_speed_ramp[0],
            self.config.boost_speed_ramp[1],
            *self.config.boost_speed_prob,
        )

        max_shooting_range = self.config.shoot_distance_ramp[1]
        boost_metric = power_ratio - (1.0 - closest_dist / max_shooting_range)
        p_far_power = self._linear_ramp(boost_metric, -0.2, 0.2, 0.0, 1.0)

        p_want_boost = self._prob_or(p_slow, p_far_power)
        p_boost = self._prob_and(p_want_boost, 1.0 - p_is_close)
        p_coast = (1.0 - self._prob_or(p_want_boost, p_reverse)).clamp(0.0, 1.0)

        power_probs = torch.stack([p_coast, p_boost, p_reverse], dim=-1)
        power_probs = power_probs / (power_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # --- 3. Shoot Probabilities (2 options) ---
        target_angular_size = 2.0 * torch.atan(
            self.ship_config.collision_radius / (closest_dist + 1e-8)
        )
        shoot_threshold = target_angular_size.clamp(np.deg2rad(1.0), np.deg2rad(45.0))

        shoot_rel_angle = torch.angle(dir_shoot * torch.conj(att))
        angle_ratio = shoot_rel_angle.abs() / (shoot_threshold + 1e-8)
        p_aligned = self._linear_ramp(
            angle_ratio,
            self.config.shoot_angle_ramp[0],
            self.config.shoot_angle_ramp[1],
            *self.config.shoot_angle_prob,
        )
        p_in_range = self._linear_ramp(
            closest_dist,
            self.config.shoot_distance_ramp[0],
            self.config.shoot_distance_ramp[1],
            *self.config.shoot_distance_prob,
        )

        p_shoot = self._prob_and(p_aligned, p_in_range)
        p_no_shoot = 1.0 - p_shoot

        shoot_probs = torch.stack([p_no_shoot, p_shoot], dim=-1)
        shoot_probs = shoot_probs / (shoot_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Mask inactive ships (dead or no target) → default no-op action
        active_expanded = active_mask.unsqueeze(-1)

        def apply_mask(probs: torch.Tensor, default_idx: int) -> torch.Tensor:
            mask_probs = torch.zeros_like(probs)
            mask_probs[..., default_idx] = 1.0
            return torch.where(active_expanded, probs, mask_probs)

        power_probs = apply_mask(power_probs, 0)
        turn_probs = apply_mask(turn_probs, 0)
        shoot_probs = apply_mask(shoot_probs, 0)

        return power_probs, turn_probs, shoot_probs

    def _frontline_targets(
        self,
        state: TensorState,
        closest_dist: torch.Tensor,
        target_idx: torch.Tensor,
        has_target: torch.Tensor,
        team_visibility: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose a local enemy, spawn, or strategic frontline destination.

        Each ship keeps an offensive, defensive, or timid tendency for the whole
        episode. One contested point pulls the fleet there; when both or neither
        are contested, non-timid ships follow their tendency and timid ships follow
        that team's non-timid majority. A point is contested from its owner's
        perspective whenever one or more enemy ships currently occupy it.

        Nearby fights override strategy and non-timid healing. Timid ships below
        the retreat threshold (and timid ships that just respawned) instead latch a
        retreat to their spawn until fully healed. Other respawned ships heal to
        full only while neither point is contested.

        Returns:
            distance:     ``(B, N)`` toroidal distance to the chosen destination.
            bearing:      ``(B, N)`` unit complex bearing to that destination.
            engage_enemy: ``(B, N)`` whether shooting should use the intercept.
        """

        team = state.ship_team_id
        team0 = team == 0
        roles = state.zone_roles

        def zone_for_role(role: ZoneRole) -> torch.Tensor:
            zone_idx = (roles == int(role)).long().argmax(dim=1)
            return state.zone_pos.gather(1, zone_idx.unsqueeze(1)).squeeze(1)

        team0_spawn = zone_for_role(ZoneRole.TEAM0_SPAWN)
        team1_spawn = zone_for_role(ZoneRole.TEAM1_SPAWN)
        own_spawn = torch.where(team0, team0_spawn.unsqueeze(1), team1_spawn.unsqueeze(1))

        team0_defense = zone_for_role(ZoneRole.TEAM0_DEFENSE)
        team1_defense = zone_for_role(ZoneRole.TEAM1_DEFENSE)
        own_defense = torch.where(team0, team0_defense.unsqueeze(1), team1_defense.unsqueeze(1))
        enemy_defense = torch.where(team0, team1_defense.unsqueeze(1), team0_defense.unsqueeze(1))

        world_size = self.ship_config.world_size
        spawnward = toroidal_displacement(own_spawn - own_defense, world_size)
        spawnward = spawnward / spawnward.abs().clamp(min=1e-8)
        # The Gate-1 map currently gives every zone one radius. Read it from
        # authoritative state so the holding position follows future map tuning.
        defense_patrol_radius = state.zone_radius[:, :1] + (2.0 * self.ship_config.collision_radius)
        defense_radial = toroidal_displacement(state.ship_pos - own_defense, world_size)
        defense_radial_direction = defense_radial / defense_radial.abs().clamp(min=1e-8)
        defense_radial_direction = torch.where(
            defense_radial.abs() > 1e-8,
            defense_radial_direction,
            spawnward,
        )
        rank0 = team0.long().cumsum(dim=1) - 1
        rank1 = (~team0).long().cumsum(dim=1) - 1
        team_rank = torch.where(team0, rank0, rank1)
        orbit_direction = torch.where(
            team_rank.remainder(2) == 0,
            torch.ones_like(state.ship_health),
            -torch.ones_like(state.ship_health),
        )
        orbit_lookahead = torch.polar(
            torch.ones_like(state.ship_health),
            orbit_direction * float(np.deg2rad(20.0)),
        )
        own_defense_patrol = wrap_positions(
            own_defense + defense_radial_direction * orbit_lookahead * defense_patrol_radius,
            world_size,
        )
        attack_route = toroidal_displacement(enemy_defense - own_spawn, world_size)
        attack_route_distance = attack_route.abs()
        attack_route_direction = attack_route / attack_route_distance.clamp(min=1e-8)
        rally_point = wrap_positions(own_spawn + attack_route / 3.0, world_size)

        membership = zone_membership(
            state.ship_pos,
            state.zone_pos,
            state.zone_radius,
            self.ship_config.world_size,
        )
        occupied = membership & state.ship_alive.unsqueeze(2)
        team0_occupied = occupied & team0.unsqueeze(2)
        team1_occupied = occupied & (~team0).unsqueeze(2)
        team0_present = team0_occupied.any(dim=1)
        team1_present = team1_occupied.any(dim=1)
        team0_defense_role = roles == int(ZoneRole.TEAM0_DEFENSE)
        team1_defense_role = roles == int(ZoneRole.TEAM1_DEFENSE)

        # "Contested" intentionally means enemy presence, not simultaneous
        # presence. An undefended capture attempt must trigger the same response
        # as a point where both teams are fighting.
        if team_visibility is None:
            team1_seen_by_team0 = team1_occupied
            team0_seen_by_team1 = team0_occupied
        else:
            team1_seen_by_team0 = team1_occupied & team_visibility[:, 0, :, None]
            team0_seen_by_team1 = team0_occupied & team_visibility[:, 1, :, None]
        team0_own_contested = (team1_seen_by_team0.any(dim=1) & team0_defense_role).any(dim=1)
        team0_enemy_contested = (team0_present & team1_defense_role).any(dim=1)
        team1_own_contested = (team0_seen_by_team1.any(dim=1) & team1_defense_role).any(dim=1)
        team1_enemy_contested = (team1_present & team0_defense_role).any(dim=1)
        own_contested = torch.where(
            team0, team0_own_contested.unsqueeze(1), team1_own_contested.unsqueeze(1)
        )
        enemy_contested = torch.where(
            team0, team0_enemy_contested.unsqueeze(1), team1_enemy_contested.unsqueeze(1)
        )

        memory = self._frontline_episode_memory(state)
        tendencies = memory.tendencies
        offensive = tendencies == int(FrontlineTendency.OFFENSIVE)
        defensive = tendencies == int(FrontlineTendency.DEFENSIVE)
        timid = tendencies == int(FrontlineTendency.TIMID)

        team0_offensive = (offensive & team0).sum(dim=1)
        team0_defensive = (defensive & team0).sum(dim=1)
        team1_offensive = (offensive & ~team0).sum(dim=1)
        team1_defensive = (defensive & ~team0).sum(dim=1)
        team0_majority_attack = (team0_offensive > team0_defensive) | (
            (team0_offensive == team0_defensive) & memory.tie_attack[:, 0]
        )
        team1_majority_attack = (team1_offensive > team1_defensive) | (
            (team1_offensive == team1_defensive) & memory.tie_attack[:, 1]
        )
        majority_attack = torch.where(
            team0, team0_majority_attack.unsqueeze(1), team1_majority_attack.unsqueeze(1)
        )
        tendency_attack = offensive | (timid & majority_attack)

        # Offensive waves gather one-third of the way from spawn to the enemy
        # defense. Readiness is derived entirely from visible geometry: once every
        # living offensive ship has reached or passed the rally threshold, the
        # team proceeds. A respawn naturally rearms gathering without hidden state.
        from_spawn = toroidal_displacement(state.ship_pos - own_spawn, world_size)
        attack_progress = (from_spawn * torch.conj(attack_route_direction)).real
        rally_tolerance = state.zone_radius[:, :1] * 0.5
        reached_rally = attack_progress >= (attack_route_distance / 3.0 - rally_tolerance)
        active_offensive = offensive & state.ship_alive
        team0_wave_ready = (~(active_offensive & team0) | reached_rally).all(dim=1)
        team1_wave_ready = (~(active_offensive & ~team0) | reached_rally).all(dim=1)
        wave_ready = torch.where(
            team0, team0_wave_ready.unsqueeze(1), team1_wave_ready.unsqueeze(1)
        )
        attack_objective = torch.where(
            enemy_contested,
            enemy_defense,
            torch.where(wave_ready, enemy_defense, rally_point),
        )
        # Idle defenders chase a short moving waypoint around the safe perimeter.
        # Alternating direction by stable within-team rank reduces bunching. Once
        # enemies enter, the contested-point override sends defenders into the
        # point to fight and stabilize it.
        tendency_objective = torch.where(
            tendency_attack,
            attack_objective,
            torch.where(own_contested, own_defense, own_defense_patrol),
        )

        only_enemy_contested = enemy_contested & ~own_contested
        only_own_contested = own_contested & ~enemy_contested
        objective = torch.where(
            only_enemy_contested,
            enemy_defense,
            torch.where(only_own_contested, own_defense, tendency_objective),
        )

        nearby_enemy = has_target & (closest_dist <= self.config.frontline_enemy_engage_distance)

        below_timid_threshold = state.ship_health < (
            self.config.frontline_heal_health_fraction * self.ship_config.max_health
        )
        memory.healing |= state.ship_respawned | (timid & below_timid_threshold)
        memory.healing &= state.ship_health < self.ship_config.max_health

        timid_healing = timid & memory.healing
        other_healing = (~timid) & memory.healing
        any_contested = own_contested | enemy_contested
        engage_enemy = nearby_enemy & ~timid_healing
        enemy_pos = state.ship_pos.gather(1, target_idx)
        destination = torch.where(
            timid_healing,
            own_spawn,
            torch.where(
                engage_enemy,
                enemy_pos,
                torch.where(other_healing & ~any_contested, own_spawn, objective),
            ),
        )

        world_width, world_height = world_size
        displacement = destination - state.ship_pos
        displacement = torch.complex(
            (displacement.real + world_width / 2.0) % world_width - world_width / 2.0,
            (displacement.imag + world_height / 2.0) % world_height - world_height / 2.0,
        )
        distance = displacement.abs()
        bearing = displacement / distance.clamp(min=1e-8)
        return distance, bearing, engage_enemy

    def _get_frontline_actions_and_probs(
        self,
        state: TensorState,
        team_visibility: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the existing flight controller against frontline destinations."""

        closest_dist, target_idx, has_target, _ = select_targets(
            state, self.ship_config, team_visibility
        )
        objective_dist, objective_bearing, engage_enemy = self._frontline_targets(
            state, closest_dist, target_idx, has_target, team_visibility
        )
        intercept = predict_interception(state, self.ship_config, target_idx, closest_dist)
        intercept = torch.where(engage_enemy, intercept, torch.zeros_like(intercept))

        p_power, p_turn, p_shoot = self._compute_action_probs(
            state,
            objective_dist,
            torch.where(engage_enemy, intercept, objective_bearing),
            intercept,
            state.ship_alive,
        )
        # Navigation targets are not things to shoot. This explicit gate avoids
        # treating an aligned nearby zone center like a ship-sized target.
        no_shoot = torch.zeros_like(p_shoot)
        no_shoot[..., 0] = 1.0
        p_shoot = torch.where(engage_enemy.unsqueeze(-1), p_shoot, no_shoot)

        batch_size, num_ships = state.ship_pos.shape
        if self.config.flat_action_sampling:
            joint_probs = (
                p_power.unsqueeze(-1).unsqueeze(-1)
                * p_turn.unsqueeze(-2).unsqueeze(-1)
                * p_shoot.unsqueeze(-2).unsqueeze(-2)
            ).reshape(batch_size, num_ships, 42)
            expert_probs = joint_probs
            sampled_flat = torch.multinomial(joint_probs.view(-1, 42), num_samples=1).view(
                batch_size, num_ships
            )
            actions_shoot = sampled_flat % 2
            sampled_flat = sampled_flat // 2
            actions_turn = sampled_flat % 7
            actions_power = sampled_flat // 7
            actions = torch.stack([actions_power, actions_turn, actions_shoot], dim=-1)
        else:
            expert_probs = torch.cat([p_power, p_turn, p_shoot], dim=-1)
            actions = torch.stack(
                [
                    torch.multinomial(p_power.view(-1, 3), 1).view(batch_size, num_ships),
                    torch.multinomial(p_turn.view(-1, 7), 1).view(batch_size, num_ships),
                    torch.multinomial(p_shoot.view(-1, 2), 1).view(batch_size, num_ships),
                ],
                dim=-1,
            )
        return actions, expert_probs

    def get_actions_and_probs(
        self,
        state: TensorState,
        team_visibility: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return the expert probability distribution as soft labels.

        Returns:
            actions:      (B, N, 3) int tensor
            expert_probs: (B, N, 12) float tensor (independent marginals) or
                          (B, N, 42) float tensor (joint, if flat_action_sampling=True)
        """
        # A zero-length zone axis is the exact legacy combat contract. Keep its
        # control path below unchanged so adding frontline objectives cannot alter
        # existing scripted anchors, BC targets, or calibrated ratings.
        if state.num_zones > 0:
            return self._get_frontline_actions_and_probs(state, team_visibility)

        closest_dist, target_idx, has_target, _ = select_targets(
            state, self.ship_config, team_visibility
        )
        dir_pred = predict_interception(state, self.ship_config, target_idx, closest_dist)

        # Guard against NaN in dir_pred when there is no target (closest_dist = inf)
        dir_pred = torch.where(has_target, dir_pred, torch.zeros_like(dir_pred))

        active_mask = state.ship_alive & has_target

        # Blend turn direction: personal intercept at close range, team target at far range
        team_bearing, _, _, team_has_target = compute_team_target_bearings(
            state, self.ship_config, team_visibility
        )
        p_team = (
            self._linear_ramp(
                closest_dist,
                self.config.team_target_distance_ramp[0],
                self.config.team_target_distance_ramp[1],
                *self.config.team_target_distance_prob,
            )
            * team_has_target.float()
        )
        dir_turn = (1.0 - p_team) * dir_pred + p_team * team_bearing
        dir_turn = dir_turn / (torch.abs(dir_turn) + 1e-8)

        # Suppress enemy-proximity reversal when there is no actual target
        # (closest_dist = inf gives p_is_close ≈ 1 which would force spurious reversal)
        effective_combat_dist = torch.where(
            has_target, closest_dist, torch.zeros_like(closest_dist)
        )

        p_power, p_turn, p_shoot = self._compute_action_probs(
            state, effective_combat_dist, dir_turn, dir_pred, active_mask
        )

        batch_size, num_ships = state.ship_pos.shape

        if self.config.flat_action_sampling:
            joint_probs = (
                p_power.unsqueeze(-1).unsqueeze(-1)
                * p_turn.unsqueeze(-2).unsqueeze(-1)
                * p_shoot.unsqueeze(-2).unsqueeze(-2)
            )
            joint_probs = joint_probs.reshape(batch_size, num_ships, 42)
            expert_probs = joint_probs

            flat_probs = joint_probs.view(-1, 42)
            sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(batch_size, num_ships)

            actions_shoot = sampled_flat % 2
            sampled_flat = sampled_flat // 2
            actions_turn = sampled_flat % 7
            actions_power = sampled_flat // 7
            actions = torch.stack([actions_power, actions_turn, actions_shoot], dim=-1)
        else:
            expert_probs = torch.cat([p_power, p_turn, p_shoot], dim=-1)  # (B, N, 12)

            a_p = torch.multinomial(p_power.view(-1, 3), num_samples=1).view(batch_size, num_ships)
            a_t = torch.multinomial(p_turn.view(-1, 7), num_samples=1).view(batch_size, num_ships)
            a_s = torch.multinomial(p_shoot.view(-1, 2), num_samples=1).view(batch_size, num_ships)
            actions = torch.stack([a_p, a_t, a_s], dim=-1)

        return actions, expert_probs

    def get_actions(
        self,
        state: TensorState,
        team_visibility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard interface — returns (B, N, 3) int tensor of sampled actions."""
        actions, _ = self.get_actions_and_probs(state, team_visibility)
        return actions
