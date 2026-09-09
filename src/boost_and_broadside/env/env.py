"""TensorEnv: pure physics engine for vectorized parallel game environments.

This class owns the physics simulation only. Observation construction, reward
computation, and episode tracking are handled by YemongEnvWrapper (env/wrapper.py).
"""

from typing import Any

import numpy as np
import torch

from boost_and_broadside.config import EnvConfig, MatchResult, ShipConfig
from boost_and_broadside.env.field_generation import generate_field_layout
from boost_and_broadside.env.field_physics import evaluate_fields
from boost_and_broadside.env.frontline import (
    FRONTLINE_WORLD_SIZE,
    NUM_FRONTLINE_ZONES,
    apply_frontline_tick,
    apply_timeout_result,
    clear_previous_life_attribution,
    initialize_frontline_map,
    place_ships_at_spawns,
)
from boost_and_broadside.env.physics import (
    _combat_damage_tensors,
    advance_bullets,
    resolve_collisions,
    update_ships,
)
from boost_and_broadside.env.state import TensorState


class TensorEnv:
    """Vectorized GPU physics engine for B parallel game instances.

    Attributes:
        num_envs: Number of parallel environments (B).
        ship_config: Physics constants.
        env_config: Environment sizing (ships, bullets, episode length).
        device: Torch device for all tensors.
        state: Live TensorState; updated each call to step().
    """

    def __init__(
        self,
        num_envs: int,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        device: str | torch.device,
        collision_compile_mode: str | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.ship_config = ship_config
        self.env_config = env_config
        self.device = torch.device(device)
        self._combat_damage_fn = (
            torch.compile(_combat_damage_tensors, mode=collision_compile_mode)
            if collision_compile_mode is not None and self.device.type == "cuda"
            else None
        )
        if env_config.frontline is not None:
            if tuple(ship_config.world_size) != FRONTLINE_WORLD_SIZE:
                raise ValueError(
                    "frontline mode requires the design-contract world size "
                    f"{FRONTLINE_WORLD_SIZE}, got {ship_config.world_size}"
                )
            if env_config.single_team:
                raise ValueError("frontline mode requires two teams")
            if env_config.max_episode_steps is None:
                raise ValueError("frontline mode requires a finite maximum match duration")
            if env_config.frontline.respawn_health > ship_config.max_health:
                raise ValueError("frontline respawn_health cannot exceed ship max_health")
        self.state: TensorState | None = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Allocate state tensors and reset all environments.

        Args:
            options: Optional dict. Supported keys:
                - "team_sizes": (n_team0, n_team1) tuple.
            seed: Optional RNG seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self._allocate_state()
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_envs(mask, options)

    def _allocate_state(self) -> None:
        """Pre-allocate all state tensors on device."""
        B = self.num_envs
        N = self.env_config.num_ships
        K = self.env_config.max_bullets
        M = self.env_config.num_fields
        Z = NUM_FRONTLINE_ZONES if self.env_config.frontline is not None else 0
        dev = self.device

        self.state = TensorState(
            step_count=torch.zeros((B,), dtype=torch.int32, device=dev),
            ship_pos=torch.zeros((B, N), dtype=torch.complex64, device=dev),
            ship_vel=torch.zeros((B, N), dtype=torch.complex64, device=dev),
            ship_attitude=torch.zeros((B, N), dtype=torch.complex64, device=dev),
            ship_ang_vel=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_health=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_power=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_cooldown=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_team_id=torch.zeros((B, N), dtype=torch.int32, device=dev),
            ship_alive=torch.zeros((B, N), dtype=torch.bool, device=dev),
            ship_is_shooting=torch.zeros((B, N), dtype=torch.bool, device=dev),
            map_center=torch.zeros((B,), dtype=torch.complex64, device=dev),
            playable_boundary_radius=torch.zeros((B,), dtype=torch.float32, device=dev),
            front_position=torch.zeros((B,), dtype=torch.long, device=dev),
            front_delta=torch.zeros((B,), dtype=torch.int8, device=dev),
            front_win_threshold=torch.full(
                (B,),
                self.env_config.frontline.front_win_threshold
                if self.env_config.frontline is not None
                else 0,
                dtype=torch.long,
                device=dev,
            ),
            match_max_steps=torch.full(
                (B,), self.env_config.max_episode_steps or 0, dtype=torch.long, device=dev
            ),
            match_result=torch.full((B,), int(MatchResult.ONGOING), dtype=torch.int8, device=dev),
            zone_pos=torch.zeros((B, Z), dtype=torch.complex64, device=dev),
            zone_radius=torch.zeros((B, Z), dtype=torch.float32, device=dev),
            zone_roles=torch.zeros((B, Z), dtype=torch.int8, device=dev),
            zone_capture_progress=torch.zeros((B, Z), dtype=torch.float32, device=dev),
            zone_capture_direction=torch.zeros((B, Z), dtype=torch.int8, device=dev),
            team0_captured=torch.zeros((B,), dtype=torch.bool, device=dev),
            team1_captured=torch.zeros((B,), dtype=torch.bool, device=dev),
            simultaneous_capture=torch.zeros((B,), dtype=torch.bool, device=dev),
            prev_action=torch.zeros((B, N, 3), dtype=torch.float32, device=dev),
            bullet_pos=torch.zeros((B, N, K), dtype=torch.complex64, device=dev),
            bullet_vel=torch.zeros((B, N, K), dtype=torch.complex64, device=dev),
            bullet_time=torch.zeros((B, N, K), dtype=torch.float32, device=dev),
            bullet_active=torch.zeros((B, N, K), dtype=torch.bool, device=dev),
            bullet_remaining_damage=torch.zeros((B, N, K), dtype=torch.float32, device=dev),
            bullet_field_alpha=torch.zeros((B, N, K, M), dtype=torch.float32, device=dev),
            bullet_local_index=torch.ones((B, N, K), dtype=torch.float32, device=dev),
            bullet_field_gradient=torch.zeros((B, N, K), dtype=torch.complex64, device=dev),
            bullet_cursor=torch.zeros((B, N), dtype=torch.long, device=dev),
            damage_matrix=torch.zeros((B, N, N), dtype=torch.float32, device=dev),
            cumulative_damage_matrix=torch.zeros((B, N, N), dtype=torch.float32, device=dev),
            field_pos=torch.zeros((B, M), dtype=torch.complex64, device=dev),
            field_radius=torch.zeros((B, M), dtype=torch.float32, device=dev),
            field_transition_width=torch.zeros((B, M), dtype=torch.float32, device=dev),
            field_index_level=torch.zeros((B, M), dtype=torch.int8, device=dev),
            field_index=torch.ones((B, M), dtype=torch.float32, device=dev),
            field_damage_level=torch.zeros((B, M), dtype=torch.int8, device=dev),
            field_damage=torch.zeros((B, M), dtype=torch.float32, device=dev),
            ship_field_alpha=torch.zeros((B, N, M), dtype=torch.float32, device=dev),
            ship_local_index=torch.ones((B, N), dtype=torch.float32, device=dev),
            ship_field_gradient=torch.zeros((B, N), dtype=torch.complex64, device=dev),
            ship_field_damage=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_combat_damage=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_field_death=torch.zeros((B, N), dtype=torch.bool, device=dev),
            ship_combat_death=torch.zeros((B, N), dtype=torch.bool, device=dev),
            ship_zone_damage=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_spawn_damage=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_boundary_damage=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_zone_death=torch.zeros((B, N), dtype=torch.bool, device=dev),
            ship_spawn_death=torch.zeros((B, N), dtype=torch.bool, device=dev),
            ship_boundary_death=torch.zeros((B, N), dtype=torch.bool, device=dev),
            ship_respawned=torch.zeros((B, N), dtype=torch.bool, device=dev),
            ship_spawn_healing=torch.zeros((B, N), dtype=torch.float32, device=dev),
        )

    def reset_envs(
        self,
        mask: torch.Tensor,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environments selected by the boolean mask.

        Fully branchless: new values are generated for every env and applied
        only where mask is True. Generating full-size randoms is trivially
        cheap compared to the host-device sync that counting/indexing the mask
        would force — this runs every step of the training hot path.

        Args:
            mask: (B,) bool — True for envs that need resetting.
            options: Same as reset() options.
        """
        B = self.num_envs
        world_w, world_h = self.ship_config.world_size
        N = self.env_config.num_ships

        n_team0 = N // 2
        n_team1 = N - n_team0
        if options and "team_sizes" in options:
            n_team0, n_team1 = options["team_sizes"]
            if n_team0 < 0 or n_team1 < 0 or n_team0 + n_team1 > N:
                raise ValueError(
                    f"team_sizes {(n_team0, n_team1)} invalid for num_ships={N}: counts "
                    "must be non-negative and sum to at most num_ships."
                )

        s = self.state
        m = mask.unsqueeze(1)  # (B, 1) — broadcasts over ships/fields

        s.step_count = torch.where(mask, 0, s.step_count)
        s.match_result = torch.where(mask, int(MatchResult.ONGOING), s.match_result)
        if self.env_config.frontline is not None:
            initialize_frontline_map(
                s,
                mask,
                self.env_config.frontline,
                self.ship_config.world_size,
            )

        # Positions — uniformly random in world
        rand_x = torch.rand((B, N), device=self.device) * world_w
        rand_y = torch.rand((B, N), device=self.device) * world_h
        s.ship_pos = torch.where(m, torch.complex(rand_x, rand_y), s.ship_pos)

        # Attitude — random unit vectors
        rand_angle = torch.rand((B, N), device=self.device) * 2 * np.pi
        att = torch.polar(torch.ones_like(rand_angle), rand_angle)
        s.ship_attitude = torch.where(m, att, s.ship_attitude)

        # Each reset receives a fresh independent layout. Frontline fields share
        # the same map translation as its zones and practical boundary.
        if self.env_config.num_fields > 0:
            sampled = generate_field_layout(
                B,
                self.ship_config,
                self.env_config,
                self.device,
                map_center=(s.map_center if self.env_config.frontline is not None else None),
                playable_radius=(
                    s.playable_boundary_radius if self.env_config.frontline is not None else None
                ),
            )
            field_names = (
                "field_pos",
                "field_radius",
                "field_transition_width",
                "field_index_level",
                "field_index",
                "field_damage_level",
                "field_damage",
            )
            for name, value in zip(field_names, sampled, strict=True):
                setattr(s, name, torch.where(m, value, getattr(s, name)))

        field_eval = evaluate_fields(
            s.ship_pos,
            s.field_pos,
            s.field_radius,
            s.field_transition_width,
            s.field_index,
            self.ship_config.world_size,
        )
        field_mask = mask.view(B, 1, 1)
        s.ship_field_alpha = torch.where(field_mask, field_eval.alpha, s.ship_field_alpha)
        s.ship_local_index = torch.where(m, field_eval.index, s.ship_local_index)
        s.ship_field_gradient = torch.where(m, field_eval.grad_index, s.ship_field_gradient)

        # Configured speeds are proper speeds u=n*v_world. Spawning in a medium
        # therefore never injects generalized kinetic energy.
        if self.ship_config.random_speed:
            proper_speed = self.ship_config.min_speed + torch.rand((B, N), device=self.device) * (
                self.ship_config.max_speed - self.ship_config.min_speed
            )
        else:
            proper_speed = torch.full((B, N), self.ship_config.default_speed, device=self.device)
        world_speed = proper_speed / field_eval.index
        s.ship_vel = torch.where(m, world_speed * att, s.ship_vel)

        # Resources. Spawning every episode at full health and power makes
        # health an almost deterministic function of elapsed time early on, which
        # the critic can read off the clock instead of the state, and it means
        # damaged-fleet positions are only ever reached by playing two hundred
        # steps to get there. Randomising the start exposes those states directly.
        #
        # Draws are per-ship but centred per env, so both teams get the same
        # expected resources: an episode that started lopsided would put outcome
        # variance into the win signal that no policy could have influenced.
        health = torch.full((B, N), self.ship_config.max_health, device=self.device)
        power = torch.full((B, N), self.ship_config.max_power, device=self.device)
        cooldown = torch.zeros((B, N), device=self.device)
        spread = self.env_config.spawn_resource_spread
        if spread > 0.0:
            lo = 1.0 - spread
            health = health * (lo + torch.rand((B, N), device=self.device) * spread)
            power = power * (lo + torch.rand((B, N), device=self.device) * spread)
            cooldown = torch.rand((B, N), device=self.device) * self.ship_config.firing_cooldown
        s.ship_health = torch.where(m, health, s.ship_health)
        s.ship_power = torch.where(m, power, s.ship_power)
        s.ship_cooldown = torch.where(m, cooldown, s.ship_cooldown)
        s.ship_ang_vel = torch.where(m, 0.0, s.ship_ang_vel)

        if self.env_config.single_team:
            # All ships share one randomly chosen team id (0 or 1) per env.
            # Random team prevents the policy overfitting to always seeing itself as team 0.
            team_id = torch.randint(0, 2, (B,), device=self.device, dtype=torch.int32)
            s.ship_team_id = torch.where(m, team_id.unsqueeze(1), s.ship_team_id)
            s.ship_alive = s.ship_alive | m
        else:
            # Two-team setup: randomly shuffle ship slots across both teams.
            new_alive = torch.zeros((B, N), dtype=torch.bool, device=self.device)
            new_alive[:, : n_team0 + n_team1] = True

            base_team_ids = torch.zeros((B, N), dtype=torch.int32, device=self.device)
            base_team_ids[:, n_team0 : n_team0 + n_team1] = 1  # last n_team1 slots = team 1

            # Independent random permutation per env → any slot can be any team.
            perm = torch.rand((B, N), device=self.device).argsort(dim=1)
            new_team_ids = base_team_ids.gather(1, perm)

            s.ship_team_id = torch.where(m, new_team_ids, s.ship_team_id)
            s.ship_alive = torch.where(m, new_alive, s.ship_alive)

        if self.env_config.frontline is not None:
            active_reset = m & s.ship_alive
            place_ships_at_spawns(s, active_reset, self.ship_config, health)

        # Clear bullets
        m3 = mask.view(B, 1, 1)
        s.bullet_active = s.bullet_active & ~m3
        s.bullet_time = torch.where(m3, 0.0, s.bullet_time)
        s.bullet_remaining_damage = torch.where(m3, 0.0, s.bullet_remaining_damage)
        s.bullet_field_alpha = torch.where(
            mask.view(B, 1, 1, 1),
            0.0,
            s.bullet_field_alpha,
        )
        s.bullet_local_index = torch.where(m3, 1.0, s.bullet_local_index)
        s.bullet_field_gradient = torch.where(m3, 0.0, s.bullet_field_gradient)
        s.bullet_cursor = torch.where(m, 0, s.bullet_cursor)

        # Clear damage attribution
        s.cumulative_damage_matrix = torch.where(m3, 0.0, s.cumulative_damage_matrix)

        # Clear previous action
        s.prev_action = torch.where(m3, 0.0, s.prev_action)

        # Resetting/spawning initializes alpha rather than comparing against
        # ambient, so it cannot cause artificial crossing damage.
        s.ship_field_damage = torch.where(m, 0.0, s.ship_field_damage)
        s.ship_combat_damage = torch.where(m, 0.0, s.ship_combat_damage)
        s.ship_field_death = s.ship_field_death & ~m
        s.ship_combat_death = s.ship_combat_death & ~m
        s.ship_zone_damage = torch.where(m, 0.0, s.ship_zone_damage)
        s.ship_spawn_damage = torch.where(m, 0.0, s.ship_spawn_damage)
        s.ship_boundary_damage = torch.where(m, 0.0, s.ship_boundary_damage)
        s.ship_zone_death &= ~m
        s.ship_spawn_death &= ~m
        s.ship_boundary_death &= ~m
        s.ship_respawned &= ~m
        s.ship_spawn_healing = torch.where(m, 0.0, s.ship_spawn_healing)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: torch.Tensor,
        *,
        unlimited_resources: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance all environments by one *decision*, honouring action_repeat.

        This is the default because every consumer that is not accumulating
        per-tick rewards wants a decision, and getting it wrong is silent: a
        policy trained to hold an action for N ticks but evaluated one tick per
        action turns a fraction of its intended per decision, mistimes every
        lead, and advances its recurrent state N times too fast for the game
        clock. It still plays, just far worse, and nothing in the metrics says
        why. YemongEnvWrapper opts out via ``tick`` because it has to compute
        rewards and episode statistics per physics tick.

        An environment that finishes partway through the hold keeps being
        simulated for the remainder; the returned flags are sticky, so the
        caller sees the episode as ended either way.

        Args:
            actions: (B, N, 3) int tensor — [power, turn, shoot].
            unlimited_resources: Protect and refill alive ships.

        Returns:
            (dones, truncated) — each a (B,) bool tensor, accumulated over the hold.
        """
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(self.env_config.action_repeat):
            tick_dones, tick_truncated = self.tick(actions, unlimited_resources=unlimited_resources)
            dones |= tick_dones
            truncated |= tick_truncated
        return dones, truncated

    def tick(
        self,
        actions: torch.Tensor,
        *,
        unlimited_resources: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance all environments by one physics tick.

        The caller (wrapper) is responsible for:
        - Snapshotting health/alive BEFORE calling step() if needed for rewards.
        - Calling reset_envs() on done environments AFTER computing rewards.

        Args:
            actions: (B, N, 3) int tensor — [power, turn, shoot].
            unlimited_resources: Protect currently alive ships from damage and
                refill their power. Used only by the interactive play toggle.

        Returns:
            (dones, truncated) — each is a (B,) bool tensor.
        """
        protected_alive = self.state.ship_alive.clone() if unlimited_resources else None
        if self.env_config.frontline is not None:
            clear_previous_life_attribution(self.state)
        if protected_alive is not None:
            # A very large temporary health value prevents a lethal field or
            # bullet hit from setting ``alive=False`` before game-over is
            # checked. The user-visible value is restored to max_health below.
            protected_health = torch.full_like(
                self.state.ship_health,
                torch.finfo(self.state.ship_health.dtype).max,
            )
            self.state.ship_health = torch.where(
                protected_alive, protected_health, self.state.ship_health
            )
            self.state.ship_power = torch.where(
                protected_alive,
                torch.full_like(self.state.ship_power, self.ship_config.max_power),
                self.state.ship_power,
            )

        self.state.prev_action = actions.float()
        self.state = update_ships(self.state, actions, self.ship_config)
        bullet_trajectory = None
        if self.env_config.max_bullets > 0:
            self.state, bullet_trajectory = advance_bullets(
                self.state,
                self.ship_config,
            )
        self.state, dones = resolve_collisions(
            self.state,
            self.ship_config,
            self._combat_damage_fn,
            bullet_trajectory,
        )

        if self.env_config.frontline is not None:
            dones = apply_frontline_tick(
                self.state,
                self.env_config.frontline,
                self.ship_config,
            )
        else:
            team0_alive = ((self.state.ship_team_id == 0) & self.state.ship_alive).any(dim=1)
            team1_alive = ((self.state.ship_team_id == 1) & self.state.ship_alive).any(dim=1)
            result = torch.where(
                team0_alive & ~team1_alive,
                int(MatchResult.TEAM0_WIN),
                torch.where(
                    team1_alive & ~team0_alive,
                    int(MatchResult.TEAM1_WIN),
                    int(MatchResult.DRAW),
                ),
            )
            self.state.match_result = torch.where(dones, result, self.state.match_result)

        if protected_alive is not None:
            self.state.ship_alive |= protected_alive
            self.state.ship_health = torch.where(
                protected_alive,
                torch.full_like(self.state.ship_health, self.ship_config.max_health),
                self.state.ship_health,
            )
            self.state.ship_power = torch.where(
                protected_alive,
                torch.full_like(self.state.ship_power, self.ship_config.max_power),
                self.state.ship_power,
            )
            self.state.ship_field_damage = torch.where(
                protected_alive, 0.0, self.state.ship_field_damage
            )
            self.state.ship_combat_damage = torch.where(
                protected_alive, 0.0, self.state.ship_combat_damage
            )
            self.state.ship_field_death &= ~protected_alive
            self.state.ship_combat_death &= ~protected_alive
            if self.env_config.frontline is None:
                dones &= ~protected_alive.any(dim=1)

        self.state.step_count += 1
        if self.env_config.max_episode_steps is None:
            truncated = torch.zeros_like(dones)
        else:
            truncated = self.state.step_count >= self.env_config.max_episode_steps

        if self.env_config.frontline is not None:
            apply_timeout_result(self.state, truncated)
        else:
            unresolved_timeout = truncated & (self.state.match_result == int(MatchResult.ONGOING))
            self.state.match_result = torch.where(
                unresolved_timeout,
                int(MatchResult.DRAW),
                self.state.match_result,
            )

        if protected_alive is not None and self.env_config.frontline is not None:
            self.state.ship_health = torch.where(
                protected_alive,
                torch.full_like(self.state.ship_health, self.ship_config.max_health),
                self.state.ship_health,
            )
            self.state.ship_zone_damage = torch.where(
                protected_alive, 0.0, self.state.ship_zone_damage
            )
            self.state.ship_spawn_damage = torch.where(
                protected_alive, 0.0, self.state.ship_spawn_damage
            )
            self.state.ship_boundary_damage = torch.where(
                protected_alive, 0.0, self.state.ship_boundary_damage
            )

        return dones, truncated
