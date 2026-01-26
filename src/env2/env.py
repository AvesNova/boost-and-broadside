
import math
import numpy as np
import torch
import gymnasium as gym

from env.env import Environment
from env.ship import default_ship_config
from env.constants import (
    MAX_SHIPS,
    RewardConstants,
    NORM_VELOCITY,
    NORM_ACCELERATION,
    NORM_ANGULAR_VELOCITY,
    NORM_HEALTH,
    NORM_POWER,
)
from .state import TensorState
from .physics import update_ships, update_bullets, check_collisions

class TensorEnv:
    """
    Vectorized environment running on GPU.
    Supports NvM games in parallel.
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        max_ships: int = MAX_SHIPS, # Per env
        dt: float = 0.015,
        world_size: tuple[float, float] = (2000.0, 2000.0),
        # Configs
        fixed_dt: bool = True, # Use fixed physics steps?
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_ships = max_ships # Total ships capacity per env (N)
        self.dt = dt
        self.world_size = world_size
        
        # Determine K (max bullets per ship)
        # ceil(lifetime / cooldown)
        self.max_bullets_per_ship = int(math.ceil(default_ship_config.bullet_lifetime / default_ship_config.firing_cooldown)) + 1
        
        # State Container
        self.state: TensorState | None = None
        
        # RNG for initialization
        self.rng = np.random.default_rng()
        
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        """
        Reset all environments.
        Options:
            'team_sizes': tuple[int, int] (ships_team_0, ships_team_1)
            'random_pos': bool
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            # Seed torch as well for deterministic runs on GPU?
            # torch.manual_seed(seed) # Global seed might be invasive.
            
        options = options or {}
        
        # Config
        team_sizes = options.get("team_sizes", (self.max_ships // 2, self.max_ships // 2))
        random_pos = options.get("random_pos", False)
        
        n_team0, n_team1 = team_sizes
        assert n_team0 + n_team1 <= self.max_ships, "Total ships exceed max_ships"
        
        # Initialize State Tensors
        B = self.num_envs
        N = self.max_ships
        K = self.max_bullets_per_ship
        
        # Create empty state
        self.state = TensorState(
            time=torch.zeros((B,), device=self.device),
            ships_pos=torch.zeros((B, N), dtype=torch.complex64, device=self.device),
            ships_vel=torch.zeros((B, N), dtype=torch.complex64, device=self.device),
            ships_power=torch.ones((B, N), device=self.device) * default_ship_config.max_power,
            ships_cooldown=torch.zeros((B, N), device=self.device),
            ships_team=torch.zeros((B, N), dtype=torch.long, device=self.device),
            ships_alive=torch.zeros((B, N), dtype=torch.bool, device=self.device),
            ships_health=torch.ones((B, N), device=self.device) * default_ship_config.max_health,
            ships_acc=torch.zeros((B, N), dtype=torch.complex64, device=self.device),
            ships_ang_vel=torch.zeros((B, N), dtype=torch.float32, device=self.device),
            ships_attitude=torch.zeros((B, N), dtype=torch.complex64, device=self.device), # Init below
            bullets_pos=torch.zeros((B, N, K), dtype=torch.complex64, device=self.device),
            bullets_vel=torch.zeros((B, N, K), dtype=torch.complex64, device=self.device),
            bullets_time=torch.zeros((B, N, K), dtype=torch.float32, device=self.device),
            bullets_team=torch.zeros((B, N, K), dtype=torch.long, device=self.device),
            bullet_cursor=torch.zeros((B, N), dtype=torch.long, device=self.device)
        )
        
        # Generate initial positions (CPU logic vectorized or loop)
        # Using existing fractal logic (replicated for batch or just reused).
        # Since logic is CPU based and fast enough for reset, we can just generate numpy and convert.
        
        # Note: Fractal positions are deterministic per level.
        # If we use random_pos, we use uniform.
        
        # For simple NvM, we can assign index 0..n0-1 to team 0, n0..n0+n1-1 to team 1.
        
        # Generate raw positions on CPU
        # Shape (B, N)
        pos_np = np.zeros((B, N), dtype=np.complex64)
        vel_np = np.zeros((B, N), dtype=np.complex64)
        
        # Team indices
        team_np = np.zeros((B, N), dtype=np.int64)
        alive_np = np.zeros((B, N), dtype=bool)
        
        # Fill for each Env (slow loop in Python for reset is acceptable for now vs complex vectorization)
        # Optimize later if reset is bottle neck.
        
        base_fractal = Environment.get_ship_positions(self.max_ships) # Gets sequence
        
        for b in range(B):
            # Team 0
            # Reuse logic from env.py create_squad... roughly.
            # Team 0 starts left/bottom, Team 1 top/right typically?
            # 1v1 logic: 0.25, 0.40 and 0.75, 0.60.
            # Fractal: origin + offsets.
            
            # Team 0 setup
            idx_start = 0
            idx_end = n_team0
            
            if random_pos:
                # Random per ship
                for i in range(idx_start, idx_end):
                    p = self.rng.uniform(0, self.world_size[0]) + 1j * self.rng.uniform(0, self.world_size[1])
                    att = np.exp(1j * self.rng.uniform(0, 2*np.pi))
                    v = att * 100.0
                    pos_np[b, i] = p
                    vel_np[b, i] = v
            else:
                # Team 0 fractal
                origin = self.rng.uniform(0, self.world_size[0]) + 1j * self.rng.uniform(0, self.world_size[1])
                att = np.exp(1j * self.rng.uniform(0, 2*np.pi))
                offsets = Environment.get_ship_positions(n_team0) * att * default_ship_config.collision_radius * 4
                pos_np[b, idx_start:idx_end] = origin + offsets
                vel_np[b, idx_start:idx_end] = att * 100.0
                
            team_np[b, idx_start:idx_end] = 0
            alive_np[b, idx_start:idx_end] = True
            
            # Team 1 setup
            idx_start = n_team0
            idx_end = n_team0 + n_team1
            
            if random_pos:
                 for i in range(idx_start, idx_end):
                    p = self.rng.uniform(0, self.world_size[0]) + 1j * self.rng.uniform(0, self.world_size[1])
                    att = np.exp(1j * self.rng.uniform(0, 2*np.pi))
                    v = att * 100.0
                    pos_np[b, i] = p
                    vel_np[b, i] = v
            else:
                 origin = self.rng.uniform(0, self.world_size[0]) + 1j * self.rng.uniform(0, self.world_size[1])
                 att = np.exp(1j * self.rng.uniform(0, 2*np.pi))
                 offsets = Environment.get_ship_positions(n_team1) * att * default_ship_config.collision_radius * 4
                 pos_np[b, idx_start:idx_end] = origin + offsets
                 vel_np[b, idx_start:idx_end] = att * 100.0
            
            team_np[b, idx_start:idx_end] = 1
            alive_np[b, idx_start:idx_end] = True
            
        # Copy to GPU
        self.state.ships_pos.copy_(torch.from_numpy(pos_np))
        self.state.ships_vel.copy_(torch.from_numpy(vel_np))
        self.state.ships_team.copy_(torch.from_numpy(team_np))
        self.state.ships_alive.copy_(torch.from_numpy(alive_np))
        
        # Init Attitude
        speed = self.state.ships_vel.abs()
        safe_speed = torch.maximum(speed, torch.tensor(1e-6, device=self.device))
        self.state.ships_attitude = self.state.ships_vel / safe_speed
        
        return self.get_observations()

    def step(self, actions_dict: dict[str, torch.Tensor]) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment.
        actions_dict: {'action': (B, N, 3)} usually. Or broken down?
        Input expected: Tensor (B, N, 3) Int64. [Power, Turn, Shoot]
        """
        if self.state is None:
            raise RuntimeError("Env not reset")
            
        # Parse actions
        # Assume actions is a Tensor (B, N, 3)
        actions = actions_dict["action"]
        act_power = actions[..., 0]
        act_turn = actions[..., 1]
        act_shoot = actions[..., 2]
        
        B = self.num_envs
        N = self.max_ships

        # Advance Time
        self.state.time += self.dt
        
        # 1. Update Ships
        (
            self.state.ships_pos,
            self.state.ships_vel,
            self.state.ships_power,
            self.state.ships_cooldown,
            should_shoot,
            self.state.ships_acc,
            self.state.ships_ang_vel,
            self.state.ships_attitude
        ) = update_ships(
            self.state.ships_pos,
            self.state.ships_vel,
            self.state.ships_power,
            self.state.ships_cooldown,
            self.state.ships_team,
            self.state.ships_alive,
            act_power,
            act_turn,
            act_shoot,
            self.dt,
            self.world_size
        )
        
        # 2. Spawn Bullets for 'should_shoot'
        # Need to insert bullets into ring buffer.
        # Vectorized insert?
        # Indices: (b, n) where should_shoot is True.
        # We need to compute target indices in bullets_pos (B, N, K).
        
        if should_shoot.any():
            # Get current cursor for shooting ships
            cursors = self.state.bullet_cursor # (B, N)
            
            # Calculate bullet properties
            # Pos = ship pos
            # Vel = ship vel + bullet_speed * ship_attitude + noise?
            # rng noise not easy in pure torch without creating tensor?
            # `torch.normal` supported.
            
            shooting_mask = should_shoot
            
            ship_pos = self.state.ships_pos
            ship_vel = self.state.ships_vel
            ship_att = self.state.ships_attitude
            
            # Bullet Speed
            b_speed = default_ship_config.bullet_speed
            b_spread = default_ship_config.bullet_spread
            
            # Velocity
            # Defined in ship.py: 
            # bullet_vx = self.velocity.real + self.config.bullet_speed * self.attitude.real + self.rng.normal(0, self.config.bullet_spread)
            
            # Add spread noise
            noise_real = torch.randn_like(ship_pos.real) * b_spread
            noise_imag = torch.randn_like(ship_pos.imag) * b_spread
            
            bullet_v_real = ship_vel.real + b_speed * ship_att.real + noise_real
            bullet_v_imag = ship_vel.imag + b_speed * ship_att.imag + noise_imag
            bullet_vel = torch.complex(bullet_v_real, bullet_v_imag)
            
            bullet_pos = ship_pos # Start at ship center
            
            # Update only slots where shooting
            # Use gather/scatter or indexing
            
            # B, N indices
            # K index is cursors
            
            # Create indexing tuple
            # We want to update self.state.bullets_pos[b, n, cursor[b,n]] where shoud_shoot[b,n]
            
            # This is scattered write.
            # Masked write.
            
            # Let's start with flattened view if strictly necessary, but scatter_ is okay.
            # Or assume we update ALL, masked?
            # scatter is better.
            
            # Prepare src tensors (B, N) -> (B, N, 1) -> Scatter to (B, N, K)?
            
            target_indices = cursors.unsqueeze(2) # (B, N, 1)
            
            # We need to write ONLY if should_shoot is true. 
            # If we write typically to ring buffer, we overwrite old bullets.
            # But if NOT shooting, we shouldn't overwrite valid bullets with garbage.
            # So masking is critical.
            
            # Actually easier:
            # 1. Expand properties to (B, N, 1)
            # 2. Scatter them into (B, N, K) using cursor
            # 3. BUT only apply where should_shoot.
            
            # scatter usage: self.bullets_pos.scatter_(2, index, src)
            
            # If src has same size as dest slice? 
            # src broadcast to (B, N, 1).
            # If we just scatter, it writes for All ships (even those not shooting).
            # We must mask the action.
            
            # Simplest:
            # Loop over batch? No.
            # Advanced Indexing.
            # Get indices where shooting.
            batch_idx, ship_idx = torch.where(shooting_mask) # 1D tensors of length M (shooters)
            
            if len(batch_idx) > 0:
                cur = cursors[batch_idx, ship_idx] # (M,)
                
                # Update bullets
                self.state.bullets_pos[batch_idx, ship_idx, cur] = bullet_pos[batch_idx, ship_idx]
                self.state.bullets_vel[batch_idx, ship_idx, cur] = bullet_vel[batch_idx, ship_idx]
                self.state.bullets_time[batch_idx, ship_idx, cur] = default_ship_config.bullet_lifetime
                self.state.bullets_team[batch_idx, ship_idx, cur] = self.state.ships_team[batch_idx, ship_idx]
                
                # Advance cursor for shooters
                new_cur = (cur + 1) % self.max_bullets_per_ship
                self.state.bullet_cursor[batch_idx, ship_idx] = new_cur
        
        # 3. Update Bullets
        self.state.bullets_pos, self.state.bullets_time = update_bullets(
            self.state.bullets_pos,
            self.state.bullets_vel,
            self.state.bullets_time,
            self.dt,
            self.world_size
        )
        
        # 4. Check Collisions & Rewards
        collision_matrix, bullet_mask = check_collisions(
            self.state.ships_pos,
            self.state.ships_team,
            self.state.ships_alive,
            self.state.bullets_pos,
            self.state.bullets_team,
            self.state.bullets_time,
            ship_collision_radius=default_ship_config.collision_radius
        )
        
        # Remove active bullets that hit
        # Set mask bullets_time = 0
        self.state.bullets_time[bullet_mask] = -1.0
        
        # Calculate Damage
        # collision_matrix: (B, N_targets, S_sources) where S = N * K
        # Sum hits per target-source pair? No collision_matrix is boolean.
        # But wait, collision_matrix structure:
        # (B, N_targets, M*K)
        # We need to know TEAM of source to attribute reward.
        
        # Source Team Lookup
        # Reshape bullets_team to (B, M*K)
        flat_bullets_team = self.state.bullets_team.view(self.num_envs, -1) # (B, M*K)
        
        # We want to sum hits by (target_ship, source_team).
        # (B, N_target, M*K) -> (B, N_target, Num_Teams)
        # Multiply (and) collision matrix with one-hot of source teams?
        
        # Assume 2 teams for NvM?
        # Or just specific teams.
        # Let's support an arbitrary number, say 2.
        
        # Construct source_team_mask: (B, M*K) -> (B, 1, M*K)
        # hits_from_team0 = collision_matrix & (flat_bullets_team == 0).unsqueeze(1)
        # count_team0 = hits_from_team0.sum(dim=2)
        
        # Optimization: iterate unique teams present? usually just 0 and 1.
        
        hits_per_ship_by_team = torch.zeros((B, N, 2), device=self.device) # Assume 2 teams
        
        for t in [0, 1]:
            mask_t = (flat_bullets_team == t).unsqueeze(1) # (B, 1, MK)
            hits_t = (collision_matrix & mask_t).sum(dim=2) # (B, N)
            hits_per_ship_by_team[..., t] = hits_t.float()
            
        # Apply Damage
        total_hits_received = hits_per_ship_by_team.sum(dim=2) # (B, N)
        damage = total_hits_received * default_ship_config.bullet_damage
        
        self.state.ships_health -= damage
        
        # Check Deaths
        still_alive = self.state.ships_health > 0
        just_died = self.state.ships_alive & (~still_alive)
        self.state.ships_alive = still_alive & self.state.ships_alive # Update alive status
        
        # Rewards
        # Shape (B, 2) for team rewards? Or (B, N) individual?
        # Env returns (B, 2) typically for 2 teams, or (B,) if competitive sum?
        # Original env: `rewards = {team_id: total_reward}`.
        # Here we return tensor (B, Num_Teams) or (B, N_agents).
        # Training usually takes (B, N) rewards.
        
        # Reward consts:
        # DAMAGE: +0.001 (ENEMY), -0.001 (ALLY)
        # DEATH: +0.1 (ENEMY), -0.1 (ALLY)
        
        # We need to sum up rewards for each TEAM.
        # Then broadcast to agents? Or return team rewards?
        # Let's compute Team Rewards (B, 2).
        
        team_rewards = torch.zeros((B, 2), device=self.device)
        
        for team_id in [0, 1]:
            enemy_id = 1 - team_id
            
            # 1. Damage Dealt (by this team to enemy)
            # hits on enemy ships by team_id
            hits_on_enemy = hits_per_ship_by_team[..., team_id] * (self.state.ships_team == enemy_id) # (B, N)
            damage_dealt = hits_on_enemy.sum(dim=1) * default_ship_config.bullet_damage
            team_rewards[:, team_id] += damage_dealt * RewardConstants.ENEMY_DAMAGE
            
            # 2. Friendly Fire (damage to self)
            hits_on_ally = hits_per_ship_by_team[..., team_id] * (self.state.ships_team == team_id)
            damage_friendly = hits_on_ally.sum(dim=1) * default_ship_config.bullet_damage
            team_rewards[:, team_id] += damage_friendly * RewardConstants.ALLY_DAMAGE
            
            # 3. Kills (Enemy Deaths caused by team)
            # How to know WHO caused death?
            # Approximation: If multiple hits from mixed teams killed it, split credit? 
            # Or simplified: if just_died and hits from team_id > 0?
            # Or proportional?
            # Original env logic:
            # `damage_ship` -> if health <= 0 -> Death Event.
            # Event has `ship_id` (victim).
            # Who gets credit?
            # Original code LOOPED over bullets.
            # `if np.any(hit_mask) ... damage ... if not alive ... Death`
            # But the Event doesn't store killer?
            # `GameEvent.team_id` is the VICTIM's team.
            # `case EventType.DEATH: if event.team_id == team_id: +ALLY_DEATH else: +ENEMY_DEATH`
            # Wait. Rewards are calculated based on events.
            # If Team 0 ship dies, it generates DEATH event with team_id=0.
            # Then Team 0 gets ALLY_DEATH (-0.1).
            # Team 1 gets ENEMY_DEATH (+0.1).
            # So it DOESN'T MATTER who killed it. It just matters that it died.
            # This simplifies things!
            
            deaths = just_died # (B, N)
            
            # Team 0 deaths
            deaths_t0 = (deaths & (self.state.ships_team == 0)).sum(dim=1)
            # Team 1 deaths
            deaths_t1 = (deaths & (self.state.ships_team == 1)).sum(dim=1)
            
            if team_id == 0:
                team_rewards[:, 0] += deaths_t0 * RewardConstants.ALLY_DEATH
                team_rewards[:, 0] += deaths_t1 * RewardConstants.ENEMY_DEATH
            else:
                team_rewards[:, 1] += deaths_t1 * RewardConstants.ALLY_DEATH
                team_rewards[:, 1] += deaths_t0 * RewardConstants.ENEMY_DEATH
            
            # Win/Loss
            # Check active teams
            # If one team wiped out.
            
        # 5. Termination
        
        # Check active counts
        active_t0 = (self.state.ships_alive & (self.state.ships_team == 0)).sum(dim=1)
        active_t1 = (self.state.ships_alive & (self.state.ships_team == 1)).sum(dim=1)
        
        done_mask = (active_t0 == 0) | (active_t1 == 0) # (B,)
        
        # Apply Win/Loss rewards
        # If t0 == 0 and t1 > 0 -> Team 1 Win (Reward VICTORY), Team 0 Loss (DEFEAT)
        # If t0 > 0 and t1 == 0 -> Team 0 Win
        # If both 0 -> Draw (DRAW)
        
        win_t0 = (active_t0 > 0) & (active_t1 == 0)
        win_t1 = (active_t0 == 0) & (active_t1 > 0)
        draw   = (active_t0 == 0) & (active_t1 == 0)
        
        team_rewards[:, 0] += win_t0.float() * RewardConstants.VICTORY
        team_rewards[:, 0] += win_t1.float() * RewardConstants.DEFEAT
        team_rewards[:, 0] += draw.float() * RewardConstants.DRAW
        
        team_rewards[:, 1] += win_t1.float() * RewardConstants.VICTORY
        team_rewards[:, 1] += win_t0.float() * RewardConstants.DEFEAT
        team_rewards[:, 1] += draw.float() * RewardConstants.DRAW
        
        obs = self.get_observations()
        
        # Auto-Reset if done?
        # Usually vector envs auto-reset.
        # I will leave auto-reset to a wrapper or handle it here if requested.
        # For simplicity, returning done mask.
        
        return obs, team_rewards, done_mask, torch.zeros_like(done_mask), {}

    def get_observations(self) -> dict[str, torch.Tensor]:
        """
        Produce "tokens" tensor (B, N, 15).
        """
        # [team_id, health, power, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y, ang_vel, att_x, att_y, shoot]
        # Wait, 13 dims listed but STATE_DIM is 15?
        # Lets check env/ship.py get_token again.
        # [team, health, power, px, py, vx, vy, ax, ay, angv, attx, atty, shoot] -> 13.
        # src/core/constants.py says STATE_DIM = 15.
        # Maybe padding? Or I missed something.
        # Let's count again.
        # 1+1+1+2+2+2+1+2+1 = 13?
        # Check actual code in ship.py
        # 1 (team)
        # 1 (health)
        # 1 (power)
        # 2 (pos)
        # 2 (vel)
        # 2 (acc)
        # 1 (ang)
        # 2 (att)
        # 1 (shoot)
        # Total: 13.
        # Why is STATE_DIM 15? Maybe cooldown? Or future expansion?
        # Observation space says (max_ships, 15).
        # We should pad to 15 to match expectation if that's what downstream expects.
        
        s = self.state
        B, N = s.ships_pos.shape
        
        obs = torch.zeros((B, N, 15), device=self.device)
        
        # 0: Team
        obs[..., 0] = s.ships_team.float()
        # 1: Health
        obs[..., 1] = s.ships_health / NORM_HEALTH # Or ship config max
        # 2: Power
        obs[..., 2] = s.ships_power / NORM_POWER
        
        # 3,4: Pos
        obs[..., 3] = s.ships_pos.real / self.world_size[0]
        obs[..., 4] = s.ships_pos.imag / self.world_size[1]
        
        # 5,6: Vel
        obs[..., 5] = s.ships_vel.real / NORM_VELOCITY
        obs[..., 6] = s.ships_vel.imag / NORM_VELOCITY
        
        # 7,8: Acc
        obs[..., 7] = s.ships_acc.real / NORM_ACCELERATION
        obs[..., 8] = s.ships_acc.imag / NORM_ACCELERATION
        
        # 9: Ang Vel
        obs[..., 9] = s.ships_ang_vel / NORM_ANGULAR_VELOCITY
        
        # 10,11: Attitude
        obs[..., 10] = s.ships_attitude.real
        obs[..., 11] = s.ships_attitude.imag
        
        # 12: Shoot (is_shooting?)
        # We don't store "is_shooting" state explicitly across steps except derivation?
        # In ship.py: "self.is_shooting = True" if fired.
        # This is strictly derived from action in current step.
        # Since obs comes AFTER step, we should know if it shot.
        # `update_ships` returned `should_shoot`.
        # But we didn't save it in state.
        # I should output it in update_ships or derive it from cooldown?
        # If cooldown == firing_cooldown, it just shot.
        # Or close to it.
        # Precise way: return `should_shoot` mask from step and store in state (transiently)?
        # Or just compute: (cooldown > firing_cooldown - dt_eps).
        
        # I'll use logic: ships_cooldown approx firing_cooldown.
        # Or add local var in Step to pass to obs?
        # `get_observations` is stateless method.
        # I will deduce from cooldown.
        # If cooldown is high (near max), it implies shot recently.
        
        just_fired = (s.ships_cooldown > (default_ship_config.firing_cooldown - self.dt * 1.5))
        obs[..., 12] = just_fired.float()
        
        # 13, 14: Padding (zeros)
        
        return {"tokens": obs}
