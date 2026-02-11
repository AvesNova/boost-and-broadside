"""
Vectorized sticky agent wrapper for adding behavioral diversity to data collection.
"""

from typing import Any, Literal

import torch

from boost_and_broadside.core.constants import NUM_POWER_ACTIONS, NUM_SHOOT_ACTIONS, NUM_TURN_ACTIONS


class VectorStickyAgent:
    """Wraps a vectorized agent to add sticky action logic and skill-based noise.

    This implementation maintains state for each ship in each environment:
    1. Timers: How long the current action persists.
    2. Mode: Whether the current persistent action is 'Expert' or 'Random'.
    3. Sticky Value: The random action chosen if Mode is 'Random'.

    Attributes:
        agent: The base vectorized agent producing expert actions.
        num_envs: Number of parallel environments.
        max_ships: Maximum number of ships per team.
        device: Torch device (cpu/cuda).
        min_skill: Minimum skill level [0, 1].
        max_skill: Maximum skill level [0, 1].
        expert_ratio: Fraction of environments that are pure expert (skill=1.0).
        random_dist: Distribution for sticky action duration.
        mean_sticky_steps: Mean number of steps an action sticks.
    """

    def __init__(
        self,
        base_agent: Any,
        num_envs: int,
        max_ships: int,
        device: torch.device,
        min_skill: float = 0.1,
        max_skill: float = 1.0,
        expert_ratio: float = 0.0,
        random_dist: Literal["beta", "exponential"] = "beta",
        mean_sticky_steps: float = 16.0,
    ):
        """Initializes the sticky agent wrapper.

        Args:
            base_agent: The base agent.
            num_envs: Number of environments.
            max_ships: Max ships per team.
            device: Computing device.
            min_skill: Min skill probability.
            max_skill: Max skill probability.
            expert_ratio: Fraction of Pure Expert envs.
            random_dist: Duration distribution type.
            mean_sticky_steps: Mean steps per sticky choice.
        """
        self.agent = base_agent
        self.num_envs = num_envs
        self.max_ships = max_ships
        self.device = device

        self.min_skill = min_skill
        self.max_skill = max_skill
        self.expert_ratio = expert_ratio
        self.random_dist = random_dist
        self.mean_sticky_steps = mean_sticky_steps

        # State Tensors (B, N, 3) -> [Power, Turn, Shoot]
        # Timers: Int
        self.timers = torch.zeros((num_envs, max_ships, 3), dtype=torch.long, device=device)

        # Expert Mode: Bool (True = Use Expert Action, False = Use Sticky Random)
        self.is_expert_mode = torch.ones((num_envs, max_ships, 3), dtype=torch.bool, device=device)

        # Sticky Values: Int (The random action selected)
        self.sticky_values = torch.zeros((num_envs, max_ships, 3), dtype=torch.long, device=device)

        # Skill Levels: Float (Probability of Expert)
        num_experts = int(num_envs * expert_ratio)
        self.skills = torch.empty((num_envs, max_ships, 1), device=device)

        # First N envs are experts
        if num_experts > 0:
            self.skills[:num_experts] = 1.0

        # Remaining envs are sampled
        if num_experts < num_envs:
            self.skills[num_experts:] = torch.empty(
                (num_envs - num_experts, max_ships, 1), device=device
            ).uniform_(min_skill, max_skill)

        # Expand to 3 channels for checks
        self.skills_3ch = self.skills.expand(-1, -1, 3)

    def _sample_durations(self, mask: torch.Tensor) -> torch.Tensor:
        """Samples durations for expired sticky actions.

        Args:
            mask: Boolean mask of (B, N, 3) indicating expired timers.

        Returns:
            Tensor of sampled durations.
        """
        count = mask.sum()
        if count == 0:
            return torch.empty(0, device=self.device)

        if self.random_dist == "beta":
            # Beta(2, 6) * (mean * 4) -> Mean is 0.25. 0.25 * X = mean -> X = mean * 4
            scale = self.mean_sticky_steps * 4.0
            alpha = torch.tensor(2.0, device=self.device)
            beta = torch.tensor(6.0, device=self.device)
            dist = torch.distributions.Beta(alpha, beta)
            samples = dist.sample((count,))
            steps = (samples * scale).long()

        elif self.random_dist == "exponential":
            # Exponential(lambda). Mean = 1/lambda.
            lambd = 1.0 / self.mean_sticky_steps
            dist = torch.distributions.Exponential(torch.tensor(lambd, device=self.device))
            steps = dist.sample((count,)).long()

        else:
            steps = torch.full((count,), int(self.mean_sticky_steps), device=self.device)

        return torch.clamp(steps, min=1)

    def get_actions(self, state: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes taken actions, expert actions, and skill levels.

        Args:
            state: The current environment state.

        Returns:
            Tuple containing:
                - taken_actions: Actions actually executed (B, N, 3).
                - expert_actions: Optimal actions (B, N, 3).
                - skills: Skill levels (B, N).
        """
        # 1. Get Expert Actions
        expert_actions = self.agent.get_actions(state)  # (B, N, 3)

        # 2. Decrement Timers
        self.timers -= 1

        # 3. Handle Expired Timers
        expired_mask = self.timers <= 0  # (B, N, 3)

        if expired_mask.any():
            # A. Sample New Durations
            new_durations = self._sample_durations(expired_mask)
            self.timers[expired_mask] = new_durations

            # B. Sample New Mode (Expert vs Random) based on Skill
            active_skills = self.skills_3ch[expired_mask]
            rand_check = torch.rand_like(active_skills)
            new_modes = rand_check < active_skills
            self.is_expert_mode[expired_mask] = new_modes

            # C. Sample New Sticky Values (Random Actions)
            # Power: 0-2, Turn: 0-6, Shoot: 0-1
            max_vals = torch.tensor(
                [NUM_POWER_ACTIONS, NUM_TURN_ACTIONS, NUM_SHOOT_ACTIONS], device=self.device
            ).expand(self.num_envs, self.max_ships, 3)

            # Limit random values
            rand_vals = (
                torch.rand(expired_mask.sum(), device=self.device) * max_vals[expired_mask]
            ).long()
            self.sticky_values[expired_mask] = rand_vals

        # 4. Composite Final Actions
        taken_actions = expert_actions.clone()
        random_mask = ~self.is_expert_mode
        taken_actions[random_mask] = self.sticky_values[random_mask]

        return taken_actions, expert_actions, self.skills.squeeze(-1)

