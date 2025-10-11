"""
Unified Agent System - supports human, scripted, RL, and data collection
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from copy import deepcopy

from .scripted_agent import ScriptedAgent


class Agent(ABC):
    """Base class for all agent types"""

    @abstractmethod
    def get_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """Get actions for specified ships"""
        pass

    @abstractmethod
    def get_agent_type(self) -> str:
        """Return agent type for logging"""
        pass


class ScriptedAgentProvider(Agent):
    """Provides scripted agent actions"""

    def __init__(
        self,
        scripted_config: dict,
        world_size: tuple[int, int],
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.agents: dict[int, ScriptedAgent] = {}
        self.scripted_config = scripted_config
        self.world_size = world_size
        self.rng = rng

    def get_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        actions = {}
        for ship_id in ship_ids:
            if ship_id not in self.agents:
                self.agents[ship_id] = ScriptedAgent(
                    controlled_ship_id=ship_id,
                    world_size=self.world_size,
                    **self.scripted_config,
                )

            if self._is_ship_alive(obs_dict, ship_id):
                actions[ship_id] = self.agents[ship_id](obs_dict)
            else:
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        return actions

    def _is_ship_alive(self, obs_dict: dict, ship_id: int) -> bool:
        if ship_id >= obs_dict["alive"].shape[0]:
            return False
        return obs_dict["alive"][ship_id, 0].item() > 0

    def get_agent_type(self) -> str:
        return "scripted"


class RLAgentProvider(Agent):
    """Provides RL model actions (both PPO and BC models)"""

    def __init__(self, model, team_controller, model_type: str = "transformer"):
        self.model = model
        self.team_controller = team_controller
        self.model_type = model_type  # "transformer", "ppo", "bc"

    def get_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        try:
            if self.model_type == "ppo":
                return self._get_ppo_actions(obs_dict, ship_ids)
            else:
                return self._get_transformer_actions(obs_dict, ship_ids)

        except Exception as e:
            print(f"RL agent error: {e}, using random actions")
            return {
                ship_id: torch.randint(0, 2, (6,), dtype=torch.float32)
                for ship_id in ship_ids
            }

    def _get_transformer_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """Get actions from transformer-based model (BC or direct transformer)"""
        tokens = obs_dict["tokens"].numpy()
        ship_mask = self._create_ship_mask(tokens)

        with torch.no_grad():
            obs_batch = {"tokens": torch.from_numpy(tokens).unsqueeze(0).float()}
            all_actions = self.model.get_actions(
                obs_batch, ship_mask, deterministic=False
            )["actions"][0]

            # Extract actions for requested ships
            actions = {}
            for ship_id in ship_ids:
                if ship_id < all_actions.shape[0]:
                    actions[ship_id] = all_actions[ship_id]
                else:
                    actions[ship_id] = torch.zeros(6, dtype=torch.float32)

            return actions

    def _get_ppo_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """Get actions from PPO model"""
        tokens = obs_dict["tokens"].numpy()

        with torch.no_grad():
            # PPO model expects flattened observation
            action_array, _ = self.model.predict(tokens, deterministic=False)

            # Convert back to ship actions using team controller
            # This assumes the PPO model was trained with specific team assignments
            actions = {}
            actions_per_ship = 6

            for i, ship_id in enumerate(sorted(ship_ids)):
                start_idx = i * actions_per_ship
                end_idx = start_idx + actions_per_ship
                if end_idx <= len(action_array):
                    actions[ship_id] = torch.from_numpy(
                        action_array[start_idx:end_idx]
                    ).float()
                else:
                    actions[ship_id] = torch.zeros(6, dtype=torch.float32)

            return actions

    def _create_ship_mask(self, tokens: np.ndarray) -> torch.Tensor:
        batch_size = 1
        max_ships = tokens.shape[0]
        mask = torch.ones(batch_size, max_ships, dtype=torch.bool)

        for ship_id in range(max_ships):
            if tokens[ship_id, 1] > 0:  # Health > 0
                mask[0, ship_id] = False

        return mask

    def get_agent_type(self) -> str:
        return f"rl_{self.model_type}"


class HumanAgentProvider(Agent):
    """Provides human input actions"""

    def __init__(self, renderer):
        self.renderer = renderer

    def get_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        # Update human input
        self.renderer.update_human_actions()
        human_actions = self.renderer.get_human_actions()

        # Return only requested ship actions
        return {
            ship_id: human_actions.get(ship_id, torch.zeros(6, dtype=torch.float32))
            for ship_id in ship_ids
        }

    def get_agent_type(self) -> str:
        return "human"


class RandomAgentProvider(Agent):
    """Provides random actions"""

    def __init__(self, rng: np.random.Generator = np.random.default_rng()):
        self.rng = rng

    def get_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        actions = {}
        for ship_id in ship_ids:
            # Check if ship is alive (within observation bounds and alive==1)
            if (
                ship_id < obs_dict.get("alive", torch.empty(0)).shape[0]
                and obs_dict.get("alive", torch.zeros(1, 1))[ship_id, 0].item() == 1
            ):
                # Ship is alive, generate random actions
                actions[ship_id] = torch.from_numpy(self.rng.integers(0, 2, 6)).float()
            else:
                # Ship is dead or not found, return zero actions
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)
        return actions

    def get_agent_type(self) -> str:
        return "random"


class SelfPlayAgentProvider(Agent):
    """Manages self-play with memory of past models"""

    def __init__(
        self,
        team_controller,
        memory_size: int = 50,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.team_controller = team_controller
        self.model_memory: list[dict] = []
        self.current_opponent = None
        self.memory_size = memory_size
        self.rng = rng

    def add_model_to_memory(self, model):
        """Add a model snapshot to memory"""
        model_state = deepcopy(model.state_dict())
        self.model_memory.append(model_state)

        # Keep only recent models
        if len(self.model_memory) > self.memory_size:
            self.model_memory.pop(0)

    def update_opponent(self, model_class, model_config: dict):
        """Update current opponent from memory"""
        if not self.model_memory:
            self.current_opponent = RandomAgentProvider()
            return

        # Create new model instance
        opponent_model = model_class(**model_config)

        model_state = self.rng.choice(self.model_memory)
        opponent_model.load_state_dict(model_state)
        opponent_model.eval()

        # Wrap as RL agent
        self.current_opponent = RLAgentProvider(opponent_model, self.team_controller)

    def get_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        if self.current_opponent is None:
            # Fallback to random if no opponent set
            return RandomAgentProvider().get_actions(obs_dict, ship_ids)

        return self.current_opponent.get_actions(obs_dict, ship_ids)

    def get_agent_type(self) -> str:
        if self.current_opponent:
            return f"self_play_{self.current_opponent.get_agent_type()}"
        return "self_play_random"


def create_scripted_agent(
    world_size: tuple[int, int], config: dict | None = None
) -> ScriptedAgentProvider:
    """Factory function for scripted agents"""
    default_config = {
        "max_shooting_range": 500.0,
        "angle_threshold": 5.0,
        "bullet_speed": 500.0,
        "target_radius": 10.0,
        "radius_multiplier": 1.5,
    }

    if config:
        default_config.update(config)

    return ScriptedAgentProvider(default_config, world_size)


def create_rl_agent(
    model, team_controller, model_type: str = "transformer"
) -> RLAgentProvider:
    """Factory function for RL agents"""
    return RLAgentProvider(model, team_controller, model_type)


def create_human_agent(renderer) -> HumanAgentProvider:
    """Factory function for human agents"""
    return HumanAgentProvider(renderer)


def create_selfplay_agent(
    team_controller, memory_size: int = 50
) -> SelfPlayAgentProvider:
    """Factory function for self-play agents"""
    return SelfPlayAgentProvider(team_controller, memory_size)
