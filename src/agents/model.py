from typing import Any
import torch

from .base import Agent


class ModelAgent(Agent):
    """Agent that controls n ships using a neural network model"""

    def __init__(
        self,
        agent_id: str,
        team_id: int,
        squad: list[int],
        model_path: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize model agent

        Args:
            agent_id: Unique identifier for this agent
            team_id: Which team this agent belongs to
            squad: List of ship IDs this agent controls
            model_path: Path to trained model file
            config: Configuration for model behavior
        """
        super().__init__(agent_id, team_id, squad)

        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model if path provided
        if model_path:
            self._load_model()

    def _load_model(self):
        """Load the trained model"""
        try:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load model from {self.model_path}: {e}")
            self.model = None

    def get_actions(self, obs_dict: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        """
        Get actions from neural network model

        Args:
            obs_dict: Observation dictionary from environment

        Returns:
            Dictionary mapping ship_id to action tensor
        """
        actions: dict[int, torch.Tensor] = {}
        valid_ships = self.get_ship_ids_for_obs(obs_dict)

        if not valid_ships or self.model is None:
            # No valid ships or no model, return zero actions
            for ship_id in self.squad:
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)
            return actions

        try:
            # Prepare observation for model
            model_input = self._prepare_model_input(obs_dict, valid_ships)

            with torch.no_grad():
                # Get model predictions
                model_output = self.model(model_input)

                # Convert model output to actions for each ship
                for i, ship_id in enumerate(valid_ships):
                    if i < model_output.shape[0]:
                        # Apply sigmoid to get binary actions
                        ship_actions = torch.sigmoid(model_output[i])

                        # Threshold to get binary actions
                        binary_actions = (ship_actions > 0.5).float()
                        actions[ship_id] = binary_actions
                    else:
                        actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        except Exception as e:
            print(f"Error getting actions from model: {e}")
            # Fallback to random actions
            for ship_id in valid_ships:
                actions[ship_id] = torch.randint(0, 2, (6,), dtype=torch.float32)

        # Ensure all ships in squad have actions
        for ship_id in self.squad:
            if ship_id not in actions:
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        return actions

    def _prepare_model_input(
        self, obs_dict: dict[str, torch.Tensor], valid_ships: list[int]
    ) -> torch.Tensor:
        """
        Prepare observation for model input

        Args:
            obs_dict: Observation dictionary from environment
            valid_ships: List of valid ship IDs

        Returns:
            Tensor ready for model input
        """
        # Extract tokens for valid ships
        tokens = obs_dict["tokens"]

        # Create input tensor for valid ships only
        ship_tokens = []
        for ship_id in valid_ships:
            if ship_id < tokens.shape[0]:
                ship_tokens.append(tokens[ship_id])

        if not ship_tokens:
            # No valid ships, return empty tensor
            return torch.zeros(1, 10, dtype=torch.float32)

        # Stack tokens into batch
        batch_tokens = torch.stack(ship_tokens)

        return batch_tokens

    def get_agent_type(self) -> str:
        """Return agent type for logging/debugging"""
        return "model"

    def reset(self) -> None:
        """Reset agent state for new episode"""
        # Model agent doesn't need to reset any state
        pass

    def set_model(self, model: Any) -> None:
        """Set the model directly (useful for training)"""
        self.model = model
        self.model.eval()
