"""
World Model Agent for gameplay.

Wraps the WorldModel to act as an agent in the environment.
Maintains a history of observations and uses the model to predict the next action.
"""

import logging
import torch
from collections import deque

from agents.world_model import WorldModel
from agents.tokenizer import observation_to_tokens

log = logging.getLogger(__name__)


import torch.nn.functional as F


class WorldModelAgent:
    """
    Agent that uses the World Model to predict actions.

    It maintains a sliding window of history (tokens) and at each step:
    1. Appends the current observation token to history.
    2. Appends a MASK token for the next step.
    3. Runs the World Model to predict the action for the masked token.
    4. Samples/selects the action and returns it.
    """

    def __init__(
        self,
        agent_id: str,
        team_id: int,
        squad: list[int],
        model_path: str | None = None,
        context_len: int = 128,
        device: str = "cpu",
        state_dim: int = 12,
        action_dim: int = 12,
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_ships: int = 8,
        dropout: float = 0.1,
        world_size: tuple[float, float] = (1024.0, 1024.0),
        model: WorldModel | None = None,
        **kwargs,
    ):
        """
        Initialize the WorldModelAgent.

        Args:
            agent_id: Unique identifier for the agent.
            team_id: Team ID (0 or 1).
            squad: List of ship IDs controlled by this agent.
            model_path: Path to a pretrained model checkpoint.
            context_len: Maximum context length for the model.
            device: Device to run the model on ("cpu" or "cuda").
            state_dim: Dimension of state tokens.
            action_dim: Dimension of action tokens (12 total for categorical).
            embed_dim: Transformer embedding dimension.
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
            max_ships: Maximum number of ships in the environment.
            dropout: Dropout rate.
            model: Optional pre-loaded WorldModel instance.
            **kwargs: Additional configuration parameters (ignored).
        """
        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad
        self.device = torch.device(device)
        self.context_len = context_len
        self.max_ships = max_ships
        self.action_dim = action_dim
        self.world_size = world_size

        # Initialize model
        if model is not None:
            self.model = model
            # Ensure model is on correct device
            self.model.to(self.device)
        else:
            self.model = WorldModel(
                state_dim=state_dim,
                action_dim=action_dim,
                embed_dim=embed_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                max_ships=max_ships,
                max_context_len=context_len,
                dropout=dropout,
            ).to(self.device)

            if model_path:
                self.load_model(model_path)
            else:
                log.warning(
                    "WorldModelAgent initialized with random weights (no model_path provided)"
                )

        self.model.eval()

        # History buffer: list of (token, action) tuples
        # token: (1, num_ships, state_dim)
        # action: (1, num_ships, action_dim) - ONE HOT
        self.history = deque(maxlen=context_len - 1)

        # Track previous action to feed into the model (Action t-1)
        # Stored as One-Hot for model input
        self.last_action = torch.zeros(1, max_ships, action_dim, device=self.device)

    def load_model(self, path: str) -> None:
        """
        Load model weights from file.

        Args:
            path: Path to the model checkpoint.
        """
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            log.info(f"Loaded WorldModel from {path}")
        except Exception as e:
            log.error(f"Failed to load WorldModel from {path}: {e}")

    def reset(self) -> None:
        """Reset agent state (history)."""
        self.history.clear()
        self.last_action = torch.zeros(
            1, self.max_ships, self.action_dim, device=self.device
        )

    def __call__(
        self, observation: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Get actions for the specified ships.

        Args:
            observation: Observation dictionary.
            ship_ids: List of ship IDs to get actions for.

        Returns:
            Dict mapping ship_id -> action tensor (categorical indices).
        """
        # 1. Tokenize observation -> (1, num_ships, state_dim)
        current_token = observation_to_tokens(
            observation, self.team_id, self.world_size
        ).to(self.device)

        # 2. Prepare input sequence
        # Construct sequence: [past_tokens, current_token, mask_token]
        # Construct actions: [past_actions, last_action, mask_action]

        hist_tokens = [t for t, a in self.history]
        hist_actions = [a for t, a in self.history]

        input_tokens_list = hist_tokens + [current_token]
        input_actions_list = hist_actions + [self.last_action]

        # Add MASK step (target) for t+1
        mask_token = torch.zeros_like(current_token)
        mask_action = torch.zeros_like(self.last_action)

        input_tokens_list.append(mask_token)
        input_actions_list.append(mask_action)

        # Stack into tensors
        input_tokens = torch.cat(input_tokens_list, dim=0).unsqueeze(
            0
        )  # (1, Seq, N, F)
        input_actions = torch.cat(input_actions_list, dim=0).unsqueeze(
            0
        )  # (1, Seq, N, A)

        seq_len = input_tokens.shape[1]

        # Create mask: Mask ONLY the last token (t+1)
        mask = torch.zeros(
            1, seq_len, self.max_ships, dtype=torch.bool, device=self.device
        )
        mask[:, -1, :] = True

        # 3. Forward pass
        with torch.no_grad():
            _, pred_actions_logits, _, _, _ = self.model(
                input_tokens, input_actions, mask=mask
            )

        # 4. Extract prediction for the masked step (last step)
        next_action_logits = pred_actions_logits[:, -1, :, :]  # (1, N, 12)

        # 5. Extract actions
        # Split heads
        power_logits = next_action_logits[..., 0:3]
        turn_logits = next_action_logits[..., 3:10]
        shoot_logits = next_action_logits[..., 10:12]

        # Argmax selection
        power_idx = power_logits.argmax(dim=-1)
        turn_idx = turn_logits.argmax(dim=-1)
        shoot_idx = shoot_logits.argmax(dim=-1)

        # Convert to one-hot for history
        power_oh = F.one_hot(power_idx, num_classes=3)
        turn_oh = F.one_hot(turn_idx, num_classes=7)
        shoot_oh = F.one_hot(shoot_idx, num_classes=2)
        next_action_oh = torch.cat([power_oh, turn_oh, shoot_oh], dim=-1).float()

        # 6. Update history
        # Store (current_token, last_action) which represents step t
        self.history.append((current_token, self.last_action))
        # Update last_action to the action we just decided (Action t)
        self.last_action = next_action_oh

        # 7. Return categorical indices for my ships
        team_actions = {}
        for ship_id in ship_ids:
            if ship_id < self.max_ships:
                # Construct (3,) tensor of indices
                action_indices = torch.tensor(
                    [
                        power_idx[0, ship_id].item(),
                        turn_idx[0, ship_id].item(),
                        shoot_idx[0, ship_id].item(),
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                team_actions[ship_id] = action_indices

        return team_actions

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "world_model"
