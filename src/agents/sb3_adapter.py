"""
Stable Baselines3 adapter for TeamTransformerModel.

Provides a custom SB3 policy that wraps the TeamTransformerModel for use
with Stable Baselines3 RL algorithms.
"""

from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    Distribution,
    make_proba_distribution,
)
from stable_baselines3.common.type_aliases import Schedule

from agents.team_transformer_agent import TeamTransformerModel


class TeamTransformerSB3Policy(ActorCriticPolicy):
    """
    Custom Policy for Stable Baselines3 that uses the TeamTransformerModel.

    This adapter allows the TeamTransformerModel to be used with SB3's
    PPO and other on-policy algorithms.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        model_config: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the SB3 policy.

        Args:
            observation_space: Observation space from the environment.
            action_space: Action space from the environment.
            lr_schedule: Learning rate schedule function.
            model_config: Configuration for TeamTransformerModel.
            *args: Additional positional arguments for ActorCriticPolicy.
            **kwargs: Additional keyword arguments for ActorCriticPolicy.
        """
        # Initialize parent class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        # Create our custom model
        if model_config is None:
            # Default config if not provided
            model_config = {
                "token_dim": 12,
                "embed_dim": 64,
                "num_heads": 4,
                "num_layers": 3,
                "max_ships": 8,
                "dropout": 0.0,
                "use_layer_norm": True,
            }

        # Filter out keys that are not in TeamTransformerModel.__init__
        valid_keys = {
            "token_dim",
            "embed_dim",
            "num_heads",
            "num_layers",
            "max_ships",
            "dropout",
            "use_layer_norm",
        }
        filtered_config = {k: v for k, v in model_config.items() if k in valid_keys}

        self.transformer_model = TeamTransformerModel(**filtered_config)

        # Create action distribution
        self.action_dist = make_proba_distribution(action_space)

    def _predict(
        self,
        observation: torch.Tensor | dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        Args:
            observation: Observation tensor.
            deterministic: Whether to use deterministic actions.

        Returns:
            Action tensor.
        """
        # Forward pass through transformer
        output = self.transformer_model(observation)
        action_logits = output["action_logits"]  # (batch, max_ships, num_actions)

        # Flatten logits to match action space
        batch_size = action_logits.shape[0]
        action_logits_flat = action_logits.view(batch_size, -1)

        # Get distribution
        distribution = self.action_dist.proba_distribution(action_logits_flat)

        if deterministic:
            return distribution.mode()
        return distribution.sample()

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic).

        Args:
            obs: Observation tensor.
            deterministic: Whether to sample or use deterministic actions.

        Returns:
            Tuple of (action, value, log_prob).
        """
        output = self.transformer_model(obs)
        action_logits = output["action_logits"]
        values = output["value"]

        # Flatten logits
        batch_size = action_logits.shape[0]
        action_logits_flat = action_logits.view(batch_size, -1)

        distribution = self.action_dist.proba_distribution(action_logits_flat)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy.

        Args:
            obs: Observation tensor.
            actions: Actions to evaluate.

        Returns:
            Tuple of (values, log_prob, entropy).
        """
        output = self.transformer_model(obs)
        action_logits = output["action_logits"]
        values = output["value"]

        batch_size = action_logits.shape[0]
        action_logits_flat = action_logits.view(batch_size, -1)

        distribution = self.action_dist.proba_distribution(action_logits_flat)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        Args:
            obs: Observation tensor.

        Returns:
            The action distribution.
        """
        output = self.transformer_model(obs)
        action_logits = output["action_logits"]

        batch_size = action_logits.shape[0]
        action_logits_flat = action_logits.view(batch_size, -1)

        return self.action_dist.proba_distribution(action_logits_flat)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy.

        Args:
            obs: Observation tensor.

        Returns:
            The estimated values.
        """
        output = self.transformer_model(obs)
        return output["value"]
