"""
Custom SB3 policy that integrates the transformer model.
"""

from typing import Any
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3 import PPO
from gymnasium import spaces

from .team_transformer_model import TeamTransformerModel


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that uses the transformer model.
    Extracts ship embeddings for the policy and value networks.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        transformer_config: dict = None,
        team_id: int = 0,
        team_assignments: dict = None,
    ):
        # Calculate features dim based on controlled ships
        self.team_assignments = team_assignments or {0: [0], 1: [1]}
        self.controlled_ships = self.team_assignments[team_id]
        self.num_controlled_ships = len(self.controlled_ships)
        self.team_id = team_id

        transformer_config = transformer_config or {}
        embed_dim = transformer_config.get("embed_dim", 64)

        # Features = embeddings of controlled ships flattened
        features_dim = self.num_controlled_ships * embed_dim

        super().__init__(observation_space, features_dim)

        # Create transformer model
        self.transformer = TeamTransformerModel(**transformer_config)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features using transformer.

        Args:
            observations: (batch_size, max_ships, token_dim)

        Returns:
            features: (batch_size, features_dim) - flattened embeddings of controlled ships
        """
        batch_size = observations.shape[0]

        # Prepare observation dict for transformer
        obs_dict = {"tokens": observations}

        # Create ship mask (True for inactive ships)
        max_ships = observations.shape[1]
        ship_mask = torch.ones(
            batch_size, max_ships, dtype=torch.bool, device=observations.device
        )

        # Mark active ships as False (not masked)
        for ship_id in range(max_ships):
            # Check if ship is alive (health > 0, assuming health is token[1])
            alive_mask = observations[:, ship_id, 1] > 0  # (batch_size,)
            ship_mask[:, ship_id] = ~alive_mask

        # Forward through transformer
        output = self.transformer(obs_dict, ship_mask)
        ship_embeddings = output[
            "ship_embeddings"
        ]  # (batch_size, max_ships, embed_dim)

        # Extract embeddings for controlled ships only
        controlled_embeddings = []
        for ship_id in sorted(self.controlled_ships):
            if ship_id < max_ships:
                controlled_embeddings.append(ship_embeddings[:, ship_id, :])
            else:
                # Pad with zeros if ship_id out of bounds
                embed_dim = ship_embeddings.shape[-1]
                controlled_embeddings.append(
                    torch.zeros(batch_size, embed_dim, device=observations.device)
                )

        # Flatten controlled ship embeddings
        features = torch.cat(
            controlled_embeddings, dim=1
        )  # (batch_size, num_controlled * embed_dim)

        return features


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using transformer features.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        transformer_config: dict = None,
        team_id: int = 0,
        team_assignments: dict = None,
        **kwargs,
    ):
        self.transformer_config = transformer_config or {}
        self.team_id = team_id
        self.team_assignments = team_assignments or {0: [0], 1: [1]}
        self.controlled_ships = self.team_assignments[team_id]

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs={
                "transformer_config": transformer_config,
                "team_id": team_id,
                "team_assignments": team_assignments,
            },
            **kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "transformer_config": self.transformer_config,
                "team_id": self.team_id,
                "team_assignments": self.team_assignments,
            }
        )
        return data

    def get_transformer_model(self) -> TeamTransformerModel:
        """Get the underlying transformer model for self-play memory."""
        return self.features_extractor.transformer


def create_team_ppo_model(
    env,
    transformer_config: dict = None,
    team_id: int = 0,
    team_assignments: dict = None,
    ppo_config: dict = None,
):
    """
    Create PPO model with transformer policy.

    Args:
        env: The wrapped environment
        transformer_config: Config for transformer model
        team_id: Which team this policy controls
        team_assignments: Team assignments dict
        ppo_config: Config for PPO algorithm

    Returns:
        PPO model ready for training
    """

    # Default configs
    transformer_config = transformer_config or {
        "token_dim": 10,
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 3,
        "max_ships": 4,
        "num_actions": 6,
        "dropout": 0.1,
    }

    ppo_config = ppo_config or {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    # Create policy kwargs
    policy_kwargs = {
        "transformer_config": transformer_config,
        "team_id": team_id,
        "team_assignments": team_assignments,
    }

    # Create PPO model
    model = PPO(
        policy=TransformerActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **ppo_config,
    )

    return model
