from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.type_aliases import Schedule

from agents.team_transformer_agent import TeamTransformerModel


class TeamTransformerSB3Policy(ActorCriticPolicy):
    """
    Custom Policy for Stable Baselines3 that uses the TeamTransformerModel.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        model_config: dict = None,
        *args,
        **kwargs,
    ):
        # We need to initialize the parent class, but we'll disable its network creation
        # by passing custom arguments or just overriding the attributes after init.
        # However, SB3 init does a lot of setup.
        
        # Pass dummy values for net_arch to avoid creating default networks if possible,
        # or just let it create them and we replace them.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        # Disable orthogonal initialization for now to keep our model's init
        # (or re-init if we want)

        # Create our custom model
        # We expect model_config to be passed in kwargs or we extract it
        if model_config is None:
            # Default config if not provided
            model_config = {
                "token_dim": 12, # Default from config
                "embed_dim": 64,
                "num_heads": 4,
                "num_layers": 3,
                "max_ships": 8,
                "dropout": 0.0, # Usually 0 for RL? Or keep it?
                "use_layer_norm": True,
            }
            
        # Filter out keys that are not in TeamTransformerModel.__init__
        # specifically num_actions which is in config but not in init
        valid_keys = {
            "token_dim", "embed_dim", "num_heads", "num_layers", 
            "max_ships", "dropout", "use_layer_norm"
        }
        filtered_config = {k: v for k, v in model_config.items() if k in valid_keys}
            
        self.transformer_model = TeamTransformerModel(**filtered_config)
        
        # We need to ensure the action distribution matches
        # Action space is MultiBinary(max_ships * num_actions)
        # The distribution should be Bernoulli
        self.action_dist = make_proba_distribution(action_space)

    def _predict(self, observation: torch.Tensor | Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        """
        # Forward pass through transformer
        output = self.transformer_model(observation)
        action_logits = output["action_logits"] # (batch, max_ships, num_actions)
        
        # Flatten logits to match action space
        batch_size = action_logits.shape[0]
        action_logits_flat = action_logits.view(batch_size, -1)
        
        # Get distribution
        distribution = self.action_dist.proba_distribution(action_logits_flat)
        
        if deterministic:
            return distribution.mode()
        return distribution.sample()

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
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

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
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

        :param obs:
        :return: the action distribution.
        """
        output = self.transformer_model(obs)
        action_logits = output["action_logits"]
        
        batch_size = action_logits.shape[0]
        action_logits_flat = action_logits.view(batch_size, -1)
        
        return self.action_dist.proba_distribution(action_logits_flat)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        output = self.transformer_model(obs)
        return output["value"]
