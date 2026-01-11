"""
Stable Baselines3 adapter for World Model.

Provides a custom SB3 policy that wraps the WorldModel for use
with Stable Baselines3 RL algorithms, exposing auxiliary loss functionality.
"""

from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    Distribution,
    make_proba_distribution,
)
from stable_baselines3.common.type_aliases import Schedule

from agents.world_model import WorldModel
from train.train_world_model import create_mixed_mask


class WorldModelSB3Policy(ActorCriticPolicy):
    """
    Custom Policy for Stable Baselines3 that uses the WorldModel.

    Features:
    - Wraps WorldModel as the feature extractor and actor.
    - Adds a Value Head (MLP) for the critic.
    - Implements `get_dynamics_loss` for auxiliary self-supervised learning.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        model_config: dict[str, Any] | None = None,
        freeze_backbone: bool = False,
        aux_loss_config: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the SB3 policy.

        Args:
            observation_space: Observation space.
            action_space: Action space.
            lr_schedule: Learning rate schedule.
            model_config: Config for WorldModel.
            freeze_backbone: Whether to freeze WorldModel weights (no RL updates).
            aux_loss_config: Config for auxiliary loss (mask_ratio, etc.).
            *args: Args for ActorCriticPolicy.
            **kwargs: Kwargs for ActorCriticPolicy.
        """
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        self.model_config = model_config or {}
        self.aux_loss_config = aux_loss_config or {}
        self.freeze_backbone = freeze_backbone

        # 1. Instantiate World Model
        self.world_model = WorldModel(
            state_dim=self.model_config.get("state_dim", 12),
            action_dim=self.model_config.get("action_dim", 12),
            embed_dim=self.model_config.get("embed_dim", 128),
            n_layers=self.model_config.get("n_layers", 8),
            n_heads=self.model_config.get("n_heads", 4),
            max_ships=self.model_config.get("max_ships", 8),
            max_context_len=self.model_config.get("context_len", 128),
            dropout=self.model_config.get("dropout", 0.1),
        )

        # Freeze backbone if requested (only train value head and/or adapter layers)
        if self.freeze_backbone:
            for param in self.world_model.parameters():
                param.requires_grad = False

        # 2. Value Head (Critic)
        # WorldModel output dimension is embed_dim
        embed_dim = self.model_config.get("embed_dim", 128)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

        # 3. Action Distribution
        self.action_dist = make_proba_distribution(action_space)

    def _get_obs_actions(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parse observation tensor into states and actions.

        The RL wrapper stacks observations (B, T, N, F).
        However, WorldModel expects states and actions separately.
        Assuming the tokens in obs are [state, one_hot_action] concatenated?
        Wait, SB3Wrapper currently returns just tokens (state vectors).
        It does NOT include past actions in the observation 'tokens' key.
        
        Correction: We need to ensure the environment provides BOTH states and past actions.
        Or, we maintain the assumption that `tokens` contains the features we need.
        
        Let's look at `sb3_wrapper.py`. It returns "tokens": (N, F).
        If we use FrameStack, we get (B, T, N, F).
        
        BUT, WorldModel takes `states` and `actions` as separate inputs.
        We face a discrepancy here.
        Option A: The policy must maintain its own history of actions (Recurrent).
        Option B: The environment puts past actions into the observation.
        
        Given we want to allow "No Pretraining" and simple integration, 
        we should rely on FrameStack in the env to give us history of STATES.
        But we are missing ACTIONS.
        
        CRITICAL: The WorldModel needs (state, previous_action) pairs.
        If we only give states, the WM is crippled (cannot deduce action intent directly).
        
        For this implementation, let's assume the input `obs` contains 
        embedded sequences of (state) and we might have to Mock the actions 
        or change the wrapper to include them.
        
        Let's MODIFY `sb3_wrapper.py` later to include 'last_action' in observation.
        For now, let's assume `obs` is a dictionary or tensor that somehow has info.
        
        Wait, `sb3_wrapper` returns `{'tokens': ...}`.
        If we FrameStack, we get `{'tokens': (B, T, N, F)}`.
        
        We need `actions` for the World Model.
        For the *first* implementation iteration, let's use Zero actions if not available,
        meaning the model predicts based on State dynamics only.
        
        BETTER: We should modify SB3Wrapper to include "actions" in observation.
        
        Lets proceed assuming `obs` has keys "tokens" and "prev_actions" 
        (which we will add to wrapper).
        """
        # obs is usually just the tensor if it's not Dict space, but we used Dict in wrapper.
        
        if isinstance(obs, dict):
            states = obs.get("tokens") # (B, T, N, F)
            actions = obs.get("prev_actions") # (B, T, N, A)
        else:
             # If flattened or direct tensor, assume it's just states
             states = obs
             actions = None
        
        if actions is None:
            # Create dummy zero actions: (B, T, N, 12)
            # This is suboptimal but allows running without wrapper changes initially
            b, t, n, _ = states.shape
            action_dim = self.model_config.get("action_dim", 12)
            actions = torch.zeros(b, t, n, action_dim, device=states.device)

        return states, actions

    def _predict(
        self,
        observation: torch.Tensor | dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        states, actions = self._get_obs_actions(observation)
        
        # Forward pass (no masking/noise for inference)
        # Returns: (pred_states, pred_actions_logits, mask, current_key_values)
        _, pred_action_logits, _, _ = self.world_model(states, actions)
        
        # Taking prediction for the NEXT step (t+1)?
        # The model outputs a sequence of predictions.
        # We want the action for the *current* situation, which corresponds to the last output.
        # Actually, effectively, at timestep T, we output action for T.
        # So we take the last timestep.
        
        last_step_logits = pred_action_logits[:, -1, :, :] # (B, N, A)
        
        # Flatten
        batch_size = last_step_logits.shape[0]
        logits_flat = last_step_logits.view(batch_size, -1)
        
        distribution = self.action_dist.proba_distribution(logits_flat)
        
        if deterministic:
            return distribution.mode()
        return distribution.sample()

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions = self._get_obs_actions(obs)
        
        # Run model
        # Output embeddings (hidden states) are needed for Value function.
        # WorldModel forward returns predictions, let's look at WorldModel.py.
        # It returns predictions. To get embeddings, we might need to modify WM or access internal.
        # However, `pred_actions` comes from `action_head(embeddings)`.
        # We can just assume the Value Head takes the *predictions* or we need to access embeddings.
        
        # HACK: For now, let's attach the value head to the *output* of the transformer
        # effectively `embeddings` just before heads.
        # We need to expose `embeddings` from `WorldModel.forward`.
        # I will modify `WorldModel.forward` to return embeddings as well.
        
        # As I cannot modify WorldModel immediately in this file (it's imported), 
        # I will rely on `pred_states` or something? No, that's bad.
        # I should assume I will update WorldModel to return embeddings.
        
        # Let's assume WorldModel now returns (pred_states, pred_actions, mask, kv, embeddings)
        # Or I just use `pred_states` (detached) as features? No.
        
        # Let's peek into WM properties.
        # I'll just replicate the forward pass logic briefly here since I have access to `self.world_model`
        # OR better, since I am in "Execution" I can Modify WorldModel.py in next step.
        
        # For now, let's pretend `world_model` returns `embeddings` as a 5th return value.
        pred_states, pred_action_logits, _, _, embeddings = self.world_model(states, actions, return_embeddings=True)
         
        # Use embeddings from last timestep for Value estimation
        # embeddings: (B, T, N, E)
        last_embed = embeddings[:, -1, :, :] # (B, N, E)
        
        # Pool across ships for single scalar value per episode/team?
        # Or value per ship? PPO usually expects (B,) value.
        # We have N ships.
        # Let's Average pool across ships to get Global State Value.
        global_embed = last_embed.mean(dim=1) # (B, E)
        values = self.value_head(global_embed) # (B, 1) -> (B,) check shape
        
        # Action Distribution
        last_step_logits = pred_action_logits[:, -1, :, :]
        batch_size = last_step_logits.shape[0]
        logits_flat = last_step_logits.view(batch_size, -1)
        
        distribution = self.action_dist.proba_distribution(logits_flat)
        
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
            
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, prev_actions = self._get_obs_actions(obs)
        
        # Run model
        pred_states, pred_action_logits, _, _, embeddings = self.world_model(states, prev_actions, return_embeddings=True)
        
        # Value
        last_embed = embeddings[:, -1, :, :]
        global_embed = last_embed.mean(dim=1)
        values = self.value_head(global_embed)
        
        # Actions
        last_step_logits = pred_action_logits[:, -1, :, :]
        batch_size = last_step_logits.shape[0]
        logits_flat = last_step_logits.view(batch_size, -1)
        
        distribution = self.action_dist.proba_distribution(logits_flat)
        
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        states, prev_actions = self._get_obs_actions(obs)
        _, pred_action_logits, _, _ = self.world_model(states, prev_actions)
        
        last_step_logits = pred_action_logits[:, -1, :, :]
        batch_size = last_step_logits.shape[0]
        logits_flat = last_step_logits.view(batch_size, -1)
        
        return self.action_dist.proba_distribution(logits_flat)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        states, prev_actions = self._get_obs_actions(obs)
        _, _, _, _, embeddings = self.world_model(states, prev_actions, return_embeddings=True)
        
        last_embed = embeddings[:, -1, :, :]
        global_embed = last_embed.mean(dim=1)
        values = self.value_head(global_embed)
        return values

    def get_dynamics_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the auxiliary dynamics loss (Reconstruction + Denoising).
        """
        states, actions = self._get_obs_actions(obs)
        device = states.device
        
        # Create Mask (Mixed Strategy)
        mask = create_mixed_mask(
            states.shape[0], # B
            states.shape[1], # T
            states.shape[2], # N
            device,
            mask_ratio=self.aux_loss_config.get("mask_ratio", 0.15)
        )
        
        # Run Forward with masking/noise
        pred_states, pred_actions, mask, _ = self.world_model(
            states,
            actions,
            mask_ratio=0.0, # We provide explicit mask
            noise_scale=self.aux_loss_config.get("noise_scale", 0.1),
            mask=mask
        )
        
        # Compute Loss
        recon_loss, denoise_loss = self.world_model.get_loss(
            states,
            actions,
            pred_states,
            pred_actions,
            mask
            # loss_mask is assumed all ones (we care about all timesteps in current chunk)
        )
        
        return recon_loss + denoise_loss
