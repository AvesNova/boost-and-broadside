"""
World Model Agent for gameplay.

Wraps the InterleavedWorldModel to act as an agent in the environment.
Maintains a history of observations and uses the model to predict the next action.
"""

import logging
import torch
from collections import deque

# Changed import to InterleavedWorldModel
# from boost_and_broadside.agents.mamba_bb import MambaBB # Removed
from boost_and_broadside.core.constants import (
    STATE_DIM, 
    TARGET_DIM, 
    TOTAL_ACTION_LOGITS
)
try:
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    InferenceParams = None

from boost_and_broadside.agents.tokenizer import observation_to_tokens

log = logging.getLogger(__name__)




class WorldModelAgent:
    """
    Agent that uses the Interleaved World Model to predict actions.

    It maintains a sliding window of history (tokens) and at each step:
    1. Appends the current observation token to history.
    2. Runs the World Model to predict the action for the current step.
    
    The InterleavedWorldModel takes:
    S0 -> A0
    S1 -> A1
    
    In inference:
    We have S_t. We want to predict A_t.
    We pass [S0, A0, S1, A1, ..., S_t] (and dummy A_t which is masked/ignored by model for prediction).
    Actually, the model predicts A_t from S_t.
    """

    def __init__(
        self,
        agent_id: str,
        team_id: int,
        squad: list[int],
        model_path: str | None = None,
        context_len: int = 128,
        device: str = "cpu",
        state_dim: int = STATE_DIM,
        # embed_dim, n_layers etc. are loaded from config or default
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_ships: int = 8,
        dropout: float = 0.1,
        world_size: tuple[float, float] = (1024.0, 1024.0),
        model: torch.nn.Module | None = None,
        **kwargs,
    ):
        """
        Initialize the WorldModelAgent.

        Args:
            agent_id: Unique identifier for the agent.
            team_id: Team ID (0 or 1).
            squad: List of ship IDs controlled by this agent.
            model_path: Path to a pretrained model checkpoint.
            context_len: Maximum context length. (Not directly used by InterleavedModel which uses config)
            device: Device to run the model on ("cpu" or "cuda").
            ...
        """
        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad
        self.device = torch.device(device)
        self.max_ships = max_ships
        self.world_size = world_size
        
        # Interleaved World Model Config Arguments
        # If loading from checkpoint, these might be overwritten or unused if we load state_dict
        # But we need to initialize the architecture first.
        
        # Check if kwargs has 'world_model' config dict (from hydra)
        # The calling code in agents.py flattens it, but let's be safe.
        

        if model is not None:
            self.model = model
            self.model.to(self.device)
        if model is not None:
            self.model = model
            self.model.to(self.device)
        else:
            # Try to instantiate model from config (which is passed via kwargs)
            try:
                import hydra
                from omegaconf import OmegaConf
                
                # Re-bundle explicit arguments into kwargs for model instantiation
                mamba_cfg_args = kwargs.copy()
                mamba_cfg_args.setdefault('d_model', embed_dim)
                mamba_cfg_args.setdefault('n_layers', n_layers)
                mamba_cfg_args.setdefault('n_heads', n_heads)
                mamba_cfg_args.setdefault('input_dim', state_dim)
                mamba_cfg_args.setdefault('action_dim', kwargs.get('action_dim', TOTAL_ACTION_LOGITS))
                mamba_cfg_args.setdefault('target_dim', kwargs.get('target_dim', TARGET_DIM))

                # We expect kwargs to contains model config fields including _target_
                # If _target_ is missing, we fallback to YemongFull as a robust default
                if "_target_" not in mamba_cfg_args:
                    log.warning("No _target_ found in agent config. Falling back to YemongFull.")
                    from boost_and_broadside.models.yemong.scaffolds import YemongFull
                    cfg = OmegaConf.create(mamba_cfg_args)
                    self.model = YemongFull(cfg).to(self.device)
                else:
                    # Instantiate via Hydra
                    log.info(f"Instantiating model: {mamba_cfg_args['_target_']}")
                    cfg = OmegaConf.create(mamba_cfg_args)
                    self.model = hydra.utils.instantiate(cfg, _recursive_=False).to(self.device)
                
                log.info(f"Successfully instantiated {type(self.model).__name__}")
                
            except Exception as e:
                log.error(f"Failed to instantiate model from config: {e}")
                raise ValueError(f"WorldModelAgent requires a 'model' instance or valid config. Error: {e}")

        if model_path:
            self.load_model(model_path)
        elif model is None:
            log.warning(
                "WorldModelAgent initialized with random weights (no model or model_path provided)"
            )

        self.model.eval()

        # History buffer: list of (state_token, action_indices)
        # state_token: (1, num_ships, state_dim)
        # action_indices: (1, num_ships, 3) - Discrete Indices [Power, Turn, Shoot]
        # We need this to reconstruct the sequence.
        
        # Max length: We keep enough context.
        # Interleaved model can handle long context. 
        self.history = deque(maxlen=context_len) 
        
        # MambaBB State
        self.inference_params = None
        self.actor_cache = None 

    def load_model(self, path: str) -> None:
        """
        Load model weights from file.
        Handles both raw state_dict and full checkpoint dictionaries.
        """
        try:
            state_dict = torch.load(path, map_location=self.device)
            
            # unwrapping checkpoint if necessary
            # unwrapping checkpoint if necessary
            if "model_state_dict" in state_dict:
                log.info(f"Loading model from checkpoint dictionary: {path}")
                state_dict = state_dict["model_state_dict"]
            else:
                log.info(f"Loading model from raw state dict: {path}")

            # Fix for compiled models (strip _orig_mod prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

            # Check for input_dim mismatch
            # state_encoder.0.weight shape is (d_model, input_dim)
            if "state_encoder.0.weight" in state_dict:
                ckpt_input_dim = state_dict["state_encoder.0.weight"].shape[1]
                current_input_dim = self.model.config.input_dim
                
                if ckpt_input_dim != current_input_dim:
                    log.warning(f"Checkpoint input_dim ({ckpt_input_dim}) != Current ({current_input_dim}). Re-instantiating MambaBB.")
                    
                    # Re-instantiate with correct input_dim
                    # Re-create model using hydra.utils.instantiate if possible, or fallback
                    import hydra
                    from omegaconf import OmegaConf
                    
                    new_config = self.model.config.copy()
                    new_config.input_dim = ckpt_input_dim
                    
                    # Also check target_dim (world_head.3.weight is target_dim, d_model)
                    if "world_head.3.weight" in state_dict:
                        ckpt_target_dim = state_dict["world_head.3.weight"].shape[0]
                        new_config.target_dim = ckpt_target_dim 
                        
                    # Check for log_vars to set loss_type
                    has_log_vars = any(k.startswith("log_vars.") for k in state_dict.keys())
                    if has_log_vars:
                         new_config.loss_type = "uncertainty"
                         
                    # Re-instantiate
                    self.model = hydra.utils.instantiate(new_config, _recursive_=False).to(self.device)
            
            # Load with strict=False to handle potential minor mismatches (like missing buffers), 
            # but usually we want strict=True to catch real errors. 
            # However, since we might have unexpected keys (like "log_vars" if we didn't set loss_type correctly above),
            # let's try to handle it. 
            # If we detected log_vars above and set loss_type="uncertainty", the model should have log_vars parameters now.
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            
            if missing:
                log.warning(f"Missing keys in state_dict: {missing}")
            if unexpected:
                log.warning(f"Unexpected keys in state_dict: {unexpected}")
                
            log.info(f"Successfully loaded MambaBB from {path}")
        except Exception as e:
            log.error(f"Failed to load WorldModel from {path}: {e}")
            raise e

    def reset(self) -> None:
        """Reset agent state (history)."""
        self.history.clear()
        self.inference_params = None
        self.actor_cache = None

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
        current_state = observation_to_tokens(
            observation, self.team_id, self.world_size
        ).to(self.device)

        # 1. Prepare Inputs
        # Pos: (1, 1, N, 2)
        # We need to extract raw pos from observation (which is dict of tensors)
        # observation["position"] is complex (N,)
        pos_c = observation["position"]
        vel_c = observation["velocity"]
        
        # Stack to (1, 1, N, 2)
        def complex_to_tensor(c):
            return torch.stack([c.real, c.imag], dim=-1).float().to(self.device).view(1, 1, self.max_ships, 2)
            
        pos = complex_to_tensor(pos_c)
        vel = complex_to_tensor(vel_c)
        
        # Prev Action
        if len(self.history) > 0:
            _, last_action = self.history[-1]
            prev_action = last_action.view(1, 1, self.max_ships, 3)
        else:
            prev_action = torch.zeros(1, 1, self.max_ships, 3, device=self.device)
            
        # Inference Params
        if self.inference_params is None and InferenceParams is not None:
            # Initialize for max_batch_size=1, max_seqlen big enough
            self.inference_params = InferenceParams(max_seqlen=100000, max_batch_size=1)
            
        with torch.no_grad():
             # Preparation of inputs
             state_in = current_state.unsqueeze(1) # (1, T=1, N, D)
             
             # Prepare generic kwargs for model forward
             # All Yemong models handle **kwargs, so we can pass more than needed safely
             forward_kwargs = {
                 "state": state_in,
                 "prev_action": prev_action,
                 "pos": pos,
                 "vel": vel,
                 "att": None,
                 "team_ids": torch.tensor([self.team_id], device=self.device).view(1, 1, 1).expand(1, 1, self.max_ships),
                 "inference_params": self.inference_params,
                 "actor_cache": self.actor_cache,
                 "world_size": self.world_size
             }
             
             # Call model
             outputs = self.model(**forward_kwargs)
             
             # All Yemong scaffolds return (state_pred, action_logits, value_pred, reward_pred, x_final/cache)
             # but some might return None for some indices
             pred_s, pred_a_logits, _, _, new_cache = outputs
             
        # 3. Update Cache
        self.actor_cache = new_cache
        
        # 4. Decode
        last_step_logits = pred_a_logits[:, -1, :, :] # (1, N, 12)
        p_idx = last_step_logits[..., 0:3].argmax(dim=-1)
        t_idx = last_step_logits[..., 3:10].argmax(dim=-1)
        s_idx = last_step_logits[..., 10:12].argmax(dim=-1)
        
        next_action_indices = torch.stack([p_idx, t_idx, s_idx], dim=-1).float() # (1, N, 3)
        
        # Update History (for prev_action next step)
        self.history.append((current_state, next_action_indices))
        
        # Return
        team_actions = {}
        for ship_id in ship_ids:
            if ship_id < self.max_ships:
                team_actions[ship_id] = torch.tensor(
                    [
                        p_idx[0, ship_id].item(),
                        t_idx[0, ship_id].item(),
                        s_idx[0, ship_id].item()
                    ],
                    dtype=torch.float32,
                    device=self.device
                )
        return team_actions

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "world_model"

