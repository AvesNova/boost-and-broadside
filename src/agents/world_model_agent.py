"""
World Model Agent for gameplay.

Wraps the InterleavedWorldModel to act as an agent in the environment.
Maintains a history of observations and uses the model to predict the next action.
"""

import logging
import torch
from collections import deque

# Changed import to InterleavedWorldModel
from agents.interleaved_world_model import InterleavedWorldModel
from agents.tokenizer import observation_to_tokens

log = logging.getLogger(__name__)


import torch.nn.functional as F


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
        state_dim: int = 12,
        # embed_dim, n_layers etc. are loaded from config or default
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_ships: int = 8,
        dropout: float = 0.1,
        world_size: tuple[float, float] = (1024.0, 1024.0),
        model: InterleavedWorldModel | None = None,
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
        else:
            self.model = InterleavedWorldModel(
                state_dim=state_dim,
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

        # History buffer: list of (state_token, action_indices)
        # state_token: (1, num_ships, state_dim)
        # action_indices: (1, num_ships, 3) - Discrete Indices [Power, Turn, Shoot]
        # We need this to reconstruct the sequence.
        
        # Max length: We keep enough context.
        # Interleaved model can handle long context. 
        self.history = deque(maxlen=context_len) 

    def load_model(self, path: str) -> None:
        """
        Load model weights from file.
        Handles both raw state_dict and full checkpoint dictionaries.
        """
        try:
            state_dict = torch.load(path, map_location=self.device)
            
            # unwrapping checkpoint if necessary
            if "model_state_dict" in state_dict:
                log.info(f"Loading model from checkpoint dictionary: {path}")
                state_dict = state_dict["model_state_dict"]
            else:
                log.info(f"Loading model from raw state dict: {path}")

            # Handle SWA vs regular model keys if needed (usually keys match)
            # But sometimes SWA models might be saved differently. Assuming standard keys.
            
            self.model.load_state_dict(state_dict)
            log.info(f"Successfully loaded InterleavedWorldModel from {path}")
        except Exception as e:
            log.error(f"Failed to load InterleavedWorldModel from {path}: {e}")

    def reset(self) -> None:
        """Reset agent state (history)."""
        self.history.clear()

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
        
        if current_state.ndim == 2:
            current_state = current_state.unsqueeze(0)

        # 2. Prepare inputs
        # We need to construct the full sequence of [S, A] pair from history
        # plus the current S.
        
        hist_states = [s for s, a in self.history]
        hist_actions = [a for s, a in self.history]
        
        # Current input sequence for the model
        # States: History + Current
        states_list = hist_states + [current_state]
        
        # Actions: History + Dummy for current step (to be predicted)
        # The model needs 'actions' input aligned with states.
        # For S_t, we provide A_{t-1} usually?
        # WAIT. Let's look at InterleavedWorldModel forward.
        # It takes `states` and `actions`.
        # It computes `state_tokens` from `states`.
        # It computes `action_tokens` from `actions`.
        # It interleaves them: (S0, A0), (S1, A1)...
        
        # To predict A_t (at index 2t+1 in sequence), we generally need S_t.
        # But the model architecture is: 
        # S_t (Index 2t) -> Predicted A_t.
        # A_t (Index 2t+1) -> Predicted S_{t+1}.
        
        # So we pass S_t. The A_t input is technically "ground truth" for training.
        # For inference, if we want to predict A_t from S_t:
        # We input S_t.
        # The A_t input at that position is technically future information we don't have.
        # However, the `InterleavedWorldModel` processes pairs.
        # (S_t, A_t) -> flattened to Sequence of 2*T length.
        
        # But wait, `InterleavedWorldModel.forward` calculates embeddings for A_t using `actions` input.
        # If we don't know A_t, we can't embed it correct?
        # Actually, for the *prediction* of A_t, we use the embedding of S_t (and previous history).
        # S_t is at even index. Action Head is applied to "S tokens (index 0) -> Action Head (Predict A_t)".
        # So the embedding of A_t is NOT used to predict A_t. It is used to predict S_{t+1}.
        
        # Therefore, to predict A_t, we can pass a dummy A_t. Its value doesn't matter for the S_t->A_t prediction path
        # (assuming causal masking prevents S_t from seeing A_t embedding, or A_t embedding is not added to S_t).
        # In `InterleavedWorldModel`, S_t and A_t are stacked `torch.stack([state_tokens, action_tokens], dim=2)`.
        # S_t uses `state_tokens`. A_t uses `action_tokens` (derived from `actions` input).
        # The attention is Factorized.
        # Layer 0 (Temporal): S_t attends to self history.
        # Layer 1 (Spatial): S_t attends to {Previous Block, Current Block}.
        # Current Block is {S_t, A_t}.
        # Wait, if S_t attends to A_t, then we can't use dummy A_t if we want causality?
        # The README says:
        # "Spatial Attention... A_t attends to {S_t, A_t}. S_t attends to {A_{t-1}, S_t}."
        # This means S_t DOES NOT attend to A_t.
        # So providing dummy A_t is safe for predicting A_t from S_t.
        
        # So:
        # 1. We construct `actions` tensor with dummy for current step.
        dummy_action = torch.zeros(1, self.max_ships, 3, device=self.device)
        actions_list = hist_actions + [dummy_action]
        
        input_states = torch.cat(states_list, dim=0).unsqueeze(0) # (1, T, N, D)
        input_actions = torch.cat(actions_list, dim=0).unsqueeze(0) # (1, T, N, 3)
        
        # 3. Team IDs
        # We need to broadcast team_id
        # (1, N)
        team_ids = torch.full((1, self.max_ships), self.team_id, device=self.device, dtype=torch.long)
        
        # 4. Forward
        with torch.no_grad():
            # input_states: (1, T, N, D)
            pred_s, pred_a_logits, _ = self.model(
                input_states,
                input_actions,
                team_ids,
                noise_scale=0.0
            )
            
        # 5. Extract Prediction
        # We want the LAST step (T-1) of pred_a_logits.
        # pred_a_logits shape: (1, T, N, 12).
        last_step_logits = pred_a_logits[:, -1, :, :] # (1, N, 12)
        
        # Decode logits
        p_logits = last_step_logits[..., 0:3]
        t_logits = last_step_logits[..., 3:10]
        s_logits = last_step_logits[..., 10:12]
        
        p_idx = p_logits.argmax(dim=-1)
        t_idx = t_logits.argmax(dim=-1)
        s_idx = s_logits.argmax(dim=-1)
        
        # (1, N) indices
        
        # 6. Update History
        # We store the PREDICTED action (as indices) effectively making it the "action taken".
        # In a real loop, we should probably store what we *decided* to do.
        # Construct (1, N, 3) tensor of indices
        # We need float for consistency with 'actions' tensor type if it expects floats?
        # InterleavedWorldModel uses .long() on it, so float input is fine as long as values are integers.
        next_action_indices = torch.stack([p_idx, t_idx, s_idx], dim=-1).float() # (1, N, 3)
        
        self.history.append((current_state, next_action_indices))
        
        # 7. Return Team Actions
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

