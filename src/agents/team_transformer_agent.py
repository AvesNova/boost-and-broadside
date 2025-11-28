"""
Team transformer agent for multi-ship control.

Implements a transformer-based model that outputs actions for all ships,
with team-based filtering for multi-agent scenarios.
"""
import torch
import torch.nn as nn

from src.env.constants import Actions
from src.agents.tokenizer import observation_to_tokens


class TeamTransformerModel(nn.Module):
    """
    Transformer model that outputs actions for all ships in the environment.
    Teams/agents can then select which ships they control.

    Architecture:
    1. Project ship tokens to transformer dimension
    2. Apply transformer encoder layers with self-attention
    3. Project back to action logits for each ship
    4. Teams select their controlled ships during action execution
    """

    def __init__(
        self,
        token_dim: int = 10,  # Ship token dimension from get_token()
        embed_dim: int = 64,  # Transformer embedding dimension
        num_heads: int = 4,  # Number of attention heads
        num_layers: int = 3,  # Number of transformer layers
        max_ships: int = 8,  # Maximum ships in environment
        dropout: float = 0.1,  # Dropout rate
        use_layer_norm: bool = True,  # Whether to use layer normalization
    ):
        super().__init__()

        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.max_ships = max_ships
        self.num_actions = len(Actions)

        # Input projection: token_dim -> embed_dim
        self.token_projection = nn.Linear(token_dim, embed_dim)

        # Optional layer norm on input
        self.input_layer_norm = (
            nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # Standard 4x expansion in FFN
            dropout=dropout,
            activation="relu",
            batch_first=True,  # (batch, seq, feature) format
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # Disable for compatibility
        )

        # Output projection: embed_dim -> num_actions
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, self.num_actions),
        )

        # Value head: embed_dim -> 1
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with reasonable defaults"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, observation: dict, ship_mask: torch.Tensor | None = None) -> dict:
        """
        Forward pass through the transformer.

        Args:
            observation: Dict containing 'tokens' key with shape (batch, max_ships, token_dim)
            ship_mask: Optional mask for inactive ships (batch, max_ships).
                      True = masked (inactive), False = active

        Returns:
            Dict containing:
                - 'action_logits': (batch, max_ships, num_actions) - logits for each ship
                - 'value': (batch, 1) - value estimate for the state
                - 'ship_embeddings': (batch, max_ships, embed_dim) - final embeddings
                - 'attention_weights': Optional attention weights for analysis
        """
        tokens = observation["tokens"]  # (batch, max_ships, token_dim)
        batch_size, num_ships, token_dim = tokens.shape

        # Project tokens to embedding space
        embeddings = self.token_projection(tokens)  # (batch, max_ships, embed_dim)
        embeddings = self.input_layer_norm(embeddings)

        # Create attention mask if ship_mask provided
        # TransformerEncoder expects (batch, seq_len) where True = ignore
        attention_mask = ship_mask if ship_mask is not None else None

        # Apply transformer
        if attention_mask is not None:
            # Ensure mask is boolean and on correct device
            attention_mask = attention_mask.bool().to(embeddings.device)
            transformed = self.transformer(
                embeddings, src_key_padding_mask=attention_mask
            )
        else:
            transformed = self.transformer(embeddings)

        # Generate action logits for each ship
        action_logits = self.action_head(transformed)  # (batch, max_ships, num_actions)

        # Generate value estimate
        # Pool embeddings: mean over ships (handling mask if present)
        if attention_mask is not None:
            # mask is True for inactive, so invert for active
            active_mask = (~attention_mask).float().unsqueeze(-1)  # (batch, max_ships, 1)
            sum_embeddings = (transformed * active_mask).sum(dim=1)
            count = active_mask.sum(dim=1).clamp(min=1e-9)
            pooled_embeddings = sum_embeddings / count
        else:
            pooled_embeddings = transformed.mean(dim=1)

        value = self.value_head(pooled_embeddings)  # (batch, 1)

        return {
            "action_logits": action_logits,
            "value": value,
            "ship_embeddings": transformed,
            "tokens": tokens,  # Pass through for convenience
        }


class TeamTransformerAgent:
    """
    Agent wrapper for TeamTransformerModel.
    """

    def __init__(
        self,
        agent_id: str,
        team_id: int,
        squad: list[int],
        token_dim: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        max_ships: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        model_path: str | None = None,
    ):
        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad

        self.model = TeamTransformerModel(
            token_dim=token_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_ships=max_ships,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        """Load model weights from file"""
        try:
            state_dict = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")


    def __call__(self, observation: dict, ship_ids: list[int]) -> dict[int, torch.Tensor]:
        """
        Get actions for the specified ships.

        Args:
            observation: Observation dictionary
            ship_ids: List of ship IDs to get actions for

        Returns:
            Dict mapping ship_id -> action tensor
        """
        # Prepare input
        # Add batch dimension if needed
        if "tokens" in observation:
            tokens = observation["tokens"]
            if len(tokens.shape) == 2:
                tokens = tokens.unsqueeze(0)
                observation["tokens"] = tokens
        else:
            # Generate tokens from raw observation
            tokens = observation_to_tokens(observation, self.team_id)
            # Create a copy of observation to avoid side effects if possible, 
            # but for efficiency we might just pass a new dict
            observation = observation.copy()
            observation["tokens"] = tokens

        # Get actions from model
        output = self.get_actions(observation, deterministic=True)
        all_actions = output["actions"]  # (batch, max_ships, num_actions)

        # Extract actions for my ships
        team_actions = {}
        for ship_id in ship_ids:
            if ship_id < all_actions.shape[1]:
                # Remove batch dim
                team_actions[ship_id] = all_actions[0, ship_id]

        return team_actions

    def get_actions(
        self,
        observation: dict,
        ship_mask: torch.Tensor = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> dict:
        """
        Get actions for all ships with optional temperature scaling.
        """
        with torch.no_grad():
            output = self.model(observation, ship_mask)
            action_logits = output["action_logits"]

            # Apply temperature scaling
            scaled_logits = action_logits / temperature

            # Convert to probabilities
            action_probs = torch.sigmoid(scaled_logits)

            if deterministic:
                # Use thresholding for deterministic actions
                actions = (action_probs > 0.5).float()
            else:
                # Sample from Bernoulli distribution
                actions = torch.bernoulli(action_probs)

            return {
                "actions": actions,
                "action_probs": action_probs,
                "action_logits": action_logits,
                "ship_embeddings": output["ship_embeddings"],
            }

    def get_agent_type(self) -> str:
        """Return agent type for logging/debugging"""
        return "team_transformer"


class TeamController:
    """
    Helper class to manage which ships each team/agent controls.
    """

    def __init__(self, team_assignments: dict[int, list[int]]):
        """
        Args:
            team_assignments: Dict mapping team_id -> list of ship_ids
                             e.g., {0: [0, 2], 1: [1, 3]} for 2 teams
        """
        self.team_assignments = team_assignments
        self.num_teams = len(team_assignments)

    def extract_team_actions(
        self, all_actions: torch.Tensor, team_id: int
    ) -> dict[int, torch.Tensor]:
        """
        Extract actions for ships controlled by a specific team.

        Args:
            all_actions: (batch, max_ships, num_actions) - actions for all ships
            team_id: Which team's actions to extract

        Returns:
            Dict mapping ship_id -> action tensor for this team's ships
        """
        team_actions = {}
        ship_ids = self.team_assignments.get(team_id, [])

        for ship_id in ship_ids:
            if ship_id < all_actions.shape[1]:  # Check bounds
                # Extract action for this ship (remove batch dim if batch_size=1)
                action = (
                    all_actions[0, ship_id]
                    if all_actions.shape[0] == 1
                    else all_actions[:, ship_id]
                )
                team_actions[ship_id] = action

        return team_actions

    def get_ship_mask(
        self, batch_size: int, max_ships: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create mask for active ships across all teams.

        Returns:
            Boolean tensor (batch, max_ships) where True = inactive ship
        """
        mask = torch.ones(batch_size, max_ships, dtype=torch.bool, device=device)

        # Mark active ships as False (not masked)
        all_active_ships = set()
        for ship_ids in self.team_assignments.values():
            all_active_ships.update(ship_ids)

        for ship_id in all_active_ships:
            if ship_id < max_ships:
                mask[:, ship_id] = False

        return mask
