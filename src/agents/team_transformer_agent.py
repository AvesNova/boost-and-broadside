import torch
import torch.nn as nn
import torch.nn.functional as F

from env.constants import Actions


class TeamTransformerAgent(nn.Module):
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
        agent_id: str,
        team_id: int,
        squad: list[int],
        token_dim: int = 10,  # Ship token dimension from get_token()
        embed_dim: int = 64,  # Transformer embedding dimension
        num_heads: int = 4,  # Number of attention heads
        num_layers: int = 3,  # Number of transformer layers
        max_ships: int = 8,  # Maximum ships in environment
        dropout: float = 0.1,  # Dropout rate
        use_layer_norm: bool = True,  # Whether to use layer normalization
    ):
        super().__init__(agent_id, team_id, squad)

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

        return {
            "action_logits": action_logits,
            "ship_embeddings": transformed,
            "tokens": tokens,  # Pass through for convenience
        }

    def get_actions(
        self,
        observation: dict,
        ship_mask: torch.Tensor = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> dict:
        """
        Get actions for all ships with optional temperature scaling.

        Args:
            observation: Observation dict
            ship_mask: Mask for inactive ships
            temperature: Temperature for softmax (higher = more random)
            deterministic: If True, use argmax instead of sampling

        Returns:
            Dict containing:
                - 'actions': (batch, max_ships, num_actions) - binary action vectors
                - 'action_probs': (batch, max_ships, num_actions) - action probabilities
                - 'action_logits': (batch, max_ships, num_actions) - raw logits
        """
        with torch.no_grad():
            output = self.forward(observation, ship_mask)
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
