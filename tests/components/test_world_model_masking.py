import torch
from boost_and_broadside.agents.world_model import WorldModel


def test_world_model_masking_behavior():
    """
    Verify that masking changes the output of the World Model.
    """
    state_dim = 10
    action_dim = 12
    embed_dim = 32  # Small dim for test
    n_layers = 2
    n_heads = 4
    max_ships = 2
    context_len = 8

    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        n_layers=2,
        n_heads=4,
        max_ships=max_ships,
        max_context_len=context_len,
    )
    model.eval()  # CRITICAL: Disable dropout for deterministic check

    batch_size = 2
    time_steps = 4
    n_ships = max_ships

    states = torch.randn(batch_size, time_steps, n_ships, state_dim)
    actions = torch.randn(batch_size, time_steps, n_ships, action_dim)

    # 1. Forward WITHOUT masking
    pred_states_clean, _, _, mask_clean, _ = model(
        states, actions, mask_ratio=0.0, noise_scale=0.0
    )

    assert mask_clean is None

    # 2. Forward WITH masking
    mask_ratio = 0.5
    pred_states_masked, _, _, mask_applied, _ = model(
        states, actions, mask_ratio=mask_ratio, noise_scale=0.0
    )

    assert mask_applied is not None
    assert mask_applied.shape == (batch_size, time_steps, n_ships)
    assert mask_applied.dtype == torch.bool

    # Check that some tokens were masked
    assert mask_applied.any(), "With ratio 0.5, some tokens should be masked"

    # Check that outputs are different
    diff = (pred_states_clean - pred_states_masked).abs().max()
    assert diff > 1e-6, (
        "Masked forward pass should produce different outputs than clean pass"
    )


def test_world_model_mask_token_usage():
    """
    Verify that masked positions use the mask token embedding.
    We can inspect the returned embeddings if we use return_embeddings=True.
    """
    state_dim = 10
    action_dim = 12
    embed_dim = 32
    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        n_layers=1,  # 1 layer to minimize complexity
        n_heads=4,
    )
    model.eval()  # CRITICAL: Disable dropout

    batch_size = 1
    time_steps = 2
    n_ships = 1

    states = torch.randn(batch_size, time_steps, n_ships, state_dim)
    actions = torch.randn(batch_size, time_steps, n_ships, action_dim)

    # Force a mask where the first token is masked
    mask = torch.zeros(batch_size, time_steps, n_ships, dtype=torch.bool)
    mask[0, 0, 0] = True

    # Get embeddings
    _, _, _, _, _, embeddings = model(
        states, actions, mask=mask, return_embeddings=True
    )
    # Shape: (B, T, N, E)

    states_v2 = states.clone()
    states_v2[0, 0, 0] += 1.0  # Change content of masked token

    _, _, _, _, _, embeddings_v2 = model(
        states_v2, actions, mask=mask, return_embeddings=True
    )

    # The embedding at [0,0,0] should be IDENTICAL because the content was masked out
    diff_masked = (embeddings[0, 0, 0] - embeddings_v2[0, 0, 0]).abs().max()
    assert diff_masked < 1e-6, (
        f"Masked token embedding should be invariant to input content changes. Diff: {diff_masked}"
    )

    # Unmasked token should change
    states_v3 = states.clone()
    states_v3[0, 1, 0] += 1.0

    _, _, _, _, _, embeddings_v3 = model(
        states_v3, actions, mask=mask, return_embeddings=True
    )

    diff_unmasked = (embeddings[0, 1, 0] - embeddings_v3[0, 1, 0]).abs().max()
    assert diff_unmasked > 1e-6, (
        "Unmasked token embedding SHOULD change when input changes"
    )


def test_first_token_never_masked():
    """
    Verify that the first token (t=0) is never masked.
    """
    state_dim = 10
    action_dim = 12
    embed_dim = 32
    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        n_layers=1,
        n_heads=4,
    )
    model.eval()

    batch_size = 10
    time_steps = 4
    n_ships = 1

    states = torch.randn(batch_size, time_steps, n_ships, state_dim)
    actions = torch.randn(batch_size, time_steps, n_ships, action_dim)

    # Use a very high mask ratio
    mask_ratio = 0.99

    _, _, _, mask, _ = model(states, actions, mask_ratio=mask_ratio)

    # Check shape
    assert mask.shape == (batch_size, time_steps, n_ships)

    # Check t=0 is all False
    first_step_mask = mask[:, 0, :]
    assert not first_step_mask.any(), "First timestep tokens should NEVER be masked"

    # Check that other steps have some masking (probabilistic, but with 0.99 it's almost certain)
    other_steps_mask = mask[:, 1:, :]
    assert other_steps_mask.any(), "Other steps should be masked with high ratio"
