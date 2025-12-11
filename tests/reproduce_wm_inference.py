import sys
import os

sys.path.append(os.getcwd())
import torch
from agents.world_model import WorldModel


def test_wm_inference():
    # Config
    state_dim = 12
    action_dim = 6
    embed_dim = 128
    n_layers = 2
    n_heads = 4
    max_ships = 4
    max_context_len = 16

    # Initialize model
    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_ships=max_ships,
        max_context_len=max_context_len,
    )
    model.eval()

    # Simulate context: T=10 steps
    T = 10
    batch_size = 1

    # Random history
    states = torch.randn(batch_size, T, max_ships, state_dim)
    actions = torch.randn(batch_size, T, max_ships, action_dim)

    # Create input for inference: T+1 steps
    # The last step (T+1) is the "next step" we want to predict
    # We mask it completely

    # Append masked token placeholders (zeros or random, doesn't matter as they will be masked)
    next_state_placeholder = torch.zeros(batch_size, 1, max_ships, state_dim)
    next_action_placeholder = torch.zeros(batch_size, 1, max_ships, action_dim)

    input_states = torch.cat([states, next_state_placeholder], dim=1)
    input_actions = torch.cat([actions, next_action_placeholder], dim=1)

    # Create mask: False for history, True for next step
    # Mask shape: (batch, T+1, max_ships)
    mask = torch.zeros(batch_size, T + 1, max_ships, dtype=torch.bool)
    mask[:, -1, :] = True  # Mask ALL ships at the last timestep

    print(f"Input states shape: {input_states.shape}")
    print(f"Input actions shape: {input_actions.shape}")
    print(f"Mask shape: {mask.shape}")

    # Forward pass
    # Note: We need to manually handle masking because model.forward generates random masks
    # if mask_ratio > 0. But we want a SPECIFIC mask.
    # The current WorldModel.forward DOES NOT accept a custom mask argument for the MAE masking.
    # It only accepts mask_ratio.
    # This is a finding! We need to modify WorldModel.forward to accept an optional 'mask' argument.

    try:
        # Attempt to pass mask
        pred_states, pred_actions, _, _ = model(
            input_states, input_actions, mask_ratio=0.0, mask=mask
        )
        print("Forward pass successful with external mask!")
        print(f"Pred states shape: {pred_states.shape}")
        print(f"Pred actions shape: {pred_actions.shape}")

        # Verify output shapes
        assert pred_states.shape == input_states.shape
        assert pred_actions.shape == input_actions.shape

    except TypeError as e:
        print(f"FAILED: {e}")
        exit(1)


if __name__ == "__main__":
    test_wm_inference()
