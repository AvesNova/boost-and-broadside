
import torch
import pytest
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongDynamics
from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM, TOTAL_ACTION_LOGITS

def test_logging_correctness():
    # Setup Config
    cfg = OmegaConf.create({
        "d_model": 64,
        "n_layers": 1,
        "n_heads": 2,
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "action_dim": TOTAL_ACTION_LOGITS,
        "action_embed_dim": 8,
        "max_ships": 4,
        "use_pairwise_targets": False,
        "loss_type": "fixed",
        "num_reward_components": 3 # Test specific number
    })
    
    device = torch.device("cpu") # Run on CPU for speed/reliability in test
    model = YemongDynamics(cfg).to(device)
    model.eval()
    
    # Dummy Data
    B, T, N = 2, 8, 4
    
    pred_states = torch.randn(B, T, N, TARGET_DIM, device=device)
    pred_actions = torch.randn(B, T, N, TOTAL_ACTION_LOGITS, device=device)
    target_states = torch.randn(B, T, N, TARGET_DIM, device=device)
    # Action targets: Power(0-2), Turn(0-6), Shoot(0-1)
    target_actions = torch.stack([
        torch.randint(0, 3, (B, T, N), device=device),
        torch.randint(0, 7, (B, T, N), device=device),
        torch.randint(0, 2, (B, T, N), device=device)
    ], dim=-1).float()
    
    loss_mask = torch.ones(B, T, N, device=device).bool()
    
    # Reward Inputs
    # IMPORTANT: We need reward_components from the forward pass to test that logging!
    # But get_loss takes them as argument. 
    # Let's run forward first.
    
    # Create inputs for forward
    states = torch.randn(B, T, N, STATE_DIM, device=device)
    prev_actions = torch.randint(0, 3, (B, T, N, 3), device=device).float() # embedding expects float/int? check code. 
    # ActionEncoder expects ... wait, SeparatedActionEncoder
    # It takes (..., 3) indices.
    
    # Need to check what SeparatedActionEncoder expects.
    # It likely expects Long indices.
    
    pos = torch.randn(B, T, N, 2, device=device)
    vel = torch.randn(B, T, N, 2, device=device)
    alive = torch.ones(B, T, N, device=device).bool()
    
    # Run Forward
    # Returns: action_logits, logprob, entropy, value_pred, state, next_state_pred, reward_pred, pairwise_pred, reward_components
    out = model(states, prev_actions, pos, vel, alive=alive)
    reward_components = out[8] 
    
    print("Reward Components Shape:", reward_components.shape)
    assert reward_components.shape == (B, T, 3)
    
    # Run Get Loss
    pred_values = torch.randn(B, T, 1, device=device)
    pred_rewards = torch.randn(B, T, 1, device=device)
    target_returns = torch.randn(B, T, 1, device=device)
    target_rewards = torch.randn(B, T, 1, device=device)
    
    weights_p = torch.ones(3, device=device)
    weights_t = torch.ones(7, device=device)
    weights_s = torch.ones(2, device=device)
    
    metrics = model.get_loss(
        pred_states, pred_actions, target_states, target_actions, loss_mask,
        pred_values=pred_values, pred_rewards=pred_rewards,
        target_returns=target_returns, target_rewards=target_rewards,
        weights_power=weights_p, weights_turn=weights_t, weights_shoot=weights_s,
        reward_components=reward_components # Pass the components!
    )
    
    print("\nMetrics Keys:", metrics.keys())
    
    # Assertions
    expected_keys = [
        "loss_sub/state_DX", "loss_sub/state_DY", "loss_sub/state_DVX",
        "loss_sub/action_power", "loss_sub/action_turn", "loss_sub/action_shoot",
        "val/reward_component_0", "val/reward_component_1", "val/reward_component_2"
    ]
    
    for k in expected_keys:
        assert k in metrics, f"Missing key: {k}"
        print(f"Verified {k}: {metrics[k]}")

    print("\nTest Correctness: PASSED")

if __name__ == "__main__":
    test_logging_correctness()
