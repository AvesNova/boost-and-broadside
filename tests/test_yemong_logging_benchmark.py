
import torch
import time
import pytest
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongDynamics
from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM, TOTAL_ACTION_LOGITS

def test_logging_benchmark():
    # Setup Config
    cfg = OmegaConf.create({
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "action_dim": 12,
        "action_embed_dim": 16,
        "loss_type": "fixed",
        "spatial_layer": {
             "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
             "d_model": 128,
             "n_heads": 4
        },
        "loss": {
             "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
             "losses": [
                  {"_target_": "boost_and_broadside.models.components.losses.StateLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.ActionLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.ValueLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.RewardLoss", "weight": 1.0}
             ]
        }
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YemongDynamics(cfg).to(device)
    model.eval() # We just test get_loss overhead, usually called in train but logic is same
    
    # Dummy Data
    B, T, N = 16, 32, 4
    D = STATE_DIM
    
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
    
    # Optional args
    pred_values = torch.randn(B, T, 1, device=device)
    pred_rewards = torch.randn(B, T, 1, device=device)
    target_returns = torch.randn(B, T, 1, device=device)
    target_rewards = torch.randn(B, T, 1, device=device)
    
    weights_p = torch.ones(3, device=device)
    weights_t = torch.ones(7, device=device)
    weights_s = torch.ones(2, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model.get_loss(
            pred_states, pred_actions, target_states, target_actions, loss_mask,
            pred_values=pred_values, pred_rewards=pred_rewards,
            target_returns=target_returns, target_rewards=target_rewards,
            weights_power=weights_p, weights_turn=weights_t, weights_shoot=weights_s
        )
        
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark
    iters = 100
    t0 = time.time()
    for _ in range(iters):
        out = model.get_loss(
            pred_states, pred_actions, target_states, target_actions, loss_mask,
            pred_values=pred_values, pred_rewards=pred_rewards,
            target_returns=target_returns, target_rewards=target_rewards,
            weights_power=weights_p, weights_turn=weights_t, weights_shoot=weights_s
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    
    avg_time = (t1 - t0) / iters * 1000 # ms
    print(f"\nAverage get_loss time: {avg_time:.4f} ms")
    
    # Verification of Keys
    print("Keys in return dict:", out.keys())
    
    return avg_time

if __name__ == "__main__":
    test_logging_benchmark()
