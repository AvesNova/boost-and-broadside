
import sys
import os
import torch
import pytest
from omegaconf import OmegaConf

# Ensure project root is in sys.path

from boost_and_broadside.agents.mamba_bb import MambaBB

# Mock Config
def get_config():
    return OmegaConf.create({
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": 128, # Dummy, will be adjusted by code if needed
        "action_dim": 12,
        "target_dim": 128,
        "loss_type": "fixed",
    })

def test_memory_check_failure():
    """
    Test A: The 'Memory Check'.
    Run the model in inference mode for 2 steps with IDENTICAL inputs.
    If the model has memory, the output at t=2 must be different from t=1.
    
    If they are identical, it proves the model is resetting every step (The Bug).
    """
    config = get_config()
    model = MambaBB(config)
    model.eval()
    
    batch_size = 1
    seq_len = 1
    num_ships = 2
    d_model = config.d_model
    
    # Inputs
    state = torch.randn(batch_size, seq_len, num_ships, config.input_dim)
    prev_action = torch.zeros(batch_size, seq_len, num_ships, 3) # Power, Turn, Shoot
    pos = torch.randn(batch_size, seq_len, num_ships, 2) * 500 # World scale
    vel = torch.randn(batch_size, seq_len, num_ships, 2)
    alive = torch.ones(batch_size, seq_len, num_ships, dtype=torch.bool)
    
    # Step 1
    with torch.no_grad():
        out1 = model(state, prev_action, pos, vel, alive=alive)
        pred_actions_1 = out1[1] # Action Logits
        
    # Step 2 (Same Inputs)
    with torch.no_grad():
        out2 = model(state, prev_action, pos, vel, alive=alive)
        pred_actions_2 = out2[1]
        
    # Check difference
    diff = (pred_actions_1 - pred_actions_2).abs().max().item()
    print(f"Difference between Step 1 and Step 2 outputs: {diff}")
    
    # The bug expectation: diff == 0.0 (No memory)
    # The fix expectation: diff > 0.0 (Memory works)
    
    # For now, we assert that they ARE equal to confirm the bug exists.
    # Once fixed, this test should fail (or be inverted).
    if diff == 0.0:
        print("CONFIRMED: Model has NO MEMORY (Output identical for subsequent steps).")
    else:
        print("Model HAS MEMORY (Outputs diverged).")
        
    # We want to FAIL if the bug is present, so we can verify the fix later.
    # But for 'reproduce', we want to demonstrate the failure.
    # So I will assert diff > 0. If it raises, the bug is present.
    assert diff > 1e-6, "FAILURE: Model outputs are identical across steps! Memory is resetting."

def test_coordinate_scale_bug():
    """
    Test B: The 'Unit Scale' Probe.
    Check if normalized inputs result in tiny distances due to hardcoded World Size.
    """
    config = get_config()
    model = MambaBB(config)
    
    # Normalized positions (0.0 to 1.0) simulating inference input
    batch_size = 1
    seq_len = 1
    num_ships = 2
    
    pos_norm = torch.rand(batch_size, seq_len, num_ships, 2) # [0, 1]
    vel = torch.randn(batch_size, seq_len, num_ships, 2)
    
    # Check features internally
    # We can inspect the RelationalEncoder directly
    encoder = model.relational_encoder
    
    # Run compute_analytic_features
    # We expect the 'dist' feature to be tiny because it divides by 1024 implicitly or wraps around
    # Actually, the logic is: dx = dx - round(dx/W)*W. If dx is 0.5 and W=1024, round is 0.
    # So dx remains 0.5.
    # The analytic features (dist) will be ~0.5.
    # BUT, the model might expect distances in range [0, 1024]. 
    # If the model was trained on [0, 1024], seeing 0.5 is basically "on top of each other".
    
    # Let's see what the features look like
    feats = encoder.compute_analytic_features(pos_norm, vel)
    
    # dist is index 4 (after 2 pos, 2 vel) in base_features
    # In the code:
    # d_pos, d_vel, dist, ...
    # d_pos (2), d_vel (2), dist (1) -> Index 4
    
    # Wait, 
    # base_features = cat([d_pos(2), d_vel(2), dist(1), ...])
    # So dist is at index 4.
    
    dist_feat = feats[..., 4]
    avg_dist = dist_feat.mean().item()
    
    print(f"Average Distance Feature with Normalized Input: {avg_dist}")
    
    # If standard world is 1000x1000, avg distance between random points is ~520.
    # Here it is ~0.5.
    # This 1000x discrepancy is the bug.
    
    # We assert that the distance is suspiciously small (< 2.0)
    if avg_dist < 2.0:
         print("CONFIRMED: Distance features are microscopic! Coordinate mismatch detected.")
    
    # To pass as a "reproduction of bug", we assert that it IS small.
    # Later we will fix it and update the test to expect correct scaling or handle it.
    assert avg_dist < 2.0, "Distances are large? Maybe normalization is not applied?"

if __name__ == "__main__":
    # check if mamba_ssm is installed
    try:
        import mamba_ssm
        print("mamba_ssm is installed.")
    except ImportError:
        print("mamba_ssm is NOT installed. Skipping Mamba state checks.")
        
    try:
        test_memory_check_failure()
    except AssertionError as e:
        print(e)
        
    try:
        test_coordinate_scale_bug()
    except AssertionError as e:
        print(e)
