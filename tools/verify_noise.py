
import math
import torch
from omegaconf import OmegaConf
from agents.world_model import WorldModel
from train.data_loader import load_bc_data, create_unified_data_loaders

def verify_noise_injection():
    # Load Config (mimic config_test)
    cfg = OmegaConf.create({
        "world_model": {
            "short_batch_size": 16,
            "long_batch_size": 4,
            "short_batch_len": 32,
            "long_batch_len": 128,
            "batch_ratio": 4,
            "input_noise_ratio": 1.0, # FORCE NOISE for verification
            "input_noise_scale": 10.0, # LARGE NOISE to be obvious
        },
        "train": {
             "bc_data_path": "data/bc_pretraining/20260112_102804/aggregated_data.pkl"
        }
    })
    
    device = torch.device("cpu")
    
    print("Loading data...")
    data = load_bc_data(cfg.train.bc_data_path)
    
    train_short_loader, _, _, _ = create_unified_data_loaders(
        data,
        short_batch_size=cfg.world_model.short_batch_size,
        long_batch_size=cfg.world_model.long_batch_size,
        short_batch_len=cfg.world_model.short_batch_len,
        long_batch_len=cfg.world_model.long_batch_len,
        batch_ratio=cfg.world_model.batch_ratio,
        validation_split=0.2,
        num_workers=0,
    )
    
    # Get one batch
    loader_iter = iter(train_short_loader)
    states, actions, returns, loss_mask, action_masks = next(loader_iter)
    
    print("Original States Mean:", states.mean().item())
    
    # Simulate the training loop logic
    
    # 1. Prepare inputs/targets
    input_states_original = states[:, :-1].clone() # Keep copy to compare
    target_states = states[:, 1:]
    
    input_states = states[:, :-1]
    
    # 2. Inject Noise (Copy-paste logic from train_world_model.py)
    # Using 1.0 ratio and 10.0 scale as defined above
    if cfg.world_model.input_noise_ratio > 0:
         print(f"Injecting noise with ratio {cfg.world_model.input_noise_ratio} and scale {cfg.world_model.input_noise_scale}")
         noise_mask = (torch.rand(input_states.shape[:3], device=device) < cfg.world_model.input_noise_ratio)
         noise = torch.randn_like(input_states) * cfg.world_model.input_noise_scale
         input_states = torch.where(noise_mask.unsqueeze(-1), input_states + noise, input_states)

    # 3. Verify
    print("\n--- Verification Results ---")
    
    # Check Inputs Modified
    diff = (input_states - input_states_original).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    print(f"Input States Mean Diff (Should be large): {mean_diff:.4f}")
    if mean_diff > 1.0:
        print("PASS: Inputs are modified.")
    else:
        print("FAIL: Inputs are NOT modified significantly.")

    # Check Targets Unmodified
    # Target states come from 'states' tensor. 
    # In the real code, 'states' was sliced to get input_states and target_states.
    # The noise injection: input_states = torch.where(..., input_states + noise, input_states)
    # This creates a NEW tensor for input_states if we are careful, OR it modifies in place if we are not.
    # In my implementation: input_states = torch.where(...) returns a new tensor.
    # So 'states' tensor should remain untouched.
    # Let's verify 'target_states' (which is a slice of 'states') is clean.
    
    # Since we didn't keep a copy of 'states' before slicing, we can't compare directly unless we know...
    # Wait, input_states_original IS a copy of the slice.
    # But target_states is a slice of the original 'states'.
    # If 'states' was modified in place, target_states would be corrupted (for overlapping parts? No, targets are next step).
    # But input_states and target_states overlap in time (t0..tN-1 vs t1..tN).
    # If I modified input_states IN PLACE, then indices 1..N-1 would act as targets for 0..N-2, so specific overlap...
    # Actually, input_states is 0..T-2. target_states is 1..T-1.
    # Overlap is 1..T-2.
    # If input_states data was modified in place, then 'states' tensor data is modified.
    # Then target_states (view of states) would be modified.
    
    # My implementation:
    # input_states = torch.where(noise_mask.unsqueeze(-1), input_states + noise, input_states)
    # This returns a NEW tensor. The original 'states' tensor should remain untouched.
    
    # Let's verify that 'target_states' matches 'input_states_original[:, 1:]' (which is states[:, 1:-1])
    # logical check: target_states[:, :-1] should match input_states_original[:, 1:] exactly.
    
    target_overlap = target_states[:, :-1]
    input_overlap = input_states_original[:, 1:]
    
    overlap_diff = (target_overlap - input_overlap).abs().mean().item()
    print(f"Target/Original Overlap Diff (Should be 0.0): {overlap_diff:.6f}")
    
    if overlap_diff < 1e-6:
         print("PASS: Original tensor not modified in-place (Targets are clean).")
    else:
         print("FAIL: Original tensor MODIFIED in-place (Targets are dirty!)")

if __name__ == "__main__":
    verify_noise_injection()
