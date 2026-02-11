import torch
import pytest
from boost_and_broadside.agents.mamba_bb import MambaBB

class MambaConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

try:
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    InferenceParams = None

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_mamba_sanity_echo(device):
    """
    Sanity Echo Test:
    Verifies that the model produces identical outputs when run in:
    1. Training Mode (Parallel/Full Sequence)
    2. Inference Mode (Recurrent/Step-by-Step)
    
    This confirms that the internal state management (SSM State + Attention History) is correct.
    """
    print(f"\nRunning Sanity Echo Test on {device}...")
    
    # 1. Setup Model
    cfg = MambaConfig(
        input_dim=16, # Arbitrary
        d_model=64,
        n_layers=2,
        n_heads=2,
        action_dim=12,
        target_dim=16,
        loss_type="fixed"
    )
    
    model = MambaBB(cfg).to(device)
    model.eval()

    # 2. Generate Synthetic Episode
    B = 1
    T = 5 # Run 5 steps
    N = 2
    
    # Random inputs
    state = torch.randn(B, T, N, cfg.input_dim, device=device)
    prev_action = torch.randn(B, T, N, 3, device=device)
    pos = torch.randn(B, T, N, 2, device=device) * 1000.0 # World scale
    vel = torch.randn(B, T, N, 2, device=device)
    att = torch.randn(B, T, N, 2, device=device) # Cos, Sin
    alive = torch.ones(B, T, N, dtype=torch.bool, device=device)
    
    # 3. Training Mode (Forward on full sequence)
    print("Running Training Mode (Parallel)...")
    with torch.no_grad():
        # returns: pred_states, pred_actions, value, reward, cache
        train_out = model(
            state=state,
            prev_action=prev_action,
            pos=pos,
            vel=vel,
            att=att,
            alive=alive,
            world_size=(1024.0, 1024.0)
        )
        train_logits = train_out[1] # (B, T, N, 12)
        print("Training Mode Done.")
        
    # 4. Inference Mode (Step-by-Step)
    print("Running Inference Mode (Recurrent)...")
    inference_logits = []
    
    actor_cache = None
    inference_params = InferenceParams(max_seqlen=T, max_batch_size=B) if InferenceParams else None
    
    with torch.no_grad():
        for t in range(T):
            print(f"Step {t}/{T}")
            # Slice inputs at t -> (B, 1, N, D)
            s_t = state[:, t:t+1]
            pa_t = prev_action[:, t:t+1]
            p_t = pos[:, t:t+1]
            v_t = vel[:, t:t+1]
            at_t = att[:, t:t+1]
            al_t = alive[:, t:t+1]
            
            # Forward step
            step_out = model(
                state=s_t,
                prev_action=pa_t,
                pos=p_t,
                vel=v_t,
                att=at_t,
                alive=al_t,
                inference_params=inference_params,
                actor_cache=actor_cache,
                world_size=(1024.0, 1024.0)
            )
            
            # Helper to extract logits
            step_logits = step_out[1] # (B, 1, N, 12)
            inference_logits.append(step_logits)
            
            # Update Cache
            actor_cache = step_out[4]
            
    # Concatenate inference logits
    inf_logits_cat = torch.cat(inference_logits, dim=1) # (B, T, N, 12)
    
    # 5. Compare
    # We compare the entire sequence
    diff = (train_logits - inf_logits_cat).abs().max().item()
    print(f"Max Difference: {diff}")
    
    # Tolerance: Floating point differences expected, especially with differing parallel/recurrent kernels
    # Usually < 1e-4 or 1e-5 is good.
    limit = 1e-4
    if diff > limit:
        print("FAILURE: Outputs generated in parallel do not match recurrent generation.")
        print(f"Indices of mismatch: {torch.where((train_logits - inf_logits_cat).abs() > limit)}")
    else:
        print("SUCCESS: Parallel and Recurrent outputs match!")
        
    assert diff < limit, f"Mismatch between Training and Inference modes: {diff}"

if __name__ == "__main__":
    test_mamba_sanity_echo(torch.device("cpu"))
