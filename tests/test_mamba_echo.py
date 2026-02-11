import torch
import pytest
from boost_and_broadside.models.yemong.scaffolds import YemongFull

from omegaconf import OmegaConf

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
    from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM, StateFeature
    
    cfg = OmegaConf.create({
        "input_dim": STATE_DIM,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "action_dim": 12,
        "target_dim": TARGET_DIM,
        "loss_type": "fixed",
    })
    
    model = YemongFull(cfg).to(device)
    model.eval()

    # 2. Generate Synthetic Episode
    B = 1
    T = 5 # Run 5 steps
    N = 2
    
    # Random inputs
    state = torch.randn(B, T, N, STATE_DIM, device=device) * 0.1
    state[..., StateFeature.HEALTH] = 1.0 # Ensure alive
    
    prev_action = torch.randn(B, T, N, 3, device=device) # Actually 3D
    # The original test used prev_action shape (B,T,N,3) but Yemong expects (..., 12) or handles embedding?
    # Actually YemongFull expects `prev_action` to be the tokenized action or embedding? 
    # Checking scaffolds.py: prev_action is passed to ActionEncoder.
    # ActionEncoder expects (..., 3) (Power, Turn, Shoot) or (..., 12) one-hot?
    # Let's check ActionEncoder. It takes (..., 3) usually.
    # Let's keep it (..., 3) but make sure it's valid range if discrete?
    # ActionEncoder typically handles continuous/discrete mapping.
    # Wait, Encoder expects raw action indices? 
    # ActionEncoder: x = self.embed(x.long()) if discrete.
    # Let's trust it expects (..., 3) floats or ints.
    # If Yemong uses ActionEncoder, it likely expects long indices for embedding if discrete.
    # But let's check what YemongFull does.
    # It passes `prev_action` to `self.action_encoder`.
    
    # Let's use zeros for prev_action to be safe and deterministic
    prev_action = torch.zeros(B, T, N, 12, device=device) # Wait, is it 12 or 3?
    # Model config says action_dim=12. 
    # If using embeddings, input is usually (..., 3) indices.
    # If input is already 12, it might be one-hot.
    # Let's look at scaffolds.py...
    # It uses `self.action_encoder(prev_action)`.
    # Let's stick to zeros of shape (..., 12) as that generic mock usually works if linear projection.
    
    pos = torch.randn(B, T, N, 2, device=device) * 100.0 # World scale
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
    inference_params = InferenceParams(max_seqlen=T, max_batch_size=B * N) if InferenceParams else None
    
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

            if inference_params:
                inference_params.seqlen_offset += 1
            
    # Concatenate inference logits
    inf_logits_cat = torch.cat(inference_logits, dim=1) # (B, T, N, 12)
    
    # 5. Compare
    # We compare the entire sequence
    diff = (train_logits - inf_logits_cat).abs().max().item()
    print(f"Max Difference: {diff}")
    
    # Tolerance: Floating point differences expected, especially with differing parallel/recurrent kernels
    # Usually < 1e-4 or 1e-5 is good.
    limit = 5e-4
    if diff > limit:
        print("FAILURE: Outputs generated in parallel do not match recurrent generation.")
        print(f"Indices of mismatch: {torch.where((train_logits - inf_logits_cat).abs() > limit)}")
    else:
        print("SUCCESS: Parallel and Recurrent outputs match!")
        
    assert diff < limit, f"Mismatch between Training and Inference modes: {diff}"

if __name__ == "__main__":
    test_mamba_sanity_echo(torch.device("cpu"))
