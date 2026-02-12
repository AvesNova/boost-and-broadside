
import pytest
import torch
import hydra
from omegaconf import OmegaConf

from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.env2.gpu_wrapper import GPUEnvWrapper

def test_episode_truncation():
    """Verifies that episodes are truncated at max_episode_steps."""
    num_envs = 4
    max_steps = 10
    
    cfg = ShipConfig()
    
    # Initialize Env with short max_episode_steps
    env = TensorEnv(
        num_envs=num_envs, 
        config=cfg, 
        device=torch.device("cpu"), # CPU for testing
        max_episode_steps=max_steps
    )
    gpu_env = GPUEnvWrapper(env)
    
    obs, _ = gpu_env.reset()
    
    # Step for max_steps - 1
    for _ in range(max_steps - 1):
        action = torch.zeros((num_envs, env.max_ships, 3), dtype=torch.long)
        obs, reward, terminated, truncated, info = gpu_env.step(action)
        
        # Should not be truncated yet (assuming no collisions/termination)
        # Note: If random initialization causes collisions, terminated might be True.
        # But we check truncated specifically?
        # GPUEnvWrapper returns 'truncated' from env.
        # And TensorEnv sets truncated based on step count.
        
        # We can't guarantee 'terminated' is False (random collisions), 
        # but 'truncated' should be False.
        if truncated.any():
             # If step count < max_steps, truncated should be False.
             # Unless step count accumulates?
             pass 
        assert not truncated.any(), f"Truncated early at step {env.state.step_count[0]}"

    # Time Limit Step
    action = torch.zeros((num_envs, env.max_ships, 3), dtype=torch.long)
    obs, reward, terminated, truncated, info = gpu_env.step(action)
    
    # Now truncated should be True for all active envs (or at least those that didn't terminate earlier)
    # Actually, TensorEnv resets envs that are done. 
    # If an env terminated at step 5, it reset. Its step count became 0.
    # So it won't be truncated at global step 10.
    # We need to check individual env step counts.
    
    # step_count is reset on done.
    # So we assume no termination occurred for this test to be clean?
    # Or strict check: 
    # For any env that did NOT terminate previously, it MUST be truncated now.
    # But how do we track that?
    
    # Simpler check: step count should never exceed max_steps.
    assert (env.state.step_count <= max_steps).all()
    
    # And if step_count was max_steps, it should have triggered reset (so step_count becomes 0).
    # Wait, reset happens in step() AFTER trunc calculation but BEFORE return?
    # TensorEnv.step: 
    #   step_count += 1
    #   truncated = step_count >= max
    #   dones = ...
    #   if dones: reset() -> step_count = 0
    
    # So if it truncated, step_count is now 0.
    # If it didn't terminate/truncate, step_count is count.
    
    # So at step 10:
    # 1. step_count becomes 10.
    # 2. truncated becomes True.
    # 3. reset happens -> step_count becomes 0.
    # So returned step_count (in state) is 0.
    assert (env.state.step_count == 0).all()
    assert truncated.all()

if __name__ == "__main__":
    test_episode_truncation()
