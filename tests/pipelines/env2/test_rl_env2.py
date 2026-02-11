from omegaconf import OmegaConf

from boost_and_broadside.env2.coordinator_wrapper import TensorEnvWrapper
from boost_and_broadside.env2.sb3_wrapper import SB3Wrapper

class MockPolicy:
    """Mock policy for testing."""
    def __init__(self, action_space):
        self.action_space = action_space
        
    def predict(self, observation, deterministic=False):
        # Return random action
        batch_size = 1 # Wrapper handles 1 env
        # Action space of SB3Wrapper is MultiDiscrete for all ships
        # Shape: (12 * max_ships) ??? No, Flattened?
        # Let's check SB3Wrapper action space.
        # It's usually MultiDiscrete.
        action = self.action_space.sample()
        return action, None

def test_rl_env2_integration():
    """
    Verify that TensorEnvWrapped can be wrapped by SB3Wrapper and used in a rollout.
    """
    config = OmegaConf.create({
        "environment": {
            "max_ships": 2,
            "world_size": [1000, 1000],
            "physics_dt": 0.02,
            "agent_dt": 0.04,
            "render_mode": "none",
            "early_termination": True,
            "reward_config": None
        },
        "rl": {
           "context_len": 4 
        }
    })
    
    # Instantiate TensorEnvWrapper
    # Note: TensorEnvWrapper signature: def __init__(self, **kwargs)
    env_kwargs = OmegaConf.to_container(config.environment)
    env = TensorEnvWrapper(**env_kwargs)
    
    # Wrap with SB3Wrapper (Adapter for Gym API)
    # SB3Wrapper expects an Environment-like object (GameCoordinator or Environment)
    # TensorEnvWrapper mimics GameCoordinator/Environment API.
    sb3_env = SB3Wrapper(env, config)
    
    # Gym Check
    # This might fail if spaces are not exactly as expected, but let's try basic reset/step
    obs, info = sb3_env.reset()
    
    # Observation should be dict with "tokens" if SB3Wrapper process it, OR
    # SB3Wrapper returns dict of tokens.
    # Actually SB3Wrapper usually tokenizes raw observations.
    # TensorEnvWrapper .get_observation() returns raw tensors (pos, vel, etc.)
    # SB3Wrapper calls `observation_to_tokens`.
    
    assert isinstance(obs, dict)
    assert "tokens" in obs
    
    # Rollout loop
    for _ in range(10):
        # Sample action
        action = sb3_env.action_space.sample()
        
        obs, reward, terminated, truncated, info = sb3_env.step(action)
        
        assert "tokens" in obs
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        if terminated or truncated:
            sb3_env.reset()

if __name__ == "__main__":
    test_rl_env2_integration()
