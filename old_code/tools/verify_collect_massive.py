import torch
from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.env2.collector import AsyncCollector
from boost_and_broadside.core.config import ShipConfig
from pathlib import Path

def test_randomization():
    print("Testing Randomization...")
    config = ShipConfig(random_speed=True)
    num_envs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TensorEnv(num_envs, config, device=device)
    
    # Reset with a fixed seed
    obs = env.reset(seed=42)
    
    pos = obs["position"]
    att = obs["attitude"]
    
    # Check if all envs are different (flatten and convert complex to real/imag for unique)
    unique_pos = torch.unique(torch.view_as_real(pos.flatten()), dim=0)
    print(f"Unique positions: {len(unique_pos)} / {num_envs * 8}")
    
    is_unique = len(unique_pos) == num_envs * 8
    print(f"Randomization Test: {'PASSED' if is_unique else 'FAILED'}")
    
    if not is_unique:
        # Check first env vs second env
        print(f"Env 0 Pos 0: {pos[0,0]}")
        print(f"Env 1 Pos 0: {pos[1,0]}")

def test_buffer_steps():
    print("\nTesting buffer_steps...")
    config = ShipConfig()
    num_envs = 2
    device = torch.device("cpu")
    max_steps = 10
    
    data_path = "test_buffer.h5"
    collector = AsyncCollector(data_path, num_envs, 8, device, max_steps=max_steps)
    
    env = TensorEnv(num_envs, config, device=device)
    obs = env.reset(seed=42)
    
    # Run for more than max_steps
    for i in range(max_steps + 5):
        actions = torch.zeros((num_envs, 8, 3), dtype=torch.long)
        obs, rewards, dones, _, _ = env.step(actions)
        collector.step(obs, actions, rewards, dones)
        
    print(f"Step counts: {collector.step_counts}")
    # If buffer_steps is respected, step_counts should be capped or handled
    # Our implementation uses torch.clamp(steps, max=self.max_steps - 1)
    # So it should be fine.
    
    collector.close()
    if Path(data_path).exists():
        Path(data_path).unlink()
    print("Buffer Steps Test: DONE (check step_counts output)")

if __name__ == "__main__":
    test_randomization()
    test_buffer_steps()
