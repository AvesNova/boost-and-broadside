import os
import pytest
import torch
from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.env2.adapter import tensor_state_to_render_state
from boost_and_broadside.env2.renderer import GameRenderer
from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.core.types import RenderState

# Set HEADLESS mode to avoid opening windows
os.environ["HEADLESS"] = "1"

def test_renderer_pipeline():
    """
    Test that TensorEnv state can be successfully converted and rendered.
    """
    # 1. Setup Env
    config = ShipConfig()
    env = TensorEnv(num_envs=1, config=config, device="cpu", max_ships=2)
    env.reset()
    
    # Step once to get some bullets maybe? No actions=zeros means no shooting.
    # Let's force some state
    valid_actions = torch.zeros((1, 2, 3), dtype=torch.long)
    env.step(valid_actions)
    
    # 2. Conversion
    render_state = tensor_state_to_render_state(env.state, config, batch_idx=0)
    
    assert isinstance(render_state, RenderState)
    assert len(render_state.ships) == 2
    assert render_state.time > 0
    
    # 3. Renderer
    renderer = GameRenderer(config, target_fps=60)
    
    # 4. Render Call (Should not crash)
    try:
        renderer.render(render_state)
    except Exception as e:
        pytest.fail(f"Renderer crashed: {e}")
    finally:
        renderer.close()

if __name__ == "__main__":
    test_renderer_pipeline()
