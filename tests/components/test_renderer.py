import os
import numpy as np
import pytest
from src.env2.renderer import GameRenderer
from src.core.config import ShipConfig
from src.core.types import RenderState, RenderShip

def test_renderer_initialization():
    """Test that the renderer initializes with correct dimensions and scaling."""
    os.environ["HEADLESS"] = "1"
    config = ShipConfig(world_size=(1000.0, 1000.0))
    renderer = GameRenderer(config)
    
    assert renderer.native_width == 1024 + 250
    assert renderer.native_height == 1024
    assert renderer.game_width == 1024
    assert renderer.game_height == 1024
    assert renderer.sidebar_width == 250
    
    # Check scaling (1024 / 1000 = 1.024)
    assert renderer.scale_x == pytest.approx(1.024)
    assert renderer.scale_y == pytest.approx(1.024)
    
    renderer.close()

def test_renderer_render_frame():
    """Test that the renderer can process a frame without crashing."""
    os.environ["HEADLESS"] = "1"
    config = ShipConfig(world_size=(1024.0, 1024.0))
    renderer = GameRenderer(config)
    
    # Create a dummy state
    ships = {
        0: RenderShip(
            id=0,
            team_id=0,
            position=complex(500, 500),
            attitude=complex(1, 0),
            health=100,
            max_health=100,
            power=100,
            alive=True
        ),
        1: RenderShip(
            id=1,
            team_id=1,
            position=complex(600, 600),
            attitude=complex(0, 1),
            health=50,
            max_health=100,
            power=20,
            alive=True
        )
    }
    
    state = RenderState(
        ships=ships,
        bullet_x=np.array([550.0], dtype=np.float32),
        bullet_y=np.array([550.0], dtype=np.float32),
        bullet_owner_id=np.array([0], dtype=np.int32),
        time=1.0
    )
    
    # This should run without error
    renderer.render(state)
    renderer.close()

def test_world_to_screen():
    """Test the coordinate mapping logic."""
    os.environ["HEADLESS"] = "1"
    config = ShipConfig(world_size=(1024.0, 1024.0))
    renderer = GameRenderer(config)
    
    # 0,0 should be top-left of game area
    assert renderer._world_to_screen(0, 0) == (0, 0)
    
    # 1024, 1024 should be bottom-right of game area
    assert renderer._world_to_screen(1024, 1024) == (1024, 1024)
    
    # Middle
    assert renderer._world_to_screen(512, 512) == (512, 512)
    
    renderer.close()
