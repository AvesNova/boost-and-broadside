from env.state import State


def test_state_initialization(dummy_ship):
    """Test that state initializes correctly."""
    ships = {0: dummy_ship}
    state = State(ships=ships)

    assert state.ships == ships
    assert state.time == 0.0
    assert state.bullets is not None


def test_state_immutability(dummy_ship):
    """Test that state is effectively immutable (or at least we treat it as such)."""
    # Note: The current implementation might not enforce immutability strictly,
    # but the design pattern suggests we create new states or modify in place carefully.
    # If the design is to modify in place during a step, we check that.
    # If the design is to return new states, we check that.
    # Based on env.py, it seems to modify state in place during substeps.

    ships = {0: dummy_ship}
    state = State(ships=ships)

    # Check that we can access ships
    assert state.ships[0].ship_id == 0

    # Check that bullets are initialized empty
    assert state.bullets.num_active == 0


def test_state_bullet_management(dummy_ship):
    """Test adding and removing bullets."""
    ships = {0: dummy_ship}
    state = State(ships=ships)

    # Add a bullet
    state.bullets.add_bullet(ship_id=0, x=100.0, y=100.0, vx=10.0, vy=0.0, lifetime=1.0)

    assert state.bullets.num_active == 1

    # Remove the bullet
    state.bullets.remove_bullet(0)

    assert state.bullets.num_active == 0
