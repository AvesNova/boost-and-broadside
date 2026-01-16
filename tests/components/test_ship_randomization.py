from env.ship import Ship, default_ship_config


def test_ship_initialization_with_custom_values():
    config = default_ship_config

    # Test with custom health and power
    ship = Ship(
        ship_id=0,
        team_id=0,
        ship_config=config,
        initial_x=0.0,
        initial_y=0.0,
        initial_vx=10.0,
        initial_vy=0.0,
        initial_health=50.0,
        initial_power=25.0,
    )

    assert ship.health == 50.0
    assert ship.power == 25.0

    # Test defaulting behavior
    ship_default = Ship(
        ship_id=1,
        team_id=0,
        ship_config=config,
        initial_x=0.0,
        initial_y=0.0,
        initial_vx=10.0,
        initial_vy=0.0,
    )

    assert ship_default.health == config.max_health
    assert ship_default.power == config.max_power
