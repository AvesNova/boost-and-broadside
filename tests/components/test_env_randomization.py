from env.env import Environment
from env.ship import default_ship_config


def test_environment_random_initialization():
    # randomized env
    env = Environment(
        render_mode="none",
        world_size=(1000, 1000),
        memory_size=10,
        max_ships=8,
        agent_dt=0.1,
        physics_dt=0.01,
        random_positioning=True,
        random_speed=True,
        random_initialization=True,  # Enable random health/power
    )

    # Reset 4v4
    state, info = env.reset(game_mode="4v4")

    # Check that health/power are not all equal to max
    # We expect variance.

    varied_health = False
    varied_power = False

    max_health = default_ship_config.max_health
    max_power = default_ship_config.max_power

    observation = env.get_observation()

    # ship_ids are 0..7
    for i in range(8):
        # We need to access the state directly to get exact values,
        # as observation might be normalized (though env.get_observation currently returns raw tensors for some fields?
        # actually env.get_observation returns a dict of tensors.
        # Let's look at env.py:
        # "health": torch.zeros((self.max_ships), dtype=torch.int64) -> Wait, dtype is int64?
        # In env.py: "health": torch.zeros((self.max_ships), dtype=torch.int64) -> This seems wrong if health is float.
        # Checking ship.py: self.health is float.
        # Checking reset_from_observation: health = float(initial_obs["health"][i, 0].item())
        # Let's check the state object directly for higher precision and certainty.

        ship = env.state.ships[i]

        if abs(ship.health - max_health) > 0.01:
            varied_health = True

        if abs(ship.power - max_power) > 0.01:
            varied_power = True

    assert varied_health, "Health should be randomized (not all max)"
    assert varied_power, "Power should be randomized (not all max)"


def test_environment_default_initialization():
    # default env
    env = Environment(
        render_mode="none",
        world_size=(1000, 1000),
        memory_size=10,
        max_ships=8,
        agent_dt=0.1,
        physics_dt=0.01,
        random_positioning=True,
        random_speed=True,
        random_initialization=False,  # Disable
    )

    env.reset(game_mode="4v4")

    for i in range(8):
        ship = env.state.ships[i]
        assert ship.health == default_ship_config.max_health
        assert ship.power == default_ship_config.max_power


def test_ship_id_randomization():
    """Test that ship IDs are randomized within team bounds."""
    env = Environment(
        render_mode="none",
        world_size=(1000, 1000),
        memory_size=0,
        max_ships=8,
        agent_dt=0.1,
        physics_dt=0.01,
        random_positioning=True,
        random_speed=True,
    )

    # Test 1v1 Randomization
    ids_seen_team_0 = set()
    ids_seen_team_1 = set()

    for _ in range(10):
        _ = env.reset(game_mode="1v1")
        ships = env.state.ships

        # Should have 2 ships
        assert len(ships) == 2

        # Sort IDs
        ids = sorted(list(ships.keys()))

        # Check Team assignment
        # Team 0 should be in [0, 3], Team 1 in [4, 7] (for max_ships=8)
        ship_0 = ships[ids[0]]
        ship_1 = ships[ids[1]]

        # It's not guaranteed ids[0] is team 0 if ids are e.g. 5 and 6 (both team 1? No 1v1 means 1 per team)
        # 1v1 reset ensures 1 ship per team.
        # Find team 0 ship
        team_0_ships = [s for s in ships.values() if s.team_id == 0]
        team_1_ships = [s for s in ships.values() if s.team_id == 1]

        assert len(team_0_ships) == 1
        assert len(team_1_ships) == 1

        id_0 = team_0_ships[0].ship_id
        id_1 = team_1_ships[0].ship_id

        assert 0 <= id_0 <= 3
        assert 4 <= id_1 <= 7

        ids_seen_team_0.add(id_0)
        ids_seen_team_1.add(id_1)

    # Verify we saw some variation (probability of same ID 10 times is tiny)
    assert len(ids_seen_team_0) > 1, f"Team 0 IDs not randomized: {ids_seen_team_0}"
    assert len(ids_seen_team_1) > 1, f"Team 1 IDs not randomized: {ids_seen_team_1}"

    # Test 2v2 Randomization
    previous_ids = None
    variation_count = 0

    for _ in range(5):
        state = env.n_vs_n_reset(ships_per_team=2)
        current_ids = sorted(list(state.ships.keys()))

        if previous_ids is not None and current_ids != previous_ids:
            variation_count += 1

        previous_ids = current_ids

        # Check bounds
        team_0_ids = [s.ship_id for s in state.ships.values() if s.team_id == 0]
        team_1_ids = [s.ship_id for s in state.ships.values() if s.team_id == 1]

        assert len(team_0_ids) == 2
        assert len(team_1_ids) == 2

        for mid in team_0_ids:
            assert 0 <= mid <= 3
        for mid in team_1_ids:
            assert 4 <= mid <= 7

    assert variation_count > 0, "2v2 Ship IDs did not vary across resets"
