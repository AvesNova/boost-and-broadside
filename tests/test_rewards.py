"""
Tests for the team-based reward system.
"""

import pytest
from copy import deepcopy

from src.constants import RewardConstants
from src.env import Environment
from src.ship import Ship, default_ship_config
from src.state import State


@pytest.fixture
def reward_constants():
    """Provide reward constants for tests"""
    return RewardConstants


@pytest.fixture
def reward_test_env():
    """Environment configured for reward testing with minimal world"""
    return Environment(
        world_size=(400, 400),
        max_ships=6,
        agent_dt=0.02,
        physics_dt=0.02,
        memory_size=2,  # Need previous state for reward calculation
    )


@pytest.fixture
def state_builder():
    """Factory for creating test states with specific ship configurations"""

    def _build_state(
        team_configs: dict[
            int, list[tuple[int, float, bool]]
        ],  # team_id -> [(ship_id, health, alive), ...]
    ) -> State:
        """
        Build a state with ships configured as specified

        Args:
            team_configs: Dict mapping team_id to list of (ship_id, health, alive) tuples
        """
        ships = {}

        for team_id, ship_configs in team_configs.items():
            for ship_id, health, alive in ship_configs:
                ship = Ship(
                    ship_id=ship_id,
                    team_id=team_id,
                    ship_config=default_ship_config,
                    initial_x=100.0 + ship_id * 50.0,  # Spread ships out
                    initial_y=100.0 + team_id * 50.0,
                    initial_vx=50.0,
                    initial_vy=0.0,
                    world_size=(400, 400),
                )
                ship.health = health
                ship.alive = alive
                ships[ship_id] = ship

        return State(ships=ships)

    return _build_state


@pytest.fixture
def reward_calculator(reward_test_env):
    """Isolated reward calculation helper"""

    def _calculate_reward(
        previous_state: State,
        current_state: State,
        team_id: int,
        episode_ended: bool = False,
    ) -> float:
        """Calculate reward for team between two states"""
        # Set up environment with states
        reward_test_env.state.clear()
        reward_test_env.state.append(previous_state)
        reward_test_env.state.append(current_state)

        return reward_test_env._calculate_team_reward(
            current_state, team_id, episode_ended
        )

    return _calculate_reward


@pytest.fixture
def create_damage_transition(state_builder):
    """Create before/after states showing specific damage"""

    def _create_transition(
        ship_id: int,
        team_id: int,
        initial_health: float,
        final_health: float,
        other_teams: dict[int, list[tuple[int, float, bool]]] = None,
    ) -> tuple[State, State]:
        """
        Create transition showing ship taking damage

        Args:
            ship_id: ID of ship being damaged
            team_id: Team of ship being damaged
            initial_health: Health before damage
            final_health: Health after damage
            other_teams: Other teams/ships to include in state
        """
        # Base team config
        base_config = {team_id: [(ship_id, initial_health, True)]}

        # Add other teams if specified
        if other_teams:
            base_config.update(other_teams)

        before_state = state_builder(base_config)

        # Create after state with damage
        after_config = deepcopy(base_config)
        after_config[team_id][0] = (ship_id, final_health, final_health > 0)
        after_state = state_builder(after_config)

        return before_state, after_state

    return _create_transition


@pytest.fixture
def create_death_transition(state_builder):
    """Create before/after states showing ship death"""

    def _create_transition(
        dying_ship_id: int,
        dying_team_id: int,
        other_teams: dict[int, list[tuple[int, float, bool]]] = None,
    ) -> tuple[State, State]:
        """Create transition showing ship death"""
        # Ship starts alive with some health
        initial_health = default_ship_config.max_health / 2

        base_config = {dying_team_id: [(dying_ship_id, initial_health, True)]}
        if other_teams:
            base_config.update(other_teams)

        before_state = state_builder(base_config)

        # Ship dies
        after_config = deepcopy(base_config)
        after_config[dying_team_id][0] = (dying_ship_id, 0.0, False)
        after_state = state_builder(after_config)

        return before_state, after_state

    return _create_transition


@pytest.fixture
def two_team_scenario(state_builder):
    """Standard two-team setup for testing"""

    def _create_scenario(
        team0_health: list[float] = None, team1_health: list[float] = None
    ) -> State:
        """Create two-team scenario with specified health levels"""
        team0_health = team0_health or [default_ship_config.max_health] * 2
        team1_health = team1_health or [default_ship_config.max_health] * 2

        config = {
            0: [(i, health, health > 0) for i, health in enumerate(team0_health)],
            1: [(i + 10, health, health > 0) for i, health in enumerate(team1_health)],
        }

        return state_builder(config)

    return _create_scenario


class TestTacticalRewards:
    """Tests for damage and death rewards during episodes"""

    def test_ally_damage_penalty(
        self, create_damage_transition, reward_calculator, reward_constants
    ):
        """Test that ally taking damage gives negative reward"""
        damage_amount = reward_constants.TEST_DAMAGE_AMOUNT
        initial_health = default_ship_config.max_health
        final_health = initial_health - damage_amount

        before_state, after_state = create_damage_transition(
            ship_id=0,
            team_id=0,
            initial_health=initial_health,
            final_health=final_health,
        )

        reward = reward_calculator(before_state, after_state, team_id=0)

        expected_reward = -damage_amount * reward_constants.DAMAGE_REWARD_SCALE
        assert abs(reward - expected_reward) < 1e-6

    def test_enemy_damage_bonus(
        self, create_damage_transition, reward_calculator, reward_constants
    ):
        """Test that enemy taking damage gives positive reward"""
        damage_amount = reward_constants.TEST_DAMAGE_AMOUNT
        initial_health = default_ship_config.max_health
        final_health = initial_health - damage_amount

        # Create enemy ship on team 1, calculate reward for team 0
        before_state, after_state = create_damage_transition(
            ship_id=5,
            team_id=1,
            initial_health=initial_health,
            final_health=final_health,
        )

        reward = reward_calculator(before_state, after_state, team_id=0)

        expected_reward = damage_amount * reward_constants.DAMAGE_REWARD_SCALE
        assert abs(reward - expected_reward) < 1e-6

    def test_ally_death_penalty(
        self, create_death_transition, reward_calculator, reward_constants
    ):
        """Test that ally death gives death penalty"""
        before_state, after_state = create_death_transition(
            dying_ship_id=0, dying_team_id=0
        )

        reward = reward_calculator(before_state, after_state, team_id=0)

        assert abs(reward - reward_constants.ALLY_DEATH_PENALTY) < 1e-6

    def test_enemy_death_bonus(
        self, create_death_transition, reward_calculator, reward_constants
    ):
        """Test that enemy death gives death bonus"""
        before_state, after_state = create_death_transition(
            dying_ship_id=5, dying_team_id=1
        )

        reward = reward_calculator(before_state, after_state, team_id=0)

        assert abs(reward - reward_constants.ENEMY_DEATH_BONUS) < 1e-6

    def test_damage_reward_scaling(
        self, create_damage_transition, reward_calculator, reward_constants
    ):
        """Test that damage rewards scale linearly with damage amount"""
        test_damages = [
            reward_constants.TEST_SMALL_DAMAGE,
            reward_constants.TEST_DAMAGE_AMOUNT,
            reward_constants.TEST_LARGE_DAMAGE,
        ]

        for damage_amount in test_damages:
            initial_health = default_ship_config.max_health
            final_health = initial_health - damage_amount

            before_state, after_state = create_damage_transition(
                ship_id=5,
                team_id=1,  # Enemy ship
                initial_health=initial_health,
                final_health=final_health,
            )

            reward = reward_calculator(before_state, after_state, team_id=0)
            expected_reward = damage_amount * reward_constants.DAMAGE_REWARD_SCALE

            assert (
                abs(reward - expected_reward) < 1e-6
            ), f"Failed for damage {damage_amount}"

    def test_simultaneous_damage_and_death(
        self, state_builder, reward_calculator, reward_constants
    ):
        """Test multiple events in single transition"""
        # Team 0: ship 0 takes damage, ship 1 dies
        # Team 1: ship 10 takes damage
        before_config = {0: [(0, 80.0, True), (1, 20.0, True)], 1: [(10, 60.0, True)]}

        after_config = {
            0: [(0, 60.0, True), (1, 0.0, False)],  # Damage + death
            1: [(10, 40.0, True)],  # Enemy damage
        }

        before_state = state_builder(before_config)
        after_state = state_builder(after_config)

        reward = reward_calculator(before_state, after_state, team_id=0)

        # Expected: -20 damage to ally, -0.1 ally death, +20 damage to enemy
        expected_reward = (
            -20.0 * reward_constants.DAMAGE_REWARD_SCALE  # Ally damage
            + reward_constants.ALLY_DEATH_PENALTY  # Ally death
            + 20.0 * reward_constants.DAMAGE_REWARD_SCALE  # Enemy damage
        )

        assert abs(reward - expected_reward) < 1e-6

    def test_no_reward_for_dead_ships(self, state_builder, reward_calculator):
        """Test that dead ships don't generate damage rewards"""
        # Ship starts dead and "takes more damage" (shouldn't count)
        before_config = {0: [(0, 0.0, False)]}
        after_config = {0: [(0, -10.0, False)]}  # Negative health

        before_state = state_builder(before_config)
        after_state = state_builder(after_config)

        reward = reward_calculator(before_state, after_state, team_id=0)

        assert abs(reward) < 1e-6  # No reward for dead ship taking "damage"

    def test_no_reward_for_healing(self, create_damage_transition, reward_calculator):
        """Test that healing doesn't generate negative damage rewards"""
        # Ship "heals" (health increases)
        before_state, after_state = create_damage_transition(
            ship_id=0,
            team_id=0,
            initial_health=50.0,
            final_health=70.0,  # Health increase
        )

        reward = reward_calculator(before_state, after_state, team_id=0)

        assert abs(reward) < 1e-6  # No reward for healing

    def test_first_step_no_previous_state(
        self, reward_test_env, two_team_scenario, reward_constants
    ):
        """Test reward calculation when no previous state exists"""
        current_state = two_team_scenario()

        # Clear state history and add only current state
        reward_test_env.state.clear()
        reward_test_env.state.append(current_state)

        # Should return 0 since no previous state for tactical rewards
        reward = reward_test_env._calculate_team_reward(current_state, team_id=0)

        assert abs(reward) < 1e-6


class TestOutcomeRewards:
    """Tests for episode termination rewards"""

    def test_clear_victory(self, state_builder, reward_calculator, reward_constants):
        """Test victory reward when all enemies are dead"""
        final_state = state_builder(
            {
                0: [(0, 50.0, True), (1, 30.0, True)],  # Our team alive
                1: [(10, 0.0, False), (11, 0.0, False)],  # Enemy team dead
            }
        )

        # Previous state doesn't matter for outcome rewards
        previous_state = final_state

        reward = reward_calculator(
            previous_state, final_state, team_id=0, episode_ended=True
        )

        # Should get victory reward (tactical rewards would be additional)
        assert reward >= reward_constants.VICTORY_REWARD

    def test_clear_defeat(self, state_builder, reward_calculator, reward_constants):
        """Test defeat reward when all allies are dead"""
        final_state = state_builder(
            {
                0: [(0, 0.0, False), (1, 0.0, False)],  # Our team dead
                1: [(10, 50.0, True), (11, 30.0, True)],  # Enemy team alive
            }
        )

        previous_state = final_state

        reward = reward_calculator(
            previous_state, final_state, team_id=0, episode_ended=True
        )

        # Should get defeat reward (tactical rewards would be additional)
        assert reward <= reward_constants.DEFEAT_REWARD

    def test_mutual_destruction_draw(
        self, state_builder, reward_calculator, reward_constants, reward_test_env
    ):
        """Test draw reward when all ships are dead"""
        final_state = state_builder(
            {
                0: [(0, 0.0, False), (1, 0.0, False)],  # Our team dead
                1: [(10, 0.0, False), (11, 0.0, False)],  # Enemy team dead
            }
        )

        previous_state = final_state

        reward = reward_calculator(
            previous_state, final_state, team_id=0, episode_ended=True
        )

        # Should be draw reward (plus any tactical rewards from the deaths)
        # The exact value depends on whether deaths happened this step
        # But the outcome component should be 0
        outcome_reward = reward_test_env._calculate_outcome_rewards(
            final_state, team_id=0
        )
        assert abs(outcome_reward - reward_constants.DRAW_REWARD) < 1e-6

    def test_timeout_draw(
        self, state_builder, reward_calculator, reward_constants, reward_test_env
    ):
        """Test draw when episode times out with survivors"""
        final_state = state_builder(
            {
                0: [(0, 30.0, True), (1, 20.0, True)],  # Our team alive
                1: [(10, 40.0, True), (11, 50.0, True)],  # Enemy team alive
            }
        )

        # Test outcome rewards in isolation
        outcome_reward = reward_test_env._calculate_outcome_rewards(
            final_state, team_id=0
        )
        assert abs(outcome_reward - reward_constants.DRAW_REWARD) < 1e-6

    def test_no_outcome_reward_during_episode(self, state_builder, reward_calculator):
        """Test that outcome rewards only calculated at episode end"""
        # State where we would win if episode ended
        final_state = state_builder(
            {
                0: [(0, 50.0, True)],  # Our team alive
                1: [(10, 0.0, False)],  # Enemy dead
            }
        )

        previous_state = final_state

        # But episode hasn't ended
        reward = reward_calculator(
            previous_state, final_state, team_id=0, episode_ended=False
        )

        # Should not include victory reward
        assert abs(reward) < 1e-6  # No tactical changes, no outcome reward


class TestMultiTeamScenarios:
    """Tests for complex multi-team reward scenarios"""

    def test_three_team_victory(
        self, state_builder, reward_calculator, reward_constants, reward_test_env
    ):
        """Test victory in three-team scenario"""
        final_state = state_builder(
            {
                0: [(0, 50.0, True)],  # Our team survives
                1: [(10, 0.0, False)],  # Team 1 eliminated
                2: [(20, 0.0, False)],  # Team 2 eliminated
            }
        )

        outcome_reward = reward_test_env._calculate_outcome_rewards(
            final_state, team_id=0
        )
        assert abs(outcome_reward - reward_constants.VICTORY_REWARD) < 1e-6

    def test_three_team_defeat(
        self, state_builder, reward_calculator, reward_constants, reward_test_env
    ):
        """Test defeat when other teams survive"""
        final_state = state_builder(
            {
                0: [(0, 0.0, False)],  # Our team eliminated
                1: [(10, 30.0, True)],  # Team 1 survives
                2: [(20, 40.0, True)],  # Team 2 survives
            }
        )

        outcome_reward = reward_test_env._calculate_outcome_rewards(
            final_state, team_id=0
        )
        assert abs(outcome_reward - reward_constants.DEFEAT_REWARD) < 1e-6

    def test_enemy_vs_enemy_damage(
        self, state_builder, reward_calculator, reward_constants
    ):
        """Test that enemy-vs-enemy damage gives us a reward (enemy taking damage is good for us)"""
        # Team 1 ship damages Team 2 ship (neither is our team)
        before_config = {
            0: [(0, 100.0, True)],  # Our team (unchanged)
            1: [(10, 80.0, True)],  # Enemy team 1
            2: [(20, 60.0, True)],  # Enemy team 2
        }

        after_config = {
            0: [(0, 100.0, True)],  # Our team (unchanged)
            1: [(10, 80.0, True)],  # Enemy team 1 (unchanged)
            2: [(20, 40.0, True)],  # Enemy team 2 (damaged)
        }

        before_state = state_builder(before_config)
        after_state = state_builder(after_config)

        reward = reward_calculator(before_state, after_state, team_id=0)

        # Enemy taking damage should give us a reward, regardless of who caused it
        damage_amount = 20.0
        expected_reward = damage_amount * reward_constants.DAMAGE_REWARD_SCALE
        assert abs(reward - expected_reward) < 1e-6
