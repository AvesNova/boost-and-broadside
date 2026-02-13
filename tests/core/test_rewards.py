import torch
import pytest
from boost_and_broadside.core.rewards import DamageReward, DeathReward, VictoryReward
from boost_and_broadside.env2.state import TensorState

@pytest.fixture
def mock_states():
    device = torch.device("cpu")
    num_envs = 2
    num_ships = 4
    
    # Create dummy states
    # Team 0: Ships 0, 1
    # Team 1: Ships 2, 3
    teams = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], device=device, dtype=torch.int32)
    
    # Base State
    prev_state = TensorState(
        step_count=torch.zeros(num_envs),
        ship_pos=torch.zeros(num_envs, num_ships),
        ship_vel=torch.zeros(num_envs, num_ships),
        ship_attitude=torch.zeros(num_envs, num_ships),
        ship_ang_vel=torch.zeros(num_envs, num_ships),
        ship_health=torch.full((num_envs, num_ships), 100.0, device=device),
        ship_power=torch.zeros(num_envs, num_ships),
        ship_cooldown=torch.zeros(num_envs, num_ships),
        ship_team_id=teams,
        ship_alive=torch.ones((num_envs, num_ships), dtype=torch.bool, device=device),
        ship_is_shooting=torch.zeros(num_envs, num_ships, dtype=torch.bool),
        bullet_pos=torch.zeros(num_envs, num_ships, 1),
        bullet_vel=torch.zeros(num_envs, num_ships, 1),
        bullet_time=torch.zeros(num_envs, num_ships, 1),
        bullet_active=torch.zeros(num_envs, num_ships, 1, dtype=torch.bool),
        bullet_cursor=torch.zeros(num_envs, num_ships, dtype=torch.long)
    )
    
    next_state = TensorState(
        step_count=torch.zeros(num_envs),
        ship_pos=torch.zeros(num_envs, num_ships),
        ship_vel=torch.zeros(num_envs, num_ships),
        ship_attitude=torch.zeros(num_envs, num_ships),
        ship_ang_vel=torch.zeros(num_envs, num_ships),
        ship_health=torch.full((num_envs, num_ships), 100.0, device=device),
        ship_power=torch.zeros(num_envs, num_ships),
        ship_cooldown=torch.zeros(num_envs, num_ships),
        ship_team_id=teams,
        ship_alive=torch.ones((num_envs, num_ships), dtype=torch.bool, device=device),
        ship_is_shooting=torch.zeros(num_envs, num_ships, dtype=torch.bool),
        bullet_pos=torch.zeros(num_envs, num_ships, 1),
        bullet_vel=torch.zeros(num_envs, num_ships, 1),
        bullet_time=torch.zeros(num_envs, num_ships, 1),
        bullet_active=torch.zeros(num_envs, num_ships, 1, dtype=torch.bool),
        bullet_cursor=torch.zeros(num_envs, num_ships, dtype=torch.long)
    )
    
    actions = torch.zeros(num_envs, num_ships, 3)
    dones = torch.zeros(num_envs, dtype=torch.bool)
    
    return prev_state, actions, next_state, dones

def test_damage_reward(mock_states):
    prev, actions, next_s, dones = mock_states
    
    # damage reward config
    r = DamageReward(weight=1.0, pain_weight=2.0, blood_weight=0.5)
    
    # Scenario:
    # Env 0: Ship 0 (Ally) takes 10 damage.
    # Env 0: Ship 2 (Enemy) takes 20 damage.
    
    next_s.ship_health[0, 0] = 90.0
    next_s.ship_health[0, 2] = 80.0
    
    rewards = r.compute(prev, actions, next_s, dones)
    
    # Exp 0: Ship 0 (Ally) -> -10 * 2.0 = -20.0
    assert rewards[0, 0].item() == -20.0
    
    # Exp 0: Ship 2 (Enemy) -> +20 * 0.5 = +10.0
    assert rewards[0, 2].item() == 10.0
    
    # Others 0
    assert rewards[0, 1].item() == 0.0

def test_death_reward(mock_states):
    prev, actions, next_s, dones = mock_states
    r = DeathReward(weight=1.0, die_penalty=10.0, kill_reward=5.0)
    
    # Scenario:
    # Env 0: Ship 1 (Ally) dies
    # Env 0: Ship 3 (Enemy) dies
    
    next_s.ship_alive[0, 1] = False
    next_s.ship_alive[0, 3] = False
    next_s.ship_health[0, 1] = 0.0
    next_s.ship_health[0, 3] = 0.0
    
    rewards = r.compute(prev, actions, next_s, dones)
    
    # Ally Died -> Penalty
    assert rewards[0, 1].item() == -10.0
    
    # Enemy Died -> Reward
    assert rewards[0, 3].item() == 5.0
    
    # Survivors 0
    assert rewards[0, 0].item() == 0.0

def test_victory_reward(mock_states):
    prev, actions, next_s, dones = mock_states
    r = VictoryReward(weight=1.0, win_reward=100.0, lose_penalty=50.0)
    
    # Scenario:
    # Env 0: Team 1 wiped out (Win usually)
    # Env 1: Team 0 wiped out (Loss)
    
    # Env 0: Team 0 alive, Team 1 dead
    next_s.ship_alive[0, 0] = True
    next_s.ship_alive[0, 1] = True
    next_s.ship_alive[0, 2] = False
    next_s.ship_alive[0, 3] = False
    
    # Env 1: Team 0 dead, Team 1 alive
    next_s.ship_alive[1, 0] = False
    next_s.ship_alive[1, 1] = False
    next_s.ship_alive[1, 2] = True
    next_s.ship_alive[1, 3] = True
    
    dones = torch.tensor([True, True], dtype=torch.bool)
    
    rewards = r.compute(prev, actions, next_s, dones, is_terminal=True)
    
    # Env 0 (Win): All ships get +100
    assert torch.all(rewards[0] == 100.0)
    
    # Env 1 (Loss): All ships get -50
    assert torch.all(rewards[1] == -50.0)
