import torch
from env.ship import Ship, default_ship_config
from env.constants import PowerActions, TurnActions, ShootActions
from env.bullets import Bullets


def create_test_ship() -> Ship:
    return Ship(
        ship_id=0,
        team_id=0,
        ship_config=default_ship_config,
        initial_x=0.0,
        initial_y=0.0,
        initial_vx=10.0,
        initial_vy=0.0,
        world_size=(1000, 1000),
    )


def test_ship_id_initialization():
    ship = create_test_ship()
    assert ship.ship_id == 0
    assert ship.team_id == 0
    assert ship.alive


def test_ship_movement_no_action():
    ship = create_test_ship()
    # [Power, Turn, Shoot]
    action = torch.zeros(3)
    # Default is COAST, GO_STRAIGHT, NO_SHOOT (all 0)

    bullets = Bullets(max_bullets=100)
    ship.forward(action, bullets, 0.0, 1.0)

    # Ship has initial velocity of 10.0, so it should move
    assert ship.position.real > 0
    assert ship.position.imag == 0


def test_ship_movement_forward():
    ship = create_test_ship()
    action = torch.zeros(3)
    action[0] = float(PowerActions.BOOST)  # Forward/Boost
    action[1] = float(TurnActions.GO_STRAIGHT)
    action[2] = float(ShootActions.NO_SHOOT)

    bullets = Bullets(max_bullets=100)
    ship.forward(action, bullets, 0.0, 1.0)

    # Should have moved in positive x direction (default heading is 1+0j)
    assert ship.position.real > 0
    assert ship.position.imag == 0
    # Should accelerate (boost)
    assert ship.velocity.real > 10.0


def test_ship_turn_left():
    ship = create_test_ship()
    # Give it some velocity so turning does something to position if tracked,
    # but mainly checking attitude update
    ship.velocity = 100 + 0j

    action = torch.zeros(3)
    action[0] = float(PowerActions.COAST)
    action[1] = float(TurnActions.TURN_LEFT)
    action[2] = float(ShootActions.NO_SHOOT)

    bullets = Bullets(max_bullets=100)
    ship.forward(action, bullets, 0.0, 1.0)

    # Implementation defines TURN_LEFT as negative angle offset
    # Initial attitude 1+0j (0 rad). New attitude should have negative angle.
    # sin(-theta) is negative.
    assert ship.attitude.imag < 0


def test_ship_shoot():
    ship = create_test_ship()
    action = torch.zeros(3)
    action[0] = float(PowerActions.COAST)
    action[1] = float(TurnActions.GO_STRAIGHT)
    action[2] = float(ShootActions.SHOOT)

    bullets = Bullets(max_bullets=100)
    initial_bullets = bullets.num_active
    ship.forward(action, bullets, 0.0, 1.0)

    assert bullets.num_active == initial_bullets + 1


def test_ship_cooldown():
    ship = create_test_ship()
    action = torch.zeros(3)
    action[0] = float(PowerActions.BOOST)  # Forward
    action[1] = float(TurnActions.GO_STRAIGHT)
    action[2] = float(ShootActions.SHOOT)

    bullets = Bullets(max_bullets=100)
    ship.forward(action, bullets, 0.0, 1.0)
    assert bullets.num_active == 1

    # Should not fire again immediately
    ship.forward(action, bullets, 0.0, 1.0)
    assert bullets.num_active == 1
