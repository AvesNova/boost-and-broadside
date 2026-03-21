"""Interactive game modes: human play and checkpoint watch.

Two entry points:
  - run_play_mode: human controls ship 0 (WASD + Space); AI controls the rest.
  - run_watch_mode: load a checkpoint and watch self-play at 60fps.

Both share a common render loop (_run_interactive_loop).
"""

import torch

from boost_and_broadside.config import ShipConfig, EnvConfig, ModelConfig, RewardConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.wrapper import MVPEnvWrapper
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig


def run_play_mode(
    ship_config:   ShipConfig,
    env_config:    EnvConfig,
    reward_config: RewardConfig,
    model_config:  ModelConfig,
    render_config: RenderConfig,
    device:        str,
) -> None:
    """Human plays ship 0 (WASD + Space). A fresh AI policy controls ships 1-N.

    Args:
        ship_config:   Physics constants.
        env_config:    Environment sizing.
        reward_config: Reward weights.
        model_config:  Policy architecture.
        render_config: Display settings.
        device:        Torch device string.
    """
    wrapper  = MVPEnvWrapper(num_envs=1, ship_config=ship_config, env_config=env_config,
                             reward_config=reward_config, device=device)
    policy   = MVPPolicy(model_config, ship_config).to(device)
    renderer = GameRenderer(ship_config, render_config)

    policy.eval()
    try:
        _run_interactive_loop(wrapper, policy, renderer, human_ship_idx=0,
                              device=torch.device(device))
    finally:
        renderer.close()


def run_watch_mode(
    checkpoint_path: str,
    ship_config:     ShipConfig,
    env_config:      EnvConfig,
    reward_config:   RewardConfig,
    model_config:    ModelConfig,
    render_config:   RenderConfig,
    device:          str,
) -> None:
    """Load a checkpoint and watch self-play at 60fps.

    Args:
        checkpoint_path: Path to a .pt checkpoint file saved by PPOTrainer.
        ship_config:     Physics constants.
        env_config:      Environment sizing.
        reward_config:   Reward weights.
        model_config:    Policy architecture.
        render_config:   Display settings.
        device:          Torch device string.
    """
    wrapper  = MVPEnvWrapper(num_envs=1, ship_config=ship_config, env_config=env_config,
                             reward_config=reward_config, device=device)
    policy   = MVPPolicy(model_config, ship_config).to(device)
    renderer = GameRenderer(ship_config, render_config)

    ckpt = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    update = ckpt.get("update", "?")
    step   = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint: update={update}, global_step={step}")

    try:
        _run_interactive_loop(wrapper, policy, renderer, human_ship_idx=None,
                              device=torch.device(device))
    finally:
        renderer.close()


def _run_interactive_loop(
    wrapper:          MVPEnvWrapper,
    policy:           MVPPolicy,
    renderer:         GameRenderer,
    human_ship_idx:   int | None,
    device:           torch.device,
) -> None:
    """Core render loop shared by play and watch modes.

    Runs episodes back-to-back until the user closes the window.

    Args:
        wrapper:         Single-env MVPEnvWrapper (num_envs=1).
        policy:          Policy to run under torch.no_grad.
        renderer:        Pygame renderer.
        human_ship_idx:  Ship index controlled by keyboard, or None for full AI.
        device:          Torch device.
    """
    N = wrapper.num_ships

    while True:
        obs    = wrapper.reset()
        hidden = policy.initial_hidden(1, N, device)

        while True:
            with torch.no_grad():
                action, _, _, hidden = policy.get_action_and_value(obs, hidden)

            if human_ship_idx is not None:
                action[0, human_ship_idx, :] = _decode_keyboard().to(device)

            obs, _, dones, truncated, _ = wrapper.step(action)

            if (dones | truncated).any():
                hidden = policy.initial_hidden(1, N, device)
                obs    = wrapper.reset()

            running = renderer.render(wrapper.state)
            if not running:
                return
            renderer.tick()


def _decode_keyboard() -> torch.Tensor:
    """Read current pygame key state and return a (3,) int action tensor.

    Controls:
        W              → BOOST
        S              → REVERSE
        A              → TURN_LEFT  (+ Shift → SHARP_LEFT)
        D              → TURN_RIGHT (+ Shift → SHARP_RIGHT)
        Space          → SHOOT
        No key         → COAST, GO_STRAIGHT, NO_SHOOT

    Returns:
        (3,) int tensor with [power_action, turn_action, shoot_action].
    """
    import pygame
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        power = PowerActions.BOOST
    elif keys[pygame.K_s]:
        power = PowerActions.REVERSE
    else:
        power = PowerActions.COAST

    shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
    if keys[pygame.K_a]:
        turn = TurnActions.SHARP_LEFT if shift else TurnActions.TURN_LEFT
    elif keys[pygame.K_d]:
        turn = TurnActions.SHARP_RIGHT if shift else TurnActions.TURN_RIGHT
    else:
        turn = TurnActions.GO_STRAIGHT

    shoot = ShootActions.SHOOT if keys[pygame.K_SPACE] else ShootActions.NO_SHOOT

    return torch.tensor([int(power), int(turn), int(shoot)], dtype=torch.int32)
