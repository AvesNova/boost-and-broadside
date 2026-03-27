"""Interactive game modes: watch and human play.

Entry point:
  - run_watch_mode: render live gameplay between two specified agents at 60fps.

Supported agent specs (--team0 / --team1):
    null        — human keyboard (WASD + Space)
    random      — uniform random actions
    scripted    — StochasticScriptedAgent
    latest      — most recently modified checkpoint
    <path.pt>   — specific checkpoint file
"""

import torch

from boost_and_broadside.config import ShipConfig, EnvConfig, ModelConfig, RewardConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.wrapper import MVPEnvWrapper
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig


def run_watch_mode(
    team0_spec:    str,
    team1_spec:    str,
    ship_config:   ShipConfig,
    env_config:    EnvConfig,
    reward_config: RewardConfig,
    model_config:  ModelConfig,
    render_config: RenderConfig,
    device:        str,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Render live gameplay between two agents at 60fps.

    Args:
        team0_spec:     Agent spec for team 0 (null, random, scripted, latest, or path.pt).
        team1_spec:     Agent spec for team 1.
        ship_config:    Physics constants.
        env_config:     Environment sizing.
        reward_config:  Reward weights (needed to build the env wrapper).
        model_config:   Policy architecture (needed if either spec is a checkpoint).
        render_config:  Display settings.
        device:         Torch device string.
        checkpoint_dir: Root directory searched when a spec is "latest".
    """
    agent0 = resolve_agent_spec(team0_spec, ship_config, model_config, device, checkpoint_dir)
    agent1 = resolve_agent_spec(team1_spec, ship_config, model_config, device, checkpoint_dir)

    wrapper  = MVPEnvWrapper(num_envs=1, ship_config=ship_config, env_config=env_config,
                             reward_config=reward_config, device=device)
    renderer = GameRenderer(ship_config, render_config)

    try:
        _run_interactive_loop(wrapper, agent0, agent1, renderer, torch.device(device))
    finally:
        renderer.close()


def _run_interactive_loop(
    wrapper:  MVPEnvWrapper,
    agent0:   ResolvedAgent,
    agent1:   ResolvedAgent,
    renderer: GameRenderer,
    device:   torch.device,
) -> None:
    """Core render loop.  Runs episodes back-to-back until the window is closed.

    Args:
        wrapper:  Single-env MVPEnvWrapper (num_envs=1).
        agent0:   Agent controlling team-0 ships.
        agent1:   Agent controlling team-1 ships.
        renderer: Pygame renderer.
        device:   Torch device.
    """
    N = wrapper.num_ships

    while True:
        obs = wrapper.reset()
        init_hidden(agent0, 1, N, device)
        init_hidden(agent1, 1, N, device)

        while True:
            state = wrapper.state

            action0 = get_actions(agent0, obs, state, 1, N, device)
            action1 = get_actions(agent1, obs, state, 1, N, device)

            # Select each agent's actions for their respective team
            team_id = obs["team_id"]  # (1, N)
            action  = torch.where((team_id == 0).unsqueeze(-1), action0, action1)

            # Human keyboard overrides for null agents
            if agent0.kind == "null" or agent1.kind == "null":
                keyboard = _decode_keyboard().to(device)
                for ship_idx in range(N):
                    t = int(team_id[0, ship_idx].item())
                    if (t == 0 and agent0.kind == "null") or (t == 1 and agent1.kind == "null"):
                        action[0, ship_idx] = keyboard

            obs, _, dones, truncated, _ = wrapper.step(action)

            if (dones | truncated).any():
                reset_done_envs(agent0, dones | truncated, N)
                reset_done_envs(agent1, dones | truncated, N)
                obs = wrapper.reset()

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
