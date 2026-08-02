"""Interactive game modes: watch and human play.

Entry points:
  - run_play_mode: fixed 1v1 player-vs-null match with four fields.
  - run_watch_mode: render live gameplay between two specified agents at 60fps.

Agent specs (--team0 / --team1) are resolved by modes/agent_factory.py —
`null` maps to human keyboard control (WASD to fly, Shift for sharp turns,
Space to shoot); see that module for the full spec list.
"""

from dataclasses import replace

import torch

from boost_and_broadside.config import (
    EnvConfig,
    FieldMapConfig,
    ModelConfig,
    RewardConfig,
    ShipConfig,
)
from boost_and_broadside.constants import (
    DEFAULT_MAX_BULLETS_PER_SHIP,
    PowerActions,
    ShootActions,
    TurnActions,
)
from boost_and_broadside.env.field_cache import FieldMapCache
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
    imagine_trajectory,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig

PLAY_ENV_CONFIG = EnvConfig(
    num_ships=2,
    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
    max_episode_steps=None,
    num_fields=4,
)


def run_play_mode(
    ship_config: ShipConfig,
    rewards: RewardConfig,
    model_config: ModelConfig,
    render_config: RenderConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Run the fixed player-vs-null game preset.

    The player controls the blue team-0 ship. The red team-1 ship receives the
    null (all-zero) action. Matches contain four static refractive fields, have
    no time horizon, and restart automatically as soon as either ship dies.
    An on-screen button toggles unlimited health and power for both ships.
    """
    render_config = replace(render_config, show_unlimited_button=True)
    agent0 = resolve_agent_spec(
        "null",
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=PLAY_ENV_CONFIG.num_ships,
    )
    agent1 = resolve_agent_spec(
        "null",
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=PLAY_ENV_CONFIG.num_ships,
    )
    _run_resolved_interactive_mode(
        agent0,
        agent1,
        ship_config,
        PLAY_ENV_CONFIG,
        rewards,
        render_config,
        device,
        keyboard_teams=frozenset({0}),
    )


def run_watch_mode(
    team0_spec: str,
    team1_spec: str,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    rewards: RewardConfig,
    model_config: ModelConfig,
    render_config: RenderConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    fast_cache: bool = False,
) -> None:
    """Render live gameplay between two agents at 60fps.

    Args:
        team0_spec:     Agent spec for team 0 (null, random, scripted, latest, or path.pt).
        team1_spec:     Agent spec for team 1.
        ship_config:    Physics constants.
        env_config:     Environment sizing.
        rewards:        Reward weights (used to build the env wrapper).
        model_config:   Policy architecture (needed if either spec is a checkpoint).
        render_config:  Display settings.
        device:         Torch device string.
        checkpoint_dir: Root directory searched when a spec is "latest".
    """
    agent0 = resolve_agent_spec(
        team0_spec,
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=env_config.num_ships,
    )
    agent1 = resolve_agent_spec(
        team1_spec,
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=env_config.num_ships,
    )

    keyboard_teams = frozenset(
        team for team, agent in enumerate((agent0, agent1)) if agent.kind == "null"
    )
    _run_resolved_interactive_mode(
        agent0,
        agent1,
        ship_config,
        env_config,
        rewards,
        render_config,
        device,
        keyboard_teams=keyboard_teams,
    )


def _run_resolved_interactive_mode(
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    rewards: RewardConfig,
    render_config: RenderConfig,
    device: str,
    keyboard_teams: frozenset[int],
) -> None:
    """Build the single environment and render two already-resolved agents."""

    renderer = GameRenderer(ship_config, render_config)

    # Static maps need no orbital settling phase. ``fast_cache`` is retained in
    # the watch-mode signature for CLI compatibility but no longer changes map
    # construction.
    field_map = None
    if env_config.num_fields > 0:
        field_map = FieldMapCache.generate(
            ship_config,
            env_config,
            FieldMapConfig(cache_size=1, max_generation_attempts=256),
            torch.device(device),
        )

    wrapper = YemongEnvWrapper(
        num_envs=1,
        ship_config=ship_config,
        env_config=env_config,
        rewards=rewards,
        device=device,
        field_map=field_map,
    )

    try:
        _run_interactive_loop(
            wrapper,
            agent0,
            agent1,
            renderer,
            torch.device(device),
            keyboard_teams,
        )
    finally:
        renderer.close()


def _run_interactive_loop(
    wrapper: YemongEnvWrapper,
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    renderer: GameRenderer,
    device: torch.device,
    keyboard_teams: frozenset[int],
) -> None:
    """Core render loop.  Runs episodes back-to-back until the window is closed.

    Args:
        wrapper:  Single-env YemongEnvWrapper (num_envs=1).
        agent0:   Agent controlling team-0 ships.
        agent1:   Agent controlling team-1 ships.
        renderer: Pygame renderer.
        device:   Torch device.
        keyboard_teams: Team IDs whose selected actions are replaced by keyboard input.
    """
    N_IMAGINE_STEPS = 12

    N = wrapper.num_ships
    M = wrapper.env_config.num_fields
    num_tokens = N + M

    first_episode = True
    while True:
        obs = wrapper.reset()
        init_hidden(agent0, 1, num_tokens, device)
        init_hidden(agent1, 1, num_tokens, device)
        pred_nexts = None

        # Show "Match starting!" for half a second on the first episode so the
        # user can see the reloaded snapshot before agents begin moving.
        if first_episode and M > 0:
            first_episode = False
            for _ in range(renderer._render_config.fps // 2):
                if not renderer.render_with_label(
                    wrapper.state, "Reloading from snapshot", color=(180, 180, 255)
                ):
                    return
                renderer.tick()

        while True:
            if not renderer.paused:
                state = wrapper.state

                # Imagined trajectories use the hidden state BEFORE the real forward pass.
                imag_nexts0 = imagine_trajectory(agent0, obs, N_IMAGINE_STEPS, N, device)
                imag_nexts1 = imagine_trajectory(agent1, obs, N_IMAGINE_STEPS, N, device)

                action0, _ = get_actions(agent0, obs, state, 1, N, device, return_pred_next=True)
                action1, _ = get_actions(agent1, obs, state, 1, N, device, return_pred_next=True)

                # Select each agent's actions for their respective team (ship tokens only)
                team_id = obs["team_id"][:, :N]  # (1, N) — exclude field tokens
                action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)

                # Merge imagined trajectories by team into a single list of per-step tensors.
                pred_nexts = None
                if imag_nexts0 or imag_nexts1:
                    n_steps = max(len(imag_nexts0), len(imag_nexts1))
                    mask = (team_id == 0).unsqueeze(-1)  # (1, N, 1)
                    merged = []
                    for k in range(n_steps):
                        pn0 = (
                            imag_nexts0[k]
                            if k < len(imag_nexts0)
                            else torch.zeros_like(imag_nexts1[k])
                        )
                        pn1 = (
                            imag_nexts1[k]
                            if k < len(imag_nexts1)
                            else torch.zeros_like(imag_nexts0[k])
                        )
                        merged.append(torch.where(mask, pn0, pn1))
                    pred_nexts = merged

                if keyboard_teams:
                    keyboard = _decode_keyboard().to(device)
                    action = _apply_keyboard_override(action, team_id, keyboard, keyboard_teams)

                obs, _, dones, truncated, _ = wrapper.step(
                    action,
                    unlimited_resources=renderer.unlimited_resources,
                )

                if (dones | truncated).any():
                    reset_done_envs(agent0, dones | truncated, num_tokens)
                    reset_done_envs(agent1, dones | truncated, num_tokens)
                    pred_nexts = None

            running = renderer.render(wrapper.state, pred_nexts=pred_nexts)
            if not running:
                return
            renderer.tick()


def _apply_keyboard_override(
    action: torch.Tensor,
    team_id: torch.Tensor,
    keyboard: torch.Tensor,
    keyboard_teams: frozenset[int],
) -> torch.Tensor:
    """Replace actions for only the explicitly player-controlled teams."""
    keyboard_mask = torch.zeros_like(team_id, dtype=torch.bool)
    for team in keyboard_teams:
        keyboard_mask |= team_id == team
    return torch.where(keyboard_mask.unsqueeze(-1), keyboard.view(1, 1, 3), action)


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
