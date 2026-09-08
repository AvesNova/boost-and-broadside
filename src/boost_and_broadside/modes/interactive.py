"""Interactive game modes: watch and human play.

Entry points:
  - run_play_mode: human plus scripted allies in the Gate-1 frontline prototype.
  - run_watch_mode: render live gameplay between two specified agents at 60fps.

Agent specs (--team0 / --team1) are resolved by evaluation/agents.py —
`null` maps to human keyboard control (WASD to fly, Shift for sharp turns,
Space to shoot); see that module for the full spec list.
"""

from dataclasses import replace

import torch

from boost_and_broadside.config import (
    EnvConfig,
    FieldMapConfig,
    FrontlineConfig,
    MatchResult,
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
from boost_and_broadside.env.frontline import FRONTLINE_WORLD_SIZE
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.evaluation.agents import (
    ResolvedAgent,
    agents_read_bullets,
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.evaluation.environment import (
    create_evaluation_field_map,
    resolve_evaluation_environment,
)
from boost_and_broadside.evaluation.match import agent_view, merge_team_actions
from boost_and_broadside.evaluation.next_state import imagine_trajectory
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig

_PLAY_ZONE_RADIUS = 330.0
_PLAY_ZONE_RING_RADIUS = 1200.0

PLAY_ENV_CONFIG = EnvConfig(
    num_ships=8,
    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
    max_episode_steps=18_000,
    num_fields=0,
    action_repeat=2,
    spawn_resource_spread=0.0,
    frontline=FrontlineConfig(
        zone_radius=_PLAY_ZONE_RADIUS,
        zone_ring_radius=_PLAY_ZONE_RING_RADIUS,
        playable_radius=2600.0,
        capture_seconds=20.0,
        defense_damage_per_second=2.0,
        respawn_health=25.0,
        spawn_heal_per_second=12.0,
        enemy_spawn_damage_per_second=8.0,
        boundary_damage_per_second=5.0,
        boundary_damage_per_pixel_second=0.05,
        front_win_threshold=5,
    ),
)


def run_play_mode(
    ship_config: ShipConfig,
    rewards: RewardConfig,
    model_config: ModelConfig,
    render_config: RenderConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Run the provisional playable Gate-1 frontline preset.

    One selected blue ship is keyboard-controlled; the remaining blue ships and
    all red ships use the crude frontline scripted controller. Tab cycles the
    human ship and C toggles camera follow. Tuning values are intentionally
    provisional pending the Gate-1 playtest.
    """
    ship_config = replace(ship_config, world_size=FRONTLINE_WORLD_SIZE)
    render_config = replace(render_config, show_unlimited_button=True)
    agent0 = resolve_agent_spec(
        "scripted",
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=PLAY_ENV_CONFIG.num_ships,
    )
    agent1 = resolve_agent_spec(
        "scripted",
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
) -> None:
    """Render live gameplay between two agents at 60fps.

    Args:
        team0_spec:     Exact agent name or checkpoint path for team 0.
        team1_spec:     Agent spec for team 1.
        ship_config:    Physics constants.
        env_config:     Environment sizing.
        rewards:        Reward weights (used to build the env wrapper).
        model_config:   Policy architecture (needed if either spec is a checkpoint).
        render_config:  Display settings.
        device:         Torch device string.
        checkpoint_dir: Checkpoint root supplied by the CLI adapter.
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
    env_config, field_map_config = resolve_evaluation_environment(
        env_config,
        (agent0, agent1),
        ship_config=ship_config,
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
        field_map_config=field_map_config,
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
    field_map_config: FieldMapConfig | None = None,
) -> None:
    """Build the single environment and render two already-resolved agents."""

    renderer = GameRenderer(ship_config, render_config)

    # Static maps need no orbital settling phase.
    field_map = None
    if env_config.num_fields > 0:
        field_map = create_evaluation_field_map(
            ship_config,
            env_config,
            field_map_config or FieldMapConfig(cache_size=1, max_generation_attempts=256),
            torch.device(device),
        )

    wrapper = YemongEnvWrapper(
        num_envs=1,
        ship_config=ship_config,
        env_config=env_config,
        rewards=rewards,
        device=device,
        field_map=field_map,
        include_bullets=agents_read_bullets(agent0, agent1),
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
    N_IMAGINE_STEPS = 0

    N = wrapper.num_ships
    M = wrapper.env_config.num_fields
    num_tokens = N + M

    first_episode = True
    while True:
        obs = wrapper.reset()
        init_hidden(agent0, 1, num_tokens, device)
        init_hidden(agent1, 1, num_tokens, device)
        pred_nexts = None
        terminal_label: str | None = None
        terminal_frames = 0

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
            if not renderer.paused and terminal_frames == 0:
                state = wrapper.state

                controllable = tuple(
                    index
                    for index, team in enumerate(state.ship_team_id[0].tolist())
                    if team in keyboard_teams and bool(state.ship_alive[0, index].item())
                )
                renderer.set_selectable_ships(controllable)

                team0_view = agent_view(
                    agent0,
                    obs,
                    N,
                    torch.zeros(1, dtype=torch.bool, device=device),
                )
                team1_view = agent_view(
                    agent1,
                    obs,
                    N,
                    torch.ones(1, dtype=torch.bool, device=device),
                )

                # Imagined trajectories use the hidden state BEFORE the real forward pass.
                imag_nexts0 = imagine_trajectory(agent0, team0_view, N_IMAGINE_STEPS, N, device)
                imag_nexts1 = imagine_trajectory(agent1, team1_view, N_IMAGINE_STEPS, N, device)

                action0, _ = get_actions(
                    agent0, team0_view, state, 1, N, device, return_pred_next=True
                )
                action1, _ = get_actions(
                    agent1, team1_view, state, 1, N, device, return_pred_next=True
                )

                # Select each agent's actions for their respective team (ship tokens only)
                team_id = obs["team_id"][:, :N]  # (1, N) — exclude field tokens
                action = merge_team_actions(action0, action1, team_id)

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
                    action = _apply_keyboard_override(
                        action,
                        team_id,
                        keyboard,
                        keyboard_teams,
                        renderer.selected_ship,
                    )

                obs, _, dones, truncated, info = wrapper.step(
                    action,
                    unlimited_resources=renderer.unlimited_resources,
                    auto_reset=False,
                )

                if (dones | truncated).any():
                    reset_done_envs(agent0, dones | truncated, num_tokens)
                    reset_done_envs(agent1, dones | truncated, num_tokens)
                    pred_nexts = None
                    result = int(info["match_result"][0].item())
                    terminal_label = {
                        int(MatchResult.TEAM0_WIN): "TEAM 0 WINS",
                        int(MatchResult.TEAM1_WIN): "TEAM 1 WINS",
                        int(MatchResult.DRAW): "DRAW",
                    }[result]
                    terminal_frames = renderer.target_fps

            if terminal_frames > 0:
                running = renderer.render_with_label(wrapper.state, terminal_label or "")
                terminal_frames -= 1
            else:
                running = renderer.render(wrapper.state, pred_nexts=pred_nexts)
            if not running:
                return
            renderer.tick()
            if terminal_label is not None and terminal_frames == 0:
                break


def _apply_keyboard_override(
    action: torch.Tensor,
    team_id: torch.Tensor,
    keyboard: torch.Tensor,
    keyboard_teams: frozenset[int],
    selected_ship: int | None,
) -> torch.Tensor:
    """Replace only the selected eligible ship action with keyboard input."""
    keyboard_mask = torch.zeros_like(team_id, dtype=torch.bool)
    if selected_ship is not None and 0 <= selected_ship < team_id.shape[1]:
        eligible = any(
            bool((team_id[:, selected_ship] == team).all().item()) for team in keyboard_teams
        )
        if eligible:
            keyboard_mask[:, selected_ship] = True
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
