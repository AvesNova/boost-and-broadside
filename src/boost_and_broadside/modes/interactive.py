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
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.evaluation.agents import (
    ResolvedAgent,
    agents_read_bullets,
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.evaluation.environment import resolve_evaluation_environment
from boost_and_broadside.evaluation.match import agent_view, merge_team_actions
from boost_and_broadside.evaluation.next_state import imagine_trajectory
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig, VisionMode

_PLAY_ZONE_RADIUS = 330.0
_PLAY_ZONE_RING_RADIUS = 1200.0

PLAY_ENV_CONFIG = EnvConfig(
    num_ships=10,
    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
    max_episode_steps=9_000,
    # Ten low-discrepancy fields with the larger Frontline radius distribution
    # preserve slightly more nominal coverage than the earlier 20 × 490 px
    # uniform proposal while halving physics work and policy-map tokens.
    num_fields=10,
    action_repeat=1,
    spawn_resource_spread=0.0,
    # Adjacent objectives remain mutually scoutable, while the opposite side of
    # the playable disk does not.
    vision_range=1024.0,
    zones_occlude=True,
    frontline=FrontlineConfig(
        zone_radius=_PLAY_ZONE_RADIUS,
        zone_ring_radius=_PLAY_ZONE_RING_RADIUS,
        playable_radius=2600.0,
        capture_seconds=10.0,
        respawn_health=15.0,
        respawn_power=40.0,
        respawn_speed=30.0,
        shield_recharge_delay=5.0,
        shield_recharge_per_second=15.0,
        boundary_damage_per_second=5.0,
        boundary_damage_per_pixel_second=0.05,
        front_win_threshold=3,
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
    """Run the provisional playable Frontline preset.

    One selected blue ship is keyboard-controlled; the remaining blue ships and
    all red ships use the crude frontline scripted controller. Tab cycles the
    human ship, C toggles camera follow, V cycles whose vision is drawn, and Z
    toggles whether capture zones block sight as well as fields. Tuning values
    are intentionally provisional pending the current human playtest gate.
    """
    # A single tiny environment is dominated by CUDA launch/synchronization
    # overhead. Play is scripted/human-only, so keep its simulation and agents
    # on CPU even when training defaults to CUDA.
    play_device = "cpu"
    ship_config = frontline_ship_config(ship_config)
    # A policy decision holds for action_repeat physics ticks. The Frontline
    # contract is 30 Hz with repeat one, hence 30 rendered decisions per second.
    decision_fps = round(1.0 / (ship_config.dt * PLAY_ENV_CONFIG.action_repeat))
    render_config = replace(
        render_config,
        fps=decision_fps,
        show_unlimited_button=True,
        vision_mode=VisionMode.TEAM_0,
    )
    agent0 = resolve_agent_spec(
        "scripted",
        ship_config,
        model_config,
        play_device,
        checkpoint_dir,
        num_ships=PLAY_ENV_CONFIG.num_ships,
    )
    # One controller draws independent per-ship tendencies for both teams. It
    # can therefore supply both sides without doing the same state analysis twice.
    agent1 = agent0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_resolved_interactive_mode(
            agent0,
            agent1,
            ship_config,
            PLAY_ENV_CONFIG,
            rewards,
            render_config,
            play_device,
            keyboard_teams=frozenset({0}),
            state_only=True,
        )
    finally:
        torch.set_num_threads(previous_threads)


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
    env_config = resolve_evaluation_environment(
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
    state_only: bool = False,
) -> None:
    """Build the single environment and render two already-resolved agents."""

    renderer = GameRenderer(
        ship_config, replace(render_config, zone_occlusion=env_config.zones_occlude)
    )

    wrapper = YemongEnvWrapper(
        num_envs=1,
        ship_config=ship_config,
        env_config=env_config,
        rewards=rewards,
        device=device,
        include_bullets=agents_read_bullets(agent0, agent1),
        # The renderer draws projectiles from the perception masks whether or
        # not the policies read bullet tokens, so ask for them explicitly.
        perceive_bullets=True,
    )

    try:
        _run_interactive_loop(
            wrapper,
            agent0,
            agent1,
            renderer,
            torch.device(device),
            keyboard_teams,
            state_only=state_only,
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
    *,
    state_only: bool = False,
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
    policy_teams = frozenset(
        team for team, agent in enumerate((agent0, agent1)) if agent.kind == "policy"
    )

    first_episode = True
    while True:
        if state_only:
            wrapper.env.reset()
            obs = None
            visibility = team_visibility_from_state(
                wrapper.state, wrapper.ship_config, wrapper.env_config
            )
        else:
            obs = wrapper.reset()
            visibility = wrapper.last_visibility
        init_hidden(agent0, 1, device)
        init_hidden(agent1, 1, device)
        pred_nexts = None
        # NN policies were trained with a one-decision actuator delay. Scripted,
        # random, and human controllers remain immediate. The buffer starts at
        # the neutral action on every episode, matching PPO rollout collection.
        policy_action_buffer = torch.zeros(
            (1, N, 3), dtype=torch.int32, device=device
        )
        terminal_label: str | None = None
        terminal_frames = 0

        # Show "Match starting!" for half a second on the first episode so the
        # user can see the reloaded snapshot before agents begin moving.
        if first_episode and M > 0:
            first_episode = False
            for _ in range(renderer._render_config.fps // 2):
                if not renderer.render_with_label(
                    wrapper.state,
                    "Reloading from snapshot",
                    color=(180, 180, 255),
                    visibility=visibility,
                ):
                    return
                renderer.tick()

        while True:
            # Zone occlusion is an environment rule, not a display filter, so
            # the Z key has to reach the environment the agents perceive.
            if renderer.zone_occlusion != wrapper.env_config.zones_occlude:
                wrapper.env_config = replace(
                    wrapper.env_config, zones_occlude=renderer.zone_occlusion
                )
            if not renderer.paused and terminal_frames == 0:
                state = wrapper.state
                visibility = (
                    team_visibility_from_state(state, wrapper.ship_config, wrapper.env_config)
                    if state_only
                    else wrapper.last_visibility
                )

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

                # Select each agent's actions for their respective team (ship tokens only)
                team_id = (
                    state.ship_team_id
                    if obs is None
                    else obs["team_id"][:, :N]  # (1, N) — exclude field tokens
                )
                action0, _ = get_actions(
                    agent0,
                    team0_view,
                    state,
                    1,
                    N,
                    device,
                    return_pred_next=True,
                    team_visibility=visibility.ship,
                )
                if agent1 is agent0:
                    action1 = action0
                else:
                    action1, _ = get_actions(
                        agent1,
                        team1_view,
                        state,
                        1,
                        N,
                        device,
                        return_pred_next=True,
                        team_visibility=visibility.ship,
                    )
                decided_action = merge_team_actions(action0, action1, team_id).int()
                if keyboard_teams:
                    decided_action = _apply_keyboard_override(
                        decided_action,
                        team_id,
                        _decode_keyboard().to(device),
                        keyboard_teams,
                        renderer.selected_ship,
                    )
                action, policy_action_buffer = _apply_policy_action_delay(
                    decided_action,
                    policy_action_buffer,
                    team_id,
                    policy_teams,
                )
                if state_only:
                    dones, truncated = wrapper.env.step(
                        action,
                        unlimited_resources=renderer.unlimited_resources,
                    )
                    result_tensor = wrapper.state.match_result
                    visibility = team_visibility_from_state(
                        wrapper.state, wrapper.ship_config, wrapper.env_config
                    )
                else:
                    obs, _, dones, truncated, info = wrapper.step(
                        action,
                        unlimited_resources=renderer.unlimited_resources,
                        auto_reset=False,
                    )

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

                if not state_only:
                    # Physics records the action it just consumed. For an NN
                    # ship, the policy state instead includes the newly queued
                    # action that will be consumed next tick, as it does in PPO.
                    _set_observation_previous_action(obs, decided_action, N)
                    result_tensor = info["match_result"]
                    visibility = wrapper.last_visibility

                if (dones | truncated).any():
                    policy_action_buffer.zero_()
                    reset_done_envs(agent0, dones | truncated)
                    reset_done_envs(agent1, dones | truncated)
                    pred_nexts = None
                    result = int(result_tensor[0].item())
                    terminal_label = {
                        int(MatchResult.TEAM0_WIN): "TEAM 0 WINS",
                        int(MatchResult.TEAM1_WIN): "TEAM 1 WINS",
                        int(MatchResult.DRAW): "DRAW",
                    }[result]
                    terminal_frames = renderer.target_fps

            if terminal_frames > 0:
                running = renderer.render_with_label(
                    wrapper.state, terminal_label or "", visibility=visibility
                )
                terminal_frames -= 1
            else:
                running = renderer.render(
                    wrapper.state, pred_nexts=pred_nexts, visibility=visibility
                )
            if not running:
                return
            renderer.tick()
            if terminal_label is not None and terminal_frames == 0:
                break


def _apply_policy_action_delay(
    decided_action: torch.Tensor,
    policy_action_buffer: torch.Tensor,
    team_id: torch.Tensor,
    policy_teams: frozenset[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply buffered NN actions and immediate non-NN actions for one tick.

    Returns the action physics should consume now and the policy buffer for the
    next tick. Non-policy entries in the returned buffer are kept at zero so a
    later controller change cannot expose stale actions.
    """

    policy_mask = _policy_team_mask(team_id, policy_teams).unsqueeze(-1)
    applied_action = torch.where(policy_mask, policy_action_buffer, decided_action)
    next_buffer = torch.where(policy_mask, decided_action, torch.zeros_like(decided_action))
    return applied_action, next_buffer


def _policy_team_mask(
    team_id: torch.Tensor, policy_teams: frozenset[int]
) -> torch.Tensor:
    """Return the ship mask controlled by delayed neural-network policies."""

    policy_mask = torch.zeros_like(team_id, dtype=torch.bool)
    for team in policy_teams:
        policy_mask |= team_id == team
    return policy_mask


def _set_observation_previous_action(
    observation: YemongObservation,
    decided_action: torch.Tensor,
    num_ships: int,
) -> None:
    """Expose the current decision, including queued NN actions, to the next view."""

    observation.data[ObsKey.PREVIOUS_ACTION][:, :num_ships].copy_(decided_action)
    if observation.team1_data is not None:
        observation.team1_data[ObsKey.PREVIOUS_ACTION][:, :num_ships].copy_(decided_action)


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
