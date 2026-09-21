"""Interactive game modes: watch and human play.

Entry points:
  - run_play_mode: human plus scripted allies in the Gate-1 frontline prototype.
  - run_watch_mode: render live gameplay between two specified agents at 60fps.

Agent specs (--team0 / --team1) are resolved by evaluation/agents.py —
`null` maps to human keyboard control (WASD to fly, Shift for sharp turns,
Space to shoot); see that module for the full spec list.
"""

import time
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
from boost_and_broadside.env.frontline import (
    frontline_ship_config,
    scaled_frontline_geometry,
)
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
    ships_per_team: int = PLAY_ENV_CONFIG.num_ships // 2,
    num_fields: int = PLAY_ENV_CONFIG.num_fields,
) -> None:
    """Run the provisional playable Frontline preset.

    The first team-0 ship starts keyboard-controlled. T changes selected team,
    Tab cycles its living ships, H toggles control/watch, C toggles camera follow,
    V cycles vision, and Z toggles zone occlusion. U (or the Frame cap button)
    unlocks presentation while retaining the fixed decision rate.
    Tuning values are intentionally provisional pending the current human
    playtest gate.
    """
    env_config, ship_config = _frontline_interactive_config(
        ships_per_team, num_fields, frontline_ship_config(ship_config)
    )
    # A single tiny environment is dominated by CUDA launch/synchronization
    # overhead.  Larger interactive fleets keep the requested CUDA device so
    # their compiled/graph path is available.
    play_device = _interactive_device(device, env_config)
    render_config = _interactive_render_config(render_config, ship_config, env_config)
    agent0 = resolve_agent_spec(
        "scripted",
        ship_config,
        model_config,
        play_device,
        checkpoint_dir,
        num_ships=env_config.num_ships,
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
            env_config,
            rewards,
            render_config,
            play_device,
            start_human_control=True,
            state_only=True,
        )
    finally:
        torch.set_num_threads(previous_threads)


def run_watch_mode(
    team0_spec: str,
    team1_spec: str,
    ship_config: ShipConfig,
    rewards: RewardConfig,
    model_config: ModelConfig,
    render_config: RenderConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    ships_per_team: int = PLAY_ENV_CONFIG.num_ships // 2,
    num_fields: int = PLAY_ENV_CONFIG.num_fields,
) -> None:
    """Run the shared Frontline developer mode, initially watching two agents.

    Args:
        team0_spec:     Exact agent name or checkpoint path for team 0.
        team1_spec:     Agent spec for team 1.
        ship_config:    Physics constants.
        ships_per_team: Ships assigned to each team in the shared Frontline map.
        num_fields:     Number of fields in the shared Frontline map.
        rewards:        Reward weights (used to build the env wrapper).
        model_config:   Policy architecture (needed if either spec is a checkpoint).
        render_config:  Display settings.
        device:         Torch device string.
        checkpoint_dir: Checkpoint root supplied by the CLI adapter.
    """
    ship_config = frontline_ship_config(ship_config)
    env_config, ship_config = _frontline_interactive_config(
        ships_per_team, num_fields, ship_config
    )
    render_config = _interactive_render_config(render_config, ship_config, env_config)
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

    _run_resolved_interactive_mode(
        agent0,
        agent1,
        ship_config,
        env_config,
        rewards,
        render_config,
        device,
    )


def _run_resolved_interactive_mode(
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    rewards: RewardConfig,
    render_config: RenderConfig,
    device: str,
    start_human_control: bool = False,
    state_only: bool = False,
) -> None:
    """Build the single environment and render two already-resolved agents."""

    renderer = GameRenderer(
        ship_config, replace(render_config, zone_occlusion=env_config.zones_occlude)
    )
    renderer.configure_interaction(human_control=start_human_control)
    cuda_interactive = torch.device(device).type == "cuda"

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
        # CUDA play/watch uses the parity-checked fast path by default: graph
        # replay for fixed-shape physics and pure compiled perception.  CPU
        # play remains eager because CUDA graph capture is unavailable there.
        interactive_cuda_graph=cuda_interactive,
        interactive_perception_compile_mode="default" if cuda_interactive else None,
    )

    try:
        _run_interactive_loop(
            wrapper,
            agent0,
            agent1,
            renderer,
            torch.device(device),
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
        terminal_until: float | None = None

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
            if (
                not renderer.paused
                and terminal_until is None
                and renderer.simulation_due()
            ):
                state = wrapper.state
                visibility = (
                    team_visibility_from_state(state, wrapper.ship_config, wrapper.env_config)
                    if state_only
                    else wrapper.last_visibility
                )

                living_by_team = tuple(
                    tuple(
                        index
                        for index, team in enumerate(state.ship_team_id[0].tolist())
                        if team == selected_team and bool(state.ship_alive[0, index].item())
                    )
                    for selected_team in range(2)
                )
                renderer.set_living_ships(living_by_team)

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
                human_control_mask = _selected_human_mask(
                    team_id, renderer.human_control_enabled, renderer.selected_ship
                )
                if bool(human_control_mask.any().item()):
                    decided_action = _apply_keyboard_override(
                        decided_action,
                        _decode_keyboard().to(device),
                        renderer.selected_ship,
                    )
                action, policy_action_buffer = _apply_policy_action_delay(
                    decided_action,
                    policy_action_buffer,
                    team_id,
                    policy_teams,
                    immediate_mask=human_control_mask,
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
                    obs, dones, truncated, info = wrapper.step_interactive(
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
                    # Keep the result readable for one wall-clock second even
                    # with presentation unlocked.
                    terminal_until = time.perf_counter() + 1.0
                else:
                    # Exactly one decision/physics tick has occurred.  The
                    # renderer can now present this immutable snapshot at any
                    # rate without changing action-delay semantics.
                    renderer.mark_simulation_advanced()

            if terminal_until is not None:
                running = renderer.render_with_label(
                    wrapper.state, terminal_label or "", visibility=visibility
                )
            else:
                running = renderer.render(
                    wrapper.state, pred_nexts=pred_nexts, visibility=visibility
                )
            if not running:
                return
            renderer.tick()
            if terminal_until is not None and time.perf_counter() >= terminal_until:
                break


def _apply_policy_action_delay(
    decided_action: torch.Tensor,
    policy_action_buffer: torch.Tensor,
    team_id: torch.Tensor,
    policy_teams: frozenset[int],
    *,
    immediate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply buffered NN actions and immediate non-NN actions for one tick.

    Returns the action physics should consume now and the policy buffer for the
    next tick. Non-policy entries in the returned buffer are kept at zero so a
    later controller change cannot expose stale actions.
    """

    policy_mask = _policy_team_mask(team_id, policy_teams)
    if immediate_mask is not None:
        policy_mask &= ~immediate_mask
    policy_mask = policy_mask.unsqueeze(-1)
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
    keyboard: torch.Tensor,
    selected_ship: int | None,
) -> torch.Tensor:
    """Replace one selected ship action with immediate keyboard input."""
    keyboard_mask = torch.zeros(action.shape[:2], dtype=torch.bool, device=action.device)
    if selected_ship is not None and 0 <= selected_ship < action.shape[1]:
        keyboard_mask[:, selected_ship] = True
    return torch.where(keyboard_mask.unsqueeze(-1), keyboard.view(1, 1, 3), action)


def _selected_human_mask(
    team_id: torch.Tensor, human_control: bool, selected_ship: int | None
) -> torch.Tensor:
    """Return the one slot temporarily driven by the human, if any."""

    mask = torch.zeros_like(team_id, dtype=torch.bool)
    if human_control and selected_ship is not None and 0 <= selected_ship < team_id.shape[1]:
        mask[:, selected_ship] = True
    return mask


def _frontline_interactive_config(
    ships_per_team: int, num_fields: int, ship_config: ShipConfig
) -> tuple[EnvConfig, ShipConfig]:
    """Keep play and watch on Frontline rules, on a density-matched map.

    ``PLAY_ENV_CONFIG`` states the 5v5 reference geometry; the requested fleet
    resizes it so a 50v50 match has the same ships per unit area, zone area per
    ship and field area per ship that 5v5 was tuned for.
    """

    num_ships = 2 * ships_per_team
    scaled_ship_config, scaled_frontline = scaled_frontline_geometry(
        ship_config, PLAY_ENV_CONFIG.frontline, num_ships
    )
    return (
        replace(
            PLAY_ENV_CONFIG,
            num_ships=num_ships,
            num_fields=num_fields,
            frontline=scaled_frontline,
        ),
        scaled_ship_config,
    )


def _interactive_device(requested_device: str, env_config: EnvConfig) -> str:
    """Avoid CUDA launch overhead for the small default, retain it for large fleets."""

    return requested_device if env_config.num_ships > PLAY_ENV_CONFIG.num_ships else "cpu"


def _interactive_render_config(
    render_config: RenderConfig, ship_config: ShipConfig, env_config: EnvConfig
) -> RenderConfig:
    """Apply the shared Frontline dev UI and fixed decision cadence."""

    decision_fps = round(1.0 / (ship_config.dt * env_config.action_repeat))
    return replace(
        render_config,
        fps=decision_fps,
        show_unlimited_button=True,
        show_frame_pacing_toggle=True,
        vision_mode=VisionMode.FULL,
    )


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
