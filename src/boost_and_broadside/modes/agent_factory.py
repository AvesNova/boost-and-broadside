"""Agent specification resolution for interactive and collect-stats modes.

Supported specs:
    null        — human keyboard input (watch mode only)
    random      — uniform random actions every step
    scripted    — StochasticScriptedAgent
    latest      — most recently modified checkpoint under checkpoint_dir
    <path.pt>   — specific .pt checkpoint file
"""

import sys
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.agents.run_away import RunAwayAgent
from boost_and_broadside.agents.boom_zoom import BoomZoomAgent
from boost_and_broadside.agents.abreast import AbreastAgent
from boost_and_broadside.agents.reverse_turret import ReverseTurretAgent
from boost_and_broadside.agents.spiral_evader import SpiralEvaderAgent
from boost_and_broadside.agents.jouster import JousterAgent
from boost_and_broadside.agents.jinking import JinkingAgent
from boost_and_broadside.agents.team_jouster import TeamJousterAgent
from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.constants import (
    NUM_POWER_ACTIONS,
    NUM_SHOOT_ACTIONS,
    NUM_TURN_ACTIONS,
)
from boost_and_broadside.env.observation import MVPObservation, ObsKey
from boost_and_broadside.env.state import TensorState


class ResolvedAgent:
    """An agent resolved from a spec string, with mutable hidden state for policy agents."""

    def __init__(self, kind: str, agent, hidden=None):
        self.kind = kind  # "null" | "random" | "scripted" | "policy"
        self.agent = agent  # None | StochasticScriptedAgent | MVPPolicy
        self.hidden = hidden  # (1, B*N, D) float tensor, policy agents only

    def __repr__(self) -> str:
        return f"ResolvedAgent(kind={self.kind!r})"


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    """Return the path to the most recently modified .pt file under checkpoint_dir."""
    pts = sorted(Path(checkpoint_dir).glob("**/*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        sys.exit(f"Error: no checkpoint files found under '{checkpoint_dir}'.")
    return str(pts[-1])


def resolve_agent_spec(
    spec: str,
    ship_config: ShipConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    num_ships: int = 4,
) -> ResolvedAgent:
    """Resolve a spec string to a ResolvedAgent.

    Args:
        spec:           One of: null, random, scripted, latest, or a path ending in .pt.
        ship_config:    Physics constants (needed for scripted agent).
        model_config:   Policy architecture (needed for checkpoint agents).
        device:         Torch device string.
        checkpoint_dir: Root directory searched when spec is "latest".
        num_ships:      Number of ships (N) — required to size the policy's action/value heads.
    """
    if spec == "null":
        return ResolvedAgent("null", None)

    if spec == "random":
        return ResolvedAgent("random", None)

    if spec == "scripted":
        agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
        return ResolvedAgent("scripted", agent)

    if spec == "scripted_team":
        agent = StochasticScriptedAgent(
            ship_config, StochasticAgentConfig(team_target_distance_prob=(0.0, 1.0))
        )
        return ResolvedAgent("scripted", agent)

    if spec == "run_away":
        return ResolvedAgent("scripted", RunAwayAgent(ship_config))

    if spec == "boom_zoom":
        return ResolvedAgent("scripted", BoomZoomAgent(ship_config))

    if spec == "abreast":
        return ResolvedAgent("scripted", AbreastAgent(ship_config))

    if spec == "reverse_turret":
        return ResolvedAgent("scripted", ReverseTurretAgent(ship_config))

    if spec == "spiral_evader":
        return ResolvedAgent("scripted", SpiralEvaderAgent(ship_config))

    if spec == "jouster":
        return ResolvedAgent("scripted", JousterAgent(ship_config))

    if spec == "jinking":
        return ResolvedAgent("scripted", JinkingAgent(ship_config))

    if spec == "team_jouster":
        return ResolvedAgent("scripted", TeamJousterAgent(ship_config))

    # Checkpoint: "latest" or an explicit path
    if spec == "latest":
        path = find_latest_checkpoint(checkpoint_dir)
        print(f"Auto-selected checkpoint: {path}")
    else:
        path = spec
        if not Path(path).exists():
            sys.exit(f"Error: checkpoint not found: {path!r}")

    # Deferred import to avoid circular dependency
    from boost_and_broadside.models.mvp.policy import MVPPolicy
    from boost_and_broadside.train.rl.features import build_standard_coordinator

    ckpt = torch.load(path, map_location=device, weights_only=False)
    coordinator = build_standard_coordinator(ship_config)
    K = ckpt["policy_state_dict"]["value_head.3.weight"].shape[0]
    policy = MVPPolicy(model_config, coordinator, num_value_components=K, num_ships=num_ships).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    update = ckpt.get("update", "?")
    step = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint: update={update}  step={step}  path={path}")

    return ResolvedAgent("policy", policy)


def init_hidden(agent: ResolvedAgent, num_envs: int, num_tokens: int, device) -> None:
    """Allocate initial GRU hidden state for policy agents; no-op for all others."""
    if agent.kind == "policy":
        agent.hidden = agent.agent.initial_hidden(num_envs, num_tokens, device)


def get_actions(
    agent: ResolvedAgent,
    obs: MVPObservation | None,
    state: TensorState,
    num_envs: int,
    num_ships: int,
    device,
    return_pred_next: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Return (B, N, 3) int actions for every ship in the batch.

    Policy and scripted agents produce actions for all ships; the caller selects
    the relevant team's actions via a team-id mask.  For the null agent the
    returned tensor is all-zeros — the caller must apply keyboard overrides.
    
    If return_pred_next is True, also returns the predicted next state (B, N, AUX_DIM) 
    from the policy, or None if the agent doesn't predict it.
    """
    B, N = num_envs, num_ships

    if agent.kind == "random":
        action = torch.stack(
            [
                torch.randint(0, NUM_POWER_ACTIONS, (B, N), device=device),
                torch.randint(0, NUM_TURN_ACTIONS, (B, N), device=device),
                torch.randint(0, NUM_SHOOT_ACTIONS, (B, N), device=device),
            ],
            dim=-1,
        ).int()
        return (action, None) if return_pred_next else action

    if agent.kind == "scripted":
        with torch.no_grad():
            action = agent.agent.get_actions(state)
        return (action, None) if return_pred_next else action

    if agent.kind == "policy":
        with torch.no_grad():
            action, _, _, pred_next, agent.hidden = agent.agent.get_action_and_value(
                obs, agent.hidden
            )
        return (action, pred_next) if return_pred_next else action

    # null — zero placeholder; caller must override with keyboard input
    action = torch.zeros(B, N, 3, dtype=torch.int32, device=device)
    return (action, None) if return_pred_next else action


def reset_done_envs(
    agent: ResolvedAgent, done_mask: torch.Tensor, num_tokens: int
) -> None:
    """Reset GRU hidden state for completed envs; no-op for non-policy agents."""
    if agent.kind == "policy" and agent.hidden is not None:
        agent.hidden = agent.agent.reset_hidden_for_envs(
            agent.hidden, done_mask, num_tokens
        )


# ---------------------------------------------------------------------------
# Autoregressive imagination for watch-mode visualization
# ---------------------------------------------------------------------------

ALIVE_HEALTH_EPS = 1.0  # ship is dead when decoded health ≤ this value


def _decode_targets_to_obs(
    targets: torch.Tensor,
    prev_obs: MVPObservation,
    action: torch.Tensor,
    N: int,
    ship_config,
) -> MVPObservation:
    """Decode a coordinator target vector back to a raw MVPObservation.

    targets: (B, N, 15)
      pos_x(sin,cos)[0:2]  pos_y(sin,cos)[2:4]  vel(vx_norm,vy_norm)[4:6]
      att(cos,sin)[6:8]    ang_vel(symlog)[8:9]
      health(sin,cos)[9:11]  power(sin,cos)[11:13]  cooldown(sin,cos)[13:15]
    prev_obs: previous MVPObservation — obstacle tokens (N:) are copied unchanged
    action:   (B, N, 3) int — stored as PREVIOUS_ACTION for next step
    """
    import math
    _2pi = 2.0 * math.pi
    _hpi = math.pi / 2.0
    W = ship_config.world_size[0]

    # Position: Fourier(1, W) encodes as (sin(2πx/W), cos(2πx/W))
    pos_x = (torch.atan2(targets[..., 0], targets[..., 1]) % _2pi) * W / _2pi
    pos_y = (torch.atan2(targets[..., 2], targets[..., 3]) % _2pi) * W / _2pi
    pos = torch.stack([pos_x, pos_y], dim=-1)

    # Velocity: SymlogVelocity encodes as direction * symlog(speed)
    vel_enc = targets[..., 4:6]
    speed_enc = torch.norm(vel_enc, dim=-1, keepdim=True)
    direction = vel_enc / speed_enc.clamp(min=1e-8)
    speed = torch.expm1(speed_enc.clamp(min=0.0))
    vel = direction * speed

    # Attitude: Identity transform stores raw (cos, sin)
    att = targets[..., 6:8]

    # Angular velocity: Symlog encodes as symlog(ω)
    sym = targets[..., 8:9]
    ang_vel = torch.sign(sym) * torch.expm1(sym.abs())

    # Health/power/cooldown: UnitCircle encodes as (sin(angle), cos(angle))
    health_angle = torch.atan2(targets[..., 9], targets[..., 10]).clamp(0.0, _hpi)
    health = (health_angle / _hpi * ship_config.max_health).unsqueeze(-1)

    power_angle = torch.atan2(targets[..., 11], targets[..., 12]).clamp(0.0, _hpi)
    power = (power_angle / _hpi * ship_config.max_power).unsqueeze(-1)

    cd_angle = torch.atan2(targets[..., 13], targets[..., 14]).clamp(0.0, _hpi)
    cooldown = (cd_angle / _hpi * ship_config.firing_cooldown).unsqueeze(-1)

    alive = health.squeeze(-1) > ALIVE_HEALTH_EPS

    new_data = {k: v.clone() for k, v in prev_obs.items()}
    new_data[ObsKey.POS]             = torch.cat([pos,     prev_obs[ObsKey.POS][:, N:]],            dim=1)
    new_data[ObsKey.VEL]             = torch.cat([vel,     prev_obs[ObsKey.VEL][:, N:]],            dim=1)
    new_data[ObsKey.ATT]             = torch.cat([att,     prev_obs[ObsKey.ATT][:, N:]],            dim=1)
    new_data[ObsKey.ANG_VEL]         = torch.cat([ang_vel, prev_obs[ObsKey.ANG_VEL][:, N:]],       dim=1)
    new_data[ObsKey.HEALTH]          = torch.cat([health,  prev_obs[ObsKey.HEALTH][:, N:]],         dim=1)
    new_data[ObsKey.POWER]           = torch.cat([power,   prev_obs[ObsKey.POWER][:, N:]],          dim=1)
    new_data[ObsKey.COOLDOWN]        = torch.cat([cooldown,prev_obs[ObsKey.COOLDOWN][:, N:]],       dim=1)
    new_data[ObsKey.ALIVE]           = torch.cat([alive,   prev_obs[ObsKey.ALIVE][:, N:]],          dim=1)
    new_data[ObsKey.PREVIOUS_ACTION] = torch.cat([action,  prev_obs[ObsKey.PREVIOUS_ACTION][:, N:]], dim=1)
    return MVPObservation(data=new_data)


def imagine_trajectory(
    agent: ResolvedAgent,
    obs: MVPObservation,
    n_steps: int,
    num_ships: int,
    device,
    ship_config,
) -> list:
    """Autoregressive imagined rollout for watch-mode visualization.

    Clones the policy's hidden state and runs n_steps forward passes,
    feeding each predicted state back as the next observation.  The real
    agent.hidden is never modified.

    Returns:
        List of n_steps (B, N, pred_dim) tensors, or [] for non-policy agents.
    """
    if agent.kind != "policy" or agent.hidden is None or n_steps <= 0:
        return []

    coordinator = agent.agent.coordinator
    imag_hidden = agent.hidden.clone()
    imag_obs = MVPObservation(data={k: v.clone() for k, v in obs.items()})
    curr_ship_targets = coordinator.get_target_vector(imag_obs)[:, :num_ships]

    pred_nexts = []
    with torch.no_grad():
        for _ in range(n_steps):
            action, _, _, pred_next_scaled, imag_hidden = agent.agent.get_action_and_value(
                imag_obs, imag_hidden
            )
            pred_nexts.append(pred_next_scaled)
            next_ship_targets = coordinator.apply_scaled_predictions(
                curr_ship_targets, pred_next_scaled
            )
            imag_obs = _decode_targets_to_obs(
                next_ship_targets, imag_obs, action, num_ships, ship_config
            )
            curr_ship_targets = coordinator.get_target_vector(imag_obs)[:, :num_ships]

    return pred_nexts
