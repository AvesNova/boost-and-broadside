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
    from boost_and_broadside.config.obs_spec import obs_config_from_dict

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "obs_config" not in ckpt:
        raise KeyError(f"Checkpoint {path!r} has no 'obs_config' key — retrain from scratch or provide obs_config manually.")
    obs_config = obs_config_from_dict(ckpt["obs_config"])
    K = ckpt["policy_state_dict"]["value_head.3.weight"].shape[0]
    policy = MVPPolicy(model_config, obs_config, num_value_components=K, num_ships=num_ships).to(device)
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
    obs: dict[str, torch.Tensor] | None,
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

def _extract_aux_cont(obs: dict, N: int, ship_config) -> torch.Tensor:
    """Encode raw obs into AUX continuous state (B, N, 16).

    Matches the layout used by _apply_aux_deltas_local:
      pos_x(sin,cos) pos_y(sin,cos) vel_dir(cos,sin) symlog_speed(1) att(cos,sin) ang_vel(1)
      health(sin,cos) power(sin,cos) cooldown(sin,cos)
    """
    import math
    _2pi = 2.0 * math.pi
    _hpi = math.pi / 2.0
    W = ship_config.world_size[0]

    pos_norm = obs["pos"][:, :N].float() / W                              # (B, N, 2) in [0,1]
    pos_sin = torch.sin(_2pi * pos_norm)
    pos_cos = torch.cos(_2pi * pos_norm)
    pos = torch.stack([pos_sin[..., 0], pos_cos[..., 0],
                       pos_sin[..., 1], pos_cos[..., 1]], dim=-1)         # (B, N, 4)

    vel_raw = obs["vel"][:, :N].float()                                    # (B, N, 2)
    speed   = vel_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)          # (B, N, 1)
    vel_dir = vel_raw / speed                                              # (B, N, 2) unit vec (cos, sin)
    symlog_spd = torch.sign(speed) * torch.log1p(speed.abs())             # (B, N, 1)

    att = obs["att"][:, :N].float()                                        # (B, N, 2)

    av_raw = obs["ang_vel"][:, :N].float()
    ang_vel = torch.sign(av_raw) * torch.log1p(av_raw.abs())              # symlog, (B, N, 1)

    health_n = (obs["health"][:, :N].float().squeeze(-1) / ship_config.max_health).clamp(0.0, 1.0)
    power_n  = (obs["power"][:, :N].float().squeeze(-1)  / ship_config.max_power).clamp(0.0, 1.0)
    cd_n     = (obs["cooldown"][:, :N].float().squeeze(-1) / ship_config.firing_cooldown).clamp(0.0, 1.0)

    health   = torch.stack([torch.sin(_hpi * health_n), torch.cos(_hpi * health_n)], dim=-1)
    power    = torch.stack([torch.sin(_hpi * power_n),  torch.cos(_hpi * power_n)],  dim=-1)
    cooldown = torch.stack([torch.sin(_hpi * cd_n),     torch.cos(_hpi * cd_n)],     dim=-1)

    return torch.cat([pos, vel_dir, symlog_spd, att, ang_vel, health, power, cooldown], dim=-1)  # (B, N, 16)


def _apply_aux_deltas_local(curr: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply predicted deltas/absolutes to AUX continuous state.

    curr:  (*, 16)  pos_x(2) pos_y(2) vel_dir(2) symlog_spd(1) att(2) ang_vel(1) health(2) power(2) cooldown(2)
    delta: (*, 9)   Δφ_x(1) Δφ_y(1) Δφ_vel(1) Δspd(1) Δφ_att(1) ang_vel_abs(1) Δφ_h(1) Δφ_p(1) Δφ_c(1)
    Returns: (*, 16)
    """
    def _phase_shift(sc: torch.Tensor, d: torch.Tensor, cos_first: bool = False) -> torch.Tensor:
        cd, sd = d.cos(), d.sin()
        if cos_first:
            c, s = sc[..., 0], sc[..., 1]
            return torch.stack([c * cd - s * sd, s * cd + c * sd], dim=-1)
        else:
            s, c = sc[..., 0], sc[..., 1]
            return torch.stack([s * cd + c * sd, c * cd - s * sd], dim=-1)

    return torch.cat([
        _phase_shift(curr[..., 0:2],   delta[..., 0:1].squeeze(-1)),                 # pos_x  (sin, cos)
        _phase_shift(curr[..., 2:4],   delta[..., 1:2].squeeze(-1)),                 # pos_y  (sin, cos)
        _phase_shift(curr[..., 4:6],   delta[..., 2:3].squeeze(-1), cos_first=True), # vel_dir (cos, sin)
        curr[..., 6:7] + delta[..., 3:4],                                            # symlog_speed
        _phase_shift(curr[..., 7:9],   delta[..., 4:5].squeeze(-1), cos_first=True), # att    (cos, sin)
        delta[..., 5:6],                                                              # ang_vel absolute
        _phase_shift(curr[..., 10:12], delta[..., 6:7].squeeze(-1)),                 # health (sin, cos)
        _phase_shift(curr[..., 12:14], delta[..., 7:8].squeeze(-1)),                 # power  (sin, cos)
        _phase_shift(curr[..., 14:16], delta[..., 8:9].squeeze(-1)),                 # cooldown (sin, cos)
    ], dim=-1)


ALIVE_HEALTH_EPS = 1.0  # ship is dead when decoded health ≤ this value


def _imagined_obs_from_aux(
    aux_next: torch.Tensor,
    prev_obs: dict,
    action: torch.Tensor,
    N: int,
    ship_config,
) -> dict:
    """Reconstruct raw obs dict ship fields from predicted AUX continuous state.

    aux_next:  (B, N, 16) pos_x(2) pos_y(2) vel_dir(2) symlog_spd(1) att(2) ang_vel(1) health(2) power(2) cooldown(2)
    prev_obs:  previous obs dict — obstacle tokens (N:) copied unchanged
    action:    (B, N, 3) int — imagined action used as prev_power/turn/shoot
    """
    import math
    _2pi = 2.0 * math.pi
    _hpi = math.pi / 2.0
    W = ship_config.world_size[0]

    pos_x = (torch.atan2(aux_next[..., 0], aux_next[..., 1]) % _2pi) * W / _2pi
    pos_y = (torch.atan2(aux_next[..., 2], aux_next[..., 3]) % _2pi) * W / _2pi
    pos = torch.stack([pos_x, pos_y], dim=-1)                             # (B, N, 2)

    vel_dir = aux_next[..., 4:6]                                          # (B, N, 2) unit vec (cos, sin)
    sym_spd = aux_next[..., 6:7]
    speed   = torch.sign(sym_spd) * (torch.exp(sym_spd.abs()) - 1)       # inv_symlog
    vel     = vel_dir * speed                                              # (B, N, 2)

    att = aux_next[..., 7:9]                                              # (B, N, 2) unit vec

    sym = aux_next[..., 9:10]
    ang_vel = torch.sign(sym) * (torch.exp(sym.abs()) - 1)               # inv_symlog, (B, N, 1)

    health_n = torch.atan2(aux_next[..., 10], aux_next[..., 11]) / _hpi
    power_n  = torch.atan2(aux_next[..., 12], aux_next[..., 13]) / _hpi
    cd_n     = torch.atan2(aux_next[..., 14], aux_next[..., 15]) / _hpi
    health   = (health_n * ship_config.max_health).unsqueeze(-1)          # (B, N, 1)
    power    = (power_n  * ship_config.max_power).unsqueeze(-1)
    cooldown = (cd_n     * ship_config.firing_cooldown).unsqueeze(-1)

    alive = health.squeeze(-1) > ALIVE_HEALTH_EPS                         # (B, N) bool

    new_obs = {k: v.clone() for k, v in prev_obs.items()}
    new_obs["pos"]        = torch.cat([pos,     prev_obs["pos"][:, N:]],      dim=1)
    new_obs["vel"]        = torch.cat([vel,     prev_obs["vel"][:, N:]],      dim=1)
    new_obs["att"]        = torch.cat([att,     prev_obs["att"][:, N:]],      dim=1)
    new_obs["ang_vel"]    = torch.cat([ang_vel, prev_obs["ang_vel"][:, N:]],  dim=1)
    new_obs["health"]     = torch.cat([health,  prev_obs["health"][:, N:]],   dim=1)
    new_obs["power"]      = torch.cat([power,   prev_obs["power"][:, N:]],    dim=1)
    new_obs["cooldown"]   = torch.cat([cooldown,prev_obs["cooldown"][:, N:]], dim=1)
    new_obs["alive"]      = torch.cat([alive,   prev_obs["alive"][:, N:]],    dim=1)
    new_obs["prev_power"] = torch.cat([action[..., 0], prev_obs["prev_power"][:, N:]], dim=1)
    new_obs["prev_turn"]  = torch.cat([action[..., 1], prev_obs["prev_turn"][:, N:]],  dim=1)
    new_obs["prev_shoot"] = torch.cat([action[..., 2], prev_obs["prev_shoot"][:, N:]], dim=1)
    return new_obs


def imagine_trajectory(
    agent: ResolvedAgent,
    obs: dict,
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
        List of n_steps (B, N, AUX_PRED_DIM=12) tensors, or [] for non-policy agents.
    """
    if agent.kind != "policy" or agent.hidden is None or n_steps <= 0:
        return []

    imag_hidden = agent.hidden.clone()
    imag_obs = {k: v.clone() for k, v in obs.items()}
    curr_aux = _extract_aux_cont(imag_obs, num_ships, ship_config)        # (B, N, 16)

    pred_nexts = []
    with torch.no_grad():
        for _ in range(n_steps):
            action, _, _, pred_next, imag_hidden = agent.agent.get_action_and_value(
                imag_obs, imag_hidden
            )
            pred_nexts.append(pred_next)
            next_aux = _apply_aux_deltas_local(curr_aux, pred_next)
            imag_obs = _imagined_obs_from_aux(
                next_aux, imag_obs, action, num_ships, ship_config
            )
            curr_aux = next_aux

    return pred_nexts
