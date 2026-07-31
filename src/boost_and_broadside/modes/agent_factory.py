"""Agent specification resolution for interactive and evaluation modes.

Supported specs:
    null           — human keyboard input (watch mode only)
    random         — uniform random actions every step
    latest         — most recently modified checkpoint under checkpoint_dir
    <path.pt>      — specific .pt checkpoint file
    scripted       — StochasticScriptedAgent
    scripted_team  — StochasticScriptedAgent with team target selection
    jouster / team_jouster / boom_zoom / abreast / reverse_turret /
    run_away / spiral_evader / jinking
                   — deterministic scripted agents (see agents/)
"""

import sys
from pathlib import Path

import torch

from boost_and_broadside.agents.abreast import AbreastAgent
from boost_and_broadside.agents.boom_zoom import BoomZoomAgent
from boost_and_broadside.agents.jinking import JinkingAgent
from boost_and_broadside.agents.jouster import JousterAgent
from boost_and_broadside.agents.reverse_turret import ReverseTurretAgent
from boost_and_broadside.agents.run_away import RunAwayAgent
from boost_and_broadside.agents.semi_random_scripted import SemiRandomScriptedAgent
from boost_and_broadside.agents.spiral_evader import SpiralEvaderAgent
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.agents.team_jouster import TeamJousterAgent
from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.constants import (
    NUM_POWER_ACTIONS,
    NUM_SHOOT_ACTIONS,
    NUM_TURN_ACTIONS,
)
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.env.state import TensorState


def infer_num_value_components(ckpt: dict) -> int:
    """Return the critic width K (number of value components) for a checkpoint.

    Newer checkpoints store this directly under "num_value_components". Older ones
    predate the field, so fall back to reading the final value-head Linear's output
    width straight from the state dict — the same shape introspection every loader
    used before the field existed.
    """
    if "num_value_components" in ckpt:
        return int(ckpt["num_value_components"])
    return ckpt["policy_state_dict"]["value_head_local.3.weight"].shape[0]


def infer_team_pma_k(ckpt: dict) -> tuple[int, ...]:
    """Return the win/loss value-component indices for a checkpoint.

    Newer checkpoints store this directly under "team_pma_k". Older ones
    (e.g. avg-model saves from before the key was added) only carry the
    team_pma weights in the state_dict, so reconstruct the active-component
    ordering from the stored reward weights — the same filter the reward
    wrapper applies at training time.
    """
    if "team_pma_k" in ckpt:
        return tuple(ckpt["team_pma_k"])
    if "team_pma.seeds" not in ckpt["policy_state_dict"]:
        return ()

    from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES

    reward_weights = ckpt["train_config"]["rewards"]
    active = [
        name for name in REWARD_COMPONENT_NAMES if reward_weights.get(f"{name}_weight", 0.0) != 0.0
    ]
    win_k = tuple(i for i, name in enumerate(active) if name in ("ally_win", "enemy_win"))
    n_win = ckpt["policy_state_dict"]["value_head_win.3.weight"].shape[0]
    if len(win_k) != n_win:
        sys.exit(
            f"Error: inferred {len(win_k)} win components from checkpoint "
            f"train_config but value_head_win outputs {n_win}."
        )
    return win_k


class ResolvedAgent:
    """An agent resolved from a spec string, with mutable hidden state for policy agents."""

    def __init__(self, kind: str, agent, hidden=None):
        self.kind = kind  # "null" | "random" | "scripted" | "semi_random" | "policy"
        self.agent = agent  # None | StochasticScriptedAgent | YemongPolicy
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
        spec:           One of: null, random, scripted, semi_scripted:P, latest, or a
                        path ending in .pt.
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

    if spec.startswith("semi_scripted:"):
        try:
            probability = float(spec.partition(":")[2])
        except ValueError as error:
            raise ValueError(f"invalid semi-scripted probability in {spec!r}") from error
        agent = SemiRandomScriptedAgent(ship_config, probability)
        return ResolvedAgent("semi_random", agent)

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
    from boost_and_broadside.models.yemong.policy import YemongPolicy
    from boost_and_broadside.train.rl.features import build_standard_coordinator

    ckpt = torch.load(path, map_location=device, weights_only=False)
    coordinator = build_standard_coordinator(ship_config)
    K = infer_num_value_components(ckpt)
    team_pma_k = infer_team_pma_k(ckpt)
    policy = YemongPolicy(
        model_config,
        coordinator,
        num_value_components=K,
        num_ships=num_ships,
        team_pma_k=team_pma_k,
    ).to(device)
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
    obs: YemongObservation | None,
    state: TensorState,
    num_envs: int,
    num_ships: int,
    device: str | torch.device,
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

    if agent.kind == "semi_random":
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


def reset_done_envs(agent: ResolvedAgent, done_mask: torch.Tensor, num_tokens: int) -> None:
    """Reset GRU hidden state for completed envs; no-op for non-policy agents."""
    if agent.kind == "policy" and agent.hidden is not None:
        agent.hidden = agent.agent.reset_hidden_for_envs(agent.hidden, done_mask, num_tokens)


# ---------------------------------------------------------------------------
# Autoregressive imagination for watch-mode visualization
# ---------------------------------------------------------------------------

ALIVE_HEALTH_EPS = 1.0  # ship is dead when decoded health ≤ this value


def _decode_targets_to_obs(
    targets: torch.Tensor,
    prev_obs: YemongObservation,
    action: torch.Tensor,
    N: int,
    coordinator,
) -> YemongObservation:
    """Decode a coordinator target vector back to a raw YemongObservation.

    Per-feature target encodings are inverted by the coordinator — each feature's
    target Transform owns its own inverse — so this function only reassembles the
    raw values into an YemongObservation: obstacle tokens (N:) are copied unchanged,
    `alive` is derived from decoded health, and `action` becomes PREVIOUS_ACTION.

    prev_obs: previous YemongObservation — obstacle tokens (N:) are copied unchanged
    action:   (B, N, 3) int — stored as PREVIOUS_ACTION for next step
    """
    raw = coordinator.decode_targets(targets)

    pos = torch.cat([raw["position_x"], raw["position_y"]], dim=-1)  # (B, N, 2)
    vel = raw["velocity"]  # (B, N, 2)
    att = raw["attitude"]  # (B, N, 2)
    ang_vel = raw["angular_velocity"]  # (B, N, 1)
    health = raw["health"]  # (B, N, 1)
    power = raw["power"]  # (B, N, 1)
    cooldown = raw["cooldown"]  # (B, N, 1)

    alive = health.squeeze(-1) > ALIVE_HEALTH_EPS

    new_data = {k: v.clone() for k, v in prev_obs.items()}
    new_data[ObsKey.POS] = torch.cat([pos, prev_obs[ObsKey.POS][:, N:]], dim=1)
    new_data[ObsKey.VEL] = torch.cat([vel, prev_obs[ObsKey.VEL][:, N:]], dim=1)
    new_data[ObsKey.ATT] = torch.cat([att, prev_obs[ObsKey.ATT][:, N:]], dim=1)
    new_data[ObsKey.ANG_VEL] = torch.cat([ang_vel, prev_obs[ObsKey.ANG_VEL][:, N:]], dim=1)
    new_data[ObsKey.HEALTH] = torch.cat([health, prev_obs[ObsKey.HEALTH][:, N:]], dim=1)
    new_data[ObsKey.POWER] = torch.cat([power, prev_obs[ObsKey.POWER][:, N:]], dim=1)
    new_data[ObsKey.COOLDOWN] = torch.cat([cooldown, prev_obs[ObsKey.COOLDOWN][:, N:]], dim=1)
    new_data[ObsKey.ALIVE] = torch.cat([alive, prev_obs[ObsKey.ALIVE][:, N:]], dim=1)
    new_data[ObsKey.PREVIOUS_ACTION] = torch.cat(
        [action, prev_obs[ObsKey.PREVIOUS_ACTION][:, N:]], dim=1
    )
    return YemongObservation(data=new_data)


def imagine_trajectory(
    agent: ResolvedAgent,
    obs: YemongObservation,
    n_steps: int,
    num_ships: int,
    device,
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
    label_scale = coordinator.label_scale_vector(device)
    imag_hidden = agent.hidden.clone()
    imag_obs = YemongObservation(data={k: v.clone() for k, v in obs.items()})
    curr_ship_targets = coordinator.get_target_vector(imag_obs)[:, :num_ships]

    pred_nexts = []
    with torch.no_grad():
        for _ in range(n_steps):
            action, _, _, pred_next_scaled, imag_hidden = agent.agent.get_action_and_value(
                imag_obs, imag_hidden
            )
            # Unscale before storing: the renderer expects raw phase deltas, not
            # the scaled values the network predicts (scaled by 1/std for O(1) outputs).
            pred_nexts.append(pred_next_scaled / label_scale)
            next_ship_targets = coordinator.apply_scaled_predictions(
                curr_ship_targets, pred_next_scaled
            )
            imag_obs = _decode_targets_to_obs(
                next_ship_targets, imag_obs, action, num_ships, coordinator
            )
            curr_ship_targets = coordinator.get_target_vector(imag_obs)[:, :num_ships]

    return pred_nexts
