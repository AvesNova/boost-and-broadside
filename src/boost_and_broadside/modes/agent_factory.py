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
) -> ResolvedAgent:
    """Resolve a spec string to a ResolvedAgent.

    Args:
        spec:           One of: null, random, scripted, latest, or a path ending in .pt.
        ship_config:    Physics constants (needed for scripted agent).
        model_config:   Policy architecture (needed for checkpoint agents).
        device:         Torch device string.
        checkpoint_dir: Root directory searched when spec is "latest".
    """
    if spec == "null":
        return ResolvedAgent("null", None)

    if spec == "random":
        return ResolvedAgent("random", None)

    if spec == "scripted":
        agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
        return ResolvedAgent("scripted", agent)

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

    ckpt = torch.load(path, map_location=device, weights_only=False)
    K = ckpt["policy_state_dict"]["value_head.3.weight"].shape[0]
    policy = MVPPolicy(model_config, ship_config, num_value_components=K).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    update = ckpt.get("update", "?")
    step = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint: update={update}  step={step}  path={path}")

    return ResolvedAgent("policy", policy)


def init_hidden(agent: ResolvedAgent, num_envs: int, num_ships: int, device) -> None:
    """Allocate initial GRU hidden state for policy agents; no-op for all others."""
    if agent.kind == "policy":
        agent.hidden = agent.agent.initial_hidden(num_envs, num_ships, device)


def get_actions(
    agent: ResolvedAgent,
    obs: dict[str, torch.Tensor] | None,
    state: TensorState,
    num_envs: int,
    num_ships: int,
    device,
) -> torch.Tensor:
    """Return (B, N, 3) int actions for every ship in the batch.

    Policy and scripted agents produce actions for all ships; the caller selects
    the relevant team's actions via a team-id mask.  For the null agent the
    returned tensor is all-zeros — the caller must apply keyboard overrides.
    """
    B, N = num_envs, num_ships

    if agent.kind == "random":
        return torch.stack(
            [
                torch.randint(0, NUM_POWER_ACTIONS, (B, N), device=device),
                torch.randint(0, NUM_TURN_ACTIONS, (B, N), device=device),
                torch.randint(0, NUM_SHOOT_ACTIONS, (B, N), device=device),
            ],
            dim=-1,
        ).int()

    if agent.kind == "scripted":
        with torch.no_grad():
            return agent.agent.get_actions(state)

    if agent.kind == "policy":
        with torch.no_grad():
            action, _, _, agent.hidden = agent.agent.get_action_and_value(
                obs, agent.hidden
            )
        return action

    # null — zero placeholder; caller must override with keyboard input
    return torch.zeros(B, N, 3, dtype=torch.int32, device=device)


def reset_done_envs(
    agent: ResolvedAgent, done_mask: torch.Tensor, num_ships: int
) -> None:
    """Reset GRU hidden state for completed envs; no-op for non-policy agents."""
    if agent.kind == "policy" and agent.hidden is not None:
        agent.hidden = agent.agent.reset_hidden_for_envs(
            agent.hidden, done_mask, num_ships
        )
