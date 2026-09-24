"""Shared action representation and pending-actuator state.

The methods here are deliberately tensor-level. High-throughput and low-latency
schedulers own their control flow, but they share these queue, lifecycle, and
perspective operations instead of reimplementing their semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from boost_and_broadside.constants import (
    NUM_JOINT_ACTIONS,
    NUM_SHOOT_ACTIONS,
    NUM_TURN_ACTIONS,
    PRIVATE_POWER_ACTION,
    PRIVATE_SHOOT_ACTION,
    PRIVATE_TURN_ACTION,
)
from boost_and_broadside.env.observation import (
    ObsKey,
    YemongObservation,
    write_pending_action_view,
)


class AutonomousPhysicsEnv(Protocol):
    """Minimum physics interface consumed by an autonomous decision advance."""

    last_actuator_contiguous: torch.Tensor

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


def encode_joint_action(action: torch.Tensor) -> torch.Tensor:
    """Encode ``[..., power, turn, shoot]`` physical triples as joint IDs."""
    validate_physical_actions(action)
    return encode_joint_action_unchecked(action)


def encode_joint_action_unchecked(action: torch.Tensor) -> torch.Tensor:
    """Hot-path encoder for action tensors already guaranteed physical."""
    return (
        action[..., 0].long() * NUM_TURN_ACTIONS + action[..., 1].long()
    ) * NUM_SHOOT_ACTIONS + action[..., 2].long()


def decode_joint_action(action_id: torch.Tensor) -> torch.Tensor:
    """Decode joint IDs in ``[0, 42)`` into physical action triples."""
    if not bool(((action_id >= 0) & (action_id < NUM_JOINT_ACTIONS)).all()):
        raise ValueError(f"joint action IDs must be in [0, {NUM_JOINT_ACTIONS})")
    return decode_joint_action_unchecked(action_id)


def decode_joint_action_unchecked(action_id: torch.Tensor) -> torch.Tensor:
    """Hot-path decoder for IDs sampled from the 42-way policy distribution."""
    action_id = action_id.long()
    power = action_id // (NUM_TURN_ACTIONS * NUM_SHOOT_ACTIONS)
    turn = (action_id // NUM_SHOOT_ACTIONS) % NUM_TURN_ACTIONS
    shoot = action_id % NUM_SHOOT_ACTIONS
    return torch.stack((power, turn, shoot), dim=-1)


def validate_physical_actions(action: torch.Tensor) -> None:
    """Reject observation-only private categories at a non-hot boundary."""
    if action.shape[-1] != 3:
        raise ValueError(f"physical actions need a final dimension of 3, got {action.shape}")
    encoded = encode_joint_action_unchecked(action)
    valid = (
        (action[..., 0] >= 0)
        & (action[..., 0] < PRIVATE_POWER_ACTION)
        & (action[..., 1] >= 0)
        & (action[..., 1] < PRIVATE_TURN_ACTION)
        & (action[..., 2] >= 0)
        & (action[..., 2] < PRIVATE_SHOOT_ACTION)
        & (encoded >= 0)
        & (encoded < NUM_JOINT_ACTIONS)
    )
    if not bool(valid.all()):
        raise ValueError("observation-only private categories are not physical actions")


def neutralize_invalidated_actions_(
    action: torch.Tensor,
    ship_alive: torch.Tensor,
    ship_respawned: torch.Tensor,
) -> torch.Tensor:
    """Cancel held commands after death/respawn, returning ``action`` in place."""
    action.masked_fill_((~ship_alive | ship_respawned).unsqueeze(-1), 0)
    return action


def write_pending_action_observation(
    observation: YemongObservation,
    pending_action: torch.Tensor,
    team_id: torch.Tensor,
    num_ships: int,
) -> None:
    """Install a pending queue into both policy views with opponent privacy.

    This is the intentionally tiny post-overlap operation. Physics/perception
    may run concurrently with policy inference; after both complete, schedulers
    call this once to make the already-built next observation describe the
    command that physics will consume next.
    """
    if pending_action.shape != (*team_id.shape, 3):
        raise ValueError(
            "pending_action must have shape (*team_id.shape, 3), got "
            f"{pending_action.shape} for team_id {team_id.shape}"
        )
    write_pending_action_view(
        observation.data[ObsKey.PREVIOUS_ACTION][:, :num_ships],
        pending_action,
        team_id,
        observer_team=0,
    )
    if observation.team1_data is not None:
        write_pending_action_view(
            observation.team1_data[ObsKey.PREVIOUS_ACTION][:, :num_ships],
            pending_action,
            team_id,
            observer_team=1,
        )


@dataclass
class PendingActionState:
    """Preallocated pending command tensor for one scheduler's environments."""

    pending: torch.Tensor

    @classmethod
    def allocate(
        cls,
        num_envs: int,
        num_ships: int,
        device: str | torch.device,
    ) -> PendingActionState:
        return cls(
            torch.zeros(
                (num_envs, num_ships, 3),
                dtype=torch.int32,
                device=device,
            )
        )

    def applied_action(
        self,
        selected_action: torch.Tensor | None = None,
        immediate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the physical command for this advance.

        The autonomous fast path returns the preallocated queue without a copy.
        Human override supplies ``selected_action`` and a per-ship immediate
        mask, paying for one ``where`` only in that specialized mode.
        """
        if immediate_mask is None:
            return self.pending
        if selected_action is None:
            raise ValueError("selected_action is required for immediate override")
        return torch.where(immediate_mask.unsqueeze(-1), selected_action, self.pending)

    def commit(
        self,
        selected_action: torch.Tensor,
        actuator_contiguous: torch.Tensor,
        terminated: torch.Tensor,
        *,
        autonomous_mask: torch.Tensor | None = None,
        immediate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Install selected autonomous commands and return their causal mask.

        ``actuator_contiguous`` is false for a dead/respawned actuator. Full
        termination/truncation and human override also cancel the selection.
        The queue storage is reused in place.
        """
        committed = actuator_contiguous & ~terminated.unsqueeze(-1)
        if autonomous_mask is not None:
            committed &= autonomous_mask
        if immediate_mask is not None:
            committed &= ~immediate_mask
        self.pending.copy_(selected_action)
        self.pending.masked_fill_(~committed.unsqueeze(-1), 0)
        return committed

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        """Reset all queues or only the named environments to neutral."""
        if env_mask is None:
            self.pending.zero_()
        else:
            self.pending.masked_fill_(env_mask[:, None, None], 0)

    def write_observation(
        self,
        observation: YemongObservation,
        team_id: torch.Tensor,
        num_ships: int,
    ) -> None:
        write_pending_action_observation(observation, self.pending, team_id, num_ships)


def advance_autonomous_decision(
    env: AutonomousPhysicsEnv,
    action_state: PendingActionState,
    selected_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance physics under pending commands, then queue this decision.

    This is the low-latency/evaluation scheduler's authoritative transition
    primitive. The throughput scheduler deliberately keeps its CUDA-stream
    orchestration separate, but uses the same :class:`PendingActionState`
    consume/commit operations around its overlapped environment and network
    work.

    Returns ``(dones, truncated, decision_committed)``.
    """
    dones, truncated = env.step(action_state.applied_action())
    decision_committed = action_state.commit(
        selected_action,
        env.last_actuator_contiguous,
        dones | truncated,
    )
    return dones, truncated, decision_committed
