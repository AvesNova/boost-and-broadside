"""Small, test-only oracle for the decision-runtime contract.

This deliberately models boundary semantics rather than game physics. Production
executors should be tested against the same traces, not implemented by calling this
class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Action = tuple[int, int, int]

NEUTRAL_ACTION: Action = (0, 0, 0)
PRIVATE_ACTION: Action = (3, 5, 2)


def encode_joint_action(action: Action) -> int:
    """Encode one physical action using the normative 3 * 5 * 2 ordering."""
    _validate_physical_action(action)
    power, turn, shoot = action
    return ((power * 5) + turn) * 2 + shoot


def decode_joint_action(action_id: int) -> Action:
    """Decode one of the 30 physical joint-action IDs."""
    if not 0 <= action_id < 30:
        raise ValueError(f"joint action ID must be in [0, 30), got {action_id}")
    return action_id // 10, (action_id // 2) % 5, action_id % 2


def _validate_physical_action(action: Action) -> None:
    power, turn, shoot = action
    if not (0 <= power < 3 and 0 <= turn < 5 and 0 <= shoot < 2):
        raise ValueError(f"not a physical action: {action!r}")


@dataclass(frozen=True)
class PerspectiveObservation:
    """The pending-action portion of one team's policy observation."""

    team: int
    pending_action: tuple[Action, ...]


@dataclass(frozen=True)
class ReferenceTransition:
    """Observable action/lifecycle outputs from one decision transition."""

    selected_action: tuple[Action, ...]
    applied_action_by_substep: tuple[tuple[Action, ...], ...]
    decision_committed: tuple[bool, ...]
    next_pending_action: tuple[Action, ...]
    terminated: bool
    truncated: bool


class ReferenceDecisionRuntime:
    """Semantic oracle for a single environment with stable ship slots."""

    def __init__(self, team_ids: Sequence[int], *, action_repeat: int = 1) -> None:
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        if not team_ids:
            raise ValueError("at least one ship is required")
        if any(team not in (0, 1) for team in team_ids):
            raise ValueError("team IDs must be 0 or 1")

        self.team_ids = tuple(team_ids)
        self.action_repeat = action_repeat
        self.pending_action = tuple(NEUTRAL_ACTION for _ in self.team_ids)

    def observe(
        self,
        team: int,
        *,
        visible: Sequence[bool] | None = None,
    ) -> PerspectiveObservation:
        """Compose the pending-action view for one policy perspective.

        ``visible`` is accepted to make the privacy rule explicit: physical
        visibility never reveals an opponent's private queued command.
        """
        if team not in (0, 1):
            raise ValueError("team must be 0 or 1")
        if visible is not None and len(visible) != len(self.team_ids):
            raise ValueError("visibility length must match the ship count")

        pending = tuple(
            action if ship_team == team else PRIVATE_ACTION
            for action, ship_team in zip(self.pending_action, self.team_ids, strict=True)
        )
        return PerspectiveObservation(team=team, pending_action=pending)

    def advance(
        self,
        selected_action: Sequence[Action],
        *,
        respawn_after_substep: Mapping[int, Sequence[int]] | None = None,
        human_override: Mapping[int, Action] | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> ReferenceTransition:
        """Advance the action state according to the normative transition order.

        Respawn events occur after the named zero-based physical substep. Commands
        already consumed on that substep remain in the trace; later repeated
        substeps are neutral for that ship.
        """
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        if len(selected_action) != len(self.team_ids):
            raise ValueError("selected-action length must match the ship count")

        selected = tuple(selected_action)
        for action in selected:
            _validate_physical_action(action)

        respawns = dict(respawn_after_substep or {})
        overrides = dict(human_override or {})
        for substep, ship_indices in respawns.items():
            if not 0 <= substep < self.action_repeat:
                raise ValueError(f"respawn substep out of range: {substep}")
            self._validate_ship_indices(ship_indices)
        self._validate_ship_indices(overrides)
        for action in overrides.values():
            _validate_physical_action(action)

        invalidated: set[int] = set()
        applied_by_substep: list[tuple[Action, ...]] = []
        for substep in range(self.action_repeat):
            applied = list(self.pending_action)
            for ship_index in invalidated:
                applied[ship_index] = NEUTRAL_ACTION
            for ship_index, action in overrides.items():
                if ship_index not in invalidated:
                    applied[ship_index] = action
            applied_by_substep.append(tuple(applied))
            invalidated.update(respawns.get(substep, ()))

        canceled = invalidated | set(overrides)
        if terminated or truncated:
            canceled.update(range(len(self.team_ids)))

        committed = tuple(index not in canceled for index in range(len(self.team_ids)))
        next_pending = tuple(
            action if is_committed else NEUTRAL_ACTION
            for action, is_committed in zip(selected, committed, strict=True)
        )
        self.pending_action = next_pending

        return ReferenceTransition(
            selected_action=selected,
            applied_action_by_substep=tuple(applied_by_substep),
            decision_committed=committed,
            next_pending_action=next_pending,
            terminated=terminated,
            truncated=truncated,
        )

    def reset(self) -> None:
        """Reset the actuator queue before composing a new episode observation."""
        self.pending_action = tuple(NEUTRAL_ACTION for _ in self.team_ids)

    def _validate_ship_indices(self, indices: Sequence[int] | Mapping[int, Action]) -> None:
        for index in indices:
            if not 0 <= index < len(self.team_ids):
                raise ValueError(f"ship index out of range: {index}")
