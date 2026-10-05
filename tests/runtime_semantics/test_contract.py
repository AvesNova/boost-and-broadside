"""Conformance cases for the canonical decision-runtime contract."""

from __future__ import annotations

import pytest

from tests.runtime_semantics.reference import (
    NEUTRAL_ACTION,
    PRIVATE_ACTION,
    ReferenceDecisionRuntime,
    decode_joint_action,
    encode_joint_action,
)

A0 = (1, 2, 1)
A1 = (2, 3, 0)
B0 = (2, 3, 0)
B1 = (1, 4, 1)
C0 = (1, 4, 1)
C1 = (2, 1, 0)


def test_neutral_a_b_c_trace_matches_observed_pending_to_next_applied() -> None:
    runtime = ReferenceDecisionRuntime((0, 1))

    observed_team_0 = []
    observed_team_1 = []
    applied = []
    for decision in ((A0, A1), (B0, B1), (C0, C1)):
        observed_team_0.append(runtime.observe(0).pending_action)
        observed_team_1.append(runtime.observe(1).pending_action)
        transition = runtime.advance(decision)
        applied.append(transition.applied_action_by_substep[0])

    assert applied == [
        (NEUTRAL_ACTION, NEUTRAL_ACTION),
        (A0, A1),
        (B0, B1),
    ]
    assert observed_team_0 == [
        (NEUTRAL_ACTION, PRIVATE_ACTION),
        (A0, PRIVATE_ACTION),
        (B0, PRIVATE_ACTION),
    ]
    assert observed_team_1 == [
        (PRIVATE_ACTION, NEUTRAL_ACTION),
        (PRIVATE_ACTION, A1),
        (PRIVATE_ACTION, B1),
    ]
    assert runtime.pending_action == (C0, C1)


def test_opponent_pending_action_is_private_when_visible_or_hidden() -> None:
    runtime = ReferenceDecisionRuntime((0, 1))
    runtime.advance((A0, A1))

    hidden = runtime.observe(0, visible=(True, False))
    visible = runtime.observe(0, visible=(True, True))

    assert hidden.pending_action == (A0, PRIVATE_ACTION)
    assert visible.pending_action == (A0, PRIVATE_ACTION)


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_episode_boundary_discards_selected_action_and_resets_pending(ending: str) -> None:
    runtime = ReferenceDecisionRuntime((0, 1))
    runtime.advance((A0, A1))

    transition = runtime.advance(
        (B0, B1),
        terminated=ending == "terminated",
        truncated=ending == "truncated",
    )

    assert transition.applied_action_by_substep == ((A0, A1),)
    assert transition.selected_action == (B0, B1)
    assert transition.decision_committed == (False, False)
    assert transition.next_pending_action == (NEUTRAL_ACTION, NEUTRAL_ACTION)
    assert runtime.observe(0).pending_action == (NEUTRAL_ACTION, PRIVATE_ACTION)
    assert runtime.observe(1).pending_action == (PRIVATE_ACTION, NEUTRAL_ACTION)


def test_explicit_episode_reset_neutralizes_queue_before_observation() -> None:
    runtime = ReferenceDecisionRuntime((0, 1))
    runtime.advance((A0, A1))

    runtime.reset()

    assert runtime.observe(0).pending_action == (NEUTRAL_ACTION, PRIVATE_ACTION)
    assert runtime.observe(1).pending_action == (PRIVATE_ACTION, NEUTRAL_ACTION)


@pytest.mark.parametrize("respawn_substep", [0, 1, 2])
def test_respawn_cancels_decision_and_remaining_repeated_actions(
    respawn_substep: int,
) -> None:
    runtime = ReferenceDecisionRuntime((0, 1), action_repeat=3)
    runtime.advance((A0, A1))

    transition = runtime.advance(
        (B0, B1),
        respawn_after_substep={respawn_substep: (0,)},
    )

    expected_ship_0 = [A0 if substep <= respawn_substep else NEUTRAL_ACTION for substep in range(3)]
    assert [actions[0] for actions in transition.applied_action_by_substep] == expected_ship_0
    assert [actions[1] for actions in transition.applied_action_by_substep] == [A1, A1, A1]
    assert transition.decision_committed == (False, True)
    assert transition.next_pending_action == (NEUTRAL_ACTION, B1)
    assert runtime.observe(0).pending_action == (NEUTRAL_ACTION, PRIVATE_ACTION)


@pytest.mark.parametrize("controller_kind", ["neural", "scripted", "random"])
def test_every_autonomous_controller_uses_the_delayed_contract(
    controller_kind: str,
) -> None:
    runtime = ReferenceDecisionRuntime((0,))

    first = runtime.advance((A0,))
    second = runtime.advance((B0,))

    assert controller_kind in {"neural", "scripted", "random"}
    assert first.applied_action_by_substep == ((NEUTRAL_ACTION,),)
    assert first.next_pending_action == (A0,)
    assert second.applied_action_by_substep == ((A0,),)
    assert second.next_pending_action == (B0,)


def test_human_override_is_immediate_and_release_has_neutral_handoff() -> None:
    runtime = ReferenceDecisionRuntime((0, 1))
    runtime.advance((A0, A1))

    human_action = (2, 0, 1)
    overridden = runtime.advance((B0, B1), human_override={0: human_action})

    assert overridden.applied_action_by_substep == ((human_action, A1),)
    assert overridden.decision_committed == (False, True)
    assert overridden.next_pending_action == (NEUTRAL_ACTION, B1)

    released = runtime.advance((C0, C1))
    assert released.applied_action_by_substep == ((NEUTRAL_ACTION, B1),)
    assert released.next_pending_action == (C0, C1)

    resumed = runtime.advance((A0, A1))
    assert resumed.applied_action_by_substep == ((C0, C1),)


def test_respawn_cancels_held_human_override_for_remaining_substeps() -> None:
    runtime = ReferenceDecisionRuntime((0,), action_repeat=3)
    human_action = (2, 0, 1)

    transition = runtime.advance(
        (A0,),
        human_override={0: human_action},
        respawn_after_substep={0: (0,)},
    )

    assert transition.applied_action_by_substep == (
        (human_action,),
        (NEUTRAL_ACTION,),
        (NEUTRAL_ACTION,),
    )
    assert transition.decision_committed == (False,)
    assert transition.next_pending_action == (NEUTRAL_ACTION,)


def test_joint_action_codec_round_trips_all_42_physical_commands() -> None:
    actions = [
        (power, turn, shoot) for power in range(3) for turn in range(5) for shoot in range(2)
    ]

    assert [encode_joint_action(action) for action in actions] == list(range(30))
    assert [decode_joint_action(action_id) for action_id in range(30)] == actions


def test_private_observation_category_cannot_reach_physics() -> None:
    runtime = ReferenceDecisionRuntime((0,))

    with pytest.raises(ValueError, match="not a physical action"):
        runtime.advance((PRIVATE_ACTION,))
    with pytest.raises(ValueError, match="not a physical action"):
        encode_joint_action(PRIVATE_ACTION)
