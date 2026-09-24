"""Production implementations must conform to the decision-runtime oracle."""

from __future__ import annotations

import pytest
import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.runtime.actions import (
    PendingActionState,
    advance_autonomous_decision,
    decode_joint_action,
    encode_joint_action,
    validate_physical_actions,
)
from tests.runtime_semantics.reference import ReferenceDecisionRuntime

A = ((1, 2, 1), (2, 3, 0))
B = ((2, 5, 0), (1, 6, 1))
C = ((1, 4, 1), (2, 1, 0))
PRIVATE = (3, 7, 2)
NEUTRAL = (0, 0, 0)


def _tensor(actions) -> torch.Tensor:
    return torch.tensor([actions], dtype=torch.int32)


def _pending_observation() -> YemongObservation:
    team0 = torch.zeros((1, 2, 3), dtype=torch.float32)
    team1 = torch.zeros_like(team0)
    return YemongObservation(
        data={ObsKey.PREVIOUS_ACTION: team0},
        team1_data={ObsKey.PREVIOUS_ACTION: team1},
    )


def test_production_joint_codec_matches_the_reference_ordering() -> None:
    actions = torch.tensor(
        [(power, turn, shoot) for power in range(3) for turn in range(7) for shoot in range(2)],
        dtype=torch.int32,
    )

    encoded = encode_joint_action(actions)

    assert encoded.tolist() == list(range(42))
    assert torch.equal(decode_joint_action(encoded), actions.long())
    validate_physical_actions(actions)
    with pytest.raises(ValueError, match="private categories"):
        validate_physical_actions(torch.tensor([PRIVATE]))
    with pytest.raises(ValueError, match="private categories"):
        encode_joint_action(torch.tensor([PRIVATE]))
    with pytest.raises(ValueError, match="joint action IDs"):
        decode_joint_action(torch.tensor([42]))


def test_pending_state_matches_reference_neutral_a_b_c_and_privacy_trace() -> None:
    oracle = ReferenceDecisionRuntime((0, 1))
    production = PendingActionState.allocate(1, 2, "cpu")
    team_id = torch.tensor([[0, 1]], dtype=torch.int32)
    contiguous = torch.ones((1, 2), dtype=torch.bool)
    continuing = torch.zeros(1, dtype=torch.bool)

    for selected in (A, B, C):
        observation = _pending_observation()
        production.write_observation(observation, team_id, 2)
        assert tuple(map(tuple, observation.data[ObsKey.PREVIOUS_ACTION][0].int().tolist())) == (
            oracle.observe(0).pending_action
        )
        assert (
            tuple(map(tuple, observation.team1_data[ObsKey.PREVIOUS_ACTION][0].int().tolist()))
            == oracle.observe(1).pending_action
        )

        applied = production.applied_action().clone()
        expected = oracle.advance(selected)
        committed = production.commit(_tensor(selected), contiguous, continuing)

        assert tuple(map(tuple, applied[0].tolist())) == expected.applied_action_by_substep[0]
        assert tuple(committed[0].tolist()) == expected.decision_committed
        assert tuple(map(tuple, production.pending[0].tolist())) == expected.next_pending_action


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_production_episode_boundary_discards_the_terminal_selection(ending: str) -> None:
    production = PendingActionState(_tensor(A))
    committed = production.commit(
        _tensor(B),
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
    )

    assert not committed.any()
    assert torch.equal(production.pending, torch.zeros_like(production.pending)), ending


def test_production_respawn_and_human_override_cancel_decision_commit() -> None:
    production = PendingActionState(_tensor(A))
    human = torch.tensor([[True, False]])
    actuator_contiguous = torch.tensor([[True, False]])

    applied = production.applied_action(_tensor(B), human)
    committed = production.commit(
        _tensor(B),
        actuator_contiguous,
        torch.zeros(1, dtype=torch.bool),
        immediate_mask=human,
    )

    assert tuple(map(tuple, applied[0].tolist())) == (B[0], A[1])
    assert committed.tolist() == [[False, False]]
    assert tuple(map(tuple, production.pending[0].tolist())) == (NEUTRAL, NEUTRAL)


def test_shared_autonomous_advance_consumes_pending_before_committing_selection() -> None:
    class RecordingEnv:
        def __init__(self) -> None:
            self.applied: list[torch.Tensor] = []
            self.last_actuator_contiguous = torch.tensor([[True, False]])

        def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.applied.append(action.clone())
            return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)

    env = RecordingEnv()
    production = PendingActionState(_tensor(A))

    dones, truncated, committed = advance_autonomous_decision(env, production, _tensor(B))

    assert not dones.any() and not truncated.any()
    assert tuple(map(tuple, env.applied[0][0].tolist())) == A
    assert committed.tolist() == [[True, False]]
    assert tuple(map(tuple, production.pending[0].tolist())) == (B[0], NEUTRAL)


@pytest.mark.parametrize("respawn_substep", [0, 1, 2])
def test_tensor_env_neutralizes_a_respawned_ship_for_remaining_repeats(
    monkeypatch: pytest.MonkeyPatch,
    respawn_substep: int,
) -> None:
    config = EnvConfig(
        num_ships=2,
        max_bullets=0,
        max_episode_steps=100,
        action_repeat=3,
    )
    env = TensorEnv(1, ShipConfig(), config, "cpu")
    env.reset(options={"team_sizes": (1, 1)})
    held = _tensor(A)
    applied: list[torch.Tensor] = []

    def fake_tick(actions, *, unlimited_resources=False):
        del unlimited_resources
        substep = len(applied)
        applied.append(actions.clone())
        env.state.ship_respawned.zero_()
        if substep == respawn_substep:
            env.state.ship_respawned[0, 0] = True
        return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)

    monkeypatch.setattr(env, "tick", fake_tick)
    env.step(held)

    expected = ReferenceDecisionRuntime((0, 1), action_repeat=3)
    expected.advance(A)
    trace = expected.advance(B, respawn_after_substep={respawn_substep: (0,)})
    expected_ship0 = [substep[0] for substep in trace.applied_action_by_substep]
    assert [tuple(step[0, 0].tolist()) for step in applied] == expected_ship0
    assert not env.last_actuator_contiguous[0, 0]
    assert env.last_actuator_contiguous[0, 1]


class _ConstantController:
    def __init__(self, action: tuple[int, int, int]) -> None:
        self.action = action

    def get_actions(self, state, team_visibility=None):
        del team_visibility
        return torch.tensor(self.action, dtype=torch.int32).expand(
            state.num_envs, state.ship_pos.shape[1], 3
        )


def test_match_runner_uses_the_same_delayed_trace_for_scripted_agents() -> None:
    config = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=100)
    env = TensorEnv(1, ShipConfig(), config, "cpu")
    env.reset(options={"team_sizes": (1, 1)}, seed=7)
    controller0 = _ConstantController(A[0])
    controller1 = _ConstantController(A[1])
    runner = MatchRunner(
        env,
        [ResolvedAgent("scripted", controller0), ResolvedAgent("scripted", controller1)],
        torch.tensor([0]),
        torch.tensor([1]),
        ShipConfig(),
        2,
    )
    runner.init_hidden()

    runner.step()
    assert torch.equal(env.state.prev_action.int(), torch.zeros((1, 2, 3), dtype=torch.int32))
    assert tuple(map(tuple, runner.action_state.pending[0].tolist())) == A

    observation = runner.observe()
    assert tuple(map(tuple, observation[ObsKey.PREVIOUS_ACTION][0].int().tolist())) == (
        A[0],
        PRIVATE,
    )
    assert tuple(map(tuple, observation.team1_data[ObsKey.PREVIOUS_ACTION][0].int().tolist())) == (
        PRIVATE,
        A[1],
    )

    controller0.action = B[0]
    controller1.action = B[1]
    runner.step()
    assert tuple(map(tuple, env.state.prev_action[0].int().tolist())) == A
    assert tuple(map(tuple, runner.action_state.pending[0].tolist())) == B
