"""Recurrent-state allocation for resolved agents, and who decides its width.

``evaluation/agents.py`` is what every mode reaches the policy through, and it
had no tests of its own. The property worth pinning is a narrow one: the width
of a policy's recurrent state is the policy's own to decide, never the caller's.

Only ship tokens carry recurrent state. Field tokens -- and, in the Frontline
arena, zone and boundary tokens -- are static within an episode and take the
non-recurrent path, so allocating over the full entity-token axis would size the
state a third larger than the trunk consumes. Every caller used to pass its own
token total in anyway, and two of them computed it with a formula that had gone
stale; none of it was read. The parameter is gone, and this is the contract that
made removing it safe.
"""

from __future__ import annotations

import pytest
import torch

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.evaluation.agents import (
    ResolvedAgent,
    agents_read_bullets,
    init_hidden,
    reset_done_envs,
)
from boost_and_broadside.train.rl.policy_io import build_policy

_NUM_SHIPS = 4


def _policy(**overrides):
    return build_policy(
        ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1, **overrides),
        ShipConfig(),
        num_value_components=1,
        num_ships=_NUM_SHIPS,
        team_pma_k=(),
    )


@pytest.fixture
def policy_agent():
    return ResolvedAgent("policy", _policy())


def test_the_policy_sizes_its_own_state_over_ships_alone(policy_agent) -> None:
    """Not the entity-token axis. A Frontline environment presents ships, fields,
    zones and a boundary token; only the ships are recurrent."""

    init_hidden(policy_agent, 3, "cpu")

    hidden = policy_agent.hidden
    assert hidden.shape[0] == policy_agent.agent.n_hidden_layers
    assert hidden.shape[1] == 3 * policy_agent.agent.num_recurrent_tokens
    assert policy_agent.agent.num_recurrent_tokens == _NUM_SHIPS
    assert torch.count_nonzero(hidden) == 0


@pytest.mark.parametrize("kind", ["null", "random", "scripted", "semi_random"])
def test_an_agent_with_no_recurrence_is_left_alone(kind: str) -> None:
    agent = ResolvedAgent(kind, object())

    init_hidden(agent, 3, "cpu")
    reset_done_envs(agent, torch.tensor([True, False, True]))

    assert agent.hidden is None


def test_a_finished_env_is_zeroed_and_its_neighbours_are_not(policy_agent) -> None:
    """A carried-over hidden state is the previous episode leaking into the next
    one, and it is invisible in the result -- the games simply come out wrong."""

    init_hidden(policy_agent, 3, "cpu")
    policy_agent.hidden = torch.ones_like(policy_agent.hidden)
    tokens = policy_agent.agent.num_recurrent_tokens

    reset_done_envs(policy_agent, torch.tensor([False, True, False]))

    rows = policy_agent.hidden
    assert torch.count_nonzero(rows[:, tokens : 2 * tokens]) == 0
    assert (rows[:, :tokens] == 1.0).all()
    assert (rows[:, 2 * tokens :] == 1.0).all()


def test_resetting_before_allocating_is_not_an_error(policy_agent) -> None:
    """Modes reset on the first ``done`` they see, which can precede any
    allocation when a run is resumed mid-episode."""

    reset_done_envs(policy_agent, torch.tensor([True]))

    assert policy_agent.hidden is None


def test_the_allocated_state_is_the_shape_reset_expects(policy_agent) -> None:
    """The two have to agree on the token stride or the reset zeroes the wrong
    rows -- silently, since both are the right dtype and the right total size."""

    init_hidden(policy_agent, 2, "cpu")
    before = policy_agent.hidden.shape

    policy_agent.hidden = torch.ones_like(policy_agent.hidden)
    reset_done_envs(policy_agent, torch.tensor([True, True]))

    assert policy_agent.hidden.shape == before
    assert torch.count_nonzero(policy_agent.hidden) == 0


class TestBulletObservationFollowsTheWeights:
    """``include_bullets`` comes from what loaded, not from the current config.

    A policy trained with bullet cross-attention accepts a bullet-free
    observation without complaint and simply plays blind to every shot in
    flight, which reads as a weak policy rather than as a broken evaluation.
    """

    def test_a_policy_that_reads_bullets_turns_them_on(self) -> None:
        assert agents_read_bullets(ResolvedAgent("policy", _policy(n_bullet_cross_per_block=1)))

    def test_a_policy_that_does_not_leaves_them_off(self) -> None:
        assert not agents_read_bullets(ResolvedAgent("policy", _policy()))

    def test_one_reader_in_the_match_is_enough(self) -> None:
        """One observation serves every agent, and a policy that ignores bullets
        is unaffected by their presence."""

        assert agents_read_bullets(
            ResolvedAgent("policy", _policy()),
            ResolvedAgent("policy", _policy(n_bullet_cross_per_block=1)),
        )

    def test_absent_and_non_policy_agents_do_not_ask_for_them(self) -> None:
        assert not agents_read_bullets(None, ResolvedAgent("scripted", object()))
