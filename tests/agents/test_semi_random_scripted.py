"""Tests for the random-to-scripted action mixture."""

import torch

from boost_and_broadside.agents.semi_random_scripted import SemiRandomScriptedAgent
from boost_and_broadside.config import ShipConfig


class _UnusedScripted:
    def get_actions(self, _state):
        raise AssertionError("mix_actions should use the supplied action tensors")


def _agent(probability: float) -> SemiRandomScriptedAgent:
    return SemiRandomScriptedAgent(
        ShipConfig(),
        probability,
        scripted_agent=_UnusedScripted(),  # type: ignore[arg-type]
    )


def test_probability_endpoints_select_the_expected_controller() -> None:
    scripted = torch.ones(2, 3, 3, dtype=torch.int32)
    random = torch.zeros_like(scripted)

    assert torch.equal(_agent(1.0).mix_actions(scripted, random), scripted)
    assert torch.equal(_agent(0.0).mix_actions(scripted, random), random)


def test_one_bernoulli_choice_controls_all_three_action_heads(monkeypatch) -> None:
    scripted = torch.ones(1, 4, 3, dtype=torch.int32)
    random = torch.zeros_like(scripted)
    draws = torch.tensor([[0.1, 0.6, 0.2, 0.9]])
    monkeypatch.setattr(torch, "rand", lambda *_args, **_kwargs: draws)

    mixed = _agent(0.5).mix_actions(scripted, random)

    assert torch.equal(mixed[0, :, 0], torch.tensor([1, 0, 1, 0]))
    assert torch.equal(mixed[..., 0], mixed[..., 1])
    assert torch.equal(mixed[..., 1], mixed[..., 2])


def test_probability_outside_unit_interval_is_rejected() -> None:
    for probability in (-0.01, 1.01):
        try:
            _agent(probability)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid probability was accepted")
