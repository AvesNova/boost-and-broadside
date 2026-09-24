"""Autoregressive diagnostics preserve controller/runtime boundaries."""

import torch

from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.modes.ar_report import _run_ar


class _RequiresPhysicalState:
    def get_actions(self, state):
        raise AssertionError(f"imagined rollout supplied physical state {state!r}")


def _observation() -> YemongObservation:
    data = {
        ObsKey.POS: torch.zeros(1, 2, 2),
        ObsKey.VEL: torch.zeros(1, 2, 2),
        ObsKey.ATT: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        ObsKey.ANG_VEL: torch.zeros(1, 2, 1),
        ObsKey.HEALTH: torch.ones(1, 2, 1),
        ObsKey.POWER: torch.ones(1, 2, 1),
        ObsKey.COOLDOWN: torch.zeros(1, 2, 1),
        ObsKey.ALIVE: torch.ones(1, 2, dtype=torch.bool),
        ObsKey.TEAM_ID: torch.tensor([[0, 1]], dtype=torch.int32),
    }
    return YemongObservation(data=data)


def test_imagined_rollout_replays_nonpolicy_decisions_without_fake_state() -> None:
    scripted = ResolvedAgent("scripted", _RequiresPhysicalState())
    random = ResolvedAgent("random", None)
    recorded = [torch.tensor([[[1, 2, 1], [2, 3, 0]]], dtype=torch.int32)]

    history = _run_ar(
        scripted,
        random,
        _observation(),
        None,
        None,
        1,
        2,
        recorded,
        False,
    )

    assert len(history) == 1
    assert torch.equal(history[0]["pos"], torch.zeros(1, 2, 2))
