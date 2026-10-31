"""Yemong's single-step forward exports to ONNX and matches eager (client play, rule 9).

The exported surface is one view: the observation, the reset flag and the carry in; the
policy logits, the values, the outcome and the next carry out. Sampling is the client's,
and so is writing the sample into the carry's previous action.
"""

import numpy as np
import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import Action, GameConfig, Obs
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.models.yemong.model import Yemong, YemongConfig

onnxruntime = pytest.importorskip("onnxruntime")


class Step(torch.nn.Module):
    def __init__(self, model: Yemong, carry_spec) -> None:
        super().__init__()
        self.model, self.carry_spec = model, carry_spec

    def forward(self, ships, fields, globals, reset, *carry):
        decision = self.model.step(_pytree.tree_unflatten(list(carry), self.carry_spec),
                                   Obs(ships=ships, fields=fields, globals=globals), reset)
        return (decision.policy_logits, decision.values, decision.outcome_probabilities,
                *_pytree.tree_leaves(decision.carry))


def _mid_game(config: GameConfig, ticks: int):
    """One fogged match after `ticks` ticks of chaser play: team 0's view."""
    rules, types = api.rules_of(config), api.ship_types()
    generator = torch.Generator().manual_seed(3)
    state = api.reset(rules, types, api.uniforms(rules, 1, generator=generator))
    state, obs, _ = api.perceive(rules, state)
    chaser = Chaser(config.ships_per_team)
    views = []
    for _ in range(ticks):
        views.append(Obs(ships=obs.ships[:, 0], fields=obs.fields[:, 0], globals=obs.globals[:, 0]))
        action, _, _ = chaser.step(None, Obs(ships=obs.ships[0], fields=None, globals=None), None)
        packed = action.packed().reshape(1, config.num_ships, 3)
        state, _ = api.step(rules, types, state, Action.unpack(packed),
                            api.spread(rules, 1, generator=generator),
                            api.uniforms(rules, 1, generator=generator))
        state, obs, _ = api.perceive(rules, state)
    return rules, types, views


def test_yemong_step_exports_to_onnx(tmp_path):
    config = GameConfig(3, 4, 1500.0)
    rules, types, views = _mid_game(config, ticks=20)
    torch.manual_seed(0)
    model = Yemong(YemongConfig(), rules, types).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("last.weight") or ".lasts." in name and name.endswith(".weight"):
                parameter.normal_(0.0, 0.05)
        carry = model.initial_carry(1, config.ships_per_team, config.num_ships)
        reset = torch.ones(1, dtype=torch.bool)
        for view in views[:-1]:  # a carry with history and beliefs about hidden ships
            carry = model.step(carry, view, reset).carry
            reset = torch.zeros(1, dtype=torch.bool)
    assert (views[-1].ships[0, :, SHIP["visible"]] < 0.5).any(), "the last view should hide a ship"
    leaves, spec = _pytree.tree_flatten(carry)
    step = Step(model, spec).eval()
    view = views[-1]
    args = (view.ships, view.fields, view.globals, reset, *leaves)
    path = tmp_path / "yemong.onnx"
    torch.onnx.export(step, args, str(path), dynamo=True)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    got = session.run(None, {name: arg.numpy() for name, arg in zip(names, args)})
    with torch.no_grad():
        want = step(*args)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        if w.is_floating_point():
            np.testing.assert_allclose(g, w.numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_array_equal(g, w.numpy())
