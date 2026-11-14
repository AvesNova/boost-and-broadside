"""Yemong's single-step forward exports to ONNX and matches eager (client play, rule 9).

The exported surface is one view: the observation, the reset flag, the decision key (which
draws a reset view's position offset) and the carry in; the policy logits, the values, the
outcome and the next carry out. Sampling is the client's, and so is writing the sample into
the carry's previous action.
"""

import numpy as np
import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import Action, GameConfig, Obs
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
from bnb.models.yemong.state_code import SPREADS

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
# The ONNX exporter initialises CUDA when a device is visible, so the test is `gpu`.
pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.usefixtures("global_generator")]


onnxruntime = pytest.importorskip("onnxruntime")


class Step(torch.nn.Module):
    def __init__(self, model: Yemong, carry_spec) -> None:
        super().__init__()
        self.model, self.carry_spec = model, carry_spec

    def forward(self, ships, fields, globals, zones, reset, decision_key, *carry):
        decision = self.model.step(
            _pytree.tree_unflatten(list(carry), self.carry_spec),
            Obs(ships=ships, fields=fields, globals=globals, zones=zones),
            reset,
            decision_key,
        )
        return (
            decision.policy_logits,
            decision.values,
            decision.outcome_probabilities,
            *_pytree.tree_leaves(decision.carry),
        )


def _mid_game(config: GameConfig, ticks: int):
    """One fogged match after `ticks` ticks of chaser play: team 0's view."""
    rules, types = api.rules_of(config), api.ship_types()
    generator = torch.Generator().manual_seed(3)
    state = api.reset(rules, types, api.uniforms(rules, 1, generator=generator))
    state, obs, _ = api.perceive(rules, state)
    chaser = Chaser(characteristics(types, rules), config.ships_per_team)
    views = []
    for _ in range(ticks):
        views.append(
            Obs(
                ships=obs.ships[:, 0],
                fields=obs.fields[:, 0],
                globals=obs.globals[:, 0],
                zones=obs.zones[:, 0],
            )
        )
        own = Obs(ships=obs.ships[0], fields=None, globals=None)
        action, _, _ = chaser.step(None, own, None, torch.arange(own.ships.shape[0]))
        packed = action.packed().reshape(1, config.num_ships, 3)
        state, _ = api.step(
            rules,
            types,
            state,
            Action.unpack(packed),
            api.spread(rules, 1, generator=generator),
            api.uniforms(rules, 1, generator=generator),
        )
        state, obs, _ = api.perceive(rules, state)
    return rules, types, views


@pytest.mark.parametrize(
    "settings",
    [
        dict(next_state="absolute"),
        dict(next_state="delta"),
        dict(order="temporal_first", acceleration_inputs=True),
        dict(next_state="delta", position_inputs="fourier"),
    ],
    ids=["absolute", "delta", "accelerations", "fourier"],
)
def test_yemong_step_exports_to_onnx(tmp_path, settings):
    config = GameConfig(3, 4, 1500.0)
    rules, types, views = _mid_game(config, ticks=20)
    torch.manual_seed(0)
    model = Yemong(yemong_build(YemongConfig(**settings), rules, types)).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("last.weight") or ".lasts." in name and name.endswith(".weight"):
                parameter.normal_(0.0, 0.05)
        carry = model.initial_carry(1, config.ships_per_team, config.num_ships)
        reset = torch.ones(1, dtype=torch.bool)
        for view in views[:-1]:  # a carry with history and beliefs about hidden ships
            carry = model.step(carry, view, reset, torch.arange(view.ships.shape[0])).carry
            reset = torch.zeros(1, dtype=torch.bool)
    assert (views[-1].ships[0, :, SHIP["visible"]] < 0.5).any(), "the last view should hide a ship"
    leaves, spec = _pytree.tree_flatten(carry)
    step = Step(model, spec).eval()
    view = views[-1]
    key = torch.tensor([7], dtype=torch.int64)
    args = (view.ships, view.fields, view.globals, view.zones, reset, key, *leaves)
    path = tmp_path / "yemong.onnx"
    torch.onnx.export(step, args, str(path), dynamo=True)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    got = session.run(None, {name: arg.numpy() for name, arg in zip(names, args)})
    with torch.no_grad():
        want = step(*args)
    assert len(got) == len(want)
    carry_paths = [_pytree.keystr(path) for path, _ in _pytree.tree_flatten_with_path(carry)[0]]
    labels = ["policy_logits", "values", "outcome", *carry_paths]
    for label, g, w in zip(labels, got, want, strict=True):
        if w.is_floating_point():
            error = np.abs(g - w.numpy())
            allowed = _absolute_tolerance(label, w) + 1e-4 * np.abs(w.numpy())
            assert (error <= allowed).all(), (label, np.argwhere(error > allowed).tolist())
        else:
            np.testing.assert_array_equal(g, w.numpy(), err_msg=label)


VELOCITY_COVARIANCE = slice(SPREADS.index("velocity_xx"), SPREADS.index("velocity_yy") + 1)


def _absolute_tolerance(label: str, want: torch.Tensor) -> np.ndarray | float:
    """1e-4, but for a belief's velocity covariance (xx, xy, yy) also 1e-5 of the row's
    largest entry: `codes.velocity_decode` makes yy as (2 (v1 + v2) - v0) / 3, a difference
    of terms the size of xx, so a float32 rounding of a large xx shows in a small yy."""
    if not label.endswith(".belief.spreads"):
        return 1e-4
    tolerance = np.full(want.shape, 1e-4, dtype=np.float32)
    block = want[..., VELOCITY_COVARIANCE].abs().amax(-1, keepdim=True).numpy()
    tolerance[..., VELOCITY_COVARIANCE] += 1e-5 * block
    return tolerance
