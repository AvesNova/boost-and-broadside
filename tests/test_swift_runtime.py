"""The captured rollout step: equals eager `step`, and draws fresh noise on every replay."""

import pytest
import torch

from bnb.models.swift.model import Yemong
from bnb.models.swift.runtime import CapturedStep
from bnb.models.swift.synthetic import random_segment, time_slice

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

COMPARED = ("action_logits", "enemy_action_logits", "next_state_logits", "values",
            "outcome")


def relative_error(actual, expected):
    return ((actual - expected).abs().max() / max(1.0, expected.abs().max().item())).item()


@pytest.mark.parametrize("graph", [False, True], ids=["compiled", "cudagraph"])
def test_captured_step_equals_eager_step(graph):
    torch.manual_seed(0)
    model = Yemong().cuda()
    with torch.no_grad():
        model.next_state_head.last.weight.normal_(0, 0.02)
    views, steps = 8, 4
    obs, reset = random_segment(steps, views, device="cuda", reset_steps=(0, 2))
    reset[:, 1] = False
    captured = CapturedStep(model, views, time_slice(obs, 0), graph=graph)
    for static, value in zip(captured.carry, model.initial_carry(views, 10, device="cuda"),
                             strict=True):
        static.copy_(value)
    for t in range(steps):
        before = type(captured.carry)(*(leaf.clone() for leaf in captured.carry))
        out = captured(time_slice(obs, t), reset[t])
        with torch.no_grad():
            expected_carry, expected = model.step(before, time_slice(obs, t), reset[t])
        for key in COMPARED:
            if key == "next_state_logits":  # log(code + 1e-3) turns float32 ulps of the 4 m
                # position level into ~1e-1 logit noise where the code is near zero: compare
                # as probabilities
                assert (out[key].exp() - expected[key].exp()).abs().max() < 1e-3, (t, key)
            else:
                assert relative_error(out[key], expected[key]) < 1e-4, (t, key)
        for name in ("means", "spreads", "action", "state", "convolution_buffer"):
            assert relative_error(getattr(captured.carry, name),
                                  getattr(expected_carry, name)) < 1e-4, (t, name)
        assert torch.equal(captured.carry.previous_action, out["action"])


def test_replays_draw_fresh_noise():
    torch.manual_seed(0)
    model = Yemong().cuda()
    views = 16
    obs, _ = random_segment(1, views, device="cuda")
    captured = CapturedStep(model, views, time_slice(obs, 0), graph=True)
    snapshot = type(captured.carry)(*(leaf.clone() for leaf in captured.carry))
    actions = []
    for _ in range(2):
        for static, value in zip(captured.carry, snapshot, strict=True):
            static.copy_(value)
        actions.append(captured.replay()["action"].clone())
    assert not torch.equal(actions[0], actions[1])
