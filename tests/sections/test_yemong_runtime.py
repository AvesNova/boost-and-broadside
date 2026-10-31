"""Yemong in the runtime (spec 11, item 7): step compiles and captures in a CUDA graph with
fresh randomness on every replay, and the compiled, captured rollout stores what unroll
needs."""

import pytest
import torch
import yemong_cases as cases

from bnb.models.yemong import policy

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
CUDA = torch.device("cuda") if torch.cuda.is_available() else None


def test_compiled_step_captures_and_samples_fresh_actions():
    """A captured decision (no host syncs, static shapes) draws new actions on each replay."""
    net = cases.model().to(CUDA)
    controller = cases.controller(net, cases.SQUAD)
    _, trajectory = cases.rollout(cases.model(), cases.SQUAD, length=4)
    obs = torch.utils._pytree.tree_map_only(torch.Tensor, lambda x: x[-1].to(CUDA),
                                            trajectory.obs)
    carry = controller.init_carry(obs.ships.shape[0], CUDA)
    done = torch.zeros(obs.ships.shape[0], dtype=torch.bool, device=CUDA)
    step = torch.compile(controller.step, fullgraph=True, dynamic=False)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(side):
        for _ in range(2):
            step(carry, obs, done)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        action, _, outputs = step(carry, obs, done)
    draws = []
    for _ in range(3):
        graph.replay()
        draws.append(torch.stack([action.power, action.stick, action.shoot]).clone())
    assert not torch.equal(draws[0], draws[1]) and not torch.equal(draws[1], draws[2])
    assert torch.isfinite(outputs.value).all()


def test_compiled_rollout_stores_what_unroll_needs():
    net = cases.model().to(CUDA)
    _, trajectory = cases.rollout(net, envs=16, length=64, device=CUDA, compile=True)
    # The compiled, captured act stage stores exactly what its step read, so unroll agrees to
    # float32 rounding (measured 2e-6 over 64 ticks, flat in time).
    with torch.no_grad():
        unrolled = net.unroll(trajectory.initial_carry, trajectory.obs, trajectory.replay,
                              cases.resets(trajectory.obs))
    assert cases.scale_error(unrolled.values, trajectory.values) <= 1e-5
    logp, _ = policy.evaluate(unrolled.policy_logits, cases.joint_actions(trajectory))
    assert cases.scale_error(logp, trajectory.logp) <= 1e-5
    assert int(trajectory.initial_carry.nonfinite.sum()) == 0
