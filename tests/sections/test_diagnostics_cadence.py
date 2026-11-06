"""The cadence groups (diagnostics audit, group 19 and `repr/`): the gradient decomposition
and the representation's series against hand-computed cases, and the learner's cadence."""

import copy
import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts.obs import GLOBAL
from bnb.learn import gradients
from bnb.runtime import metrics

NAMES = ("policy", "entropy", "value")


def _decomposed(half_a: torch.Tensor, half_b: torch.Tensor, trunk_size: int = 2) -> dict:
    """The series of basic gradients (B, P) per half, every name its own basic term."""
    grams = gradients.grams(torch.cat([half_a, half_b]), trunk_size)
    return metrics.read(
        gradients.decomposition(
            grams,
            torch.eye(len(NAMES)),
            NAMES,
            NAMES,
            {"terms": NAMES},
            (),
            actor=("policy", "entropy"),
            critic=("value",),
        )
    )


def test_the_gradient_decomposition_matches_a_hand_computed_case():
    """policy (1, 0, 0), entropy (0, 1, 0), value (1, 0, 2), the first two coordinates the
    trunk, each half holding half of each gradient: the step is (2, 1, 2)."""
    full = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 2.0]])
    out = _decomposed(full / 2, full / 2)
    assert out["gradient/norm/model/value"] == pytest.approx(math.sqrt(5))
    assert out["gradient/norm/trunk/value"] == pytest.approx(1.0)
    shares = {name: out[f"gradient/share/model/{name}"] for name in NAMES}
    assert shares == pytest.approx(dict(policy=2 / 9, entropy=1 / 9, value=6 / 9))
    trunk = {name: out[f"gradient/share/trunk/{name}"] for name in NAMES}
    assert trunk == pytest.approx(dict(policy=2 / 5, entropy=1 / 5, value=2 / 5))
    assert out["gradient/cosine/trunk/policy/entropy"] == pytest.approx(0.0)
    assert out["gradient/cosine/trunk/policy/value"] == pytest.approx(1.0)
    assert out["gradient/agreement/model/terms"] == pytest.approx(3 / (2 + math.sqrt(5)))
    assert out["gradient/coherent_fraction/policy"] == pytest.approx(1.0)
    assert out["gradient/actor_norm"] == pytest.approx(math.sqrt(2))
    assert out["gradient/actor_share"] == pytest.approx(math.sqrt(2) / (math.sqrt(2) + 1))
    assert "gradient/cosine/model/policy/entropy" not in out  # cosines in the trunk only


def test_halves_that_disagree_are_incoherent():
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    b = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, -1.0, 0.0]])
    out = _decomposed(a, b)
    assert out["gradient/coherent_fraction/policy"] == pytest.approx(-1.0)
    # 0.5 / ((1 + 0.25) / 2): agreeing halves of unequal size
    assert out["gradient/coherent_fraction/entropy"] == pytest.approx(0.8)
    assert out["gradient/coherent_fraction/value"] == pytest.approx(0.0)


def test_the_effective_rank_and_the_dormant_share_match_hand_computed_cases():
    eye = torch.eye(4)
    assert gradients.effective_rank(torch.cat([eye, -eye])) == pytest.approx(4.0)
    line = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    assert gradients.effective_rank(line) == pytest.approx(1.0)
    hidden = torch.tensor([[1.0, -1.0, 2.0, 0.0], [1.0, 1.0, -2.0, 0.01]])
    assert gradients.dormant_share(hidden) == 0.25  # the last unit: 0.005 against 0.92


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model()
    return net, lc.rollout(net)


def test_the_taps_read_each_blocks_output_and_are_removed(segment, global_generator):
    """The last block's tapped output is what the heads read, and no hook outlives the
    context."""
    _assert_taps_read_the_block_outputs(*segment)


def test_the_taps_read_each_blocks_output_in_the_temporal_first_order(global_generator):
    """With `order = temporal_first` a block's last sublayer is spatial: the taps read it,
    not the temporal sublayer that runs first."""
    net = cases.model(order="temporal_first")
    _assert_taps_read_the_block_outputs(net, lc.rollout(net))


def _assert_taps_read_the_block_outputs(net, runner) -> None:
    trajectory = lc.whole(runner)
    heads_input = {}
    handle = net.own_heads.register_forward_pre_hook(
        lambda module, inputs: heads_input.setdefault("tokens", inputs[0])
    )
    reset = trajectory.obs.globals[..., GLOBAL["reset"]] > 0.5
    with torch.no_grad(), gradients.block_taps(net) as taps:
        net.unroll(trajectory.initial_carry, trajectory.obs, trajectory.replay, reset)
    handle.remove()
    last = taps[f"trunk_block_{len(net.trunk.blocks) - 1}"]
    own = trajectory.log_probability.shape[-1]
    torch.testing.assert_close(last["output"][..., :own, :], heads_input["tokens"])
    assert last["hidden"].shape[-1] == 4 * net.config.width
    hooked = [
        module for module in net.modules() if module._forward_hooks or module._forward_pre_hooks
    ]
    assert not hooked


@pytest.mark.slow
def test_the_cadence_measures_its_groups_and_does_not_change_the_update(segment, global_generator):
    net, runner = segment
    results = []
    for cadence in (True, False):
        model = copy.deepcopy(net)
        learner = lc.with_micro_batch(lc.learner(model, runner, epochs=2, minibatches=2), 2)
        out = metrics.read(learner.update(runner.store, runner.bootstrap(), cadence=cadence))
        results.append((out, [p.detach().clone() for p in model.parameters()]))
        if cadence:  # the cadence's gradients are taken with autograd.grad, and .grad cleared
            assert all(p.grad is None for p in model.parameters())
    (on, on_weights), (off, off_weights) = results
    assert all(torch.equal(a, b) for a, b in zip(on_weights, off_weights, strict=True))
    assert not any(name.startswith(("gradient/", "representation/")) for name in off)
    top = ("policy", "entropy_power", "entropy_stick", "entropy_shoot", "value", "next_state")
    top += ("enemy_action", "density")
    top += ("shot_trajectory", "shot_outcome")
    for scope in ("model", "trunk"):
        assert sum(on[f"gradient/share/{scope}/{term}"] for term in top) == pytest.approx(1.0)
    levels = [name for name in on if name.startswith("gradient/norm/model/value_")]
    assert len(levels) == 5
    shares = [on[f"gradient/coherent_share/next_state_{name}"] for name in ("ally",)]
    assert all(math.isfinite(value) for value in shares)
    cosines = [value for name, value in on.items() if name.startswith("gradient/cosine/")]
    assert len(cosines) == math.comb(len(top) + 5 + 3, 2)
    assert all(-1.0 - 1e-5 <= value <= 1.0 + 1e-5 for value in cosines)
    for block in range(len(net.trunk.blocks)):
        for quantity in ("activation_rms", "effective_rank", "dormant_share"):
            assert math.isfinite(on[f"representation/{quantity}/trunk_block_{block}"])
    assert learner.cadence_stopwatch.seconds() is None  # the last learner ran no cadence
