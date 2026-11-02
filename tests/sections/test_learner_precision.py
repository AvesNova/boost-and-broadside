"""The precision settings (learner spec 8): TF32 and bf16 matmuls give finite losses and
gradients that agree with float32's to their mantissas' precision; bf16 is the default;
every update reports the signals of a precision problem."""

import copy
import dataclasses
import math
import types

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.learn.ppo import Learner, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import set_precision

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]
CUDA = torch.device("cuda") if torch.cuda.is_available() else None


@pytest.fixture(scope="module")
def stored():
    net = cases.model().to(CUDA)
    config = cases.SHORT_SQUAD
    controllers = {
        "learner": cases.controller(net, config),
        "chaser": Chaser(config.ships_per_team),
    }
    runner = Rollout(
        config,
        lc.POOLS,
        controllers,
        "learner",
        16,
        64,
        CUDA,
        compile=False,
        graph_ticks=16,
        host_store=True,
    )
    runner.run()
    runner.run()
    return net, runner, [runner.bootstrap()]


def _gradient(model, runner, bootstraps):
    """Every view's minibatch: the loss terms and the flattened gradient."""
    learner = Learner(
        model,
        LearnerConfig(minibatches=1),
        runner.rules,
        runner.carry["types"],
        RewardConfig().outcome,
        CUDA,
    )
    prepared = learner.prepare(runner.store, bootstraps)
    terms, _, _ = learner.gradient(runner.store, prepared, torch.arange(runner.store.views))
    return terms, torch.cat([parameter.grad.flatten() for parameter in model.parameters()])


def _bfloat16_copy(net):
    model = Yemong(
        yemong_build(
            YemongConfig(bfloat16_matmuls=True), api.rules_of(cases.SQUAD), api.ship_types()
        )
    ).to(CUDA)
    model.load_state_dict(net.state_dict())
    return model


@pytest.mark.parametrize(
    ("precision", "tolerance"),
    [
        # TF32 keeps 10 mantissa bits (a relative rounding of 5e-4 per product), bf16 7 (4e-3);
        # the bounds allow a few stacked matmuls' errors (measured on this case, as a share of
        # the gradient's norm: TF32 8e-4, bf16 1e-2; the terms agree ten times closer).
        ("tf32", 1e-2),
        ("bfloat16", 5e-2),
    ],
)
def test_a_precision_setting_agrees_with_float32(stored, precision, tolerance):
    net, runner, bootstraps = stored
    reference_terms, reference = _gradient(net, runner, bootstraps)
    try:
        if precision == "tf32":
            set_precision("tf32")
            model = net
        else:
            model = _bfloat16_copy(net)
        terms, gradient = _gradient(model, runner, bootstraps)
    finally:
        set_precision("float32")
    assert torch.isfinite(terms).all() and torch.isfinite(gradient).all()
    torch.testing.assert_close(terms, reference_terms, rtol=tolerance, atol=1e-6)
    assert (gradient - reference).norm() <= tolerance * reference.norm()
    assert not torch.equal(gradient, reference)  # the setting took effect


def test_training_defaults_to_bfloat16():
    """TrainConfig, the command line and the model a run builds all default to bf16."""
    from bnb.cli import Train
    from bnb.runtime.train import TrainConfig, model_build

    config = TrainConfig(game=cases.SQUAD)
    assert config.precision == "bfloat16" and Train().precision == "bfloat16"
    assert model_build(config).config.bfloat16_matmuls


def test_the_update_reports_its_precision_signals(stored):
    """Every update reports the step/unroll ratio drift, the first epoch's clip fraction and
    the minibatch steps whose gradient was not finite; a non-finite gradient counts once per
    step."""
    net, runner, bootstraps = stored
    config = LearnerConfig(epochs=1, minibatches=2)
    learner = Learner(
        copy.deepcopy(net),
        config,
        runner.rules,
        runner.carry["types"],
        RewardConfig().outcome,
        CUDA,
    )
    out = learner.update(runner.store, bootstraps)
    assert {"ratio_drift_max", "ratio_drift_mean", "clip_fraction"} <= out.keys()
    assert out["nonfinite_gradients"] == 0
    model = copy.deepcopy(net)
    next(model.parameters()).register_hook(lambda gradient: gradient * math.nan)
    learner = Learner(
        model, config, runner.rules, runner.carry["types"], RewardConfig().outcome, CUDA
    )
    assert learner.update(runner.store, bootstraps)["nonfinite_gradients"] == 2


def test_the_trainer_reports_the_carrys_nonfinite_beliefs(stored):
    """The trainer's row reads the learner carry's count of non-finite belief entries."""
    from bnb.runtime.train import LEARNER, Trainer

    _, runner, _ = stored
    carry = runner.carry["ctrl"][LEARNER]
    trainer = types.SimpleNamespace(rollout=types.SimpleNamespace(carry={"ctrl": {LEARNER: carry}}))
    assert Trainer.nonfinite_beliefs(trainer) == int(carry.nonfinite.sum()) == 0
    counted = dataclasses.replace(carry, nonfinite=torch.tensor([1, 0, 2], device=CUDA))
    trainer.rollout.carry["ctrl"][LEARNER] = counted
    assert Trainer.nonfinite_beliefs(trainer) == 3
