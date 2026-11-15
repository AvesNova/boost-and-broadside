"""The auxiliary heads' build switches (arm A13, `YemongConfig.next_state_head`,
`enemy_action_head`, `density_head`, `shot_head`): every head on by default and then the
model before them; a head that is off has no parameters, no output, no labels and no loss
term; without the shot heads nothing is re-simulated; without the enemy-action head the
belief advances with uniform actions; the two A13 settings sets build, train, save, load
and come in through `--set`."""

import dataclasses
import math

import arm_goldens
import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts import GameConfig
from bnb.contracts.action import NUM_JOINT
from bnb.learn.ppo import COUNTS, HEAD_SERIES, SHOT_LABELS, TERMS
from bnb.models.yemong import belief as beliefs
from bnb.models.yemong.model import SHOT_HEADS, YemongConfig
from bnb.runs.checkpoint import load_policy, save_policy
from bnb.runtime import metrics

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = lc.OWN
EVERY_HEAD = (
    "next_state",
    "enemy_action",
    "density",
    "shot_trajectory",
    "shot_outcome",
)
# A13's two settings sets on a new run's build (`docs/explanation/yemong.md`), as `--set` lines
A13_ALL_BUT_NEXT_STATE = (
    "yemong.enemy_action_head=false",
    "yemong.density_head=false",
    "yemong.shot_head=none",
    "yemong.geometry_head=false",
)
A13_ALL = (
    "yemong.next_state_head=false",
    "yemong.delta_split=false",
    "yemong.physics_hidden_decay=null",
    "yemong.forecast_feedback=false",
    *A13_ALL_BUT_NEXT_STATE,
)
NO_NEXT_STATE = dict(next_state_head=False, forecast_feedback=False)
# one switch at a time, on the record default build (no geometry or error head)
SWITCHES = {
    "enemy_action": dict(enemy_action_head=False),
    "density": dict(density_head=False),
    "shot": dict(shot_head="none"),
    "next_state": NO_NEXT_STATE,
}
# the heads each switch removes, by the learner's term names
REMOVED = {
    "enemy_action": ("enemy_action",),
    "density": ("density",),
    "shot": ("shot_trajectory", "shot_outcome"),
    "next_state": ("next_state",),
}
# each removed head's outputs (`Unrolled`)
OUTPUTS = {
    "enemy_action": ("enemy_action_logits",),
    "density": ("density_log_rates",),
    "shot": ("shot_outcome_logits", "shot_latents"),
    "next_state": ("next_state_logits",),
}
# each removed head's labels (`Learner.labels`)
LABELS = {
    "enemy_action": ("enemy_action", "enemy_mask"),
    "density": ("density",),
    "shot": ("shot_snapshots", "shot_ages", "shot_mask", "shot_outcome"),
    "next_state": ("next_state", "next_state_mask"),
}
# each removed head's modules (`Yemong` attributes)
MODULES = {
    "density": ("density",),
    "shot": ("shot_outcome", "shot_trajectory_head"),
}


def _a13(lines: tuple[str, ...]):
    """The model build of a new run with the A13 settings `lines`."""
    from bnb.cli import parse_setting, with_settings
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), [parse_setting(line) for line in lines]
    )
    return model_build(config).config


def _parameters(net) -> int:
    return sum(parameter.numel() for parameter in net.parameters())


def test_every_head_is_on_by_default_and_the_model_before_the_switches():
    """Every switch on, every head built, and the arms' step loops and unrolls as recorded
    before the switches existed (`arm_goldens`)."""
    config = YemongConfig()
    assert config.next_state_head and config.enemy_action_head and config.density_head
    assert config.shot_head == "trajectory" == SHOT_HEADS[0]
    net = cases.model()
    assert net.auxiliary_heads == EVERY_HEAD
    assert net.ship_head_names == ("enemy_action", "next_state")
    for arm in arm_goldens.ARMS:
        arm_goldens.reproduces(arm)


def test_unknown_or_inconsistent_switches_fail():
    with pytest.raises(ValueError, match="shot_head"):
        YemongConfig(shot_head="now")
    with pytest.raises(ValueError, match="next_state_head"):
        YemongConfig(next_state_head=False)  # the forecast would still be fed back
    with pytest.raises(ValueError, match="next_state_head"):
        YemongConfig(
            next_state_head=False, forecast_feedback=False, next_state="delta", delta_split=True
        )


@pytest.mark.parametrize("switch", SWITCHES)
def test_a_head_that_is_off_is_not_built_and_gives_no_output(switch):
    full, net = cases.model(), cases.model(**SWITCHES[switch])
    assert _parameters(net) < _parameters(full)
    for name in MODULES.get(switch, ()):
        assert getattr(net, name) is None
    assert not set(REMOVED[switch]) & set(net.auxiliary_heads)
    assert set(net.auxiliary_heads) == set(EVERY_HEAD) - set(REMOVED[switch])
    _, trajectory = cases.rollout(net, environments=2, length=6)
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(trajectory))
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
        view = dataclasses.replace(
            obs,
            ships=obs.ships[0],
            fields=obs.fields[0],
            globals=obs.globals[0],
            zones=obs.zones[0],
        )
        keys = torch.arange(obs.ships.shape[1])
        _, inspected = net.inspect(carry, view, cases.resets(obs)[0], keys)
    for name in OUTPUTS[switch]:
        assert getattr(unrolled, name) is None, name
        assert getattr(inspected, name) is None, name
    for name in ("enemy_action_logits", "next_state_logits"):
        assert (getattr(decisions, name) is None) == (name in OUTPUTS[switch]), name
    # the step loop and the unroll still agree
    error = cases.scale_error(unrolled.policy_logits, decisions.policy_logits)
    assert error <= 1e-4


def test_without_the_enemy_action_head_the_belief_advances_with_uniform_actions():
    leading = (2, 3)
    blank = beliefs.blank(leading, 4, 2, NUM_JOINT)
    held = dataclasses.replace(
        blank,
        valid=torch.ones(leading, dtype=torch.bool),
        action=torch.softmax(torch.randn(leading + (NUM_JOINT,)), -1),
    )
    advanced, nonfinite = beliefs.advance(held, torch.randn(leading + (6,)), None)
    assert torch.equal(advanced.action, torch.full_like(held.action, 1.0 / NUM_JOINT))
    assert torch.equal(nonfinite, torch.zeros(2, dtype=nonfinite.dtype))
    # in the model: every enemy row's belief keeps uniform actions while the forecast is fed
    net = cases.model(enemy_action_head=False)
    _, trajectory = cases.rollout(net, environments=2, length=6)
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        _, final = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(trajectory))
    assert torch.equal(final.belief.action, torch.full_like(final.belief.action, 1 / NUM_JOINT))
    assert (final.belief.spreads[:, OWN:] != 0).any(), "the next-state forecast is fed back"


@pytest.fixture(scope="module", params=list(SWITCHES) + ["a13_all_but_next_state", "a13_all"])
def trained(request, module_global_generator):
    """A learner update (with its cadence: the gradient decomposition, the head statistics)
    of each switch's build, and of the A13 builds (a new run's, small)."""
    name = request.param
    if name.startswith("a13"):
        lines = A13_ALL if name == "a13_all" else A13_ALL_BUT_NEXT_STATE
        net = cases.model(**dataclasses.asdict(_a13(lines)))
    else:
        net = cases.model(**SWITCHES[name])
    runner = lc.rollout(net, environments=4, length=16)
    learner = lc.learner(net, runner)
    views = torch.arange(runner.store.views)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    trajectory, shot_labels = next(learner.staged(runner.store, [views]))
    label = learner.labels(trajectory, shot_labels)
    denominator = dict(zip(COUNTS, prepared["counts"].sum(0).clamp_min(1).unbind(), strict=True))
    terms, *_ = learner.cadence_terms(
        trajectory, label, prepared["derived"], denominator, prepared["scale"]
    )
    out = metrics.read(learner.update(runner.store, runner.bootstrap(), cadence=True))
    return name, net, runner, shot_labels, label, terms, out


def _removed(name: str, net) -> set[str]:
    return set(HEAD_SERIES) - set(net.auxiliary_heads)


def test_an_absent_head_has_no_labels_no_term_and_no_series(trained):
    name, net, runner, shot_labels, label, terms, out = trained
    removed = _removed(name, net)
    if name in SWITCHES:
        assert removed >= set(REMOVED[name])
        for key in LABELS[name]:
            assert key not in label, key
    for term in removed:
        assert terms[term].item() == 0.0 and not terms[term].requires_grad, term
        assert HEAD_SERIES[term] not in out, term
        assert not any(key.startswith("gradient/") and key.endswith(f"/{term}") for key in out)
    for term in net.auxiliary_heads:
        assert terms[term].requires_grad, term
        assert math.isfinite(out[HEAD_SERIES[term]]), term
    assert math.isfinite(out["critic/value_loss/outcome"])
    assert all(torch.isfinite(p).all() for p in net.parameters())
    assert set(net.auxiliary_heads) <= set(TERMS)


def test_without_the_shot_heads_no_shot_is_resimulated(trained):
    name, net, runner, shot_labels, label, _, out = trained
    shots = net.shot_trajectory_head is not None
    assert (shot_labels is not None) == shots
    assert (SHOT_LABELS in runner.store.data) == shots
    assert ("shot/supervised_share" in out) == shots


def test_the_a13_settings_lines_build_their_heads():
    assert _a13(A13_ALL_BUT_NEXT_STATE) == dataclasses.replace(
        _a13(()),
        enemy_action_head=False,
        density_head=False,
        shot_head="none",
        geometry_head=False,
    )
    every = cases.model(**dataclasses.asdict(_a13(A13_ALL_BUT_NEXT_STATE)))
    assert every.auxiliary_heads == ("next_state", "correction")
    none = cases.model(**dataclasses.asdict(_a13(A13_ALL)))
    assert none.auxiliary_heads == ()
    assert none.ship_heads is None and none.correction_head is None


@pytest.mark.parametrize("lines", [A13_ALL_BUT_NEXT_STATE, A13_ALL], ids=["but_next", "all"])
def test_an_a13_policy_saves_and_loads(tmp_path, lines):
    net = cases.model(**dataclasses.asdict(_a13(lines)))
    save_policy(tmp_path / "policy", net)
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    assert loaded.build == net.build
    for (name, a), (other, b) in zip(
        net.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert name == other and torch.equal(a, b), name
    _, trajectory = cases.rollout(net, environments=2, length=4)
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    joint = cases.joint_actions(trajectory)
    with torch.no_grad():
        saved, _ = cases.step_loop(net, net.step, carry, obs, joint)
        again, _ = cases.step_loop(loaded, loaded.step, carry, obs, joint)
    assert torch.equal(saved.policy_logits, again.policy_logits)
    assert torch.equal(saved.values, again.values)
