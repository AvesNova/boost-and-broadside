"""Arm A22's shot damage head (`YemongConfig.shot_head = "damage"`): in place of the shot
trajectory and outcome heads, for each own ship and every other ship the damage a shot
fired now would deal it, from the learner's shot re-simulation. Off by default; built
without the trajectory heads and cheaper than them; its labels are the re-simulated hit's
target and damage; it trains, saves and loads, and comes in through `--set`."""

import dataclasses
import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts import GameConfig
from bnb.learn.ppo import COUNTS, HEAD_SERIES, SHOT_LABELS, TERMS
from bnb.models.yemong import codes
from bnb.models.yemong.model import SHOT_HEADS, YemongConfig
from bnb.models.yemong.shots import (
    DAMAGE_BINS,
    DAMAGE_HIGH,
    DAMAGE_WIDTH,
    damage_code,
    damage_targets,
)
from bnb.runs.checkpoint import load_policy, save_policy
from bnb.runtime import metrics

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = lc.OWN
DAMAGE = dict(shot_head="damage")
# a crowded arena, so a short segment's own shots hit
CROWDED = GameConfig(5, 10, 600.0, episode_seconds=3.0, respawn_seconds=1.0)


def _parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_the_trajectory_heads_are_the_default_and_damage_one_more_choice():
    assert YemongConfig().shot_head == "trajectory"
    assert SHOT_HEADS == ("trajectory", "damage", "none")
    assert "shot_damage" in TERMS and HEAD_SERIES["shot_damage"] == "shot/loss/damage"


def test_the_damage_head_replaces_the_trajectory_heads_and_costs_less():
    full, net = cases.model(), cases.model(**DAMAGE)
    assert net.shot_outcome is None and net.shot_trajectory_head is None
    assert full.shot_damage_head is None
    trajectory = _parameters(full.shot_outcome) + _parameters(full.shot_trajectory_head)
    assert _parameters(net.shot_damage_head) < trajectory
    assert _parameters(net) - _parameters(net.shot_damage_head) == _parameters(full) - trajectory
    heads = ("next_state", "enemy_action", "density", "shot_damage")
    assert net.auxiliary_heads == heads


def test_the_damage_targets_and_code():
    target = torch.tensor([[2, -1, 0, 1]])
    damage = torch.tensor([[0.7, 0.0, 1.2, 0.3]])
    pairs = damage_targets(target, damage, shooters=2)
    assert pairs.shape == (1, 2, 4)
    assert torch.equal(pairs, torch.tensor([[[0.0, 0.0, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0]]]))
    shares = torch.tensor([0.0, 0.35, 1.0, DAMAGE_HIGH, 5.0])
    code = damage_code(shares)
    assert code.shape == (5, DAMAGE_WIDTH) == (5, DAMAGE_BINS + 2 * codes.PADDING_BINS)
    torch.testing.assert_close(code.sum(-1), torch.ones(5))
    mean, _ = codes.hl_gauss_scalar_decode(code, 0.0, DAMAGE_HIGH, DAMAGE_BINS)
    torch.testing.assert_close(mean, shares.clamp_max(DAMAGE_HIGH), rtol=0, atol=1e-5)


def test_a_damage_build_gives_its_latents_and_the_step_reads_no_shot_head():
    net = cases.model(**DAMAGE)
    _, trajectory = cases.rollout(net, environments=4, length=6)
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(trajectory))
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
        logits = net.shot_damage(unrolled.shot_damage_latents, OWN)
    assert unrolled.shot_latents is None and unrolled.shot_outcome_logits is None
    rows = obs.ships.shape[-2]
    assert unrolled.shot_damage_latents.shape == obs.ships.shape[:2] + (rows, 2 * net.config.width)
    assert logits.shape == obs.ships.shape[:2] + (OWN, rows, DAMAGE_WIDTH)
    assert cases.scale_error(unrolled.policy_logits, decisions.policy_logits) <= 1e-4


@pytest.fixture(scope="module")
def trained(module_global_generator):
    """A learner update of a damage build, with its cadence (the gradient decomposition
    and the head statistics), and its labels."""
    net = cases.model(**DAMAGE)
    runner = lc.rollout(net, environments=8, length=32, config=CROWDED)
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
    return net, runner, trajectory, shot_labels, label, terms, out


def test_the_damage_labels_are_the_resimulated_hits(trained):
    """Each own shot's pair labels hold its hit's damage share at the row it hit and 0 at
    every other row; the loss mask holds valid own shots only; no trajectory label."""
    net, runner, trajectory, shot_labels, label, _, _ = trained
    assert SHOT_LABELS in runner.store.data
    for name in ("shot_snapshots", "shot_ages", "shot_outcome"):
        assert name not in label, name
    pairs, mask = label["shot_damage"], label["shot_mask"]
    rows = mask.shape[-1]
    assert pairs.shape == mask.shape[:2] + (OWN, rows)
    assert not mask[..., OWN:].any() and mask[..., :OWN].any()
    shots = shot_labels["shots"]
    hit = shots.target[..., :OWN] >= 0
    assert (hit & mask[..., :OWN]).any(), "the segment should hold hits"
    torch.testing.assert_close(pairs.sum(-1), torch.where(hit, shots.damage[..., :OWN], 0.0))
    assert ((pairs > 0).sum(-1) <= 1).all()
    assert (shots.damage >= 0).all() and (shots.damage[shots.target < 0] == 0).all()


def test_the_damage_term_trains_and_reports(trained):
    net, _, _, _, _, terms, out = trained
    assert terms["shot_damage"].requires_grad and terms["shot_damage"].item() > 0
    for term in ("shot_trajectory", "shot_outcome"):
        assert terms[term].item() == 0.0 and not terms[term].requires_grad, term
        assert HEAD_SERIES[term] not in out, term
    assert math.isfinite(out["shot/loss/damage"])
    assert any(key.startswith("gradient/") and key.endswith("/shot_damage") for key in out)
    assert all(torch.isfinite(p).all() for p in net.parameters())


def _settings(lines: tuple[str, ...]):
    from bnb.cli import parse_setting, with_settings
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), [parse_setting(line) for line in lines]
    )
    return model_build(config).config


def test_the_arms_settings_line_builds_it_and_it_saves_and_loads(tmp_path):
    config = _settings(("yemong.shot_head=damage",))
    assert config == dataclasses.replace(_settings(()), shot_head="damage")
    net = cases.model(**dataclasses.asdict(config))
    assert "shot_damage" in net.auxiliary_heads and "shot_trajectory" not in net.auxiliary_heads
    save_policy(tmp_path / "policy", net)
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    assert loaded.build == net.build
    for (name, a), (other, b) in zip(
        net.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert name == other and torch.equal(a, b), name
