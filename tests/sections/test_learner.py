"""The learner (section 6), acceptance items 2, 3 and 5, and its data path: each loss term
against an independent computation on a real stored segment, micro-batches against the
unsplit minibatch, the labels' alignment, the reset flags unroll receives, the bootstrap,
and the replay compaction."""

import copy
import dataclasses
import math

import learner_cases as lc
import pytest
import torch
import torch.nn.functional as F
import yemong_cases as cases
from torch.utils import _pytree

from bnb.contracts import Obs, Outputs, Pool
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.learn import labels
from bnb.learn.ppo import COUNTS, TERMS, explained_variance, random_stream
from bnb.models.relational import RelationalConfig
from bnb.models.yemong import critic, policy
from bnb.models.yemong.density import HexGrid, poisson_terms
from bnb.models.yemong.inputs import believed, truth
from bnb.runtime import metrics

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


OWN = lc.OWN


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model()
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    prepared = learner.prepare(runner.store, [runner.bootstrap()])
    return net, runner, learner, prepared


def _resets(obs: Obs) -> torch.Tensor:
    return obs.globals[..., GLOBAL["reset"]] > 0.5


# ------------------------------------------------------------------ the data path


def test_unroll_receives_the_reset_before_each_step(segment):
    """unroll's reset[t] is done[t - 1] (the episode ended after the previous step), read off
    the observation; the trajectory's done is one step later and would be wrong."""
    net, runner, learner, _ = segment
    trajectory = lc.whole(runner)
    reset, done = _resets(trajectory.obs), trajectory.done
    assert done.any() and torch.equal(reset[1:], done[:-1])
    joint = cases.joint_actions(trajectory)
    with torch.no_grad():
        right = learner.unroll(trajectory)
        wrong = net.unroll(trajectory.initial_carry, trajectory.obs, trajectory.replay, done)
    from bnb.models.yemong import policy

    right_log_probability, _ = policy.evaluate(right.policy_logits, joint)
    wrong_log_probability, _ = policy.evaluate(wrong.policy_logits, joint)
    assert cases.scale_error(right_log_probability, trajectory.log_probability) <= 1e-5
    assert cases.scale_error(wrong_log_probability, trajectory.log_probability) > 1e-2


def test_the_bootstrap_is_the_next_segments_first_decision():
    net = cases.model()
    runner = lc.rollout(net, runs=1)
    bootstrap = runner.bootstrap()
    following = runner.run()
    assert cases.scale_error(following.values[0], bootstrap.values) <= 1e-6
    assert cases.scale_error(following.outcome[0], bootstrap.outcome) <= 1e-6


def test_the_compacted_replay_rebuilds_the_own_rows_sight_gave():
    """The replay holds the enemy rows; an own row is the observation's truth, valid, seen
    now, with no spread: what the step's belief held after sight."""
    net = cases.model(dtype=torch.float64)
    runner = lc.rollout(cases.model(), runs=1, length=16)
    trajectory = runner.trajectory(0)
    obs = cases.cast(
        Obs(
            ships=trajectory.obs.ships[-1],
            fields=trajectory.obs.fields[-1],
            globals=trajectory.obs.globals[-1],
            zones=trajectory.obs.zones[-1],
        ),
        torch.float64,
    )
    carry = net.initial_carry(obs.ships.shape[0], OWN, 10)
    with torch.no_grad():
        decision = net.step(carry, obs, torch.zeros(obs.ships.shape[0], dtype=torch.bool))
    valid, age, means, spreads = believed(obs.ships, decision.replay)
    assert decision.replay.means.shape[-2] == 10 - OWN
    assert valid[:, :OWN].all() and (age[:, :OWN] == 0).all() and (spreads[:, :OWN] == 0).all()
    assert torch.equal(means[:, :OWN], truth(obs.ships[:, :OWN], decision.replay.offset))
    assert torch.equal(valid, decision.carry.belief.valid)  # what sight left valid


def test_an_own_privileged_row_is_its_observed_row():
    """Why the store keeps only the enemy rows of the privileged view."""
    from bnb.game import api

    rules = api.rules_of(cases.SHORT_SQUAD)
    state = api.reset(rules, api.ship_types(), api.uniforms(rules, 16), torch.arange(16) % 2 == 0)
    _, obs, privileged = api.perceive(rules, state)
    assert torch.equal(obs.ships[:, :, :OWN], privileged.ships[:, :, :OWN])


# ------------------------------------------------------------------ the loss terms


def _independent_two_hot(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    values = values.clamp(bins[0], bins[-1])
    upper = torch.bucketize(values, bins, right=True).clamp(1, bins.numel() - 1)
    lower = upper - 1
    weight = ((values - bins[lower]) / (bins[upper] - bins[lower])).clamp(0, 1)
    return (
        F.one_hot(lower, bins.numel()) * (1 - weight)[..., None]
        + F.one_hot(upper, bins.numel()) * weight[..., None]
    )


def _grouped_cross_entropy(logits, target, runs) -> torch.Tensor:
    total, start = 0.0, 0
    for groups, size in runs:
        width = groups * size
        part = logits[..., start : start + width].unflatten(-1, (groups, size))
        goal = target[..., start : start + width].unflatten(-1, (groups, size))
        total = total - (goal * part.log_softmax(-1)).sum((-1, -2))
        start += width
    return total


def test_each_loss_term_matches_an_independent_computation(segment):
    net, runner, learner, prepared = segment
    views = torch.arange(runner.store.views)
    trajectory, shot_labels = next(learner.staged(runner.store, [views]))
    label = learner.labels(trajectory, shot_labels)
    denominator = dict(zip(COUNTS, prepared["counts"].sum(0).clamp_min(1).unbind()))
    derived = prepared["derived"]
    with torch.no_grad():
        _, terms, per_level, _, _, _ = learner.loss(
            trajectory, label, derived, denominator, prepared["scale"]
        )
        out = learner.unroll(trajectory)
    got = dict(zip(TERMS, terms.tolist()))
    ships = trajectory.obs.ships
    alive = ships[..., SHIP["alive"]] > 0.5
    own_alive = alive[..., :OWN]
    action = trajectory.action
    joint = to_joint(action).long()

    # policy and entropy, over living own ships
    log_probabilities = out.policy_logits.log_softmax(-1)
    log_probability = log_probabilities.gather(-1, joint[..., :OWN, None])[..., 0]
    ratio = (log_probability - derived["old_log_probability"]).exp()
    advantage = derived["advantage"] / prepared["scale"]
    surrogate = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2))
    assert math.isclose(
        got["policy"], surrogate[own_alive].mean().item(), rel_tol=1e-5, abs_tol=1e-7
    )
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
    assert math.isclose(got["entropy"], entropy[own_alive].mean().item(), rel_tol=1e-5)

    # value: per-ship levels over valued ships, the outcome per view-step
    valued = own_alive | (ships[..., :OWN, SHIP["respawn_wait"]] > 0)
    expected = []
    for level in range(5):
        if level == 4:
            ce = -(derived["outcome_targets"] * out.outcome_logits.log_softmax(-1)).sum(-1)
            expected.append(ce.mean())
        else:
            target = _independent_two_hot(derived["level_returns"][..., level], net.bins)
            ce = -(target * out.value_logits[..., level, :].log_softmax(-1)).sum(-1)
            expected.append(ce[valued].mean())
    torch.testing.assert_close(per_level, torch.stack(expected), rtol=1e-5, atol=1e-6)
    assert math.isclose(got["value"], sum(e.item() for e in expected), rel_tol=1e-5)

    # next state: every row with a valid belief at a step that did not end an episode,
    # against the sharp code of its truth at t + 1 (own rows: the observation)
    following = torch.cat(
        [torch.cat([ships[1:, :, :OWN], trajectory.truth[1:]], -2), trajectory.final_truth[None]], 0
    )
    target = net.state_code.sharp(
        truth(following, trajectory.replay.offset), following[..., SHIP["ship_type"]].long()
    )
    valid = torch.cat([torch.ones_like(trajectory.replay.valid), trajectory.replay.valid], -1)
    mask = valid & ~trajectory.done[..., None]
    ce = _grouped_cross_entropy(out.next_state_logits, target, net.state_code.runs)
    assert math.isclose(got["next_state"], ce[mask].mean().item(), rel_tol=1e-5)

    # enemy action: living enemies' actual joint actions
    enemy = (
        -out.enemy_action_logits[..., OWN:, :]
        .log_softmax(-1)
        .gather(-1, joint[..., OWN:, None])[..., 0]
    )
    assert math.isclose(got["enemy_action"], enemy[alive[..., OWN:]].mean().item(), rel_tol=1e-5)

    # density: per view-step, the Poisson terms over the cells over the valued-ship count
    rows = torch.cat([ships[..., :OWN, :], trajectory.truth], -2)
    grid = HexGrid(net.config.hex_rings)
    counts = grid.counts(
        rows[..., SHIP["x"]],
        rows[..., SHIP["y"]],
        (torch.arange(10) < OWN).expand(rows.shape[:-1]),
        rows[..., SHIP["alive"]] > 0.5,
        trajectory.obs.globals[..., GLOBAL["arena_radius"]][..., None],
    )
    per_cell, saturated = poisson_terms(out.density_log_rates, counts)
    per_step = (per_cell - saturated).sum(-1) / valued.sum(-1).clamp_min(1)
    assert math.isclose(got["density"], per_step.mean().item(), rel_tol=1e-5)

    # shots: valid shots of own ships and visible enemies, the trajectory averaged over
    # each shot's valid ages
    shot = shot_labels["shots"]
    visible = ships[..., SHIP["visible"]] > 0.5
    shooters = shot.valid & torch.cat([torch.ones_like(visible[..., :OWN]), visible[..., OWN:]], -1)
    assert shooters.any() and (shot.hit[shooters] > 0).any()
    ages = torch.arange(1, 13, dtype=torch.float32)
    logits = net.shot_trajectory(out.shot_latents, ages)
    from bnb.models.yemong import shots as shot_codes

    goal = shot_codes.trajectory_code(net.shot_code, shot.snapshots)
    per_age = _grouped_cross_entropy(logits, goal, learner.shot_runs)  # (..., ages)
    per_shot = torch.stack(
        [
            per_age[tuple(index)][shot.age_valid[tuple(index)]].mean()
            for index in shooters.nonzero().unbind(0)
        ]
    )
    assert math.isclose(got["shot_trajectory"], per_shot.mean().item(), rel_tol=1e-5)
    outcome_goal = shot_codes.outcome_code(net.shot_code, shot.hit, shot.distance, shot.side)
    outcome = -(outcome_goal * out.shot_outcome_logits.log_softmax(-1)).sum(-1)
    assert math.isclose(got["shot_outcome"], outcome[shooters].mean().item(), rel_tol=1e-5)


# ------------------------------------------------------------------ micro-batching


def test_micro_batches_accumulate_to_the_unsplit_minibatch(segment):
    net, runner, learner, prepared = segment
    views = torch.arange(runner.store.views)

    def gradient(size: int):
        lc.with_micro_batch(learner, size)
        terms, levels, statistics, _, _ = learner.gradient(runner.store, prepared, views)
        return terms, [parameter.grad.clone() for parameter in net.parameters()]

    whole_terms, whole = gradient(runner.store.views)
    split_terms, split = gradient(runner.store.views // 4)
    lc.with_micro_batch(learner, runner.store.views)
    torch.testing.assert_close(split_terms, whole_terms, rtol=1e-5, atol=1e-6)
    for a, b in zip(split, whole, strict=True):
        scale = max(1e-12, b.abs().max().item())
        assert (a - b).abs().max().item() <= 1e-5 * scale + 1e-8
    assert sum(g.abs().sum().item() for g in whole) > 0


# ------------------------------------------------------------------ labels


class Constant:
    """Every ship of the team takes one fixed action."""

    reads = ()

    def __init__(self, ships: int, joint: int) -> None:
        self.ships, self.joint = ships, joint

    def init_carry(self, views, device):
        return None

    def step(self, carry, obs, reset):
        from bnb.contracts.action import from_joint

        joint = torch.full((reset.shape[0], self.ships), self.joint, dtype=torch.int32)
        return from_joint(joint), carry, Outputs()


def test_labels_are_row_aligned_and_in_the_models_frame():
    net = cases.model()
    config = cases.SHORT_SQUAD
    pools = (Pool("constant", 1.0, ("learner", "constant"), 0.5),)
    controllers = {"learner": cases.controller(net, config), "constant": Constant(OWN, 31)}
    runner = lc.rollout(net, pools=pools, controllers=controllers, runs=1)
    trajectory = lc.whole(runner)
    enemy, mask = labels.enemy_actions(trajectory)
    assert mask.any() and (enemy[mask] == 31).all()

    rows = labels.privileged_rows(trajectory)
    target = labels.next_state_targets(net.state_code, trajectory, rows)
    ships = trajectory.obs.ships
    # A row visible at t + 1, in the same episode: its target is the sharp code of what
    # the view then observed, shifted by the view's offset.
    seen = (ships[1:, ..., SHIP["visible"]] > 0.5) & ~trajectory.done[:-1, :, None]
    observed = net.state_code.sharp(
        truth(ships[1:], trajectory.replay.offset[:-1]), ships[1:, ..., SHIP["ship_type"]].long()
    )
    assert seen[..., OWN:].any()
    torch.testing.assert_close(target[:-1][seen], observed[seen], rtol=0, atol=1e-6)

    # The density target sums to the living ships per side, for every ship on the grid:
    # with the arena radius read as 4 times larger, every ship (all within 1.3 radii) is.
    grid = HexGrid(net.config.hex_rings)
    globals_ = trajectory.obs.globals.clone()
    globals_[..., GLOBAL["arena_radius"]] *= 4.0
    wide = dataclasses.replace(
        trajectory, obs=dataclasses.replace(trajectory.obs, globals=globals_)
    )
    distance = torch.hypot(rows[..., SHIP["x"]], rows[..., SHIP["y"]])
    assert (distance < 3.0 * globals_[..., GLOBAL["arena_radius"]][..., None] / 4.0).all()
    counts = labels.density_targets(grid, wide, rows)
    living = rows[..., SHIP["alive"]] > 0.5
    cells = grid.cells
    torch.testing.assert_close(
        counts[..., :cells].sum(-1), living[..., :OWN].sum(-1).float(), rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        counts[..., cells:].sum(-1), living[..., OWN:].sum(-1).float(), rtol=0, atol=1e-5
    )


def test_the_store_records_every_rows_action(segment):
    """trajectory.action holds every row in view order: a view's own rows are what its
    step took (the next replay's previous action), and in self-play its enemy rows are the
    other view's own rows."""
    from bnb.contracts.action import to_joint

    _, runner, _, _ = segment
    trajectory = lc.whole(runner)
    joint = to_joint(trajectory.action)
    assert joint.shape[-1] == 10
    kept = ~_resets(trajectory.obs)[1:]
    previous = trajectory.replay.previous_action[1:]
    assert torch.equal(previous[kept], joint[:-1, :, :OWN][kept])
    lineup = runner.lineup
    environment, role = lineup.environments["learner"], lineup.role["learner"]
    first = (role == 0).nonzero()[:, 0]
    for view in first.tolist():
        partner = ((environment == environment[view]) & (role == 1)).nonzero()[:, 0]
        if partner.numel():  # a self-play environment: both teams are learner views
            assert torch.equal(joint[:, view, OWN:], joint[:, partner[0], :OWN])


@pytest.fixture(scope="module")
def decided(module_global_generator):
    """A self-play and chaser rollout whose every first episode end is team 0's win (the
    score is set to 1-0 before the segment; the clock decides team deathmatch by score), with
    the outcome weighted zero, so its reward cannot tell the result."""
    from bnb.learn.rewards import RewardConfig

    net = cases.model()
    controllers = {"learner": cases.controller(net, cases.SHORT_SQUAD), "chaser": Chaser(OWN)}
    runner = lc.Rollout(
        cases.SHORT_SQUAD,
        api.ship_types(lc.CPU),
        lc.POOLS,
        controllers,
        "learner",
        16,
        48,
        lc.CPU,
        compiled=False,
        reward_config=RewardConfig(outcome=0.0),
    )
    runner.carry["state"].mode.score[:, 0] = 1
    team = runner._team("learner", runner.carry["seat"])  # (R,) each view's, until its reset
    return net, runner, team, runner.run()


@pytest.mark.slow
def test_the_view_records_its_teams_match_result(decided):
    """Team 0's win is a team-1 view's loss, and the result is zero where no episode
    ended."""
    _, _, team, trajectory = decided
    ended = trajectory.done
    first = ended & (ended.int().cumsum(0) == 1)
    assert (first & (team == 0)).any() and (first & (team == 1)).any()
    assert (trajectory.result[~ended] == 0).all()
    expected = torch.where(team == 0, 1, -1).expand_as(first)
    assert torch.equal(trajectory.result[first], expected[first].int())


@pytest.mark.slow
def test_the_outcome_targets_read_the_match_result_at_any_outcome_weight(decided):
    """With the outcome weighted zero its reward is zero, but a decided match still ends in a
    win or a loss, and the targets' class at an episode end is the view's result."""
    net, runner, _, trajectory = decided
    ended = trajectory.done
    assert (trajectory.reward[..., -1] == 0).all()
    assert (trajectory.result[ended] > 0).any() and (trajectory.result[ended] < 0).any()
    learner = lc.learner(net, runner)
    targets = learner.prepare(runner.store, [runner.bootstrap()])["derived"]["outcome_targets"]
    assert torch.equal(targets[ended].argmax(-1), critic.outcome_class(trajectory.result[ended]))


def test_an_update_consumes_several_shards():
    """Two segments collected one after the other, each from the carry the last ended with,
    and one update over both (the relational model, so it is quick)."""
    from bnb.contracts import GameConfig
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import LEARNER, TrainConfig, Trainer

    cfg = TrainConfig(
        game=GameConfig(2, 3, 1200.0, episode_seconds=2.0),
        environments=8,
        length=16,
        shards=2,
        model="relational",
        relational=RelationalConfig(hidden=32),
        compiled=False,
        pools=(Pool("chaser", 1.0, (LEARNER, "chaser")),),
        learner=LearnerConfig(minibatches=2, micro_batch_rows=16),
    )
    trainer = Trainer(cfg, torch.device("cpu"))
    store, bootstraps = trainer.collect()
    assert store.views == 16 and len(bootstraps) == 2
    first, second = (trainer.rollout.trajectory(shard) for shard in range(2))
    # The second segment continues the first: its first observation is the first's last
    # bootstrap's (the 60-tick episode's clock moved on by one tick).
    progress = first.obs.globals[..., GLOBAL["episode_progress"]]
    following = second.obs.globals[0, :, GLOBAL["episode_progress"]]
    continued = ~first.done[-1]
    torch.testing.assert_close(following[continued], progress[-1][continued] + 1.0 / 60.0)
    stats = metrics.read(trainer.ppo.update(store, bootstraps))
    assert stats["policy/minibatch_steps"] == 8
    assert math.isfinite(stats["critic/value_loss/outcome"])


# ------------------------------------------------------------------ the update's other paths


def test_the_stored_shot_labels_are_the_updates_resimulation(segment):
    """prepare re-simulates every view once, from the learner's own spread stream, and
    stores the labels; each epoch's micro-batch reads them back unchanged."""
    _, runner, learner, _ = segment
    store = runner.store
    spread = torch.randn(
        (store.length, store.views, 10, 2), generator=random_stream(0, "shot_spread")
    )
    views = torch.arange(3, 7)  # not one of prepare's micro-batches
    trajectory, stored = next(learner.staged(store, [views]))
    shot, mask = learner.shots(trajectory, spread[:, views])
    assert mask.any() and torch.equal(stored["mask"], mask)
    for a, b in zip(_pytree.tree_leaves(stored["shots"]), _pytree.tree_leaves(shot), strict=True):
        assert torch.equal(a, b)


def test_recomputed_old_log_probabilities_are_the_unrolls(segment):
    """ "rollout" keeps the rollout's log-probabilities; "recomputed" takes unroll's, which
    differ from them only by the step/unroll drift."""
    net, runner, _, prepared = segment
    trajectory = lc.whole(runner)
    assert torch.equal(prepared["derived"]["old_log_probability"], trajectory.log_probability)
    recomputed = lc.learner(net, runner, old_log_probabilities="recomputed")
    old = recomputed.prepare(runner.store, [runner.bootstrap()])["derived"]["old_log_probability"]
    with torch.no_grad():
        expected, _ = policy.evaluate(
            recomputed.unroll(trajectory).policy_logits, cases.joint_actions(trajectory)
        )
    assert torch.equal(old, expected) and not torch.equal(old, trajectory.log_probability)
    assert cases.scale_error(old, trajectory.log_probability) <= 1e-5


def test_the_drift_is_the_step_unroll_ratio_error(segment):
    """max and mean of |exp(log_probability_unroll - log_probability_rollout) - 1| over
    living own ships, against a boolean-indexed computation; a changed model drifts."""
    net, runner, learner, prepared = segment
    trajectory = lc.whole(runner)

    def independent(model):
        with torch.no_grad():
            unrolled = model.unroll(
                trajectory.initial_carry,
                trajectory.obs,
                trajectory.replay,
                cases.resets(trajectory.obs),
            )
        log_probability, _ = policy.evaluate(
            unrolled.policy_logits, cases.joint_actions(trajectory)
        )
        error = (torch.exp(log_probability - trajectory.log_probability) - 1.0).abs()[
            trajectory.alive[..., :OWN]
        ]
        return torch.stack([error.max(), error.mean()])

    torch.testing.assert_close(
        learner.drift(runner.store, prepared), independent(net), rtol=1e-5, atol=1e-9
    )
    assert learner.drift(runner.store, prepared)[0] < 1e-5
    changed = copy.deepcopy(net)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in changed.parameters():
            parameter.add_(0.01 * torch.randn(parameter.shape, generator=generator))
    moved = lc.learner(changed, runner).drift(runner.store, prepared)
    torch.testing.assert_close(moved, independent(changed), rtol=1e-5, atol=1e-9)
    assert moved[0] > 1e-3


def test_the_explained_variance_matches_an_independent_computation(segment):
    """Per level over valued own ships, 1 - Var(return - value) / Var(return); a perfect
    critic explains everything, and returns that never vary give 0."""
    net, runner, _, prepared = segment
    trajectory = lc.whole(runner)
    returns, values = prepared["derived"]["level_returns"], trajectory.values  # weight 1
    valued = trajectory.valued
    expected = torch.stack(
        [
            1.0
            - (returns[..., k][valued] - values[..., k][valued]).var()
            / returns[..., k][valued].var()
            for k in range(5)
        ]
    )
    torch.testing.assert_close(prepared["explained_variance"], expected, rtol=1e-4, atol=1e-5)
    assert torch.equal(explained_variance(returns, returns, valued), torch.ones(5))
    flat = torch.ones_like(returns)
    assert torch.equal(explained_variance(flat, values, valued), torch.zeros(5))


def test_the_update_is_the_same_whatever_the_global_seed(segment):
    """Every draw of the update (the shot spread, the minibatch order) comes from the
    learner's own streams: the global generator's state changes nothing."""
    net, runner, _, _ = segment
    results = []
    for seed in (1, 2):
        model = copy.deepcopy(net)
        learner = lc.learner(model, runner, epochs=1, minibatches=3)
        torch.manual_seed(seed)
        out = metrics.read(learner.update(runner.store, [runner.bootstrap()]))
        results.append((out, [parameter.detach().clone() for parameter in model.parameters()]))
    (first, first_parameters), (second, second_parameters) = results
    assert first == second
    assert all(torch.equal(a, b) for a, b in zip(first_parameters, second_parameters, strict=True))


def test_the_kl_early_stop_ends_the_update_after_an_epoch(segment):
    """The epoch's mean approximate KL above target_kl stops the update after that epoch;
    a target it never reaches runs every epoch."""
    net, runner, _, _ = segment
    for target, steps, stopped in ((-1.0, 2, True), (math.inf, 4, False)):
        learner = lc.learner(copy.deepcopy(net), runner, epochs=2, minibatches=2, target_kl=target)
        out = metrics.read(learner.update(runner.store, [runner.bootstrap()]))
        assert out["policy/minibatch_steps"] == steps and out["policy/stopped_early"] == stopped
        assert out["policy/kl"] >= 0.0
