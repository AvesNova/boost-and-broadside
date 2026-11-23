"""The hit head's labels (collect) and loss (update)."""

import pytest
import torch

from lean.learn import LearnConfig, Opponent, collect, make_optimiser, start, update
from lean.learn.tests import standins
from lean.learn.update import hit_loss


def stand_in_rollout(ticks=16, envs=4):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1)
    game_config = standins.GameConfig(length=6, flight=2)
    policy = standins.TinyPolicy()
    opponent = Opponent(scripted_config=None)
    state, carry = start(config, game_config, policy, opponent, envs, generator, game=standins.game, scripted=standins.scripted)
    rollout = collect(config, game_config, policy, opponent, state, carry, ticks, generator, game=standins.game, scripted=standins.scripted)
    return config, game_config, policy, rollout, generator


def test_a_label_lands_on_the_row_its_bullet_left_the_gun():
    """Guards the mapping from an ended bullet back to its firing tick: in the stand-in a
    bullet fired on episode tick s by slot n ends on tick s + 2 with outcome (s + n) % 3,
    so a label on the ending row instead would read (s + 2 + n) % 3, a different class.
    Rows whose bullet ends after the segment or after its episode are unlabelled."""
    _, game_config, _, rollout, _ = stand_in_rollout()
    ticks, seats, slots = rollout.hit_label.shape
    row = torch.arange(ticks)[:, None, None]
    episode_tick = row % game_config.length  # every environment starts at tick 0 and lasts 6 ticks
    slot = torch.arange(slots)[None, None, :]
    ends_in_episode = episode_tick + game_config.flight <= game_config.length - 1
    ends_in_segment = row + game_config.flight < ticks
    expected = torch.where(ends_in_episode & ends_in_segment, standins.bullet_outcome(episode_tick, slot), -1)
    assert torch.equal(rollout.hit_label, expected.expand(ticks, seats, slots))
    assert (rollout.hit_label[-game_config.flight:] == -1).all()  # the segment's last ticks wait for their bullets


def test_the_hit_loss_reads_only_labelled_own_rows():
    """Guards the mask: unlabelled rows and rows the seat does not own add nothing to the
    loss and get no gradient; the loss is the mean cross-entropy over the rest."""
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(5, 3, 4, 3, generator=generator, requires_grad=True)
    label = torch.randint(-1, 3, (5, 3, 4), generator=generator)
    own = torch.rand(5, 3, 4, generator=generator) < 0.6
    loss, labelled = hit_loss(logits, label, own)
    assert torch.equal(labelled, own & (label >= 0))
    expected = torch.nn.functional.cross_entropy(logits[labelled], label[labelled])
    assert torch.isclose(loss, expected)
    loss.backward()
    assert (logits.grad[~labelled] == 0).all()
    assert (logits.grad[labelled].abs().sum(-1) > 0).all()
    empty, _ = hit_loss(logits, torch.full_like(label, -1), own)
    assert empty == 0


def test_the_update_trains_the_hit_head_only_from_labelled_rows():
    """Guards the wiring in update: with every label removed the hit head's parameters get
    no gradient from the update; with the labels they move."""
    config, _, policy, rollout, generator = stand_in_rollout()
    before = [parameter.detach().clone() for parameter in policy.hit.parameters()]
    rollout.hit_label = torch.full_like(rollout.hit_label, -1)
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert metrics["hit"] == 0 and metrics["hit_labelled"] == 0
    assert all(torch.equal(old, new) for old, new in zip(before, policy.hit.parameters()))

    config, _, policy, rollout, generator = stand_in_rollout()
    before = [parameter.detach().clone() for parameter in policy.hit.parameters()]
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert metrics["hit"] > 0 and metrics["hit_labelled"] > 0
    assert not all(torch.equal(old, new) for old, new in zip(before, policy.hit.parameters()))


@pytest.mark.parametrize("ghost_bullets", [True, False])
def test_real_game_labels(ghost_bullets):
    """Guards the labels with lean.game: with ghost bullets most own rows are labelled;
    without them only rows whose applied action (the observation's pending one, chosen on the
    tick before) pulled the trigger can be (the mask covers the rest), and every label is one
    of the three classes."""
    game = pytest.importorskip("lean.game")
    scripted = pytest.importorskip("lean.scripted")
    model = pytest.importorskip("lean.model")
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2)
    game_config = game.GameConfig(ghost_bullets=ghost_bullets)
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted_config=scripted.ScriptedConfig())
    state, carry = start(config, game_config, policy, opponent, 4, generator)
    rollout = collect(config, game_config, policy, opponent, state, carry, 24, generator)
    label, own = rollout.hit_label, rollout.observation.own
    assert ((label >= -1) & (label <= 2)).all()
    if ghost_bullets:
        # Every living ship emits every tick; a bullet lives up to 11 ticks.
        assert (label[:13][own[:13]] >= 0).float().mean() > 0.9
    else:
        pulled = rollout.observation.pending % 2 == 1  # NO_PENDING (42) is even
        assert not (label[own] >= 0)[~pulled[own]].any()
