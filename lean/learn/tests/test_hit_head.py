"""The hit head's labels (collect) and loss (update)."""

import pytest
import torch

from lean import game
from lean.learn import collect, make_optimiser, update
from lean.learn.tests.setting import StepRecorder, small
from lean.learn.training import hit_loss


def test_a_label_lands_on_the_row_its_bullet_left_the_gun(monkeypatch):
    """Guards the mapping from an ended bullet back to its firing tick: every label equals the
    outcome of the bullet the ship emitted on that row, found by searching the recorded game
    steps for the row of the bullet's shot tick within its episode (games end inside the
    rollout); rows whose bullet had not ended by the rollout's end are unlabelled."""
    recorder = StepRecorder().install(monkeypatch)
    config, game_config, policy, opponent, state, carry, generator = small(
        game_config=game.GameConfig(training_spawn=True, clock_ticks=20)
    )
    rollout = collect(config, game_config, policy, opponent, state, carry, 40, generator)
    expected = recorder.hit_labels().index_select(1, rollout.seat_environment)
    # Ghost bullets: a bullet lives up to 11 ticks and a game 20, so about half the rows have one.
    assert (expected >= 0).float().mean() > 0.3
    assert torch.equal(rollout.hit_label, expected)


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


def hit_head(policy):
    """The hit head's own parameters (its norm and last layer; the first layer is shared)."""
    return [
        *policy.auxiliary_heads.norms[1].parameters(),
        *policy.auxiliary_heads.lasts[1].parameters(),
    ]


@pytest.mark.parametrize("labelled", [False, True])
def test_the_update_trains_the_hit_head_only_from_labelled_rows(labelled):
    """Guards the wiring in update: with every label removed the hit head's own parameters get
    a zero gradient and do not move; with the labels they move."""
    config, game_config, policy, opponent, state, carry, generator = small()
    rollout = collect(config, game_config, policy, opponent, state, carry, 16, generator)
    if not labelled:
        rollout.hit_label = torch.full_like(rollout.hit_label, -1)
    before = [parameter.detach().clone() for parameter in hit_head(policy)]
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    moved = [not torch.equal(old, new) for old, new in zip(before, hit_head(policy), strict=True)]
    if labelled:
        assert metrics["hit"] > 0 and metrics["hit_labelled"] > 0 and all(moved)
    else:
        assert metrics["hit"] == 0 and metrics["hit_labelled"] == 0 and not any(moved)


@pytest.mark.parametrize("ghost_bullets", [True, False])
def test_real_game_labels(ghost_bullets):
    """Guards the labels with lean.game: with ghost bullets most own rows are labelled;
    without them only rows whose applied action (the observation's pending one, chosen on the
    tick before) pulled the trigger can be (the mask covers the rest), and every label is one
    of the three classes."""
    config, game_config, policy, opponent, state, carry, generator = small(
        game_config=game.GameConfig(ghost_bullets=ghost_bullets)
    )
    rollout = collect(config, game_config, policy, opponent, state, carry, 24, generator)
    label, own = rollout.hit_label, rollout.observation.own
    assert ((label >= -1) & (label <= 2)).all()
    if ghost_bullets:
        # Every living ship emits every tick; a bullet lives up to 11 ticks.
        assert (label[:13][own[:13]] >= 0).float().mean() > 0.9
    else:
        pulled = rollout.observation.pending % 2 == 1  # NO_PENDING (42) is even
        assert not (label[own] >= 0)[~pulled[own]].any()
