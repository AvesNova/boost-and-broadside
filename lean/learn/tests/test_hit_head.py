"""The hit head's labels (collect) and loss (update)."""

import pytest
import torch

from lean import game
from lean.contract import BULLET_SLOTS
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
    # Not vacuous: both sides of the rule occur (labelled rows and rows whose bullet had not
    # ended), and labelled rows lie in a later episode than the first (the mapping crosses a
    # game's end), whatever share of rows the bullet life and the clock give.
    done = torch.stack(recorder.done).index_select(1, rollout.seat_environment)  # (T, E)
    later = (done.long().cumsum(0) - done.long()) > 0  # the row is after a game ended
    assert (expected >= 0).any() and (expected < 0).any()
    assert ((expected >= 0) & later[..., None]).any()
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


def test_real_game_labels():
    """Guards the labels with lean.game: every ship writes a bullet every tick (a ghost unless
    it fired) and a bullet reports within its BULLET_SLOTS ticks, so the own rows that many
    ticks before the rollout's end are labelled; every label is one of the three classes."""
    config, game_config, policy, opponent, state, carry, generator = small(
        game_config=game.GameConfig()
    )
    rollout = collect(config, game_config, policy, opponent, state, carry, 24, generator)
    label, own = rollout.hit_label, rollout.observation.own
    assert ((label >= -1) & (label <= 2)).all()
    ended = label.shape[0] - BULLET_SLOTS
    assert ended > 0
    assert (label[:ended][own[:ended]] >= 0).float().mean() > 0.9


@pytest.mark.parametrize("ticks", [3, 12, 20])
def test_assembled_labels_put_each_outcome_on_its_firing_row(ticks):
    """assemble_hit_labels against the rule written out: the outcome reported on tick t by the
    bullet of age a lands on row T + t - a (rows from 0, the previous segment's first), and is
    dropped when that row is negative; segments shorter and longer than a bullet's life."""
    from lean.learn.rollout import assemble_hit_labels

    generator = torch.Generator().manual_seed(ticks)
    envs, slots = 3, 4
    # Each (row, environment, slot) reports at most once: draw its end age, then its outcome.
    expected = torch.full((2 * ticks, envs, slots), -1, dtype=torch.long)
    outcomes = torch.full((ticks, envs, slots, BULLET_SLOTS), -1, dtype=torch.int8)
    for row in range(2 * ticks):
        for environment in range(envs):
            for slot in range(slots):
                age = int(torch.randint(0, BULLET_SLOTS, (), generator=generator))
                tick = row - ticks + age
                if 0 <= tick < ticks and torch.rand((), generator=generator) < 0.7:
                    outcome = int(torch.randint(0, 3, (), generator=generator))
                    outcomes[tick, environment, slot, age] = outcome
                    expected[row, environment, slot] = outcome
    if ticks < BULLET_SLOTS - 1:  # a bullet fired before the previous segment: dropped
        outcomes[0, 0, 0, BULLET_SLOTS - 1] = 1
    assert torch.equal(assemble_hit_labels(outcomes), expected.view(2, ticks, envs, slots))


def test_unresolved_rows_are_the_bullet_life_before_an_end():
    """unresolved_rows against the rule: a bullet from row r misses on row r + BULLET_SLOTS
    - 1, so with an episode end on row 20 rows 10 to 20 cannot report a miss (row 9's miss
    lands on the end's own tick); a closed window also loses its last BULLET_SLOTS - 1 rows,
    an open one (a later segment follows) keeps them open."""
    from lean.learn.store import unresolved_rows

    reach = BULLET_SLOTS - 1
    done = torch.zeros(40, 2, dtype=torch.bool)
    done[20, 0] = True
    closed, opened = unresolved_rows(done, True), unresolved_rows(done, False)
    rows = torch.arange(40)
    end = (rows >= 20 - reach + 1) & (rows <= 20)
    tail = rows >= 40 - reach
    assert torch.equal(closed[:, 0], end | tail) and torch.equal(opened[:, 0], end)
    assert torch.equal(closed[:, 1], tail) and not opened[:, 1].any()
    assert end.sum() == reach and tail.sum() == reach


@pytest.mark.parametrize("ticks", [8, 24])
def test_the_store_unlabels_bullets_that_cannot_report_a_miss(ticks):
    """The hit-label bias (training-review-2 part 1; author 2026-11-28 ~13:30), a constructed
    end-of-episode case: one ship fires on every row of two segments, an episode ends on row
    E; each bullet hits at a drawn age with probability 0.3 or misses at its last age, and
    reports only by its episode's end and the window's end (the game's rule), into the
    segment of its report (the previous segment's rows through `previous_hit_label`, rows
    older than that dropped, as assemble_hit_labels does). Before the mask the labelled
    rows near the ends are all hits; after it the store keeps exactly the rows whose miss
    could have reported, each with its true outcome, so the kept rows carry no selection
    on the outcome. Segments shorter (8: four of them; a miss two segments on is dropped, so
    its row is unresolved too) and longer (24) than a bullet's life."""
    from lean.learn.store import RolloutStore

    reach = BULLET_SLOTS - 1
    segments = 4 if ticks < reach else 2
    rows = segments * ticks
    end = rows // 2 + 3
    generator = torch.Generator().manual_seed(ticks)
    hit = torch.rand(rows, generator=generator) < 0.3
    age = torch.randint(0, BULLET_SLOTS, (rows,), generator=generator)
    truth = hit.long()  # 1 hit an enemy, 0 missed
    report = torch.where(hit, torch.arange(rows) + age, torch.arange(rows) + reach)
    episode_end = torch.where(torch.arange(rows) <= end, end, rows - 1)
    reported = (report <= episode_end) & (report < rows)
    own = torch.full((segments, ticks), -1)  # each segment's own rows' labels
    previous = torch.full((segments, ticks), -1)  # the earlier segment's, from this one
    labelled = torch.zeros(rows, dtype=torch.bool)  # labelled before the mask
    for row in range(rows):
        if not reported[row]:
            continue
        here, there = row // ticks, int(report[row]) // ticks
        if there == here:
            own[here, row % ticks] = truth[row]
        elif there == here + 1:
            previous[there, row % ticks] = truth[row]
        labelled[row] = there <= here + 1
    store = RolloutStore(segments, stream=False)
    for segment in range(segments):
        store.begin_segment(
            segment, ticks, torch.zeros(1, 1, 1), {"position_offset": torch.zeros(1, 2)}
        )
        if segment == 0:
            store.resident["done"] = torch.zeros(ticks, store.views + 1, dtype=torch.bool)
            store.resident["done"][end % ticks, end // ticks] = True
        store.end_segment(
            segment,
            own[segment].view(ticks, 1, 1),
            torch.zeros(1, 1, 3),
            torch.zeros(3, 4),
            previous[segment].view(ticks, 1, 1) if segment > 0 else None,
        )
    stored = store.resident["hit_label"][:, :segments, 0].T.reshape(rows).long()
    fired = torch.arange(rows)
    resolvable = (
        (fired + reach <= episode_end)
        & (fired + reach < rows)
        & ((fired + reach) // ticks <= fired // ticks + 1)
    )
    assert resolvable.sum() > 0 and (~resolvable).sum() >= 2 * reach
    assert torch.equal(stored, torch.where(resolvable, truth, -1))
    # The bias it removes: rows labelled that could not have missed were all hits.
    unmasked = labelled & ~resolvable
    assert unmasked.any() and (truth[unmasked] == 1).all()
    assert (stored[~resolvable] == -1).all() and (store.resident["hit_label"][:, -1] == -1).all()
