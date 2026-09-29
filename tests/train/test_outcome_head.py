"""The categorical win / loss / tie head.

It is a classifier run beside the scalar ``outcome`` component rather than in
place of it: nothing here reaches the advantage path. Run 739 found the single
scalar beat both split win heads on explained variance at all thirty late-phase
points, and this tests whether three classes beat the scalar in turn -- a scalar
regressed onto {-1, 0, +1} cannot distinguish "confident tie" from "even odds",
and those are different game states.
"""

import math

import pytest
import torch

from boost_and_broadside.constants import (
    NUM_OUTCOME_CLASSES,
    OUTCOME_LOSS_INDEX,
    OUTCOME_TIE_INDEX,
    OUTCOME_WIN_INDEX,
)


class _Head:
    """The loss under test, lifted off the trainer so it needs no rollout."""

    from boost_and_broadside.train.rl.ppo import PPOTrainer

    loss = PPOTrainer._outcome_categorical_loss


def _loss(logits, outcome_class):
    """The scalar loss alone; the mask and graded class are tested separately."""
    alive = torch.ones(logits.shape[:3])
    return _Head.loss(None, logits, outcome_class, alive, alive.sum())[0]


def _graded(logits, outcome_class):
    """``(mask, class)`` of the steps whose target is a realised result."""
    alive = torch.ones(logits.shape[:3])
    _, mask, cls = _Head.loss(None, logits, outcome_class, alive, alive.sum())
    return mask, cls


def test_the_classes_are_ordered_so_the_index_carries_the_result() -> None:
    """Loss below tie below win, so a probability-weighted sum of (-1, 0, +1)
    recovers the signed expectation the scalar component regresses."""
    assert OUTCOME_LOSS_INDEX < OUTCOME_TIE_INDEX < OUTCOME_WIN_INDEX
    assert NUM_OUTCOME_CLASSES == 3


def test_a_realised_result_is_learned_from() -> None:
    """A confident, correct prediction on a labelled step costs nearly nothing;
    a confident wrong one costs a lot. Without this the head could be trained
    entirely by its own bootstrap and never contradicted by a real match."""
    logits = torch.zeros(1, 1, 1, NUM_OUTCOME_CLASSES)
    label = torch.full((1, 1, 1), OUTCOME_WIN_INDEX, dtype=torch.int8)

    logits[..., OUTCOME_WIN_INDEX] = 8.0
    assert float(_loss(logits, label)) < 0.01

    logits[..., OUTCOME_WIN_INDEX] = -8.0
    logits[..., OUTCOME_LOSS_INDEX] = 8.0
    assert float(_loss(logits, label)) > 8.0


def test_an_unlabelled_step_is_not_graded_at_all() -> None:
    """Nothing is known about a match still in progress, so nothing is claimed
    about it.

    These steps used to be graded against the head's own belief at the chunk's
    end, to densify a label that is otherwise carried by under 1% of steps.
    That target had a fixed point: every unlabelled step took the *same* one,
    so any prediction constant across the chunk satisfied it exactly, and for
    softmax cross-entropy ``d/dlogits = p - target`` is then identically zero.
    Uniform is such a fixed point and is where the head initializes. Run 748
    sat on it for 84.5M steps at cross-entropy ln 3 with chance accuracy.
    """
    logits = torch.randn(6, 2, 3, NUM_OUTCOME_CLASSES)
    unlabelled = torch.full((6, 2, 3), -1, dtype=torch.int8)

    # No realised result anywhere in the chunk: nothing to learn from, so the
    # term is zero rather than a number the head can satisfy by standing still.
    assert float(_loss(logits, unlabelled)) == pytest.approx(0.0)

    # And it is zero because nothing is graded, not because the head agrees
    # with itself -- a wildly inconsistent head pays the same nothing.
    drifting = logits.clone()
    drifting[0] = -drifting[-1] * 8.0
    assert float(_loss(drifting, unlabelled)) == pytest.approx(0.0)


def test_a_constant_prediction_is_no_longer_a_free_fixed_point() -> None:
    """The regression that cost run 748 its outcome head: a time-constant head
    must pay for being wrong, not be excused by its own consistency."""
    logits = torch.zeros(4, 1, 1, NUM_OUTCOME_CLASSES)  # uniform everywhere
    outcome = torch.full((4, 1, 1), -1, dtype=torch.int8)
    outcome[-1, 0, 0] = OUTCOME_WIN_INDEX

    # Uniform against a realised one-hot costs ln 3, and it is a real gradient
    # rather than the zero a self-target produced.
    assert float(_loss(logits, outcome)) == pytest.approx(math.log(3.0), abs=1e-5)
    leaf = logits.clone().requires_grad_(True)
    _loss(leaf, outcome).backward()
    assert float(leaf.grad.abs().sum()) > 0.0


def test_the_label_travels_backwards_to_the_steps_that_led_to_it() -> None:
    """gamma is 1.0 and nothing pays before the terminal, so the return from any
    state *is* the result. Earlier steps must be graded against it, not against
    the bootstrap."""
    logits = torch.zeros(4, 1, 1, NUM_OUTCOME_CLASSES)
    logits[..., OUTCOME_LOSS_INDEX] = 8.0  # confidently wrong everywhere
    outcome = torch.full((4, 1, 1), -1, dtype=torch.int8)
    outcome[3] = OUTCOME_WIN_INDEX  # the match is won at the last step

    # Every step is graded against the win, so the loss is large at all four.
    assert float(_loss(logits, outcome)) > 8.0


def test_a_step_after_one_match_ends_takes_the_next_result_not_the_last() -> None:
    """Instant respawn means a rollout can span an episode boundary. A step
    sitting after a finished match belongs to the *next* one, and inheriting the
    previous result would train the head on an outcome that already happened."""
    outcome = torch.full((3, 1, 1), -1, dtype=torch.int8)
    outcome[0] = OUTCOME_WIN_INDEX  # one match ends immediately
    outcome[2] = OUTCOME_LOSS_INDEX  # the next ends at the chunk's end

    # A head that calls step 0 a win and steps 1-2 a loss is right about every
    # step -- but only if step 1 looks *forward* to the loss at step 2.
    logits = torch.zeros(3, 1, 1, NUM_OUTCOME_CLASSES)
    logits[0, ..., OUTCOME_WIN_INDEX] = 8.0
    logits[1:, ..., OUTCOME_LOSS_INDEX] = 8.0
    assert float(_loss(logits, outcome)) < 0.01

    # Had step 1 inherited the finished match's win, this would be the cheap
    # arrangement instead. It must not be.
    inherited = torch.zeros(3, 1, 1, NUM_OUTCOME_CLASSES)
    inherited[:2, ..., OUTCOME_WIN_INDEX] = 8.0
    inherited[2, ..., OUTCOME_LOSS_INDEX] = 8.0
    assert float(_loss(inherited, outcome)) > 2.0


def test_the_tie_class_is_reachable_from_a_drawn_match() -> None:
    """Run 739 drew 20% of late episodes, so tie is a real mode. A truncation at
    the step cap is a genuine draw under the frontline rules."""
    logits = torch.zeros(1, 1, 1, NUM_OUTCOME_CLASSES)
    logits[..., OUTCOME_TIE_INDEX] = 8.0
    label = torch.full((1, 1, 1), OUTCOME_TIE_INDEX, dtype=torch.int8)
    assert float(_loss(logits, label)) < 0.01


def test_grading_covers_the_steps_the_label_reached_not_just_the_terminal() -> None:
    """Accuracy is read off realised results. Restricting that to the terminal
    step alone would sample about a thousandth of a batch and read as noise, so
    the mask has to include the steps the backward pass labelled from it."""
    outcome = torch.full((4, 1, 1), -1, dtype=torch.int8)
    outcome[3] = OUTCOME_WIN_INDEX
    logits = torch.zeros(4, 1, 1, NUM_OUTCOME_CLASSES)

    mask, graded_class = _graded(logits, outcome)
    assert bool(mask.all()), "every step leads to the terminal, so every step is graded"
    assert torch.equal(graded_class, torch.full((4, 1, 1), OUTCOME_WIN_INDEX, dtype=torch.int8))


def test_a_batch_with_no_finished_match_grades_nothing() -> None:
    """Bootstrapped steps are not ground truth, so they must not be counted as
    correct or incorrect -- an accuracy over them would measure only how
    self-consistent the head is."""
    outcome = torch.full((5, 2, 3), -1, dtype=torch.int8)
    mask, _ = _graded(torch.randn(5, 2, 3, NUM_OUTCOME_CLASSES), outcome)
    assert not bool(mask.any())
