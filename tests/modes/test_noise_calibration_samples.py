"""Bounded raw-sample retention for noise calibration.

D17 keeps raw samples when practical and never tracks them. What has to hold is
that the payload stays bounded whatever the run length, that it records which
step each row came from, and that an empty run yields an empty payload rather
than a malformed one.
"""

from __future__ import annotations

import numpy as np
import torch

from boost_and_broadside.modes.noise_calibration import _RawSampleBuffer


def test_retention_stops_at_the_cap_and_keeps_collection_order() -> None:
    buffer = _RawSampleBuffer(max_rows=5, num_targets=3)
    for step in range(4):
        buffer.add(torch.full((4, 3), float(step)), step)

    errors = buffer.errors()
    assert errors.shape == (5, 3)
    assert errors.dtype == np.float16
    assert buffer.steps().tolist() == [0, 0, 0, 0, 1]
    assert errors[:, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_an_empty_run_yields_an_empty_but_well_shaped_payload() -> None:
    buffer = _RawSampleBuffer(max_rows=5, num_targets=3)
    buffer.add(torch.zeros((0, 3)), 0)

    assert buffer.errors().shape == (0, 3)
    assert buffer.steps().size == 0


def test_rows_are_detached_from_the_graph_and_moved_off_device() -> None:
    rows = torch.ones((2, 3), requires_grad=True) * 2.0
    buffer = _RawSampleBuffer(max_rows=8, num_targets=3)
    buffer.add(rows, 7)

    assert buffer.errors().tolist() == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    assert buffer.steps().tolist() == [7, 7]
