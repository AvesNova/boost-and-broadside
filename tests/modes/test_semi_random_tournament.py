"""Tests for semi-random ladder configuration."""

import pytest

from boost_and_broadside.modes.semi_random_tournament import (
    _greatest_divisor_at_most,
    parse_probabilities,
)


def test_probability_ladder_is_sorted_and_deduplicated() -> None:
    assert parse_probabilities("1,0.2,0,0.2") == [0.0, 0.2, 1.0]


@pytest.mark.parametrize(
    "text",
    ["0,0.5", "0.5,1", "0,-0.1,1", "0,1.1,1", "0,nan,1", "0,inf,1"],
)
def test_probability_ladder_requires_valid_endpoints(text: str) -> None:
    with pytest.raises(ValueError):
        parse_probabilities(text)


def test_batch_chunk_divides_target_without_exceeding_capacity() -> None:
    assert _greatest_divisor_at_most(256, 11) == 8
    assert _greatest_divisor_at_most(256, 200) == 128
