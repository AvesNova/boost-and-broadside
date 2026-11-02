"""The batch arithmetic of pools and the tick (contracts: `Pool`, `ticks_of`): a share
must give whole environments, the batch multiple counts the fogged share, and a time must
be whole ticks. Mutation-sweep items 20 to 22."""

import pytest

from bnb.contracts import Pool
from bnb.contracts.config import ticks_of
from bnb.contracts.pools import environment_multiple, environments_per_pool

TICK = 1.0 / 30.0


def test_a_share_that_gives_a_fraction_of_an_environment_is_refused():
    pools = (Pool("a", 0.25, ("x", "x")), Pool("b", 0.75, ("x", "x")))
    with pytest.raises(ValueError, match="is not whole"):
        environments_per_pool(pools, 10)
    assert environments_per_pool(pools, 12) == (3, 9)


def test_the_batch_multiple_counts_the_fogged_share():
    pools = (Pool("a", 0.5, ("x", "x"), fog_share=0.5), Pool("b", 0.5, ("x", "x")))
    assert environment_multiple(pools) == 4
    environments_per_pool(pools, 4)
    with pytest.raises(ValueError, match="fog share"):
        environments_per_pool(pools, 2)


@pytest.mark.parametrize("seconds", [1.01, 1.0 + 1e-6, 0.99])
def test_a_time_just_off_a_whole_tick_is_refused(seconds):
    with pytest.raises(ValueError, match="whole number"):
        ticks_of(seconds, TICK)


def test_a_whole_number_of_ticks_converts():
    assert ticks_of(1.0, TICK) == 30 and ticks_of(0.5, TICK) == 15
