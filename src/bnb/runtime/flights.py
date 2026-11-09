"""Compiled flights for the ship card's handles: the play shell's live tuning measures and
solves cards by flying the game's flight tick thousands of times, which is several times
faster compiled (a measurement 0.9 s instead of 12.5 s on a CPU; the first compile of each
batch size takes about a minute, then comes from the on-disk cache)."""

import torch

from bnb.game import api


def compiled_flight_tick():
    """The game's flight tick, compiled (on its first call, for each batch size): what
    `api.measure_handles` and `api.solve_handles` take as their `tick_function`."""
    return torch.compile(api.flight_tick, dynamic=False)
