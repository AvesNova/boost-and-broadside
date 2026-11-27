"""The steady-flight table (core.steady_lookup): the game's steady flight tabulated once per
ship type, read by the scripted agent's look-ahead in place of the flight functions."""

from types import SimpleNamespace

import torch

from lean.contract import POWERS
from lean.game import card, core

torch.set_num_threads(1)
LEVEL_INDEX = [int(abs(level)) for level in card.STICK_LEVELS]
FIRST_OF_LEVEL = [LEVEL_INDEX.index(level) for level in range(max(LEVEL_INDEX) + 1)]


def _direct_and_table(speed, energy, level, power):
    """(direct, looked up), each (points, 4): steady_flight and steady_lookup at the points."""
    config = core.GameConfig()
    card_ = core.ship_card(config)
    rows = core.per_type_constants(config, torch.zeros_like(level))
    p = SimpleNamespace(
        **vars(rows),
        **{
            name: value[torch.zeros_like(level)]
            for name, value in core.steady_characteristics(core._per_type(config)).items()
            if name != "steady_table"
        },
    )
    index = torch.tensor(FIRST_OF_LEVEL)[level][:, None]
    shape = [p.stick_table.gather(1, index)[:, 0].abs()]
    shape += [
        getattr(p, name).gather(1, index)[:, 0]
        for name in ("sustain_stick_table", "deep_stall_table")
    ]
    direct = torch.stack(
        [
            torch.stack(
                core.steady_flight(p, speed, shape[0], each, energy, shape[1], shape[2]), -1
            )
            for each in range(POWERS)
        ],
        1,
    )
    direct = direct[torch.arange(len(speed)), power]
    table = card_.per_type["steady_table"]
    looked = core.steady_lookup(p, table, torch.zeros_like(level), speed, level, power, energy)
    return direct, torch.stack(looked, -1)


def test_the_table_holds_the_flight_functions_at_its_nodes():
    """At the grid's speeds (mid store, where the store does not matter) the lookup is the
    game's steady flight itself, every level and power."""
    c = core.ship_card(core.GameConfig())
    nodes = torch.arange(0, core.STEADY_SPEED_NODES - 1, 7)
    count = len(nodes) * 4 * POWERS
    speed = (c["steady_low_speed"] + nodes.float() / c["inverse_steady_step"]).repeat(4 * POWERS)
    level = torch.arange(4).repeat_interleave(len(nodes)).repeat(POWERS)
    power = torch.arange(POWERS).repeat_interleave(4 * len(nodes))
    energy = torch.full((count,), 0.5 * c["max_energy"])
    direct, looked = _direct_and_table(speed, energy, level, power)
    scale = direct.abs().amax(0).clamp_min(1e-6)
    assert ((looked - direct).abs() / scale).max() < 1e-4


def test_between_nodes_and_in_the_store_bands_the_table_is_close():
    """Random speeds over the flown range and stores weighted to the empty band: the median
    error is tiny and the 99th percentile small against each output's range."""
    c = core.ship_card(core.GameConfig())
    generator = torch.Generator().manual_seed(0)
    points = 20000
    speed = 0.4 * c["stall_speed"] + 1.4 * c["cruise_speed"] * torch.rand(
        points, generator=generator
    )
    energy = c["max_energy"] * torch.rand(points, generator=generator) ** 2
    level = torch.randint(0, 4, (points,), generator=generator)
    power = torch.randint(0, POWERS, (points,), generator=generator)
    direct, looked = _direct_and_table(speed, energy, level, power)
    error = (looked - direct).abs() / direct.abs().amax(0).clamp_min(1e-6)
    assert float(error.median()) < 1e-4
    assert bool((error.quantile(0.99, dim=0) < 0.01).all())
