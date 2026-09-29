"""The hex count target: grid layout, barycentric exactness, and the rim.

Every living ship deposits one unit of mass, split across the three cells whose
centres form the triangle containing it. There is no kernel and no radius, so
what has to hold is arithmetic rather than a calibration: mass is conserved, at
most three cells carry any of it, the weights invert to the ship's exact
position, and a ship leaving the map takes its mass with it continuously.
"""

import math

import pytest
import torch

from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.hex_density import (
    HEX_DENSITY_RINGS,
    HexDensityTarget,
    barycentric_cells,
    hex_axial_cells,
    hex_axial_lookup,
    hex_cell_count,
    hex_cell_offsets,
    hex_counts,
)

SMALL_RINGS = 2
SMALL_WORLD = (4000.0, 3000.0)
WORLD = (65536.0, 65536.0)


def _counts(points, teams=None, alive=None, rings=HEX_DENSITY_RINGS, playable=1.0):
    """Target for one environment whose ships sit at ``points``, in grid units.

    Positions are in units of the playable radius about a map centre at the
    origin, which is the space the grid itself is defined in.
    """
    count = len(points)
    teams = [0] * count if teams is None else teams
    alive = [True] * count if alive is None else alive
    return hex_counts(
        torch.tensor([[x for x, _ in points]], dtype=torch.float32) * playable,
        torch.tensor([[y for _, y in points]], dtype=torch.float32) * playable,
        torch.tensor([[t == 0 for t in teams]], dtype=torch.bool),
        torch.tensor([alive], dtype=torch.bool),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.full((1, 1), float(playable)),
        WORLD,
        rings,
        hex_axial_lookup(rings),
    )[0]


class TestGrid:
    @pytest.mark.parametrize("rings", [1, 2, HEX_DENSITY_RINGS])
    def test_cell_count_is_the_hexagonal_number(self, rings):
        assert hex_cell_offsets(rings).shape == (hex_cell_count(rings), 2)

    def test_production_grid_is_in_the_planned_range(self):
        assert 271 <= hex_cell_count(HEX_DENSITY_RINGS) <= 331

    def test_cells_are_distinct(self):
        cells = hex_cell_offsets(HEX_DENSITY_RINGS)
        gaps = torch.cdist(cells, cells) + torch.eye(cells.shape[0]) * 10.0
        assert gaps.min() > 0.5 / HEX_DENSITY_RINGS

    def test_outer_corners_sit_on_the_playable_boundary(self):
        radii = hex_cell_offsets(HEX_DENSITY_RINGS).norm(dim=-1)
        assert radii.max().item() == pytest.approx(1.0, abs=1e-6)

    def test_ordering_is_ring_by_ring_from_the_positive_x_axis(self):
        cells = hex_cell_offsets(SMALL_RINGS) * SMALL_RINGS
        expected_first_ring = [
            (math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)
        ]
        assert cells[0].tolist() == [0.0, 0.0]
        assert torch.allclose(cells[1:7], torch.tensor(expected_first_ring), atol=1e-6)
        assert cells[7].tolist() == pytest.approx([2.0, 0.0])


class TestBarycentricTarget:
    """Mass conservation and sparsity, which is all the target promises inside."""

    def test_a_ship_on_a_cell_centre_lands_entirely_on_that_cell(self):
        """Every one of the 331, not a sample: an ordering or rounding error
        would show on some cells and not others."""
        cells = hex_cell_offsets(HEX_DENSITY_RINGS)
        for column in range(cells.shape[0]):
            got = _counts([(float(cells[column, 0]), float(cells[column, 1]))])
            assert float(got[column]) == pytest.approx(1.0, abs=1e-5), column

    def test_every_ship_deposits_exactly_one_unit(self):
        torch.manual_seed(0)
        angles = torch.rand(300) * 2 * math.pi
        radii = torch.rand(300).sqrt() * 0.85
        for angle, radius in zip(angles.tolist(), radii.tolist(), strict=True):
            got = _counts([(radius * math.cos(angle), radius * math.sin(angle))])
            assert float(got.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_at_most_three_cells_carry_a_ship(self):
        torch.manual_seed(1)
        for _ in range(300):
            angle = float(torch.rand(()) * 2 * math.pi)
            radius = float(torch.rand(()).sqrt() * 0.85)
            got = _counts([(radius * math.cos(angle), radius * math.sin(angle))])
            assert int((got > 1e-9).sum()) <= 3

    def test_the_weights_invert_to_the_exact_position(self):
        """The property a Gaussian cannot have: nothing about where the ship is
        has been thrown away, only which three cells were used to say it."""
        cells = hex_cell_offsets(HEX_DENSITY_RINGS)
        points = torch.rand(400, 2) * 1.2 - 0.6
        column, weight = barycentric_cells(
            points[:, 0], points[:, 1], HEX_DENSITY_RINGS, hex_axial_lookup(HEX_DENSITY_RINGS)
        )
        reconstructed = (cells[column.clamp(min=0)] * weight.unsqueeze(-1)).sum(1)
        assert torch.allclose(reconstructed, points, atol=1e-5)

    def test_ships_stack_rather_than_saturating(self):
        """Three ships on one cell read as three. A ``log1p`` density could not
        say this, and it is the count half of what the head predicts."""
        centre = (0.0, 0.0)
        assert float(_counts([centre] * 3).sum()) == pytest.approx(3.0, abs=1e-5)

    def test_the_two_halves_separate_by_team(self):
        cells = hex_cell_count(HEX_DENSITY_RINGS)
        got = _counts([(0.1, 0.0), (-0.1, 0.0)], teams=[0, 1])
        assert float(got[:cells].sum()) == pytest.approx(1.0, abs=1e-5)
        assert float(got[cells:].sum()) == pytest.approx(1.0, abs=1e-5)

    def test_a_dead_ship_deposits_nothing(self):
        assert float(_counts([(0.0, 0.0)], alive=[False]).sum()) == pytest.approx(0.0)

    def test_it_holds_at_the_production_scale(self):
        """The unit-radius checks above hide a 2600x scaling and a 65536 px
        torus, and the minimum-image step is where float32 precision goes."""
        got = _counts([(0.3, -0.2)], playable=2600.0)
        assert float(got.sum()) == pytest.approx(1.0, abs=1e-5)
        assert int((got > 1e-9).sum()) <= 3


class TestOffTheGrid:
    """The lattice is infinite; the grid is not. A ship outside keeps only the
    part of itself that falls on cells this grid actually has."""

    def test_a_ship_well_outside_contributes_nothing(self):
        for distance in (1.2, 1.5, 3.0, 20.0):
            assert float(_counts([(distance, 0.0)]).sum()) == pytest.approx(0.0, abs=1e-6)

    def test_mass_falls_away_continuously_rather_than_at_a_cliff(self):
        masses = [float(_counts([(d, 0.0)]).sum()) for d in (0.98, 1.02, 1.06, 1.10)]
        assert masses[0] == pytest.approx(1.0, abs=1e-5)
        assert masses[-1] == pytest.approx(0.0, abs=1e-6)
        assert all(later <= earlier + 1e-6 for earlier, later in zip(masses, masses[1:]))
        assert any(0.01 < mass < 0.99 for mass in masses)

    def test_an_outside_ship_still_lands_on_the_rim_it_is_next_to(self):
        """Not clamped onto the boundary -- partially kept, which is what makes
        the falloff continuous."""
        cells = hex_cell_offsets(HEX_DENSITY_RINGS)
        got = _counts([(1.04, 0.0)])
        assert 0.0 < float(got.sum()) < 1.0
        lit = (got[: cells.shape[0]] > 1e-9).nonzero().flatten()
        assert lit.numel() > 0
        assert float(cells[lit].norm(dim=-1).max()) == pytest.approx(1.0, abs=1e-6)

    def test_nothing_wraps_around_the_lookup_table(self):
        """A far-off ship indexes outside the table; it must read as absent
        rather than as some unrelated cell."""
        assert float(_counts([(50.0, -37.0)]).sum()) == 0.0


class TestGridOrdering:
    def test_the_axial_cells_and_the_offsets_are_one_ordering(self):
        axial = hex_axial_cells(SMALL_RINGS)
        offsets = hex_cell_offsets(SMALL_RINGS) * SMALL_RINGS
        expected = torch.tensor(
            [(q + r / 2.0, r * math.sqrt(3.0) / 2.0) for q, r in axial], dtype=torch.float32
        )
        assert torch.allclose(offsets, expected, atol=1e-6)

    def test_the_lookup_inverts_the_ordering(self):
        lookup = hex_axial_lookup(HEX_DENSITY_RINGS)
        pad = HEX_DENSITY_RINGS + 1
        for column, (q, r) in enumerate(hex_axial_cells(HEX_DENSITY_RINGS)):
            assert int(lookup[q + pad, r + pad]) == column

    def test_the_lookup_says_absent_off_the_hexagon(self):
        lookup = hex_axial_lookup(SMALL_RINGS)
        pad = SMALL_RINGS + 1
        # (rings + 1, 0) is one step beyond the outer ring's +x corner.
        assert int(lookup[SMALL_RINGS + 1 + pad, 0 + pad]) == -1


class TestCompileParity:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_compiled_target_equals_eager(self):
        profile = PROFILES["rl"]
        config = EnvConfig(
            num_ships=10,
            num_fields=4,
            max_bullets=profile.max_bullets,
            max_episode_steps=profile.max_episode_steps,
            frontline=profile.frontline,
            vision_range=profile.vision_range,
        )
        env = TensorEnv(8, profile.ship_config, config, "cuda")
        env.reset(seed=2)
        observer = torch.zeros(8, dtype=torch.int32, device="cuda")
        world = profile.ship_config.world_size
        eager = HexDensityTarget(world, "cuda", None)(env.state, observer)
        compiled = HexDensityTarget(world, "cuda", "default")(env.state, observer)
        assert torch.allclose(eager, compiled, atol=1e-5)
