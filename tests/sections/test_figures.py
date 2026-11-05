"""The figures: the decoder layer against the codes it inverts, the shot measures on
hand-built flights, and a capture drawn end to end with the tiny Yemong policy."""

import math
import pathlib

import pytest
import torch

from bnb.models.yemong import decode
from bnb.models.yemong.shots import shot_code_parameters, trajectory_code

POLICY = pathlib.Path(__file__).resolve().parents[1] / "runs" / "checkpoints" / "yemong-v1"


def test_the_shot_path_decodes_a_sharp_trajectory_code_back_to_its_snapshot():
    parameters = shot_code_parameters(500.0, 300.0, 0.4, 8.0)
    snapshots = torch.tensor(
        [[12.0, -3.5, 610.0, -40.0, 1.3], [-0.6, 47.25, 200.0, 150.0, 0.8]], dtype=torch.float64
    )
    code = trajectory_code(parameters, snapshots)
    path = decode.shot_path(parameters, torch.log(code.clamp_min(1e-30)))
    assert path.displacement.double() == pytest.approx(snapshots[:, :2], abs=1e-3)
    angle = torch.atan2(snapshots[:, 3], snapshots[:, 2])
    assert path.direction.double() == pytest.approx(angle, abs=1e-4)
    assert path.speed.double() == pytest.approx(snapshots[:, 2:4].norm(dim=-1), rel=1e-4)
    assert path.log_index.double() == pytest.approx(torch.log(snapshots[:, 4]), abs=1e-4)
    assert path.displacement_sigma.max() < 1e-2


def test_the_launch_frame_and_the_world_are_inverse():
    heading = torch.tensor([0.3, -2.0])
    displacement = torch.tensor([[[10.0, 2.0], [20.0, -5.0]], [[1.0, 1.0], [0.0, 3.0]]])
    x, y = torch.tensor([100.0, -50.0]), torch.tensor([7.0, 8.0])
    world = decode.launch_frame_to_world(displacement, x, y, heading.cos(), heading.sin())
    offset = world - torch.stack([x, y], -1)[:, None]
    back = decode.world_to_launch_frame(offset, heading.cos()[:, None], heading.sin()[:, None])
    assert back == pytest.approx(displacement, abs=1e-5)


def _flights(predicted_bend: float) -> dict:
    """Two labelled shots of 12 ages flying +x at 600 m/s, tick 1/30 s: one straight, one
    bending left by 0.5 m times the age squared (72 m at the last age); the head predicts
    the straight one exactly and predicted_bend times the bent one's lateral offset. The
    bent one was launched inside a band."""
    tick, ages = 1.0 / 30.0, 12
    along = 600.0 * tick * torch.arange(1, ages + 1)
    lateral = 0.5 * torch.arange(1, ages + 1) ** 2
    true = torch.zeros((1, 1, 2, ages, 5))
    true[..., 0] = along
    true[0, 0, 1, :, 1] = lateral
    predicted = true[..., :2].clone()
    predicted[0, 0, 1, :, 1] = predicted_bend * lateral
    return dict(
        exact_snapshots=true,
        exact_age_valid=torch.ones((1, 1, 2, ages), dtype=torch.bool),
        labelled=torch.ones((1, 1, 2), dtype=torch.bool),
        launch_velocity=torch.tensor([600.0, 0.0]).expand(1, 1, 2, 2),
        launch_gradient=torch.tensor([0.0, 0.01]).expand(1, 1, 2),
        predicted=dict(displacement=predicted, displacement_sigma=torch.ones((1, 1, 2, ages))),
    )


@pytest.mark.parametrize("bend", [0.0, 0.5, 1.0])
def test_the_shot_measures_split_bent_flights_and_fit_the_head_s_bend(bend):
    from bnb.figures import shots

    numbers = shots.measures(_flights(bend), 1.0 / 30.0)
    assert numbers["straight"]["shots"] == 1 and numbers["bent"]["shots"] == 1
    assert numbers["straight"]["head_error_by_age"] == [0.0] * 12
    fit = numbers["lateral"]["launched_in_band"]
    assert fit["shots"] == 1
    assert numbers["lateral"]["met_band_after_launch"]["shots"] == 0
    last = 0.5 * 144
    assert numbers["bent"]["head_error_by_age"][-1] == pytest.approx((1 - bend) * last, abs=1e-3)
    assert numbers["bent"]["straight_line_error_by_age"][-1] == pytest.approx(last, abs=1e-3)
    two = shots._lateral_fit(torch.tensor([10.0, -20.0]), bend * torch.tensor([10.0, -20.0]))
    assert two["slope"] == pytest.approx(bend)


def test_the_scenario_field_meets_the_flight_at_its_incidence():
    from bnb.figures.scenarios import Scenario, field_centre

    start, direction = torch.tensor([10.0, 20.0]), torch.tensor([0.0, 1.0])
    for inside in (False, True):
        scenario = Scenario(
            "case", "", level=2, radius=300.0, along=0.5, incidence=0.6, inside=inside
        )
        centre = field_centre(scenario, start, direction, 200.0)
        boundary = start + direction * 100.0
        assert (boundary - centre).norm() == pytest.approx(300.0)
        assert ((start - centre).norm() < 300.0) == inside
        normal = (boundary - centre) / 300.0
        cosine = float((normal * direction).sum()) * (1 if inside else -1)
        assert cosine == pytest.approx(math.cos(0.6), abs=1e-5)


@pytest.mark.slow
def test_a_capture_draws_every_figure_set_with_its_provenance(
    tmp_path, monkeypatch, global_generator
):
    pytest.importorskip("matplotlib")
    from PIL import Image

    from bnb.figures import capture, scenarios
    from bnb.figures.sets import FIGURE_SETS

    monkeypatch.setattr(scenarios, "CLEARANCE", 0.0)  # two games: no ship need be clear
    config = capture.CaptureConfig(environments=1, ticks=24, spread_samples=1)
    artifact = capture.capture(
        POLICY / "policy", config, torch.device("cpu"), scenarios=scenarios.measure
    )
    path = capture.save(artifact, tmp_path / "captures" / "capture.pt")
    artifact = capture.load(path)
    for name, render in FIGURE_SETS.items():
        written, numbers = render(artifact, tmp_path)
        assert written and isinstance(numbers, dict), name
        for png in written:
            text = Image.open(png).text
            assert text["Source"] == str(POLICY / "policy")
            assert "Commit" in text and "Command" in text and "Capture commit" in text


def test_the_belief_measures_count_only_hidden_living_believed_enemies():
    from bnb.contracts.obs import SHIP, SHIP_COLUMNS
    from bnb.figures import belief

    L, R, own, enemies = 1, 1, 1, 3
    rows = torch.zeros((L, R, own + enemies, len(SHIP_COLUMNS)))
    rows[..., SHIP["alive"]] = 1.0
    ships = rows.clone()
    ships[..., own + 2, SHIP["visible"]] = 1.0  # enemy 2 is seen: not counted
    position = torch.zeros((L, R, enemies, 2))
    position[..., 0, 0] = 1.0  # enemy 0: 1 m off with sigma 1, normalised square 0.5
    position[..., 1, 0] = 3.0  # enemy 1: 3 m off, normalised square 4.5
    artifact = dict(
        meta=dict(ships_per_team=own, tick_length=0.1),
        rollout=dict(
            rows=rows,
            ships=ships,
            belief=dict(
                valid=torch.ones((L, R, enemies), dtype=torch.bool),
                age=torch.full((L, R, enemies), 3, dtype=torch.int32),  # 0.3 s
                position=position,
                position_sigma=torch.ones((L, R, enemies)),
                velocity=torch.zeros((L, R, enemies, 2)),
                velocity_covariance=torch.tensor([1.0, 0.0, 1.0]).expand(L, R, enemies, 3),
            ),
        ),
    )
    numbers = belief.measures(artifact)["age_0.1_0.5_seconds"]
    assert numbers["count"] == 2
    assert numbers["position_error"] == pytest.approx(2.0)
    assert numbers["position_within_one_sigma"] == 0.5
    assert numbers["position_within_two_sigma"] == 0.5  # 4.5 > 4
    assert numbers["velocity_within_one_sigma"] == 1.0


def test_the_next_state_measures_score_the_forecast_against_the_next_row():
    from bnb.contracts.obs import SHIP, SHIP_COLUMNS
    from bnb.figures import next_state

    L, R, N, own = 2, 1, 2, 1
    rows = torch.zeros((L, R, N, len(SHIP_COLUMNS)))
    rows[..., SHIP["alive"]] = 1.0
    rows[..., SHIP["visible"]] = 1.0
    rows[..., SHIP["velocity_x"]] = 30.0  # 3 m per tick of 0.1 s
    following = rows.clone()
    following[..., SHIP["x"]] = 3.0
    forecast = torch.zeros((L, R, N, 2))
    forecast[..., 0] = 2.0  # 1 m short
    artifact = dict(
        meta=dict(ships_per_team=own, tick_length=0.1),
        rollout=dict(
            rows=rows,
            next_rows=following,
            ships=rows,
            next_state_mask=torch.ones((L, R, N), dtype=torch.bool),
            next_state=dict(position=forecast, position_sigma=torch.full((L, R, N), 2.0)),
            belief=dict(
                position=torch.zeros((L, R, N - own, 2)), velocity=torch.zeros((L, R, N - own, 2))
            ),
        ),
    )
    numbers = next_state.measures(artifact)
    assert numbers["own"]["rows"] == 2 and numbers["enemy_seen"]["rows"] == 2
    assert numbers["own"]["head_error"] == pytest.approx(1.0)
    assert numbers["own"]["stay_error"] == pytest.approx(3.0)
    assert numbers["own"]["constant_velocity_error"] == pytest.approx(0.0, abs=1e-5)
    assert numbers["own"]["within_one_sigma"] == 1.0
    assert "enemy_hidden" not in numbers


def test_the_density_rings_count_six_more_cells_per_ring():
    from bnb.figures.density import rings_of

    rings = rings_of(10)
    counts = torch.bincount(rings)
    assert counts.tolist() == [1] + [6 * k for k in range(1, 11)]
