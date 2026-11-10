"""The nearest-neighbour MLP (`bnb.models.nearest`): a fixed input width at every fleet size,
the nearest allies, visible enemies and fields in order with padding, the zones by role,
stateless unroll, BC and PPO updates on the CPU without auxiliary terms, and policy files."""

import dataclasses
import math
import pathlib
import sys

import pytest
import torch

from bnb.contracts import GameConfig, Obs, Pool
from bnb.contracts.obs import (
    FIELD_COLUMNS,
    GLOBAL,
    GLOBAL_COLUMNS,
    SHIP,
    SHIP_COLUMNS,
    ZONE,
    ZONE_COLUMNS,
    ZONE_SLOTS,
)
from bnb.controllers.relational import RelationalController
from bnb.game.modes.rules import PRESETS
from bnb.learn import bc
from bnb.learn.ppo import HEAD_SERIES, LearnerConfig
from bnb.models import nearest
from bnb.models.nearest import Nearest, NearestBuild, NearestConfig
from bnb.runs import checkpoint
from bnb.runtime.lanes import LaneConfig
from bnb.runtime.train import LEARNER, TrainConfig, Trainer, model_build, model_controller

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "runs"))
import runs_cases as cases  # noqa: E402

CPU = torch.device("cpu")
pytestmark = pytest.mark.usefixtures("global_generator")
SMALL = NearestConfig(hidden=16, value_bins=11)
# where each block of an own ship's input starts
ALLIES = nearest.SELF_FEATURES
ENEMIES = ALLIES + SMALL.allies * nearest.SHIP_FEATURES
FIELDS = ENEMIES + SMALL.enemies * nearest.SHIP_FEATURES
ZONES = FIELDS + SMALL.fields * nearest.FIELD_FEATURES


def _config(game: GameConfig, **changes) -> TrainConfig:
    values = dict(
        game=game,
        model="nearest",
        nearest=SMALL,
        lanes=LaneConfig(environments=0),
        precision="float32",
    )
    return TrainConfig(**{**values, **changes})


def test_a_set_line_builds_the_model():
    from bnb.cli import with_settings

    settings = [("nearest.hidden", 32), ("nearest.allies", 2), ("nearest.fields", 1)]
    config = with_settings(_config(PRESETS["deathmatch"]), settings)
    build = model_build(config)
    assert build == NearestBuild(NearestConfig(hidden=32, allies=2, fields=1, value_bins=11))
    model = Nearest(build)
    assert model.body[0].in_features == nearest.input_width(build.config)
    assert model.body[2].out_features == 32
    assert isinstance(model_controller(model, config.game), RelationalController)


@pytest.mark.parametrize("ships", (2, 5, 8))
def test_the_input_width_is_the_same_at_every_fleet_and_field_count(ships):
    game = dataclasses.replace(PRESETS["frontline"], ships_per_team=ships, num_fields=ships + 1)
    build = model_build(_config(game))
    assert build == model_build(_config(PRESETS["frontline"]))
    obs = cases.observation(game=game)
    inputs = nearest.features(obs, SMALL)
    assert inputs.shape == (2, ships, nearest.input_width(SMALL)) == (2, ships, 196)
    assert torch.isfinite(inputs).all()


def test_a_model_built_at_5v5_runs_at_8v8():
    model = cases.tiny_model("nearest", width=16, game=PRESETS["frontline"])
    big = dataclasses.replace(PRESETS["frontline"], ships_per_team=8, num_fields=13)
    out = model(cases.observation(game=big))
    assert out.policy_logits.shape == (2, 8, 42) and out.values.shape == (2, 8, 5)
    assert out.outcome_logits.shape == (2, 4)
    assert torch.isfinite(out.values).all()
    # under bf16 autocast, as a bfloat16 run builds it
    bf16 = model_build(_config(big, precision="bfloat16"))
    assert bf16.config.bfloat16_matmuls
    out = Nearest(bf16)(cases.observation(game=big))
    assert out.policy_logits.dtype == torch.float32 and torch.isfinite(out.values).all()


def _constructed() -> Obs:
    """One 4v4 view: own ship 0 at the origin heading +x; allies 1 (300 m), 2 (100 m, dead),
    3 (200 m); enemies 4 (500 m), 5 (hidden), 6 (400 m), 7 (10 m, dead); two fields; the
    zones in a scrambled role order, the one of role 3 absent."""
    ships = torch.zeros(1, 8, len(SHIP_COLUMNS))
    rows = [  # x, alive, visible
        (0.0, 1, 1),
        (300.0, 1, 1),
        (100.0, 0, 1),
        (200.0, 1, 1),
        (500.0, 1, 1),
        (0.0, 1, 0),
        (400.0, 1, 1),
        (10.0, 0, 1),
    ]
    for row, (x, alive, visible) in enumerate(rows):
        ships[0, row, SHIP["x"]] = x
        ships[0, row, SHIP["heading_cos"]] = 1.0 if visible else 0.0
        ships[0, row, SHIP["alive"]] = alive
        ships[0, row, SHIP["visible"]] = visible
        ships[0, row, SHIP["team_relative"]] = float(row >= 4)
    fields = torch.zeros(1, 2, len(FIELD_COLUMNS))
    fields[0, :, 0] = torch.tensor([900.0, -600.0])  # x
    fields[0, :, 2] = torch.tensor([100.0, 50.0])  # radius
    fields[0, :, 3] = 40.0
    fields[0, :, 4] = 2.0
    globals_ = torch.zeros(1, len(GLOBAL_COLUMNS))
    globals_[0, GLOBAL["arena_radius"]] = 2000.0
    zones = torch.zeros(1, ZONE_SLOTS, len(ZONE_COLUMNS))
    for slot, role in enumerate((2, 0, 4, 1, 3)):
        zones[0, slot, ZONE["x"]] = 1000.0 * (role + 1)
        zones[0, slot, ZONE["role"]] = role
        zones[0, slot, ZONE["radius"]] = 150.0
        zones[0, slot, ZONE["progress"]] = 0.1 * role
        zones[0, slot, ZONE["present"]] = float(role != 3)
    zones[0, 4] = torch.where(zones[0, 4, ZONE["present"]] > 0, zones[0, 4], 0.0)
    return Obs(ships=ships, fields=fields, globals=globals_, zones=zones)


def _blocks(inputs: torch.Tensor, start: int, count: int, width: int) -> torch.Tensor:
    return inputs[..., start : start + count * width].unflatten(-1, (count, width))


def test_the_nearest_allies_and_visible_enemies_in_order_with_padding():
    inputs = nearest.features(_constructed(), SMALL)[0, 0]
    allies = _blocks(inputs, ALLIES, SMALL.allies, nearest.SHIP_FEATURES)
    enemies = _blocks(inputs, ENEMIES, SMALL.enemies, nearest.SHIP_FEATURES)
    # forward offset (km) and presence: the dead ally and the dead and hidden enemies are out
    assert allies[:, 0].tolist() == pytest.approx([0.2, 0.3, 0.0])
    assert allies[:, -1].tolist() == [1.0, 1.0, 0.0]
    assert enemies[:, 0].tolist() == pytest.approx([0.4, 0.5, 0.0, 0.0])
    assert enemies[:, -1].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert enemies[:, 4].tolist() == pytest.approx([0.4, 0.5, 0.0, 0.0])  # distance
    assert enemies[:2, 7].tolist() == [1.0, 1.0] and allies[:2, 7].tolist() == [0.0, 0.0]
    assert not allies[2].any() and not enemies[2:].any()  # padding is zero


def test_the_nearest_fields_by_centre_distance_with_padding():
    inputs = nearest.features(_constructed(), SMALL)[0, 0]
    fields = _blocks(inputs, FIELDS, SMALL.fields, nearest.FIELD_FEATURES)
    assert fields[:, 0].tolist() == pytest.approx([-0.6, 0.9, 0.0])  # forward offset
    assert fields[:, 2].tolist() == pytest.approx([0.6, 0.9, 0.0])  # centre distance
    assert fields[:, 3].tolist() == pytest.approx([0.55, 0.8, 0.0])  # edge distance
    assert fields[:2, 4:7].flatten().tolist() == pytest.approx([0.05, 1, 1, 0.1, 1, 1])
    assert fields[:, -1].tolist() == [1.0, 1.0, 0.0] and not fields[2].any()


def test_the_zones_in_role_order_with_presence_and_absent_ones_zero():
    inputs = nearest.features(_constructed(), SMALL)[0, 0]
    zones = _blocks(inputs, ZONES, ZONE_SLOTS, nearest.ZONE_FEATURES)
    assert inputs.shape[-1] == ZONES + ZONE_SLOTS * nearest.ZONE_FEATURES
    present = [role != 3 for role in range(ZONE_SLOTS)]
    assert zones[:, -1].tolist() == [float(one) for one in present]
    for role in range(ZONE_SLOTS):
        if not present[role]:
            assert not zones[role].any(), role
            continue
        assert zones[role, 0].item() == pytest.approx(role + 1.0)  # forward offset, km
        assert zones[role, 3].item() == pytest.approx(0.15)  # radius
        one_hot = zones[role, 4 : 4 + nearest.ZONE_ROLES]
        assert one_hot.tolist() == [float(r == role) for r in range(nearest.ZONE_ROLES)]
        assert zones[role, 9].item() == pytest.approx(0.1 * role)  # progress


def test_a_mode_without_zones_reads_them_all_zero():
    obs = cases.observation(game=PRESETS["deathmatch"])
    inputs = nearest.features(obs, SMALL)
    assert not inputs[..., ZONES:].any()


def test_unroll_is_a_loop_of_step():
    model = cases.tiny_model("nearest", width=16, game=PRESETS["frontline"])
    steps = [cases.observation(seed, game=PRESETS["frontline"]) for seed in range(3)]
    segment = Obs(
        **{
            field.name: torch.stack([getattr(one, field.name) for one in steps])
            for field in dataclasses.fields(Obs)
        }
    )
    reset = torch.zeros(3, 2, dtype=torch.bool)
    with torch.no_grad():
        unrolled = model.unroll(None, segment, None, reset)
        for t, one in enumerate(steps):
            step = model(one)
            for field in dataclasses.fields(step):
                torch.testing.assert_close(
                    getattr(unrolled, field.name)[t], getattr(step, field.name)
                )


def _tiny(objective: str, pools) -> TrainConfig:
    learner = bc.config(minibatches=2) if objective == "bc" else LearnerConfig(minibatches=2)
    return _config(
        GameConfig(2, 2, 1200.0, episode_seconds=2.0),
        pools=pools,
        environments=8,
        length=8,
        shards=1,
        lookahead=0,
        compiled=False,
        host_store=False,
        learner=learner,
        objective=objective,
    )


def _no_auxiliary(row: dict) -> None:
    """The auxiliary heads' series are zero, as for the relational model: the learner has
    no labels for them (`Learner.auxiliary` False) and the model no such heads."""
    for series in HEAD_SERIES.values():
        assert row.get(series, 0.0) == 0.0, series


def test_a_bc_update_runs_on_the_cpu_without_auxiliary_terms():
    from bnb.runtime.cloning import LINEUP

    trainer = Trainer(_tiny("bc", LINEUP), CPU)
    assert not trainer.ppo.auxiliary
    before = [parameter.clone() for parameter in trainer.model.parameters()]
    row = trainer.update()
    assert math.isfinite(row["bc/kl"])
    _no_auxiliary(row)
    assert any(
        not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters(), strict=True)
    )


def test_a_ppo_update_runs_on_the_cpu_without_auxiliary_terms():
    trainer = Trainer(_tiny("ppo", (Pool("chaser", 1.0, (LEARNER, "chaser")),)), CPU)
    assert not trainer.ppo.auxiliary
    row = trainer.update()
    assert math.isfinite(row["policy/kl"]) and math.isfinite(row["critic/value_loss/outcome"])
    _no_auxiliary(row)


def test_a_policy_file_round_trips_and_loads_as_ratings_do(tmp_path):
    model = cases.tiny_model("nearest", width=16, game=PRESETS["frontline"])
    checkpoint.save_policy(tmp_path / "policy", model)
    policy = checkpoint.read_policy(tmp_path / "policy")
    assert policy.name == "nearest" and policy.build == model.build and not policy.filled
    loaded = checkpoint.load_policy(tmp_path / "policy", CPU).eval()
    assert type(loaded) is Nearest
    obs = cases.observation(game=PRESETS["frontline"])
    cases.assert_equal(cases.single_step(loaded, obs), cases.single_step(model, obs))
    # the league's controllers (`bnb.league.matches`) wrap a loaded file the trainer's way
    controller = model_controller(loaded.requires_grad_(False), PRESETS["frontline"])
    action, carry, outputs = controller.step(None, obs, torch.zeros(2, dtype=torch.bool))
    assert carry is None and outputs.values.shape == (2, 5, 5)
