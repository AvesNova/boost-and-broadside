"""Policy files (runs spec 3 and acceptance 1, 2, 4, 5): the round trip, two builds side by
side, and each refusal with its message."""

import dataclasses
import json

import pytest
import runs_cases as cases
import torch

from bnb.contracts import Obs
from bnb.game import api
from bnb.models.yemong.critic import ReturnScale
from bnb.models.yemong.model import Yemong
from bnb.runs import checkpoint
from bnb.runs.checkpoint import CheckpointError, FilledDefaults, load_policy, save_policy
from bnb.runs.serialise import SerialiseError

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


MODELS = ("yemong", "relational", "mlp")


def _inputs(model, device=cases.CPU):
    obs = cases.observation(device=device)
    carry = cases.yemong_carry(model, 1, 2, device=device) if isinstance(model, Yemong) else None
    return obs, carry


@pytest.mark.parametrize("name", MODELS)
def test_a_loaded_policy_steps_bit_for_bit_like_the_saved_one(tmp_path, name):
    model = cases.tiny_model(name)
    obs, carry = _inputs(model)
    save_policy(tmp_path / "policy", model)
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    assert type(loaded) is type(model) and loaded.build == model.build
    cases.assert_equal(cases.single_step(loaded, obs, carry), cases.single_step(model, obs, carry))


@pytest.mark.gpu
@pytest.mark.parametrize("name", MODELS)
def test_a_policy_loads_bit_for_bit_on_the_gpu(tmp_path, name):
    cuda = torch.device("cuda")
    model = cases.tiny_model(name).to(cuda)
    obs, carry = _inputs(model, cuda)
    save_policy(tmp_path / "policy", model)
    loaded = load_policy(tmp_path / "policy", cuda).eval()
    assert next(loaded.parameters()).device.type == "cuda"
    cases.assert_equal(cases.single_step(loaded, obs, carry), cases.single_step(model, obs, carry))


def test_the_return_scale_and_steps_travel_with_the_policy(tmp_path):
    scale = ReturnScale(0.99, 1.0)
    scale.low, scale.high = torch.tensor(-0.3721), torch.tensor(2.718281)
    save_policy(
        tmp_path / "policy",
        cases.tiny_model("relational"),
        return_scale=scale,
        steps=dict(environment_steps=12345),
        game=cases.GAME,
    )
    policy = checkpoint.read_policy(tmp_path / "policy")
    assert policy.return_scale.keys() == {"low", "high"}
    for name in ("low", "high"):  # the state tensors themselves, bit for bit
        assert torch.equal(policy.return_scale[name], getattr(scale, name))
    assert (tmp_path / "policy" / checkpoint.RETURN_SCALE).is_file()
    assert checkpoint.read_policy(tmp_path / "policy").raw["files"].keys() == {
        checkpoint.WEIGHTS,
        checkpoint.RETURN_SCALE,
    }  # hashed like the weights
    assert policy.steps == dict(environment_steps=12345)
    assert policy.raw["game"]["ships_per_team"] == 2


def test_two_widths_load_side_by_side_and_play_each_other(tmp_path):
    """Acceptance 2: a width-128 and a width-256 Yemong (tiny otherwise) in one process,
    each built from its own record, each playing one step of a match against the other."""
    from bnb.contracts import Pool
    from bnb.controllers.yemong import YemongController
    from bnb.runtime.rollout import Rollout

    for width in (128, 256):
        save_policy(tmp_path / f"width_{width}", cases.tiny_model("yemong", width=width))
    narrow = load_policy(tmp_path / "width_128", cases.CPU)
    wide = load_policy(tmp_path / "width_256", cases.CPU)
    assert (narrow.config.width, wide.config.width) == (128, 256)
    game = cases.GAME
    controllers = {
        name: YemongController(model, game.ships_per_team, game.num_ships)
        for name, model in (("narrow", narrow), ("wide", wide))
    }
    rollout = Rollout(
        game,
        api.ship_types(cases.CPU),
        (Pool("match", 1.0, ("narrow", "wide")),),
        controllers,
        "narrow",
        2,
        1,
        cases.CPU,
        compiled=False,
    )
    trajectory = rollout.run()
    assert (
        torch.isfinite(trajectory.log_probability).all() and torch.isfinite(trajectory.value).all()
    )


def _edit_record(directory, change):
    path = directory / checkpoint.RECORD
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def test_a_policy_file_records_the_game_version(tmp_path):
    from bnb.game import VERSION

    save_policy(tmp_path / "policy", cases.tiny_model("relational"))
    record = json.loads((tmp_path / "policy" / "checkpoint.json").read_text())
    assert record["game_version"] == VERSION


def test_a_changed_weights_file_is_refused_by_its_hash(tmp_path):
    save_policy(tmp_path / "policy", cases.tiny_model("relational"))
    weights = tmp_path / "policy" / checkpoint.WEIGHTS
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1
    weights.write_bytes(bytes(data))
    with pytest.raises(CheckpointError, match="SHA-256 .* the file changed after it was written"):
        load_policy(tmp_path / "policy", cases.CPU)


def test_a_schema_version_mismatch_is_refused_naming_both(tmp_path):
    save_policy(tmp_path / "policy", cases.tiny_model("relational"))
    _edit_record(tmp_path / "policy", lambda record: record["schemas"].update(Obs=2))
    with pytest.raises(
        CheckpointError,
        match=r"schema versions \{'Obs': 2, 'Action': 1\}; "
        rf"the code has \{{'Obs': {Obs.SCHEMA_VERSION}",
    ):
        load_policy(tmp_path / "policy", cases.CPU)


def test_a_model_version_mismatch_is_refused_naming_both(tmp_path):
    save_policy(tmp_path / "policy", cases.tiny_model("yemong"))
    _edit_record(tmp_path / "policy", lambda record: record["model"].update(version=7))
    with pytest.raises(
        CheckpointError,
        match=r"yemong version 7; the code has version 1\. "
        r"Load it from a worktree",
    ):
        load_policy(tmp_path / "policy", cases.CPU)


def test_an_unknown_config_field_is_refused_naming_it(tmp_path):
    save_policy(tmp_path / "policy", cases.tiny_model("yemong"))
    _edit_record(
        tmp_path / "policy", lambda record: record["model"]["build"]["config"].update(dropout=0.1)
    )
    with pytest.raises(SerialiseError, match="unknown field config.dropout in YemongConfig"):
        load_policy(tmp_path / "policy", cases.CPU)


def test_a_missing_build_field_takes_its_default_and_the_load_reports_it(tmp_path):
    """Acceptance 5: a field added since the file was written loads with its default."""
    model = cases.tiny_model("yemong")
    save_policy(tmp_path / "policy", model)
    _edit_record(
        tmp_path / "policy",
        lambda record: record["model"]["build"]["config"].pop("presence_radius"),
    )
    assert checkpoint.read_policy(tmp_path / "policy").filled == ("config.presence_radius",)
    with pytest.warns(FilledDefaults, match="lacks config.presence_radius"):
        loaded = load_policy(tmp_path / "policy", cases.CPU)
    default = type(model.config)().presence_radius
    assert loaded.config == dataclasses.replace(model.config, presence_radius=default)


def test_a_policy_file_is_never_overwritten(tmp_path):
    model = cases.tiny_model("relational")
    save_policy(tmp_path / "policy", model)
    with pytest.raises(FileExistsError):
        save_policy(tmp_path / "policy", model)
