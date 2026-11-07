"""A policy file's record of its observation columns (runs spec 3): the round trip writes
it, a file from before the record loads with the columns of the time, a record of fewer
columns than the contract loads (by name for Yemong, selected by name for the MLP), and a
recorded column the code lacks is refused naming it."""

import json

import pytest
import runs_cases as cases
import torch

from bnb.contracts import Obs
from bnb.contracts.obs import COLUMNS, COLUMNS_BEFORE_RECORDS
from bnb.models.mlp import MLP, MLPBuild
from bnb.runs import checkpoint
from bnb.runs.checkpoint import CheckpointError, load_policy, save_policy

pytestmark = pytest.mark.usefixtures("global_generator")


def _edit_record(directory, change):
    path = directory / checkpoint.RECORD
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def _mlp(observation=None, seed: int = 3) -> MLP:
    game = cases.GAME
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MLP(MLPBuild(game.ships_per_team, game.num_ships, game.num_fields, 8), observation)
    return cases._randomised(model, seed)


def _narrowed(obs: Obs, observation: dict) -> Obs:
    """The observation with only the named columns of each part, in their order: what the
    code's contract gave when it had just these."""
    index = {
        part: [COLUMNS[part].index(name) for name in names] for part, names in observation.items()
    }
    return Obs(
        **{
            part: getattr(obs, part)[..., index[part]]
            for part in ("ships", "fields", "globals", "zones")
        }
    )


def test_the_frozen_columns_are_the_contracts_first_columns_in_order():
    """Columns are only ever added: the lists before the record are each part's, in order,
    with the new columns between or after them."""
    for part, names in COLUMNS_BEFORE_RECORDS.items():
        positions = [COLUMNS[part].index(name) for name in names]
        assert positions == sorted(positions), part


@pytest.mark.parametrize("name", ("yemong", "relational", "mlp"))
def test_the_round_trip_writes_the_record(tmp_path, name):
    save_policy(tmp_path / "policy", cases.tiny_model(name))
    record = json.loads((tmp_path / "policy" / checkpoint.RECORD).read_text())
    assert record["observation"] == {part: list(names) for part, names in COLUMNS.items()}
    assert checkpoint.read_policy(tmp_path / "policy").observation == COLUMNS


@pytest.mark.parametrize("name", ("yemong", "relational"))
def test_a_policy_from_before_the_record_loads_and_steps_bit_for_bit(tmp_path, name):
    model = cases.tiny_model(name)
    obs = cases.observation()
    carry = cases.yemong_carry(model, 1, 2) if name == "yemong" else None
    save_policy(tmp_path / "policy", model)
    _edit_record(tmp_path / "policy", lambda record: record.pop("observation"))
    assert checkpoint.read_policy(tmp_path / "policy").observation == COLUMNS_BEFORE_RECORDS
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    cases.assert_equal(cases.single_step(loaded, obs, carry), cases.single_step(model, obs, carry))


def test_an_mlp_from_before_the_record_reads_its_old_columns_bit_for_bit(tmp_path):
    """An MLP file written before the record (24 ship columns) loads built for them and,
    on today's observation, computes what it computed on the observation of its time."""
    old = _mlp(COLUMNS_BEFORE_RECORDS)
    obs = cases.observation()
    save_policy(tmp_path / "policy", old)
    _edit_record(tmp_path / "policy", lambda record: record.pop("observation"))
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    assert loaded.observation == COLUMNS_BEFORE_RECORDS
    assert loaded.own.in_features == len(COLUMNS_BEFORE_RECORDS["ships"])
    narrow = _narrowed(obs, COLUMNS_BEFORE_RECORDS)
    with torch.no_grad():
        want = _mlp_as_of(old, narrow)
        got = loaded(obs)
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g, w)


def _mlp_as_of(model: MLP, narrow: Obs):
    """The model's forward on an observation that holds exactly its columns: the selection
    switched off, so the rows are read as they are."""
    selected = model.selected
    model.selected = dict.fromkeys(selected)
    try:
        return model(narrow)
    finally:
        model.selected = selected


def test_a_record_of_fewer_columns_loads_for_the_mlp(tmp_path):
    fewer = dict(
        COLUMNS,
        ships=tuple(name for name in COLUMNS["ships"] if name not in ("energy", "fired")),
        globals=tuple(reversed(COLUMNS["globals"])),  # the recorded order, not the contract's
    )
    model = _mlp(fewer)
    obs = cases.observation()
    save_policy(tmp_path / "policy", model)
    assert checkpoint.read_policy(tmp_path / "policy").observation == fewer
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    assert loaded.own.in_features == len(COLUMNS["ships"]) - 2
    cases.assert_equal(cases.single_step(loaded, obs), cases.single_step(model, obs))
    with torch.no_grad():
        want = _mlp_as_of(model, _narrowed(obs, fewer))
    cases.assert_equal(cases.single_step(loaded, obs), dict(logits=want[0], value=want[1]))


def test_a_record_of_fewer_columns_loads_for_yemong(tmp_path):
    """Yemong reads its columns by name from the code's contract: a record that lists fewer
    loads, and the model steps as saved."""
    model = cases.tiny_model("yemong")
    obs = cases.observation()
    carry = cases.yemong_carry(model, 1, 2)
    save_policy(tmp_path / "policy", model)
    _edit_record(
        tmp_path / "policy", lambda record: record["observation"].update(ships=["x", "y", "alive"])
    )
    assert checkpoint.read_policy(tmp_path / "policy").observation["ships"] == ("x", "y", "alive")
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    cases.assert_equal(cases.single_step(loaded, obs, carry), cases.single_step(model, obs, carry))


@pytest.mark.parametrize("name", ("yemong", "mlp"))
def test_a_recorded_column_the_code_lacks_is_refused_naming_it(tmp_path, name):
    save_policy(tmp_path / "policy", cases.tiny_model(name))
    _edit_record(
        tmp_path / "policy",
        lambda record: record["observation"]["ships"].append("warp_charge"),
    )
    with pytest.raises(
        CheckpointError, match=r"columns the code does not have: ships\.warp_charge"
    ):
        load_policy(tmp_path / "policy", cases.CPU)
