"""The three landmark runs still load through the ordinary loader.

This replaces ``tests/migration/`` (813 lines), which pinned the same property
for run 682 alone by comparing sixteen checkpoints against a frozen npz of
historical activations. That apparatus guarded a one-time migration which has
long since run and cannot run again. What is still worth checking every commit
is much smaller: the runs the documents cite can be read by the code as it
stands today, with no migration path and no special case.

682 is the reason the zero-field width has to keep working. It trained before
fields existed, and it is still the only run at ``num_fields=0``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.config.run_config import latest_config
from boost_and_broadside.evaluation.agents import resolve_agent_spec
from boost_and_broadside.evaluation.run_catalog import (
    resolve_exact_run,
    select_final_training_checkpoint,
)
from boost_and_broadside.evaluation.tournament import load_run_config

# run name -> (num_ships, num_fields, records a field map). Only the two runs
# kept in-repo: 716 was a local run whose checkpoints were never committed, so a
# clean clone has nothing to check it against.
_LANDMARKS = {
    "resilient-resonance-682": (8, 0, False),
    "good-leaf-719": (8, 4, True),
}

_LFS_POINTER = b"version https://git-lfs"


def _fetched_run(run: str) -> Path:
    run_dir = Path("checkpoints") / run
    if not run_dir.is_dir():
        pytest.skip(f"{run} is not present in this checkout")
    checkpoint = select_final_training_checkpoint(run_dir).path
    if checkpoint.read_bytes()[: len(_LFS_POINTER)] == _LFS_POINTER:
        pytest.skip(f"{run} is an unfetched lfs pointer; run `git lfs pull`")
    return run_dir


@pytest.mark.parametrize(("run", "expected"), sorted(_LANDMARKS.items()))
def test_a_landmark_run_loads_with_the_environment_it_trained_in(run, expected) -> None:
    run_dir = _fetched_run(run)
    resolve_exact_run(run, "checkpoints")
    checkpoint = select_final_training_checkpoint(run_dir).path

    env_config, model_config, paradigm, field_map = load_run_config(run_dir)
    num_ships, num_fields, has_field_map = expected

    assert (env_config.num_ships, env_config.num_fields) == (num_ships, num_fields)
    assert (field_map is not None) is has_field_map
    assert paradigm == "ego_pass"

    # The loader, not a fixture: this is the path every evaluation mode takes.
    agent = resolve_agent_spec(
        str(checkpoint), ShipConfig(), model_config, "cpu", num_ships=env_config.num_ships
    )
    assert agent.bundle is not None
    assert agent.bundle.env_config.num_fields == num_fields


@pytest.mark.parametrize("run", sorted(_LANDMARKS))
def test_a_landmark_run_records_the_config_it_actually_trained_under(run) -> None:
    """Migrated histories, preserving absence rather than backfilling it.

    682 predates ``resolved_config`` entirely; its segment is assembled from the
    three configs its checkpoint does name, and is short exactly where the run
    was. Filling the gaps from today's profile would produce a record of a run
    that never happened -- 682 trained with no fields, so it has no field reward
    weights and no field map, and its config says so.
    """

    run_dir = _fetched_run(run)
    segment = latest_config(run_dir)
    assert segment is not None, f"{run} has no recorded config history"
    assert segment.from_step == 0

    train_config = segment.config["train_config"]
    rewards = train_config["rewards"]
    _, num_fields, _ = _LANDMARKS[run]

    if num_fields == 0:
        # Absent, not zero: these fields did not exist when 682 ran.
        assert "field_damage_taken_weight" not in rewards
        assert "field_death_weight" not in rewards
        assert "field_map" not in train_config
        # The profile name was never recorded either, and is not invented.
        assert segment.profile == "unrecorded"
    else:
        assert rewards["field_damage_taken_weight"] == 0.5
        assert train_config["field_map"] is not None
        # The name the profile had when the run happened, not the merged one.
        assert segment.profile == "rl-fields"
