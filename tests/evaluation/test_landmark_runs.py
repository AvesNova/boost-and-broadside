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

import pytest

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.evaluation.agents import resolve_agent_spec
from boost_and_broadside.evaluation.run_catalog import (
    resolve_exact_run,
    select_final_training_checkpoint,
)
from boost_and_broadside.evaluation.tournament import load_run_config

# run name -> (num_ships, num_fields, records a field map)
_LANDMARKS = {
    "resilient-resonance-682": (8, 0, False),
    "lunar-cosmos-716": (8, 4, True),
    "good-leaf-719": (8, 4, True),
}

_LFS_POINTER = b"version https://git-lfs"


@pytest.mark.parametrize(("run", "expected"), sorted(_LANDMARKS.items()))
def test_a_landmark_run_loads_with_the_environment_it_trained_in(run, expected) -> None:
    run_dir = resolve_exact_run(run, "checkpoints").path
    checkpoint = select_final_training_checkpoint(run_dir).path
    if checkpoint.read_bytes()[: len(_LFS_POINTER)] == _LFS_POINTER:
        pytest.skip(f"{run} is an unfetched lfs pointer; run `git lfs pull`")

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
