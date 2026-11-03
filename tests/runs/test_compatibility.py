"""The compatibility set (runs spec 4, acceptance 6): every registered model has a tiny
policy file at its current version whose single step still gives the stored outputs. A
change to what a model computes without a version bump fails here."""

import compatibility
import pytest
import runs_cases as cases
import safetensors.torch
import torch

from bnb.runs.checkpoint import load_policy
from bnb.runs.registry import MODELS

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


# float32 on the CPU. The same machine and library versions reproduce the outputs bit for
# bit; another CPU or BLAS may order a sum differently, which moves an output by a few ulps
# of the largest term (the weights are of order 0.3, the outputs of order 10). A change of
# computation moves them by orders of magnitude more.
OUTPUT_TOLERANCE = dict(rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_model_has_exactly_one_entry_at_its_current_version(name):
    entries = sorted(path.name for path in compatibility.SET.glob(f"{name}-v*"))
    assert entries == [compatibility.entry(name).name], (
        f"{name} is at version {MODELS[name].VERSION}; write its entry with "
        f"tests/runs/compatibility.py {name}"
    )


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_stored_policy_still_computes_its_stored_outputs(name):
    directory = compatibility.entry(name)
    model = load_policy(directory / "policy", cases.CPU).eval()
    obs, carry = compatibility.inputs(model, directory)
    want = safetensors.torch.load_file(directory / compatibility.OUTPUTS)
    got = cases.single_step(model, obs, carry)
    assert got.keys() == want.keys()
    for key, tensor in want.items():
        torch.testing.assert_close(
            got[key],
            tensor,
            **OUTPUT_TOLERANCE,
            msg=lambda message: (
                f"{name} version {MODELS[name].VERSION} computes another {key}: a change to what "
                f"saved weights compute needs a version bump (runs spec 4). {message}"
            ),
        )
