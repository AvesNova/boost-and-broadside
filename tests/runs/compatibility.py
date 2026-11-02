"""The compatibility set (runs spec 4): one tiny policy file per registered model at its
current version, with a fixed input and the outputs of its single step.

    uv run --no-sync python tests/runs/compatibility.py MODEL

writes `tests/runs/checkpoints/<MODEL>-v<VERSION>/` and removes the model's older entry.
Run it only with a version bump, and say in the commit why the version rose.
"""

import pathlib
import shutil
import sys

import runs_cases as cases
import safetensors.torch
import torch

from bnb.contracts import Obs
from bnb.models.yemong.model import Yemong
from bnb.runs.checkpoint import save_policy
from bnb.runs.registry import model_class
from bnb.runs.serialise import named_tensors, tree_from_named

SET = pathlib.Path(__file__).with_name("checkpoints")
SEED = 20261102  # the entry's own: weights, observation and carry
INPUTS, OUTPUTS = "inputs.safetensors", "outputs.safetensors"


def entry(name: str) -> pathlib.Path:
    return SET / f"{name}-v{model_class(name).VERSION}"


def inputs(model, directory: pathlib.Path):
    """The stored observation and (for Yemong) carry of an entry."""
    stored = safetensors.torch.load_file(directory / INPUTS)
    obs = Obs(**{part: stored.get(f"obs.{part}") for part in cases.PARTS})
    carry = None
    if isinstance(model, Yemong):
        views = obs.ships.shape[0]
        blank = model.initial_carry(views, cases.GAME.ships_per_team, cases.GAME.num_ships)
        carry = tree_from_named(blank, stored, "carry.")
    return obs, carry


def write(name: str) -> pathlib.Path:
    """Write the model's entry at its current version, removing its older ones."""
    for old in SET.glob(f"{name}-v*"):
        shutil.rmtree(old)
    directory = entry(name)
    model = cases.tiny_model(name, seed=SEED)
    save_policy(directory / "policy", model)
    obs = cases.observation(seed=SEED)
    stored = {f"obs.{part}": getattr(obs, part) for part in cases.PARTS}
    carry = None
    if isinstance(model, Yemong):
        carry = cases.yemong_carry(model, SEED, obs.ships.shape[0])
        stored.update(named_tensors(carry, "carry."))
    safetensors.torch.save_file({key: value.clone() for key, value in stored.items()},
                                directory / INPUTS)
    outputs = cases.single_step(model, obs, carry)
    safetensors.torch.save_file({key: value.clone() for key, value in outputs.items()},
                                directory / OUTPUTS)
    return directory


if __name__ == "__main__":
    torch.set_num_threads(1)
    for model_name in sys.argv[1:]:
        print(f"wrote {write(model_name)}")
