"""Policy files and training checkpoints (runs spec 3).

A checkpoint is a directory: `checkpoint.json` (the record) and safetensors files, never
pickle. A policy file holds what an agent needs to play: the model's registered name,
version and build record, its weights, the schema versions, and the return scale and step
counts for a warm start. A training checkpoint is a policy file plus everything a resume
needs (`training.safetensors` and the record's training part); `load_policy` reads the
policy part of either kind.

Writes are atomic: everything goes into `<name>.tmp/`, every file is flushed, the
directory is renamed to `<name>/` and the parent flushed, so a crash leaves no half-written
checkpoint. Loading checks the format, the hashes, the schema versions and the model's
version against the code's, and fails naming both sides of a difference.
"""

import dataclasses
import json
import os
import pathlib
import shutil
import warnings

import safetensors.torch
import torch

from bnb.contracts import Action, GameConfig, Obs
from bnb.game.ship_types import ShipTypes
from bnb.models.yemong.critic import ReturnScale

from . import provenance
from .registry import BUILDS, model_class, name_of
from .serialise import from_json, to_json

FORMAT = 1
RECORD = "checkpoint.json"
WEIGHTS = "weights.safetensors"
TRAINING = "training.safetensors"


class CheckpointError(ValueError):
    """A checkpoint that cannot be loaded as it is, with the reason."""


class FilledDefaults(UserWarning):
    """A loaded record lacked fields, which took their defaults (runs spec 4)."""


def schemas() -> dict[str, int]:
    """The code's schema versions of the contracts a policy depends on."""
    return {"Obs": Obs.SCHEMA_VERSION, "Action": Action.SCHEMA_VERSION}


@dataclasses.dataclass(frozen=True)
class PolicyRecord:
    """A policy file's record, checked: the model's name and version, its build record (the
    model's dataclass), the build fields its defaults filled, the return scale's (low, high)
    or None, the step counts, and the raw record."""

    name: str
    version: int
    build: object
    filled: tuple[str, ...]
    return_scale: tuple[float, float] | None
    steps: dict
    raw: dict


def _ship_card(types: ShipTypes) -> dict:
    return {field.name: getattr(types, field.name).tolist() for field in dataclasses.fields(types)}


def policy_record(model: torch.nn.Module, return_scale: ReturnScale | None = None,
                  steps: dict | None = None, game: GameConfig | None = None,
                  types: ShipTypes | None = None) -> dict:
    """The record of a policy file for `model` (its file hashes added on writing)."""
    head, dirty = provenance.commit()
    scale = None if return_scale is None else dict(low=float(return_scale.low),
                                                    high=float(return_scale.high))
    return dict(format=FORMAT, kind="policy",
                model=dict(name=name_of(model), version=type(model).VERSION,
                           build=to_json(model.build)),
                schemas=schemas(), game=None if game is None else to_json(game),
                ship_card=None if types is None else _ship_card(types),
                return_scale=scale, steps=dict(steps or {}), commit=head, dirty=dirty,
                time=provenance.now())


def weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The model's state as host tensors that own their memory (safetensors refuses shared
    storage)."""
    return {name: tensor.detach().to("cpu", copy=True).contiguous()
            for name, tensor in model.state_dict().items()}


def _flush(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_checkpoint(directory: str | os.PathLike, record: dict,
                     files: dict[str, dict[str, torch.Tensor]],
                     stage: object = None) -> pathlib.Path:
    """Write `files` (file name to named tensors) and `record` (with each file's SHA-256
    added) as the checkpoint `directory`, atomically. An existing checkpoint there is never
    replaced. `stage(name)` is called after each file is flushed (the atomicity tests
    raise from it)."""
    final = pathlib.Path(directory)
    if final.exists():
        raise FileExistsError(f"{final} exists; a checkpoint is never overwritten")
    temporary = final.with_name(final.name + ".tmp")
    if temporary.exists():  # a crashed write's leftovers
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    hashes = {}
    for name, tensors in files.items():
        path = temporary / name
        safetensors.torch.save_file(tensors, path)
        _flush(path)
        hashes[name] = provenance.file_hash(path)
        if stage is not None:
            stage(name)
    path = temporary / RECORD
    path.write_text(json.dumps(dict(record, files=hashes), indent=1) + "\n")
    _flush(path)
    if stage is not None:
        stage(RECORD)
    temporary.rename(final)
    _flush(final.parent)
    return final


def save_policy(directory: str | os.PathLike, model: torch.nn.Module, *,
                return_scale: ReturnScale | None = None, steps: dict | None = None,
                game: GameConfig | None = None,
                types: ShipTypes | None = None) -> pathlib.Path:
    """Write `model` as a policy file at `directory`: its registered name, version and build
    record, its weights, the schema versions, and for a warm start the return scale and the
    step counts (`steps`, such as {"environment_steps": n}). game and types (the ship card)
    are recorded for information (ratings are keyed by them)."""
    record = policy_record(model, return_scale, steps, game, types)
    return write_checkpoint(directory, record, {WEIGHTS: weights(model)})


def read_record(directory: str | os.PathLike) -> dict:
    """A checkpoint's record, after checking its format and every file's hash."""
    directory = pathlib.Path(directory)
    path = directory / RECORD
    if not path.is_file():
        raise CheckpointError(f"{directory} is not a checkpoint: it has no {RECORD}")
    record = json.loads(path.read_text())
    if record.get("format") != FORMAT:
        raise CheckpointError(f"{directory} is checkpoint format {record.get('format')}; "
                              f"this code reads format {FORMAT}")
    for name, expected in record["files"].items():
        if not (directory / name).is_file():
            raise CheckpointError(f"{directory / name} is missing")
        actual = provenance.file_hash(directory / name)
        if actual != expected:
            raise CheckpointError(f"{directory / name} has SHA-256 {actual}, but the record "
                                  f"says {expected}: the file changed after it was written")
    return record


def read_policy(directory: str | os.PathLike) -> PolicyRecord:
    """The checked policy part of a checkpoint of either kind: the schema versions and the
    model's version must equal the code's, and the build record must read."""
    record = read_record(directory)
    if record["schemas"] != schemas():
        raise CheckpointError(f"{directory} was trained on schema versions "
                              f"{record['schemas']}; the code has {schemas()}")
    model = record["model"]
    cls = model_class(model["name"])
    if model["version"] != cls.VERSION:
        raise CheckpointError(
            f"{directory} holds {model['name']} version {model['version']}; the code has "
            f"version {cls.VERSION}. Load it from a worktree at a commit with version "
            f"{model['version']}")
    build, filled = from_json(BUILDS[cls], model["build"])
    scale = record["return_scale"]
    return PolicyRecord(name=model["name"], version=model["version"], build=build,
                        filled=tuple(filled),
                        return_scale=None if scale is None else (scale["low"], scale["high"]),
                        steps=record["steps"], raw=record)


def build_model(policy: PolicyRecord, device: torch.device) -> torch.nn.Module:
    """A fresh model from a policy record's build (its weights not loaded)."""
    if policy.filled:
        warnings.warn(f"the {policy.name} build record lacks {', '.join(policy.filled)}; "
                      f"they take their defaults", FilledDefaults, stacklevel=3)
    return model_class(policy.name)(policy.build).to(device)


def load_policy(directory: str | os.PathLike, device: torch.device) -> torch.nn.Module:
    """The model of a policy file (or of a training checkpoint), built from the file's own
    build record, never from the current configuration, with its weights, on `device`.
    Fields the record lacks take their defaults, with a `FilledDefaults` warning naming
    them (`read_policy` lists them too)."""
    policy = read_policy(directory)
    model = build_model(policy, device)
    model.load_state_dict(safetensors.torch.load_file(pathlib.Path(directory) / WEIGHTS,
                                                      device=str(device)))
    return model
