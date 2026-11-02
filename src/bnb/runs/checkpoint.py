"""Policy files and training checkpoints (runs spec 3).

A checkpoint is a directory: `checkpoint.json` (the record) and safetensors files, never
pickle. A policy file holds what an agent needs to play: the model's registered name,
version and build record, its weights, the schema versions, and for a warm start the
return scale's state tensors (`return_scale.safetensors`) and the step counts. A training
checkpoint is a policy file plus everything a resume needs (`training.safetensors` and the
record's training part); `load_policy` reads the policy part of either kind.

Writes are atomic: everything goes into `<name>.tmp/`, every file and then the directory
itself are flushed, the directory is renamed to `<name>/` and the parent flushed, so after a
crash `<name>/` is complete or absent. Loading checks the format, the hashes, the schema
versions and the model's version against the code's, and fails naming both sides of a
difference.
"""

import dataclasses
import json
import os
import pathlib
import shutil
import warnings
from collections.abc import Callable

import safetensors.torch
import torch

from bnb.contracts import Action, GameConfig, Obs
from bnb.game.ship_types import ShipTypes
from bnb.models.yemong.critic import ReturnScale
from bnb.runtime.train import TrainConfig, Trainer, model_build

from . import provenance
from .registry import BUILDS, model_class, name_of
from .serialise import from_json, named_tensors, to_json, tree_from_named

FORMAT = 1
RECORD = "checkpoint.json"
WEIGHTS = "weights.safetensors"
RETURN_SCALE = "return_scale.safetensors"
SCALE_STATE = ("low", "high")  # the ReturnScale tensors a warm start continues from
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
    model's dataclass), the build fields its defaults filled, the return scale's state
    tensors (SCALE_STATE, on the CPU) or None, the step counts, and the raw record."""

    name: str
    version: int
    build: object
    filled: tuple[str, ...]
    return_scale: dict[str, torch.Tensor] | None
    steps: dict
    raw: dict


def _ship_card(types: ShipTypes) -> dict:
    return {field.name: getattr(types, field.name).tolist() for field in dataclasses.fields(types)}


def policy_record(model: torch.nn.Module, return_scale: ReturnScale | None = None,
                  steps: dict | None = None, game: GameConfig | None = None,
                  types: ShipTypes | None = None) -> dict:
    """The record of a policy file for `model` (its file hashes added on writing)."""
    head, dirty = provenance.commit()
    return dict(format=FORMAT, kind="policy",
                model=dict(name=name_of(model), version=type(model).VERSION,
                           build=to_json(model.build)),
                schemas=schemas(), game=None if game is None else to_json(game),
                ship_card=None if types is None else _ship_card(types),
                return_scale=return_scale is not None, steps=dict(steps or {}), commit=head,
                dirty=dirty,
                time=provenance.now())


def weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The model's state as host tensors that own their memory (safetensors refuses shared
    storage)."""
    return {name: tensor.detach().to("cpu", copy=True).contiguous()
            for name, tensor in model.state_dict().items()}


def policy_files(model: torch.nn.Module,
                 return_scale: ReturnScale | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """A policy file's tensor files: the weights and, when given, the return scale's state
    tensors."""
    files = {WEIGHTS: weights(model)}
    if return_scale is not None:
        files[RETURN_SCALE] = {name: getattr(return_scale, name).detach().to("cpu", copy=True)
                               for name in SCALE_STATE}
    return files


def _flush(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_checkpoint(directory: str | os.PathLike, record: dict,
                     files: dict[str, dict[str, torch.Tensor]],
                     stage: Callable[[str], None] | None = None) -> pathlib.Path:
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
    _flush(temporary)
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
    return write_checkpoint(directory, record, policy_files(model, return_scale))


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
    scale = None
    if record["return_scale"]:
        scale = safetensors.torch.load_file(pathlib.Path(directory) / RETURN_SCALE)
    return PolicyRecord(name=model["name"], version=model["version"], build=build,
                        filled=tuple(filled),
                        return_scale=scale,
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


# ---------------------------------------------------------------- training checkpoints

# TrainConfig fields a resume may change (runs spec 6): the budget, the learning rate and
# its warm-up, the loss coefficients and the entropy, the reward weights (coefficients of
# the return), and what changes speed but not shapes (the micro-batch rows, the precision).
# Everything else is structural, or not listed and so fixed: a resume takes it from the
# checkpoint.
OVERRIDABLE = ("updates", "precision", "learner.learning_rate", "learner.warmup_start",
               "learner.warmup_steps", "learner.policy", "learner.entropy", "learner.value",
               "learner.next_state", "learner.enemy_action", "learner.density",
               "learner.shot_trajectory", "learner.shot_outcome", "learner.micro_batch_rows",
               "reward.damage", "reward.kill_death", "reward.objective_progress",
               "reward.objective", "reward.outcome")


def _generators(trainer: Trainer) -> dict[str, torch.Generator]:
    """The named streams a trainer draws from beyond the default generators."""
    return {"shot_spread": trainer.ppo.spread_generator,
            "minibatch_order": trainer.ppo.order_generator}


def default_generator_states(device: torch.device) -> dict[str, torch.Tensor]:
    """The default CPU generator's state and, on a GPU, the device's: the rollout's draws
    and the policy's sampling inside the captured graph come from it."""
    states = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    return states


def save_training(directory: str | os.PathLike, trainer: Trainer, *,
                  stage: Callable[[str], None] | None = None,
                  extra: dict | None = None) -> pathlib.Path:
    """Write a training checkpoint of `trainer` at `directory`: the policy file of its
    model, plus the full TrainConfig, the optimizer, the league slot, every generator, the
    rollout carry and the step counters. `extra` goes into the record as it is (the run's
    settings)."""
    config, ppo = trainer.config, trainer.ppo
    device = trainer.rollout.device
    steps = dict(environment_steps=ppo.env_steps, updates=trainer.updates_done)
    record = policy_record(trainer.model, ppo.return_scale, steps, config.game, ppo.types)
    optimizer = ppo.optimizer.state_dict()
    record.update(kind="training", train_config=to_json(config),
                  optimizer=dict(param_groups=to_json(optimizer["param_groups"])),
                  extra=dict(extra or {}))
    tensors = {}
    tensors.update(named_tensors(optimizer["state"], "optimizer."))
    tensors.update(named_tensors(trainer.league.state_dict(), "league."))
    tensors.update(named_tensors(trainer.rollout.saved_carry(), "carry."))
    for name, generator in _generators(trainer).items():
        tensors[f"generator.{name}"] = generator.get_state()
    for name, state in default_generator_states(device).items():
        tensors[f"default_generator.{name}"] = state
    tensors = {name: tensor.detach().to("cpu", copy=True).contiguous()
               for name, tensor in tensors.items()}
    files = policy_files(trainer.model, ppo.return_scale)
    return write_checkpoint(directory, record, dict(files, **{TRAINING: tensors}), stage)


def _paths(value, prefix: str = "") -> dict:
    """A JSON object's leaves by dotted path."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_paths(item, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: value}


def config_differences(saved, given, allowed: tuple[str, ...] = ()) -> list[str]:
    """The dotted fields where two dataclasses differ, apart from `allowed`."""
    saved, given = _paths(to_json(saved)), _paths(to_json(given))
    return sorted(path for path in saved.keys() | given.keys()
                  if saved.get(path) != given.get(path) and path not in allowed)


def run_build(policy: PolicyRecord, config: TrainConfig, directory, what: str):
    """The build of the run's model, from the file's own record: its architecture, with the
    run's precision, and the numbers the run's game and ship card derive, which must equal
    the record's (so a changed code default never blocks loading an old file). Refuses
    another model; reports fields the record lacked (`FilledDefaults`)."""
    if policy.name != config.model:
        raise CheckpointError(f"{what} refused: {directory} holds {policy.name}, the run "
                              f"trains {config.model}")
    if policy.filled:
        warnings.warn(f"the {policy.name} build record of {directory} lacks "
                      f"{', '.join(policy.filled)}; they take their defaults", FilledDefaults,
                      stacklevel=3)
    build = model_build(config, policy.build.config)
    differ = config_differences(policy.build, build, ("config.bfloat16_matmuls",))
    if differ:
        raise CheckpointError(f"{what} refused: the build record of {directory} differs from "
                              f"the run's model in {', '.join(differ)}. A new architecture "
                              f"is a new run")
    return build


def _load_weights(model: torch.nn.Module, directory: pathlib.Path) -> None:
    device = next(model.parameters()).device
    model.load_state_dict(safetensors.torch.load_file(directory / WEIGHTS, device=str(device)))


def _set_return_scale(trainer: Trainer, scale: dict[str, torch.Tensor] | None) -> None:
    if scale is not None:
        target = trainer.ppo.return_scale
        for name in SCALE_STATE:
            setattr(target, name, scale[name].to(getattr(target, name).device))


def restore_optimizer(trainer: Trainer, stored: dict[str, torch.Tensor],
                      groups: list[dict]) -> None:
    """Load the optimizer state of a training file's tensors and the record's param groups
    into `trainer`, laid out as Adam's for its model's parameters (the step and both moments
    of each), every name, shape and dtype checked."""
    target = {}
    for index, parameter in enumerate(trainer.model.parameters()):
        if f"optimizer.{index}.step" in stored:
            target[index] = dict(step=torch.zeros((), dtype=torch.float32),
                                 exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
    groups = [dict(group, betas=tuple(group["betas"])) for group in groups]
    trainer.ppo.optimizer.load_state_dict(
        dict(state=tree_from_named(target, stored, "optimizer."), param_groups=groups))


def _restore(trainer: Trainer, directory: pathlib.Path, record: dict, policy: PolicyRecord,
             carry: bool) -> None:
    """Load the model, league, optimizer, return scale and counters (and with `carry`, the
    rollout carry and the named streams) of a training checkpoint into `trainer`."""
    stored = safetensors.torch.load_file(directory / TRAINING)
    _load_weights(trainer.model, directory)
    trainer.league.load_state_dict(tree_from_named(trainer.league.state_dict(), stored,
                                                   "league."))
    restore_optimizer(trainer, stored, record["optimizer"]["param_groups"])
    _set_return_scale(trainer, policy.return_scale)
    trainer.ppo.env_steps = policy.steps["environment_steps"]
    trainer.updates_done = policy.steps["updates"]
    if carry:
        trainer.rollout.load_carry(tree_from_named(trainer.rollout.saved_carry(), stored,
                                                   "carry."))
        for name, generator in _generators(trainer).items():
            generator.set_state(stored[f"generator.{name}"])


def _check_card(trainer: Trainer, record: dict, directory) -> None:
    card = _ship_card(trainer.ppo.types)
    if record["ship_card"] != card:
        differ = sorted(name for name in card if record["ship_card"].get(name) != card[name])
        raise CheckpointError(f"{directory} trained on another ship card ({', '.join(differ)} "
                              f"differ); the card cannot change on a resume: fork with --from")


def restore_default_generators(directory: str | os.PathLike, device: torch.device) -> None:
    """Set the process's default CPU (and on a GPU, the device's) generator to a training
    checkpoint's. The one process-wide effect of a resume, so the entry point calls it."""
    stored = safetensors.torch.load_file(pathlib.Path(directory) / TRAINING)
    torch.set_rng_state(stored["default_generator.cpu"])
    if device.type == "cuda":
        if "default_generator.cuda" not in stored:
            raise CheckpointError(f"{directory} was saved on the CPU: it holds no CUDA "
                                  f"generator to resume on {device}")
        torch.cuda.set_rng_state(stored["default_generator.cuda"], device)


def training_config(directory: str | os.PathLike) -> tuple[TrainConfig, list[str]]:
    """(the TrainConfig a training checkpoint ran with, the fields its defaults filled)."""
    record = read_record(directory)
    if record.get("kind") != "training":
        raise CheckpointError(f"{directory} is a {record.get('kind')} file, not a training "
                              f"checkpoint")
    config, filled = from_json(TrainConfig, record["train_config"])
    return config, filled


def load_training(directory: str | os.PathLike, device: torch.device,
                  config: TrainConfig | None = None) -> Trainer:
    """The trainer of a training checkpoint, ready to continue exactly where it stopped:
    built from the checkpoint's own TrainConfig, or `config`, which may differ from it only
    in OVERRIDABLE fields, and its model from the checkpoint's build record. The ship card
    must equal the code's. The default generators are the caller's
    (`restore_default_generators`)."""
    directory = pathlib.Path(directory)
    saved, filled = training_config(directory)
    if filled:
        warnings.warn(f"the TrainConfig of {directory} lacks {', '.join(filled)}; they take "
                      f"their defaults", FilledDefaults, stacklevel=2)
    config = saved if config is None else config
    structural = config_differences(saved, config, OVERRIDABLE)
    if structural:
        raise CheckpointError(f"a resume cannot change {', '.join(structural)}; start a new "
                              f"run from this checkpoint with --from")
    record, policy = read_record(directory), read_policy(directory)
    trainer = Trainer(config, device, run_build(policy, config, directory, "the resume"))
    _check_card(trainer, record, directory)
    _restore(trainer, directory, record, policy, carry=True)
    return trainer


def fork_training(directory: str | os.PathLike, config: TrainConfig,
                  device: torch.device) -> Trainer:
    """A new trainer with `config` that starts from a training checkpoint's model (built
    from its build record), league slot, optimizer, return scale and environment-step count
    (so the warm-up goes on), not its carry or generators (the new run's sizes may differ).
    Its update count starts at zero: `config.updates` is what the fork trains."""
    directory = pathlib.Path(directory)
    record, policy = read_record(directory), read_policy(directory)
    if record.get("kind") != "training":
        raise CheckpointError(f"{directory} is a {record.get('kind')} file; a fork needs a "
                              f"training checkpoint (a policy file is a warm start)")
    trainer = Trainer(config, device, run_build(policy, config, directory, "the fork"))
    _restore(trainer, directory, record, policy, carry=False)
    trainer.updates_done = 0
    return trainer


def warm_start(directory: str | os.PathLike, config: TrainConfig,
               device: torch.device) -> Trainer:
    """A new trainer with `config` whose model (and league slot) start from a policy file's
    build record, weights and return scale, with a fresh optimizer and fresh step counts:
    the BC -> RL hand-off."""
    directory = pathlib.Path(directory)
    policy = read_policy(directory)
    trainer = Trainer(config, device, run_build(policy, config, directory, "the warm start"))
    _load_weights(trainer.model, directory)
    trainer.league.load_state_dict(trainer.model.state_dict())
    _set_return_scale(trainer, policy.return_scale)
    return trainer
