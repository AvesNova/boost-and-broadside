"""Policy files and training checkpoints (runs spec 3).

A checkpoint is a directory: `checkpoint.json` (the record) and safetensors files, never
pickle. A policy file holds what an agent needs to play: the model's registered name,
version and build record, its weights, the schema versions, the observation's column names
it was trained with (`observation`), and for a warm start the
return scale's state tensors (`return_scale.safetensors`) and the step counts. A training
checkpoint is a policy file plus everything a resume needs (`training.safetensors` and the
record's training part); `load_policy` reads the policy part of either kind.

Writes are atomic: everything goes into `<name>.tmp/`, every file and then the directory
itself are flushed, the directory is renamed to `<name>/` and the parent flushed, so after a
crash `<name>/` is complete or absent. Loading checks the format, the hashes, the schema
versions and the model's version against the code's, and fails naming both sides of a
difference; and that every observation column the policy recorded exists in the code's
contract under the same name (a column's meaning never changes under its name), failing
naming each missing one. A policy file from before the record has the columns of
`COLUMNS_BEFORE_RECORDS`.
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
from bnb.contracts.obs import COLUMNS, COLUMNS_BEFORE_RECORDS
from bnb.game import VERSION as GAME_VERSION
from bnb.game.ship_types import ShipTypes
from bnb.learn.bc import StopRule
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
TEACHER_CARRY = "teacher_carry."  # BC's teacher's memory, in the training file


class CheckpointError(ValueError):
    """A checkpoint that cannot be loaded as it is, with the reason."""


class FilledDefaults(UserWarning):
    """A loaded record lacked fields, which took their defaults (runs spec 4)."""


def schemas() -> dict[str, int]:
    """The code's schema versions of the contracts a policy depends on."""
    return {"Obs": Obs.SCHEMA_VERSION, "Action": Action.SCHEMA_VERSION}


def observation_columns(model: torch.nn.Module) -> dict[str, list[str]]:
    """The observation's column names `model` was trained with, by part (`COLUMNS`'s parts):
    its own (`model.observation`, a model that reads rows by width and was built for a
    recorded layout) or the code's contract."""
    columns = getattr(model, "observation", None) or COLUMNS
    return {part: list(names) for part, names in columns.items()}


def recorded_columns(record: dict, directory) -> dict[str, tuple[str, ...]]:
    """A policy record's observation columns by part (`COLUMNS_BEFORE_RECORDS` for a file
    from before the record), each checked to exist in the code's contract under its name."""
    recorded = record.get("observation") or COLUMNS_BEFORE_RECORDS
    columns = {part: tuple(names) for part, names in recorded.items()}
    missing = [
        f"{part}.{name}"
        for part, names in columns.items()
        for name in names
        if name not in COLUMNS.get(part, ())
    ]
    if missing:
        raise CheckpointError(
            f"{directory} was trained on observation columns the code does not have: "
            f"{', '.join(missing)}. Load it from a worktree at a commit that has them"
        )
    return columns


@dataclasses.dataclass(frozen=True)
class PolicyRecord:
    """A policy file's record, checked: the model's name and version, its build record (the
    model's dataclass), the build fields its defaults filled, the return scale's state
    tensors (SCALE_STATE, on the CPU) or None, the step counts, the policy's rating per mode
    when it was rated as it trained (a warm start's live rating starts there; empty when
    not), the observation columns it was trained with by part (`recorded_columns`), and the
    raw record."""

    name: str
    version: int
    build: object
    filled: tuple[str, ...]
    return_scale: dict[str, torch.Tensor] | None
    steps: dict
    ratings: dict[str, float]
    observation: dict[str, tuple[str, ...]]
    raw: dict


def _ship_card(types: ShipTypes) -> dict:
    return {field.name: getattr(types, field.name).tolist() for field in dataclasses.fields(types)}


def policy_record(
    model: torch.nn.Module,
    return_scale: ReturnScale | None = None,
    steps: dict | None = None,
    game: GameConfig | None = None,
    types: ShipTypes | None = None,
    ratings: dict[str, float] | None = None,
) -> dict:
    """The record of a policy file for `model` (its file hashes added on writing)."""
    head, dirty = provenance.commit()
    return dict(
        format=FORMAT,
        kind="policy",
        model=dict(name=name_of(model), version=type(model).VERSION, build=to_json(model.build)),
        schemas=schemas(),
        observation=observation_columns(model),
        game_version=GAME_VERSION,
        game=None if game is None else to_json(game),
        ship_card=None if types is None else _ship_card(types),
        return_scale=return_scale is not None,
        steps=dict(steps or {}),
        ratings=dict(ratings or {}),
        commit=head,
        dirty=dirty,
        time=provenance.now(),
    )


def weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The model's state as host tensors that own their memory (safetensors refuses shared
    storage)."""
    return {
        name: tensor.detach().to("cpu", copy=True).contiguous()
        for name, tensor in model.state_dict().items()
    }


def policy_files(
    model: torch.nn.Module, return_scale: ReturnScale | None = None
) -> dict[str, dict[str, torch.Tensor]]:
    """A policy file's tensor files: the weights and, when given, the return scale's state
    tensors."""
    files = {WEIGHTS: weights(model)}
    if return_scale is not None:
        files[RETURN_SCALE] = {
            name: getattr(return_scale, name).detach().to("cpu", copy=True) for name in SCALE_STATE
        }
    return files


def _flush(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_checkpoint(
    directory: str | os.PathLike,
    record: dict,
    files: dict[str, dict[str, torch.Tensor]],
    stage: Callable[[str], None] | None = None,
) -> pathlib.Path:
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


def save_policy(
    directory: str | os.PathLike,
    model: torch.nn.Module,
    *,
    return_scale: ReturnScale | None = None,
    steps: dict | None = None,
    game: GameConfig | None = None,
    types: ShipTypes | None = None,
    ratings: dict[str, float] | None = None,
) -> pathlib.Path:
    """Write `model` as a policy file at `directory`: its registered name, version and build
    record, its weights, the schema versions, and for a warm start the return scale, the
    step counts (`steps`, such as {"environment_steps": n}) and its rating per mode as it
    trained (`ratings`). game and types (the ship card) are recorded for information
    (ratings are keyed by them)."""
    record = policy_record(model, return_scale, steps, game, types, ratings)
    return write_checkpoint(directory, record, policy_files(model, return_scale))


def read_record(directory: str | os.PathLike) -> dict:
    """A checkpoint's record, after checking its format and every file's hash."""
    directory = pathlib.Path(directory)
    path = directory / RECORD
    if not path.is_file():
        raise CheckpointError(f"{directory} is not a checkpoint: it has no {RECORD}")
    record = json.loads(path.read_text())
    if record.get("format") != FORMAT:
        raise CheckpointError(
            f"{directory} is checkpoint format {record.get('format')}; "
            f"this code reads format {FORMAT}"
        )
    for name, expected in record["files"].items():
        if not (directory / name).is_file():
            raise CheckpointError(f"{directory / name} is missing")
        actual = provenance.file_hash(directory / name)
        if actual != expected:
            raise CheckpointError(
                f"{directory / name} has SHA-256 {actual}, but the record "
                f"says {expected}: the file changed after it was written"
            )
    return record


def read_policy(directory: str | os.PathLike) -> PolicyRecord:
    """The checked policy part of a checkpoint of either kind: the schema versions and the
    model's version must equal the code's, the build record must read, and every recorded
    observation column must exist (`recorded_columns`)."""
    record = read_record(directory)
    if record["schemas"] != schemas():
        raise CheckpointError(
            f"{directory} was trained on schema versions "
            f"{record['schemas']}; the code has {schemas()}"
        )
    model = record["model"]
    cls = model_class(model["name"])
    if model["version"] != cls.VERSION:
        raise CheckpointError(
            f"{directory} holds {model['name']} version {model['version']}; the code has "
            f"version {cls.VERSION}. Load it from a worktree at a commit with version "
            f"{model['version']}"
        )
    build, filled = from_json(BUILDS[cls], model["build"])
    observation = recorded_columns(record, directory)
    scale = None
    if record["return_scale"]:
        scale = safetensors.torch.load_file(pathlib.Path(directory) / RETURN_SCALE)
    return PolicyRecord(
        name=model["name"],
        version=model["version"],
        build=build,
        filled=tuple(filled),
        return_scale=scale,
        steps=record["steps"],
        ratings=record.get("ratings", {}),
        observation=observation,
        raw=record,
    )


def build_model(policy: PolicyRecord, device: torch.device) -> torch.nn.Module:
    """A fresh model from a policy record's build (its weights not loaded). A model that
    reads rows by width (`reads_by_width`) is built for the recorded columns, which it
    selects by name from the code's observation; one that reads columns by name needs
    nothing more."""
    if policy.filled:
        warnings.warn(
            f"the {policy.name} build record lacks {', '.join(policy.filled)}; "
            f"they take their defaults",
            FilledDefaults,
            stacklevel=3,
        )
    cls = model_class(policy.name)
    if getattr(cls, "reads_by_width", False):
        return cls(policy.build, policy.observation).to(device)
    return cls(policy.build).to(device)


def load_policy(directory: str | os.PathLike, device: torch.device) -> torch.nn.Module:
    """The model of a policy file (or of a training checkpoint), built from the file's own
    build record, never from the current configuration, with its weights, on `device`.
    Fields the record lacks take their defaults, with a `FilledDefaults` warning naming
    them (`read_policy` lists them too)."""
    policy = read_policy(directory)
    model = build_model(policy, device)
    model.load_state_dict(
        safetensors.torch.load_file(pathlib.Path(directory) / WEIGHTS, device=str(device))
    )
    return model


# ---------------------------------------------------------------- training checkpoints

# TrainConfig fields a resume may change (runs spec 6): the budget (PPO's updates, BC's
# environment steps), the learning rate and its warm-up, the loss coefficients and the
# entropy, the reward weights (coefficients of the return), what changes speed but not
# shapes (the micro-batch rows, the precision), and the policy snapshots' cadence.
# Everything else is structural, or not listed and so fixed: a resume takes it from the
# checkpoint.
OVERRIDABLE = (
    "updates",
    "stop.budget_steps",
    "precision",
    "learner.learning_rate",
    "learner.warmup_start",
    "learner.warmup_steps",
    "learner.policy",
    "learner.entropy_power",
    "learner.entropy_stick",
    "learner.entropy_shoot",
    "learner.value",
    "learner.next_state",
    "learner.correction",
    "learner.enemy_action",
    "learner.density",
    "learner.shot_trajectory",
    "learner.shot_outcome",
    "learner.rotary_readout",
    "learner.micro_batch_rows",
    "graph_ticks",
    "policy_snapshot_every",
    "reward.damage",
    "reward.kill_death",
    "reward.objective_progress",
    "reward.objective",
    "reward.outcome",
)


def _generators(trainer: Trainer) -> dict[str, torch.Generator]:
    """The named streams a trainer draws from beyond the default generators."""
    return {
        "shot_spread": trainer.ppo.spread_generator,
        "minibatch_order": trainer.ppo.order_generator,
    }


def default_generator_states(device: torch.device) -> dict[str, torch.Tensor]:
    """The default CPU generator's state and, on a GPU, the device's: the rollout's draws
    and the policy's sampling inside the captured graph come from it."""
    states = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    return states


def _training_tensors(trainer: Trainer, optimizer_state: dict) -> dict[str, torch.Tensor]:
    """The training file's tensors, as host copies: the optimizer's state, the opponent
    slots, the league's tensors, the rollout carry, BC's teacher's carry (once it has one)
    and every generator."""
    tensors = {}
    tensors.update(named_tensors(optimizer_state, "optimizer."))
    if trainer.opponents is not None:
        tensors.update(named_tensors(trainer.opponents.state_dict(), "opponents."))
    if trainer.league is not None:
        tensors.update(named_tensors(trainer.league.state()[1], "league."))
    tensors.update(named_tensors(trainer.rollout.saved_carry(), "carry."))
    teacher_carry = getattr(trainer.ppo, "teacher_carry", None)  # BC's teacher's memory
    if teacher_carry is not None:
        tensors.update(named_tensors(teacher_carry, TEACHER_CARRY))
    for name, generator in _generators(trainer).items():
        tensors[f"generator.{name}"] = generator.get_state()
    for name, state in default_generator_states(trainer.rollout.device).items():
        tensors[f"default_generator.{name}"] = state
    return {
        name: tensor.detach().to("cpu", copy=True).contiguous() for name, tensor in tensors.items()
    }


def training_state(trainer: Trainer) -> tuple[dict, dict[str, torch.Tensor]]:
    """Everything a training checkpoint holds, read from the live trainer without writing:
    (the record's state: counters, optimizer groups, league record; every tensor by name,
    the policy files' and the training file's). A resumed trainer's equals the saving
    one's, exactly, before any compute. It lists what the trainer has, as the checkpoint
    does, not what a file happens to hold, so a field left out of the file shows."""
    ppo = trainer.ppo
    optimizer = ppo.optimizer.state_dict()
    record = dict(
        environment_steps=ppo.environment_steps,
        updates=trainer.updates_done,
        optimizer=to_json(optimizer["param_groups"]),
        league=None if trainer.league is None else trainer.league.state()[0],
        triggers=None if trainer.triggers is None else trainer.triggers.state,
        stop_rule=None if trainer.stop_rule is None else trainer.stop_rule.state,
    )
    tensors = _training_tensors(trainer, optimizer["state"])
    for file, files in policy_files(trainer.model, ppo.return_scale).items():
        tensors.update({f"{file}:{name}": tensor for name, tensor in files.items()})
    return record, tensors


def save_training(
    directory: str | os.PathLike,
    trainer: Trainer,
    *,
    stage: Callable[[str], None] | None = None,
    extra: dict | None = None,
) -> pathlib.Path:
    """Write a training checkpoint of `trainer` at `directory`: the policy file of its
    model, plus the full TrainConfig, the optimizer, the opponent slots, the league's state
    (when one is attached: its record part under "league", its tensors in
    `training.safetensors` under "league."), every generator, the rollout carry and the step
    counters, the trigger rules' state when the trainer has them (`trainer.triggers`) and
    BC's stop rule's (`trainer.stop_rule`). `extra` goes into the record as it is (the run's
    settings)."""
    config, ppo = trainer.config, trainer.ppo
    steps = dict(environment_steps=ppo.environment_steps, updates=trainer.updates_done)
    record = policy_record(trainer.model, ppo.return_scale, steps, config.game, ppo.types)
    optimizer = ppo.optimizer.state_dict()
    record.update(
        kind="training",
        train_config=to_json(config),
        optimizer=dict(param_groups=to_json(optimizer["param_groups"])),
        extra=dict(extra or {}),
    )
    tensors = _training_tensors(trainer, optimizer["state"])
    if trainer.league is not None:
        record["league"] = trainer.league.state()[0]
    if trainer.triggers is not None:
        record["triggers"] = trainer.triggers.state
    if trainer.stop_rule is not None:
        record["stop_rule"] = trainer.stop_rule.state
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
    return sorted(
        path
        for path in saved.keys() | given.keys()
        if saved.get(path) != given.get(path) and path not in allowed
    )


def check_build(policy: PolicyRecord, config: TrainConfig, directory, what: str) -> None:
    """Refuse a run whose model (`model_build(config)`: its settings, and the numbers the
    run's game and ship card derive) differs from the file's build record in anything but
    the precision, or is another model; report fields the record lacked
    (`FilledDefaults`)."""
    if policy.name != config.model:
        raise CheckpointError(
            f"{what} refused: {directory} holds {policy.name}, the run trains {config.model}"
        )
    if policy.filled:
        warnings.warn(
            f"the {policy.name} build record of {directory} lacks "
            f"{', '.join(policy.filled)}; they take their defaults",
            FilledDefaults,
            stacklevel=3,
        )
    differ = config_differences(policy.build, model_build(config), ("config.bfloat16_matmuls",))
    if differ:
        raise CheckpointError(
            f"{what} refused: the build record of {directory} differs from "
            f"the run's model in {', '.join(differ)}. A new architecture "
            f"is a new run"
        )


def with_model_of(config: TrainConfig, directory: str | os.PathLike) -> TrainConfig:
    """`config` with the model settings of the policy file (or training checkpoint) at
    `directory` (its build record's, the precision aside): how a fork or a warm start whose
    command sets none of them takes the file's model, even after a code default changed."""
    policy = read_policy(directory)
    if policy.name != config.model:
        return config  # check_build refuses it, naming both models
    settings = dataclasses.replace(policy.build.config, bfloat16_matmuls=False)
    return dataclasses.replace(config, **{config.model: settings})


def _load_weights(model: torch.nn.Module, directory: pathlib.Path) -> None:
    device = next(model.parameters()).device
    model.load_state_dict(safetensors.torch.load_file(directory / WEIGHTS, device=str(device)))


def _set_return_scale(trainer: Trainer, scale: dict[str, torch.Tensor] | None) -> None:
    if scale is not None:
        target = trainer.ppo.return_scale
        for name in SCALE_STATE:
            setattr(target, name, scale[name].to(getattr(target, name).device))


def restore_optimizer(
    trainer: Trainer, stored: dict[str, torch.Tensor], groups: list[dict]
) -> None:
    """Load the optimizer state of a training file's tensors and the record's param groups
    into `trainer`, laid out as Adam's for its model's parameters (the step and both moments
    of each), every name, shape and dtype checked."""
    target = {}
    for index, parameter in enumerate(trainer.model.parameters()):
        if f"optimizer.{index}.step" in stored:
            target[index] = dict(
                step=torch.zeros((), dtype=torch.float32),
                exp_avg=torch.zeros_like(parameter),
                exp_avg_sq=torch.zeros_like(parameter),
            )
    groups = [dict(group, betas=tuple(group["betas"])) for group in groups]
    trainer.ppo.optimizer.load_state_dict(
        dict(state=tree_from_named(target, stored, "optimizer."), param_groups=groups)
    )


def _restore(
    trainer: Trainer, directory: pathlib.Path, record: dict, policy: PolicyRecord, carry: bool
) -> None:
    """Load the model, optimizer, return scale and counters (and with `carry`, the opponent
    slots, the rollout carry, the named streams and BC's stop rule) of a training checkpoint
    into `trainer`."""
    stored = safetensors.torch.load_file(directory / TRAINING)
    _load_weights(trainer.model, directory)
    restore_optimizer(trainer, stored, record["optimizer"]["param_groups"])
    _set_return_scale(trainer, policy.return_scale)
    trainer.ppo.environment_steps = policy.steps["environment_steps"]
    trainer.updates_done = policy.steps["updates"]
    if carry:
        if trainer.opponents is not None:
            trainer.opponents.load_state_dict(
                tree_from_named(trainer.opponents.state_dict(), stored, "opponents.")
            )
        trainer.rollout.load_carry(tree_from_named(trainer.rollout.saved_carry(), stored, "carry."))
        for name, generator in _generators(trainer).items():
            generator.set_state(stored[f"generator.{name}"])
        if trainer.stop_rule is not None:
            trainer.stop_rule = StopRule(trainer.config.stop, record["stop_rule"])
        _restore_teacher_carry(trainer, stored)


def _restore_teacher_carry(trainer: Trainer, stored: dict[str, torch.Tensor]) -> None:
    """BC's teacher's carry, when the checkpoint holds one (its rows from the stored
    target's)."""
    target = stored.get(f"{TEACHER_CARRY}target")
    if target is None:
        return
    teacher = trainer.ppo.teacher
    template = teacher.init_carry(target.shape[0], trainer.rollout.device)
    trainer.ppo.teacher_carry = tree_from_named(template, stored, TEACHER_CARRY)


def _check_card(trainer: Trainer, record: dict, directory) -> None:
    card = _ship_card(trainer.ppo.types)
    if record["ship_card"] != card:
        differ = sorted(name for name in card if record["ship_card"].get(name) != card[name])
        raise CheckpointError(
            f"{directory} trained on another ship card ({', '.join(differ)} "
            f"differ); the card cannot change on a resume: fork with --from"
        )


def restore_default_generators(directory: str | os.PathLike, device: torch.device) -> None:
    """Set the process's default CPU (and on a GPU, the device's) generator to a training
    checkpoint's. The one process-wide effect of a resume, so the entry point calls it."""
    stored = safetensors.torch.load_file(pathlib.Path(directory) / TRAINING)
    torch.set_rng_state(stored["default_generator.cpu"])
    if device.type == "cuda":
        if "default_generator.cuda" not in stored:
            raise CheckpointError(
                f"{directory} was saved on the CPU: it holds no CUDA "
                f"generator to resume on {device}"
            )
        torch.cuda.set_rng_state(stored["default_generator.cuda"], device)


def training_config(directory: str | os.PathLike) -> tuple[TrainConfig, list[str]]:
    """(the TrainConfig a training checkpoint ran with, the fields its defaults filled)."""
    record = read_record(directory)
    if record.get("kind") != "training":
        raise CheckpointError(
            f"{directory} is a {record.get('kind')} file, not a training checkpoint"
        )
    config, filled = from_json(TrainConfig, record["train_config"])
    return config, filled


def load_training(
    directory: str | os.PathLike, device: torch.device, config: TrainConfig | None = None
) -> Trainer:
    """The trainer of a training checkpoint, ready to continue exactly where it stopped:
    built from the checkpoint's own TrainConfig (its model's settings and ship card
    included), or `config`, which may differ from it only in OVERRIDABLE fields; its model
    must match the checkpoint's build record. The default generators are the caller's
    (`restore_default_generators`)."""
    directory = pathlib.Path(directory)
    saved, filled = training_config(directory)
    if filled:
        warnings.warn(
            f"the TrainConfig of {directory} lacks {', '.join(filled)}; they take their defaults",
            FilledDefaults,
            stacklevel=2,
        )
    config = saved if config is None else config
    structural = config_differences(saved, config, OVERRIDABLE)
    if structural:
        raise CheckpointError(
            f"a resume cannot change {', '.join(structural)}; start a new "
            f"run from this checkpoint with --from"
        )
    record, policy = read_record(directory), read_policy(directory)
    check_build(policy, config, directory, "the resume")
    trainer = Trainer(config, device)
    _check_card(trainer, record, directory)
    _restore(trainer, directory, record, policy, carry=True)
    return trainer


def fork_training(
    directory: str | os.PathLike, config: TrainConfig, device: torch.device
) -> Trainer:
    """A new trainer with `config` that starts from a training checkpoint's model (whose
    build record must match `config`'s model; `with_model_of` takes its settings),
    optimizer, return scale and environment-step count (so the warm-up goes on), not its
    carry, generators or league (the new run's sizes may differ; its league starts afresh).
    Its update count starts at zero: `config.updates` is what the fork trains."""
    directory = pathlib.Path(directory)
    record, policy = read_record(directory), read_policy(directory)
    if record.get("kind") != "training":
        raise CheckpointError(
            f"{directory} is a {record.get('kind')} file; a fork needs a "
            f"training checkpoint (a policy file is a warm start)"
        )
    check_build(policy, config, directory, "the fork")
    trainer = Trainer(config, device)
    _restore(trainer, directory, record, policy, carry=False)
    trainer.reset_opponents()
    trainer.updates_done = 0
    return trainer


def warm_start(directory: str | os.PathLike, config: TrainConfig, device: torch.device) -> Trainer:
    """A new trainer with `config` whose model (and opponent slots) start from a policy
    file's weights and return scale (its build record must match `config`'s model;
    `with_model_of` takes its settings), with a fresh optimizer and fresh step
    counts: the BC -> RL hand-off."""
    directory = pathlib.Path(directory)
    policy = read_policy(directory)
    check_build(policy, config, directory, "the warm start")
    trainer = Trainer(config, device)
    _load_weights(trainer.model, directory)
    trainer.reset_opponents()
    _set_return_scale(trainer, policy.return_scale)
    return trainer


def triggers_state(directory: str | os.PathLike) -> dict | None:
    """A training checkpoint's trigger-rule state (`bnb.runs.triggers.Triggers.state`); None
    when the run kept none."""
    return read_record(pathlib.Path(directory)).get("triggers")


def league_state(directory: str | os.PathLike) -> tuple[dict, dict[str, torch.Tensor]] | None:
    """A training checkpoint's league state, (its record part, its tensors), for the league
    to restore; None when the run had no league attached."""
    directory = pathlib.Path(directory)
    record = read_record(directory)
    if "league" not in record:
        return None
    stored = safetensors.torch.load_file(directory / TRAINING)
    prefix = "league."
    return record["league"], {
        name[len(prefix) :]: tensor for name, tensor in stored.items() if name.startswith(prefix)
    }
