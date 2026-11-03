"""Training checkpoints (runs spec 3, acceptance 1, 4, 7, 9): every part round-trips,
writes are atomic, and fork and warm start load only what fits."""

import dataclasses
import json

import pytest
import runs_cases as cases
import safetensors.torch
import torch

from bnb.models.relational import RelationalConfig
from bnb.runs import checkpoint
from bnb.runs.checkpoint import CheckpointError, FilledDefaults
from bnb.runs.serialise import SerialiseError
from bnb.runtime.train import Trainer, model_build

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny trainer after two updates and its training checkpoint."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        trainer = Trainer(cases.tiny_train_config(), cases.CPU)
        for _ in range(2):
            trainer.update()
    directory = tmp_path_factory.mktemp("trained") / "step"
    checkpoint.save_training(directory, trainer)
    return trainer, directory


def _assert_same_optimizer(got: Trainer, want: Trainer) -> None:
    want, got = want.ppo.optimizer.state_dict(), got.ppo.optimizer.state_dict()
    assert want["param_groups"] == got["param_groups"]
    assert want["state"].keys() == got["state"].keys()
    for index, state in want["state"].items():
        assert state.keys() == got["state"][index].keys()
        for key, tensor in state.items():
            assert torch.equal(got["state"][index][key], tensor), (index, key)


def test_every_part_of_a_training_checkpoint_round_trips(trained):
    trainer, directory = trained
    with torch.random.fork_rng(devices=[]):
        loaded = checkpoint.load_training(directory, cases.CPU)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], tensor), name
    for name, tensor in trainer.opponents.state_dict().items():
        assert torch.equal(loaded.opponents.state_dict()[name], tensor), name
    _assert_same_optimizer(loaded, trainer)
    assert torch.equal(loaded.ppo.return_scale.low, trainer.ppo.return_scale.low)
    assert torch.equal(loaded.ppo.return_scale.high, trainer.ppo.return_scale.high)
    assert torch.equal(
        loaded.ppo.spread_generator.get_state(), trainer.ppo.spread_generator.get_state()
    )
    assert torch.equal(
        loaded.ppo.order_generator.get_state(), trainer.ppo.order_generator.get_state()
    )
    want_carry = checkpoint.named_tensors(trainer.rollout.saved_carry())
    got_carry = checkpoint.named_tensors(loaded.rollout.saved_carry())
    assert want_carry.keys() == got_carry.keys()
    for name, tensor in want_carry.items():
        assert torch.equal(got_carry[name], tensor), name
    assert (loaded.updates_done, loaded.ppo.environment_steps) == (2, trainer.ppo.environment_steps)


def test_the_default_generators_are_restored_by_the_entry_points_call(trained):
    _, directory = trained
    stored = safetensors.torch.load_file(directory / checkpoint.TRAINING)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        checkpoint.restore_default_generators(directory, cases.CPU)
        assert torch.equal(torch.get_rng_state(), stored["default_generator.cpu"])


def test_a_training_checkpoint_is_also_a_policy_file(trained):
    trainer, directory = trained
    model = checkpoint.load_policy(directory, cases.CPU)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor)


def test_a_structural_change_on_resume_is_refused(trained):
    _, directory = trained
    saved, _ = checkpoint.training_config(directory)
    with pytest.raises(
        CheckpointError,
        match="a resume cannot change environments; start a new run from this checkpoint with "
        "--from",
    ):
        checkpoint.load_training(directory, cases.CPU, dataclasses.replace(saved, environments=32))


def test_a_run_time_override_on_resume_is_allowed(trained):
    _, directory = trained
    saved, _ = checkpoint.training_config(directory)
    learner = dataclasses.replace(saved.learner, learning_rate=1e-4, entropy=-0.01)
    with torch.random.fork_rng(devices=[]):
        trainer = checkpoint.load_training(
            directory, cases.CPU, dataclasses.replace(saved, updates=9, learner=learner)
        )
    assert trainer.config.updates == 9 and trainer.ppo.config.learning_rate == 1e-4


def test_the_reward_weights_are_saved_and_a_resume_may_change_them(trained):
    _, directory = trained
    saved, _ = checkpoint.training_config(directory)
    assert saved.reward == cases.tiny_train_config().reward
    reward = dataclasses.replace(saved.reward, damage=0.5, outcome=2.0)
    with torch.random.fork_rng(devices=[]):
        trainer = checkpoint.load_training(
            directory, cases.CPU, dataclasses.replace(saved, reward=reward)
        )
    assert trainer.rollout.reward_config.damage == 0.5 and trainer.ppo.outcome_weight == 2.0
    with pytest.raises(CheckpointError, match="a resume cannot change reward.damage_unit"):
        checkpoint.load_training(
            directory,
            cases.CPU,
            dataclasses.replace(saved, reward=dataclasses.replace(saved.reward, damage_unit=50.0)),
        )


def test_the_micro_batch_rows_and_the_precision_may_change_on_resume(trained):
    _, directory = trained
    saved, _ = checkpoint.training_config(directory)
    changed = dataclasses.replace(
        saved, precision="bfloat16", learner=dataclasses.replace(saved.learner, micro_batch_rows=32)
    )
    with torch.random.fork_rng(devices=[]):
        trainer = checkpoint.load_training(directory, cases.CPU, changed)
    assert trainer.model.build.config.bfloat16_matmuls
    assert trainer.ppo.config.micro_batch_rows == 32


def test_a_resume_builds_the_model_from_the_checkpoint_not_the_code_defaults(tmp_path):
    """A model saved with another architecture than the code's default (as after a default
    changed) resumes with its own."""
    config = cases.tiny_train_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        build = model_build(config, RelationalConfig(value_bins=41))
        trainer = Trainer(config, cases.CPU, build)
        trainer.update()
    checkpoint.save_training(tmp_path / "step", trainer)
    with torch.random.fork_rng(devices=[]):
        loaded = checkpoint.load_training(tmp_path / "step", cases.CPU)
    assert loaded.model.build == build and loaded.model.build != model_build(config)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], tensor), name


def test_a_build_field_the_checkpoint_lacks_is_reported_on_resume_and_fork(tmp_path, trained):
    trainer, _ = trained
    checkpoint.save_training(tmp_path / "step", trainer)
    path = tmp_path / "step" / checkpoint.RECORD
    record = json.loads(path.read_text())
    record["model"]["build"]["config"].pop("value_limit")
    path.write_text(json.dumps(record))
    with torch.random.fork_rng(devices=[]):
        with pytest.warns(FilledDefaults, match="lacks config.value_limit"):
            checkpoint.load_training(tmp_path / "step", cases.CPU)
        with pytest.warns(FilledDefaults, match="lacks config.value_limit"):
            checkpoint.fork_training(tmp_path / "step", cases.tiny_train_config(), cases.CPU)


def test_a_fork_whose_build_differs_is_refused(trained):
    _, directory = trained
    with (
        torch.random.fork_rng(devices=[]),
        pytest.raises(
            CheckpointError,
            match="the fork refused: the build record .* differs from the "
            "run's model in config.hidden. A new architecture is a "
            "new run",
        ),
    ):
        checkpoint.fork_training(
            directory, cases.tiny_train_config(relational_hidden=32), cases.CPU
        )


def test_a_fork_takes_the_model_optimizer_and_return_scale_but_not_the_carry(trained):
    trainer, directory = trained
    with torch.random.fork_rng(devices=[]):
        fork = checkpoint.fork_training(
            directory, cases.tiny_train_config(environments=32), cases.CPU
        )
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(fork.model.state_dict()[name], tensor)
    assert torch.equal(fork.ppo.return_scale.high, trainer.ppo.return_scale.high)
    _assert_same_optimizer(fork, trainer)
    assert fork.rollout.environments == 32
    assert (fork.ppo.environment_steps, fork.updates_done) == (trainer.ppo.environment_steps, 0)


def test_an_optimizer_state_of_another_shape_is_refused(tmp_path, trained):
    _, directory = trained
    with torch.random.fork_rng(devices=[]):
        parent = Trainer(cases.tiny_train_config(), cases.CPU)
    stored = safetensors.torch.load_file(directory / checkpoint.TRAINING)
    stored["optimizer.0.exp_avg"] = stored["optimizer.0.exp_avg"][:1].clone()
    safetensors.torch.save_file(stored, tmp_path / checkpoint.TRAINING)
    with pytest.raises(SerialiseError, match="optimizer.0.exp_avg: stored"):
        checkpoint.restore_optimizer(
            parent,
            safetensors.torch.load_file(tmp_path / checkpoint.TRAINING),
            parent.ppo.optimizer.state_dict()["param_groups"],
        )


def test_a_warm_start_takes_weights_and_return_scale_with_a_fresh_optimizer(tmp_path, trained):
    """Acceptance 9: the BC -> RL hand-off, from a policy file."""
    trainer, _ = trained
    checkpoint.save_policy(
        tmp_path / "bc",
        trainer.model,
        return_scale=trainer.ppo.return_scale,
        steps=dict(environment_steps=trainer.ppo.environment_steps),
    )
    with torch.random.fork_rng(devices=[]):
        rl = checkpoint.warm_start(tmp_path / "bc", cases.tiny_train_config(), cases.CPU)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(rl.model.state_dict()[name], tensor)
        for set_weights in rl.opponents.parameters[f"model.{name}"]:  # every slot's set
            assert torch.equal(set_weights, tensor), name
    assert torch.equal(rl.ppo.return_scale.low, trainer.ppo.return_scale.low)
    assert torch.equal(rl.ppo.return_scale.high, trainer.ppo.return_scale.high)
    assert float(trainer.ppo.return_scale.high) != 0.0  # a real state, not the initial zeros
    assert rl.ppo.optimizer.state_dict()["state"] == {}
    assert (rl.ppo.environment_steps, rl.updates_done) == (0, 0)


def test_a_warm_start_from_another_build_is_refused(tmp_path, trained):
    trainer, _ = trained
    checkpoint.save_policy(tmp_path / "bc", trainer.model)
    with (
        torch.random.fork_rng(devices=[]),
        pytest.raises(CheckpointError, match="the warm start refused"),
    ):
        checkpoint.warm_start(
            tmp_path / "bc", cases.tiny_train_config(relational_hidden=32), cases.CPU
        )


class Crash(Exception):
    pass


@pytest.mark.parametrize("stage", [checkpoint.WEIGHTS, checkpoint.TRAINING, checkpoint.RECORD])
def test_a_write_killed_at_any_stage_leaves_the_previous_checkpoint_and_no_partial_one(
    tmp_path, trained, stage
):
    """Acceptance 7, a crash simulated by raising after each file is flushed."""
    trainer, _ = trained
    checkpoint.save_training(tmp_path / "step_1", trainer)

    def crash(name):
        if name == stage:
            raise Crash

    with pytest.raises(Crash):
        checkpoint.save_training(tmp_path / "step_2", trainer, stage=crash)
    assert not (tmp_path / "step_2").exists() and (tmp_path / "step_2.tmp").exists()
    checkpoint.load_policy(tmp_path / "step_1", cases.CPU)
    checkpoint.save_training(tmp_path / "step_2", trainer)  # a later write clears the leftovers
    assert not (tmp_path / "step_2.tmp").exists()
    checkpoint.load_policy(tmp_path / "step_2", cases.CPU)
