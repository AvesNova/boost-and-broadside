"""Training checkpoints (runs spec 3, acceptance 1, 4, 7, 9): every part round-trips,
writes are atomic, and fork and warm start load only what fits."""

import dataclasses

import pytest
import runs_cases as cases
import safetensors.torch
import torch

from bnb.runs import checkpoint
from bnb.runs.checkpoint import CheckpointError
from bnb.runtime.train import Trainer


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny trainer after two updates and its training checkpoint."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        trainer = Trainer(cases.tiny_train_config(), cases.CPU)
        for _ in range(2):
            trainer.update()
    directory = tmp_path_factory.mktemp("trained") / "step"
    checkpoint.save_training(directory, trainer)
    return trainer, directory


def test_every_part_of_a_training_checkpoint_round_trips(trained):
    trainer, directory = trained
    with torch.random.fork_rng():
        loaded = checkpoint.load_training(directory, cases.CPU)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], tensor), name
    for name, tensor in trainer.league.state_dict().items():
        assert torch.equal(loaded.league.state_dict()[name], tensor), name
    want, got = trainer.ppo.optimizer.state_dict(), loaded.ppo.optimizer.state_dict()
    assert want["param_groups"] == got["param_groups"]
    for index, state in want["state"].items():
        for key, tensor in state.items():
            assert torch.equal(got["state"][index][key], tensor), (index, key)
    assert torch.equal(loaded.ppo.return_scale.low, trainer.ppo.return_scale.low)
    assert torch.equal(loaded.ppo.return_scale.high, trainer.ppo.return_scale.high)
    assert torch.equal(loaded.ppo.spread_generator.get_state(),
                       trainer.ppo.spread_generator.get_state())
    assert torch.equal(loaded.ppo.order_generator.get_state(),
                       trainer.ppo.order_generator.get_state())
    want_carry = checkpoint.named_tensors(trainer.rollout.saved_carry())
    got_carry = checkpoint.named_tensors(loaded.rollout.saved_carry())
    assert want_carry.keys() == got_carry.keys()
    for name, tensor in want_carry.items():
        assert torch.equal(got_carry[name], tensor), name
    assert (loaded.updates_done, loaded.ppo.env_steps) == (2, trainer.ppo.env_steps)


def test_the_default_generators_are_restored_by_the_entry_points_call(trained):
    _, directory = trained
    stored = safetensors.torch.load_file(directory / checkpoint.TRAINING)
    with torch.random.fork_rng():
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
    with pytest.raises(CheckpointError, match="a resume cannot change num_envs; start a new "
                                              "run from this checkpoint with --from"):
        checkpoint.load_training(directory, cases.CPU, dataclasses.replace(saved, num_envs=32))


def test_a_run_time_override_on_resume_is_allowed(trained):
    _, directory = trained
    saved, _ = checkpoint.training_config(directory)
    learner = dataclasses.replace(saved.learner, learning_rate=1e-4, entropy=-0.01)
    with torch.random.fork_rng():
        trainer = checkpoint.load_training(
            directory, cases.CPU, dataclasses.replace(saved, updates=9, learner=learner))
    assert trainer.cfg.updates == 9 and trainer.ppo.config.learning_rate == 1e-4


def test_a_fork_whose_build_differs_is_refused(trained):
    _, directory = trained
    with torch.random.fork_rng(), pytest.raises(
            CheckpointError, match="the fork refused: the build record .* differs from the "
                                   "run's model in config.hidden. A new architecture is a "
                                   "new run"):
        checkpoint.fork_training(directory, cases.tiny_train_config(relational_hidden=32),
                                 cases.CPU)


def test_a_fork_takes_the_model_optimizer_and_return_scale_but_not_the_carry(trained):
    trainer, directory = trained
    with torch.random.fork_rng():
        fork = checkpoint.fork_training(directory, cases.tiny_train_config(num_envs=32),
                                        cases.CPU)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(fork.model.state_dict()[name], tensor)
    assert torch.equal(fork.ppo.return_scale.high, trainer.ppo.return_scale.high)
    assert fork.ppo.optimizer.state_dict()["state"].keys() == \
        trainer.ppo.optimizer.state_dict()["state"].keys()
    assert fork.rollout.B == 32


def test_a_warm_start_takes_weights_and_return_scale_with_a_fresh_optimizer(tmp_path, trained):
    """Acceptance 9: the BC -> RL hand-off, from a policy file."""
    trainer, _ = trained
    checkpoint.save_policy(tmp_path / "bc", trainer.model, return_scale=trainer.ppo.return_scale,
                           steps=dict(environment_steps=trainer.ppo.env_steps))
    with torch.random.fork_rng():
        rl = checkpoint.warm_start(tmp_path / "bc", cases.tiny_train_config(), cases.CPU)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(rl.model.state_dict()[name], tensor)
        assert torch.equal(rl.league.state_dict()[name], tensor)
    assert torch.equal(rl.ppo.return_scale.low, trainer.ppo.return_scale.low)
    assert torch.equal(rl.ppo.return_scale.high, trainer.ppo.return_scale.high)
    assert float(trainer.ppo.return_scale.high) != 0.0  # a real state, not the initial zeros
    assert rl.ppo.optimizer.state_dict()["state"] == {}
    assert (rl.ppo.env_steps, rl.updates_done) == (0, 0)


def test_a_warm_start_from_another_build_is_refused(tmp_path, trained):
    trainer, _ = trained
    checkpoint.save_policy(tmp_path / "bc", trainer.model)
    with torch.random.fork_rng(), pytest.raises(CheckpointError, match="the warm start refused"):
        checkpoint.warm_start(tmp_path / "bc", cases.tiny_train_config(relational_hidden=32),
                              cases.CPU)


class Crash(Exception):
    pass


@pytest.mark.parametrize("stage", [checkpoint.WEIGHTS, checkpoint.TRAINING, checkpoint.RECORD])
def test_a_write_killed_at_any_stage_leaves_the_previous_checkpoint_and_no_partial_one(
        tmp_path, trained, stage):
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
