"""Gradient accumulation over micro-batches equals the unsplit minibatch.

Masked-mean loss terms divide by minibatch-total denominators precisely so that
splitting a minibatch for memory does not change the objective. A term that is a
statistic of its own batch cannot decompose that way: the triangle-window
cumulative loss was one, and while it was in the objective a split perturbed the
applied gradient by 0.1 to 0.3 percent, growing with the split count. SIGReg is
the only such term left and is disabled in the reference configuration, so the
equivalence the sizing logic assumes now actually holds.
"""

import dataclasses

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.train.rl.ppo import PPOTrainer
from tests.train.test_ppo import _make_train_config

UNSPLIT = 10**9  # more tokens than any minibatch holds, so nothing is split


def _trainer(tmp_path) -> PPOTrainer:
    """One collected rollout over 16 environments: 8 per minibatch, so up to 8 splits."""

    torch.manual_seed(7)
    config = _make_train_config(checkpoint_dir=str(tmp_path))
    config = dataclasses.replace(
        config, scales=(dataclasses.replace(config.scales[0], num_envs=16),)
    )
    ship_config = ShipConfig()
    trainer = PPOTrainer(
        train_config=config,
        model_config=ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
        ship_config=ship_config,
        device="cpu",
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(ship_config, StochasticAgentConfig()),
    )
    runtime = trainer._initialize_rollout_runtime()
    terminated = trainer._collect_rollout(runtime, False)
    trainer._compute_rollout_gae(runtime, terminated)
    trainer._precompute_lambda_aggregates(
        trainer.buffer, trainer._active_component_weights(), is_primary=True
    )
    trainer._precompute_ns_labels(trainer.buffer)
    return trainer


def _epoch_gradient(trainer: PPOTrainer, microbatch_tokens: int) -> tuple[torch.Tensor, int]:
    """Total gradient over every minibatch of one epoch, and the widest split used.

    The buffer partitions environments with ``torch.randperm`` and denominators
    are per-minibatch, so the partition is re-seeded here: otherwise the two
    calls compare different partitions rather than different splits.
    """

    trainer.optim.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    splits = []
    for chunks in trainer.buffer.get_minibatch_iterator(
        trainer.cfg.num_minibatches, microbatch_tokens
    ):
        denominators = trainer._minibatch_denominators(chunks, trainer.buffer, True)
        envs = sum(chunk.alive.shape[1] for chunk in chunks)
        count = 0
        for source, device_chunk in trainer._iter_device_chunks(chunks, trainer.buffer):
            count += 1
            loss, _ = trainer._compute_minibatch_loss(
                device_chunk, True, denominators, source.alive.shape[1] / envs
            )
            loss.backward()
        splits.append(count)
    flat = torch.cat(
        [
            parameter.grad.detach().reshape(-1)
            for parameter in trainer._policy_module.parameters()
            if parameter.grad is not None
        ]
    )
    return flat, max(splits)


def test_splitting_a_minibatch_does_not_change_the_applied_gradient(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    per_env = trainer.buffer.num_ships * trainer.cfg.num_steps
    whole, unsplit_count = _epoch_gradient(trainer, UNSPLIT)
    assert unsplit_count == 1
    assert whole.norm() > 0.0

    for envs_per_microbatch in (4, 2, 1):
        split, count = _epoch_gradient(trainer, per_env * envs_per_microbatch)
        assert count > 1, "expected the minibatch to actually split"
        error = ((split - whole).norm() / whole.norm()).item()
        # Floating-point roundoff only. The windowed term used to put this at
        # 1e-3 and rising with the split count.
        assert error < 1e-5, f"{count} micro-batches perturbed the gradient by {error:.2e}"
