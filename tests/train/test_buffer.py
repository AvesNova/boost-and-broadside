"""Tests for the rollout buffer and GAE computation."""

import pytest
import torch

from boost_and_broadside.train.rl.buffer import RolloutBuffer


def _make_buffer(T=4, B=2, N=4, D=16) -> tuple[RolloutBuffer, int, int, int, int]:
    obs_shapes = {
        "pos"    : (N, 2),
        "vel"    : (N, 2),
        "alive"  : (N,),
    }
    buf = RolloutBuffer(
        num_steps   = T,
        num_envs    = B,
        num_ships   = N,
        obs_shapes  = obs_shapes,
        gamma       = 0.99,
        gae_lambda  = 0.95,
        device      = torch.device("cpu"),
    )
    return buf, T, B, N, D


def _fill_buffer(buf: RolloutBuffer, T: int, B: int, N: int, D: int) -> None:
    """Fill a buffer with random data."""
    for _ in range(T):
        obs = {
            "pos"  : torch.rand(B, N, 2),
            "vel"  : torch.rand(B, N, 2),
            "alive": torch.ones(B, N),
        }
        buf.add(
            obs     = obs,
            action  = torch.zeros(B, N, 3, dtype=torch.int32),
            logprob = torch.zeros(B, N),
            reward  = torch.ones(B, N) * 0.1,
            done    = torch.zeros(B),
            value   = torch.ones(B, N) * 0.5,
            alive   = torch.ones(B, N, dtype=torch.bool),
        )


class TestBufferAdd:
    def test_buffer_fills_without_error(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        assert buf.ptr == T

    def test_buffer_overflow_raises(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        with pytest.raises(IndexError):
            obs = {"pos": torch.rand(B, N, 2), "vel": torch.rand(B, N, 2), "alive": torch.ones(B, N)}
            buf.add(obs, torch.zeros(B, N, 3), torch.zeros(B, N), torch.zeros(B, N),
                    torch.zeros(B), torch.zeros(B, N), torch.ones(B, N, dtype=torch.bool))

    def test_reset_clears_pointer(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        buf.reset()
        assert buf.ptr == 0

    def test_rewards_stored_correctly(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        # All rewards were 0.1
        assert torch.allclose(buf.rewards, torch.full((T, B, N), 0.1))


class TestGAEComputation:
    def test_gae_shapes(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        buf.compute_gae(
            next_value=torch.zeros(B, N),
            next_done=torch.zeros(B),
        )
        assert buf.advantages.shape == (T, B, N)
        assert buf.returns.shape    == (T, B, N)

    def test_returns_equals_advantages_plus_values(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        buf.compute_gae(
            next_value=torch.zeros(B, N),
            next_done=torch.zeros(B),
        )
        assert torch.allclose(buf.returns, buf.advantages + buf.values)

    def test_zero_reward_zero_value_gives_zero_advantage(self):
        """With all-zero rewards and values, advantages should be zero."""
        buf, T, B, N, D = _make_buffer()

        for _ in range(T):
            obs = {"pos": torch.rand(B, N, 2), "vel": torch.rand(B, N, 2), "alive": torch.ones(B, N)}
            buf.add(obs, torch.zeros(B, N, 3), torch.zeros(B, N),
                    torch.zeros(B, N),   # reward = 0
                    torch.zeros(B),
                    torch.zeros(B, N),   # value = 0
                    torch.ones(B, N, dtype=torch.bool))

        buf.compute_gae(next_value=torch.zeros(B, N), next_done=torch.zeros(B))

        assert torch.allclose(buf.advantages, torch.zeros(T, B, N), atol=1e-6)

    def test_done_envs_mask_future_rewards(self):
        """When done=1, bootstrap from next_value should be blocked."""
        T, B, N = 3, 1, 2
        buf = RolloutBuffer(
            num_steps=T, num_envs=B, num_ships=N,
            obs_shapes={"pos": (N, 2)},
            gamma=1.0,       # no discounting to make math easy
            gae_lambda=1.0,
            device=torch.device("cpu"),
        )
        for t in range(T):
            obs = {"pos": torch.zeros(B, N, 2)}
            done = torch.tensor([1.0]) if t == 1 else torch.tensor([0.0])
            buf.add(obs, torch.zeros(B, N, 3), torch.zeros(B, N),
                    torch.ones(B, N),   # reward = 1
                    done,
                    torch.zeros(B, N),  # value = 0
                    torch.ones(B, N, dtype=torch.bool))

        # next_value should not matter for t<=1 because done=1 at t=1
        buf.compute_gae(next_value=torch.full((B, N), 99.0), next_done=torch.zeros(B))

        # At t=1 (done), next value should be 0, so advantage[1] = reward[1] = 1
        adv_t1 = buf.advantages[1, 0, 0].item()
        assert abs(adv_t1 - 1.0) < 0.1  # roughly 1 since no bootstrap beyond done


class TestMinibatchIterator:
    def test_yields_correct_number_of_minibatches(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N), torch.zeros(B))

        batches = list(buf.get_minibatch_iterator(num_minibatches=2))
        assert len(batches) == 2

    def test_minibatch_obs_shape(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N), torch.zeros(B))

        mb_obs, mb_actions, *_ = next(iter(buf.get_minibatch_iterator(num_minibatches=2)))

        # 8 envs / 2 minibatches = 4 envs per batch
        B_mb = B // 2
        assert mb_obs["pos"].shape  == (T, B_mb, N, 2)
        assert mb_actions.shape     == (T, B_mb, N, 3)

    def test_minibatch_hidden_shape(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N), torch.zeros(B))

        *_, mb_hidden = next(iter(buf.get_minibatch_iterator(num_minibatches=2)))
        B_mb = B // 2
        assert mb_hidden.shape == (1, B_mb * N, D)

    def test_requires_initial_hidden(self):
        T, B, N, D = 4, 4, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        buf.compute_gae(torch.zeros(B, N), torch.zeros(B))

        with pytest.raises(AssertionError):
            next(iter(buf.get_minibatch_iterator(num_minibatches=1)))
