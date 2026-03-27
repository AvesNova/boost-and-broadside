"""Tests for the rollout buffer and GAE computation."""

import math

import pytest
import torch

from boost_and_broadside.train.rl.buffer import RolloutBuffer, ReturnScaler, symlog, symexp


K = 4  # num_components used across tests (smaller than prod K=12 for speed)


def _make_buffer(T=4, B=2, N=4, D=16, num_components=K) -> tuple[RolloutBuffer, int, int, int, int]:
    obs_shapes = {
        "pos"    : (N, 2),
        "vel"    : (N, 2),
        "alive"  : (N,),
    }
    buf = RolloutBuffer(
        num_steps      = T,
        num_envs       = B,
        num_ships      = N,
        num_components = num_components,
        obs_shapes     = obs_shapes,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        device         = torch.device("cpu"),
    )
    return buf, T, B, N, D


def _fill_buffer(buf: RolloutBuffer, T: int, B: int, N: int, D: int) -> None:
    """Fill a buffer with random data."""
    Kc = buf.num_components
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
            reward  = torch.ones(B, N, Kc) * 0.1,
            done    = torch.zeros(B),
            value   = torch.ones(B, N, Kc) * 0.5,
            alive   = torch.ones(B, N, dtype=torch.bool),
        )


class TestSymlogSymexp:
    def test_symexp_is_inverse_of_symlog(self):
        x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)

    def test_symlog_preserves_zero(self):
        assert symlog(torch.tensor(0.0)).item() == pytest.approx(0.0)

    def test_symexp_preserves_zero(self):
        assert symexp(torch.tensor(0.0)).item() == pytest.approx(0.0)


class TestReturnScaler:
    def test_normalize_denormalize_roundtrip(self):
        """denormalize(normalize(x)) ≈ x after scaler has adapted."""
        scaler = ReturnScaler(num_components=K, device=torch.device("cpu"))
        # Feed returns that span [-3, 3] so scaler can adapt
        returns = torch.randn(8, 4, 2, K) * 2.0
        scaler.update(returns)
        x = torch.randn(4, 2, K)
        assert torch.allclose(scaler.denormalize(scaler.normalize(x)), x, atol=1e-5)

    def test_normalize_maps_percentiles_to_unit_range(self):
        """After a single update the p5/p95 percentiles of the data should map near ±1."""
        scaler = ReturnScaler(num_components=1, device=torch.device("cpu"),
                              ema_alpha=1.0)  # alpha=1: EMA = current batch exactly
        returns = torch.arange(-10.0, 10.0).reshape(20, 1, 1, 1)
        scaler.update(returns)
        p5  = torch.quantile(returns.reshape(-1), 0.05)
        p95 = torch.quantile(returns.reshape(-1), 0.95)
        assert scaler.normalize(p5.unsqueeze(-1)).abs().item() < 1.1
        assert scaler.normalize(p95.unsqueeze(-1)).abs().item() < 1.1

    def test_min_span_guards_zero_returns(self):
        """Disabled components (all-zero returns) must not produce NaN."""
        scaler = ReturnScaler(num_components=2, device=torch.device("cpu"),
                              ema_alpha=1.0, min_span=1.0)
        returns = torch.zeros(4, 4, 2, 2)
        scaler.update(returns)
        x = torch.zeros(2)
        result = scaler.normalize(x)
        assert torch.isfinite(result).all()
        assert (result == 0.0).all()

    def test_state_dict_roundtrip(self):
        """save/load scaler state must preserve p5 and p95."""
        scaler = ReturnScaler(num_components=K, device=torch.device("cpu"))
        returns = torch.randn(4, 4, 2, K)
        scaler.update(returns)
        sd = scaler.state_dict()

        scaler2 = ReturnScaler(num_components=K, device=torch.device("cpu"))
        scaler2.load_state_dict(sd)
        assert torch.allclose(scaler._p5, scaler2._p5)
        assert torch.allclose(scaler._p95, scaler2._p95)


class TestBufferAdd:
    def test_buffer_fills_without_error(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        assert buf.ptr == T

    def test_buffer_overflow_raises(self):
        buf, T, B, N, D = _make_buffer()
        Kc = buf.num_components
        _fill_buffer(buf, T, B, N, D)
        with pytest.raises(IndexError):
            obs = {"pos": torch.rand(B, N, 2), "vel": torch.rand(B, N, 2), "alive": torch.ones(B, N)}
            buf.add(obs, torch.zeros(B, N, 3), torch.zeros(B, N), torch.zeros(B, N, Kc),
                    torch.zeros(B), torch.zeros(B, N, Kc), torch.ones(B, N, dtype=torch.bool))

    def test_reset_clears_pointer(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        buf.reset()
        assert buf.ptr == 0

    def test_rewards_stored_with_symlog(self):
        """Buffer applies symlog transform on storage."""
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components
        expected = symlog(torch.full((T, B, N, Kc), 0.1))
        assert torch.allclose(buf.rewards, expected)


class TestGAEComputation:
    def test_gae_shapes(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components
        buf.compute_gae(
            next_value=torch.zeros(B, N, Kc),
            next_done=torch.zeros(B),
        )
        assert buf.advantages.shape == (T, B, N, Kc)
        assert buf.returns.shape    == (T, B, N, Kc)

    def test_returns_equals_advantages_plus_values(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components
        buf.compute_gae(
            next_value=torch.zeros(B, N, Kc),
            next_done=torch.zeros(B),
        )
        assert torch.allclose(buf.returns, buf.advantages + buf.values)

    def test_zero_reward_zero_value_gives_zero_advantage(self):
        """With all-zero rewards and values, advantages should be zero."""
        buf, T, B, N, D = _make_buffer()
        Kc = buf.num_components

        for _ in range(T):
            obs = {"pos": torch.rand(B, N, 2), "vel": torch.rand(B, N, 2), "alive": torch.ones(B, N)}
            buf.add(obs, torch.zeros(B, N, 3), torch.zeros(B, N),
                    torch.zeros(B, N, Kc),   # reward = 0
                    torch.zeros(B),
                    torch.zeros(B, N, Kc),   # value = 0
                    torch.ones(B, N, dtype=torch.bool))

        buf.compute_gae(next_value=torch.zeros(B, N, Kc), next_done=torch.zeros(B))

        assert torch.allclose(buf.advantages, torch.zeros(T, B, N, Kc), atol=1e-6)

    def test_done_envs_mask_future_rewards(self):
        """When done=1, bootstrap from next_value should be blocked."""
        T, B, N, Kc = 3, 1, 2, 1
        buf = RolloutBuffer(
            num_steps=T, num_envs=B, num_ships=N, num_components=Kc,
            obs_shapes={"pos": (N, 2)},
            gamma=1.0,
            gae_lambda=1.0,
            device=torch.device("cpu"),
        )
        for t in range(T):
            obs = {"pos": torch.zeros(B, N, 2)}
            done = torch.tensor([1.0]) if t == 1 else torch.tensor([0.0])
            buf.add(obs, torch.zeros(B, N, 3), torch.zeros(B, N),
                    torch.ones(B, N, Kc),    # reward = 1
                    done,
                    torch.zeros(B, N, Kc),   # value = 0
                    torch.ones(B, N, dtype=torch.bool))

        buf.compute_gae(next_value=torch.full((B, N, Kc), 99.0), next_done=torch.zeros(B))

        # Buffer applies symlog on storage: raw reward=1 → symlog(1)=log(2)≈0.693
        adv_t1 = buf.advantages[1, 0, 0, 0].item()
        assert abs(adv_t1 - math.log(2)) < 0.05


class TestMinibatchIterator:
    def test_yields_correct_number_of_minibatches(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        batches = list(buf.get_minibatch_iterator(num_minibatches=2))
        assert len(batches) == 2

    def test_minibatch_obs_shape(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        mb_obs, mb_actions, *_ = next(iter(buf.get_minibatch_iterator(num_minibatches=2)))

        B_mb = B // 2
        assert mb_obs["pos"].shape == (T, B_mb, N, 2)
        assert mb_actions.shape    == (T, B_mb, N, 3)

    def test_minibatch_advantage_shape(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        _, _, _, mb_adv, mb_ret, mb_val, *_ = next(iter(buf.get_minibatch_iterator(num_minibatches=2)))
        B_mb = B // 2
        assert mb_adv.shape == (T, B_mb, N, Kc)
        assert mb_ret.shape == (T, B_mb, N, Kc)
        assert mb_val.shape == (T, B_mb, N, Kc)

    def test_minibatch_hidden_shape(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        *_, mb_hidden, mb_actor_mask = next(iter(buf.get_minibatch_iterator(num_minibatches=2)))
        B_mb = B // 2
        assert mb_hidden.shape == (1, B_mb * N, D)
        assert mb_actor_mask.shape == (T, B_mb, N)

    def test_requires_initial_hidden(self):
        T, B, N, D = 4, 4, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        with pytest.raises(AssertionError):
            next(iter(buf.get_minibatch_iterator(num_minibatches=1)))
