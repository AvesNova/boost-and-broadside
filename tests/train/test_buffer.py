"""Tests for the rollout buffer and GAE computation."""

import math

import pytest
import torch

from boost_and_broadside.execution import ExecutionSettings, initialize_execution
from boost_and_broadside.train.rl.buffer import (
    AdvantageScaler,
    ReturnScaler,
    RolloutBuffer,
    symexp,
    symlog,
)

K = 4  # num_components used across tests (smaller than prod K=12 for speed)


def _make_buffer(T=4, B=2, N=4, D=16, num_components=K) -> tuple[RolloutBuffer, int, int, int, int]:
    from boost_and_broadside.env.observation import YemongObservation

    obs_sample = YemongObservation(
        data={
            "pos": torch.zeros((B, N, 2)),
            "vel": torch.zeros((B, N, 2)),
            "alive": torch.zeros((B, N), dtype=torch.bool),
        }
    )
    buf = RolloutBuffer(
        num_steps=T,
        num_envs=B,
        num_ships=N,
        num_components=num_components,
        obs_sample=obs_sample,
        gamma=torch.full((num_components,), 0.99),
        gae_lambda=torch.full((num_components,), 0.95),
        device=torch.device("cpu"),
    )
    return buf, T, B, N, D


def _fill_buffer(buf: RolloutBuffer, T: int, B: int, N: int, D: int) -> None:
    """Fill a buffer with random data."""
    Kc = buf.num_components
    for _ in range(T):
        obs = {
            "pos": torch.rand(B, N, 2),
            "vel": torch.rand(B, N, 2),
            "alive": torch.ones(B, N),
        }
        buf.add(
            obs=obs,
            action=torch.zeros(B, N, 3, dtype=torch.int32),
            logprob=torch.zeros(B, N),
            reward=torch.ones(B, N, Kc) * 0.1,
            value=torch.ones(B, N, Kc) * 0.5,
            alive=torch.ones(B, N, dtype=torch.bool),
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
    @staticmethod
    def _alive(returns):
        return torch.ones(returns.shape[:3], dtype=torch.bool)

    def test_normalize_denormalize_roundtrip(self):
        """denormalize(normalize(x)) ≈ x after scaler has adapted."""
        scaler = ReturnScaler(num_components=K, device=torch.device("cpu"))
        returns = torch.randn(8, 4, 2, K) * 2.0
        scaler.update(returns, self._alive(returns))
        x = torch.randn(4, 2, K)
        assert torch.allclose(scaler.denormalize(scaler.normalize(x)), x, atol=1e-5)

    def test_normalize_maps_two_sigma_to_unit_range(self):
        """STD_MULTIPLE standard deviations from the mean map to ±1."""
        scaler = ReturnScaler(num_components=1, device=torch.device("cpu"), ema_alpha=1.0)
        returns = torch.randn(2000, 1, 1, 1) * 3.0 + 1.0
        scaler.update(returns, self._alive(returns))
        mean, std = scaler.moments
        edge = mean + std * ReturnScaler.STD_MULTIPLE
        assert scaler.normalize(edge).item() == pytest.approx(1.0, abs=1e-4)

    def test_scale_follows_sparse_tails_not_the_spike(self):
        """A spike at zero with rare excursions must be scaled by the excursions.

        This is the regression that motivated leaving p5/p95: for a component
        like ``field_death`` the 5th and 95th percentiles both sit inside the
        zero spike, so the span measured the spike's width and the floor then
        bound on every update.
        """
        returns = torch.zeros(1000, 1, 1, 1)
        returns[::100] = -1.0  # 1% of steps carry the event
        scaler = ReturnScaler(num_components=1, device=torch.device("cpu"), ema_alpha=1.0)
        scaler.update(returns, self._alive(returns))

        flat = returns.reshape(-1)
        percentile_span = torch.quantile(flat, 0.95) - torch.quantile(flat, 0.05)
        assert percentile_span.item() == pytest.approx(0.0)  # p5/p95 sees nothing
        # The event itself lands within an order of magnitude of the unit range.
        assert 0.1 < scaler.normalize(torch.tensor([-1.0])).abs().item() < 10.0
        assert not scaler.floor_bound.any()

    def test_dead_ships_do_not_shrink_the_scale(self):
        """Dead ships sit at zero; counting them narrows the very components
        that can least afford it."""
        returns = torch.zeros(100, 2, 4, 1)
        returns[:, :, :2] = torch.randn(100, 2, 2, 1)
        alive = torch.zeros(100, 2, 4, dtype=torch.bool)
        alive[:, :, :2] = True

        masked = ReturnScaler(num_components=1, device=torch.device("cpu"), ema_alpha=1.0)
        unmasked = ReturnScaler(num_components=1, device=torch.device("cpu"), ema_alpha=1.0)
        masked.update(returns, alive)
        unmasked.update(returns, torch.ones_like(alive))

        assert masked.moments[1].item() > unmasked.moments[1].item()

    def test_min_span_guards_zero_returns(self):
        """Disabled components (all-zero returns) must not produce NaN."""
        scaler = ReturnScaler(
            num_components=2, device=torch.device("cpu"), ema_alpha=1.0, min_span=1.0
        )
        returns = torch.zeros(4, 4, 2, 2)
        scaler.update(returns, self._alive(returns))
        result = scaler.normalize(torch.zeros(2))
        assert torch.isfinite(result).all()
        assert (result == 0.0).all()

    def test_state_dict_roundtrip(self):
        """save/load scaler state must preserve the moments."""
        scaler = ReturnScaler(num_components=K, device=torch.device("cpu"))
        returns = torch.randn(4, 4, 2, K)
        scaler.update(returns, self._alive(returns))
        sd = scaler.state_dict()

        scaler2 = ReturnScaler(num_components=K, device=torch.device("cpu"))
        scaler2.load_state_dict(sd)
        assert torch.allclose(scaler.moments[0], scaler2.moments[0])
        assert torch.allclose(scaler.moments[1], scaler2.moments[1])

    def test_percentile_era_state_reseeds_instead_of_failing(self):
        """A checkpoint from the p5/p95 estimator must still load — re-seeding
        costs one rollout, refusing costs the run."""
        scaler = ReturnScaler(num_components=K, device=torch.device("cpu"), min_span=1e-2)
        scaler.load_state_dict(
            {"p5": torch.zeros(K), "p95": torch.ones(K), "initialized": True, "min_span": 1e-2}
        )
        assert not scaler._initialized

    def test_sparse_component_is_scaled_by_its_own_spread(self):
        """A small-but-real spread sets the scale, rather than the floor."""
        scaler = ReturnScaler(
            num_components=1, device=torch.device("cpu"), ema_alpha=1.0, min_span=1e-3
        )
        returns = torch.linspace(-0.05, 0.05, 200).reshape(200, 1, 1, 1)
        scaler.update(returns, self._alive(returns))
        assert not scaler.floor_bound.any()
        # Scaled by its own spread, the edge of the data is order 1. Held up by a
        # floor two orders above it, it would be order 0.01.
        assert 0.5 < scaler.normalize(torch.tensor([0.05])).abs().item() < 2.0

    def test_production_floor_clears_the_narrowest_real_component(self):
        """The floor must sit far below every live component's spread.

        run 719's narrowest component, ``field_death``, has a 4-sigma span of
        about 0.0127 measured from its logged return histograms. The floor has to
        clear that by a wide margin, or the estimator change just moves which
        components get silently compressed.
        """
        from boost_and_broadside.profiles.rl import RL_PROFILE

        narrowest_sigma = 0.00317
        scaler = ReturnScaler(
            num_components=1,
            device=torch.device("cpu"),
            ema_alpha=1.0,
            min_span=RL_PROFILE.return_min_span,
        )
        returns = (torch.randn(4000, 1, 1, 1) * narrowest_sigma).float()
        scaler.update(returns, self._alive(returns))

        assert not scaler.floor_bound.any()
        span = scaler.moments[1] * 2.0 * ReturnScaler.STD_MULTIPLE
        assert (span / RL_PROFILE.return_min_span).item() > 10.0

    def test_floor_bound_flags_a_degenerate_component(self):
        scaler = ReturnScaler(
            num_components=2, device=torch.device("cpu"), ema_alpha=1.0, min_span=1e-3
        )
        returns = torch.zeros(4, 4, 2, 2)
        returns[..., 1] = torch.linspace(-1.0, 1.0, 32).reshape(4, 4, 2)
        scaler.update(returns, self._alive(returns))
        assert scaler.floor_bound.tolist() == [True, False]

    def test_changed_floor_forces_reseed_on_load(self):
        """A checkpoint written under a different floor must not carry it forward."""
        old = ReturnScaler(num_components=K, device=torch.device("cpu"), min_span=1.0)
        returns = torch.randn(4, 4, 2, K) * 0.01
        old.update(returns, self._alive(returns))
        loaded = ReturnScaler(num_components=K, device=torch.device("cpu"), min_span=1e-3)
        loaded.load_state_dict(old.state_dict())
        assert not loaded._initialized

        same = ReturnScaler(num_components=K, device=torch.device("cpu"), min_span=1e-3)
        fresh = torch.randn(4, 4, 2, K)
        same.update(fresh, self._alive(fresh))
        reloaded = ReturnScaler(num_components=K, device=torch.device("cpu"), min_span=1e-3)
        reloaded.load_state_dict(same.state_dict())
        assert reloaded._initialized

    def test_chunked_update_matches_concatenated_rollout(self):
        first = torch.randn(4, 3, 2, K)
        second = torch.randn(4, 3, 2, K)
        chunked = ReturnScaler(num_components=K, device=torch.device("cpu"))
        concatenated = ReturnScaler(num_components=K, device=torch.device("cpu"))

        chunked.update_chunks([first, second], [self._alive(first), self._alive(second)])
        both = torch.cat((first, second), dim=1)
        concatenated.update(both, self._alive(both))

        assert torch.allclose(chunked.moments[0], concatenated.moments[0], atol=1e-6)
        assert torch.allclose(chunked.moments[1], concatenated.moments[1], atol=1e-6)


class TestAdvantageScaler:
    """The actor-side counterpart to ReturnScaler.

    Its contract is that every component leaves normalization at unit RMS, so
    RewardConfig weights alone set the policy-gradient mix. A floor that binds
    breaks exactly that contract, silently and only for sparse components.
    """

    @staticmethod
    def _advantages(rms: float, shape=(8, 4, 2)) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.full((*shape, 1), rms)
        advantages[::2] *= -1.0  # zero-mean, exact RMS
        alive = torch.ones(shape, dtype=torch.bool)
        return advantages, alive

    def test_normalize_gives_unit_rms(self):
        scaler = AdvantageScaler(num_components=1, device=torch.device("cpu"), ema_alpha=1.0)
        advantages, alive = self._advantages(0.25)
        scaler.update(advantages, alive)
        assert scaler.normalize(advantages).pow(2).mean().sqrt().item() == pytest.approx(
            1.0, abs=1e-3
        )

    def test_sparse_component_reaches_unit_rms_too(self):
        """A component two orders of magnitude below a dense one still normalizes to 1.

        This is the regression: with min_rms=0.1 an advantage RMS of 0.0075 came
        out at 0.075 after normalization, a 13x silent downweight of the win
        signal relative to the dense damage components.
        """
        scaler = AdvantageScaler(
            num_components=1, device=torch.device("cpu"), ema_alpha=1.0, min_rms=1e-4
        )
        advantages, alive = self._advantages(0.0075)
        scaler.update(advantages, alive)
        assert not scaler.floor_bound.any()
        assert scaler.normalize(advantages).pow(2).mean().sqrt().item() == pytest.approx(
            1.0, abs=1e-2
        )

    def test_floor_binds_only_on_a_collapsed_component(self):
        scaler = AdvantageScaler(
            num_components=2, device=torch.device("cpu"), ema_alpha=1.0, min_rms=1e-4
        )
        advantages = torch.zeros(8, 4, 2, 2)
        advantages[..., 1] = 0.0075
        alive = torch.ones(8, 4, 2, dtype=torch.bool)
        scaler.update(advantages, alive)
        assert scaler.floor_bound.tolist() == [True, False]
        assert torch.isfinite(scaler.normalize(advantages)).all()

    def test_changed_floor_forces_reseed_on_load(self):
        old = AdvantageScaler(num_components=K, device=torch.device("cpu"), min_rms=0.1)
        advantages, alive = self._advantages(0.0075, shape=(8, 4, K))
        old.update(advantages.expand(8, 4, K, K).contiguous(), alive)
        loaded = AdvantageScaler(num_components=K, device=torch.device("cpu"), min_rms=1e-4)
        loaded.load_state_dict(old.state_dict())
        assert not loaded._initialized

    def test_state_dict_roundtrip(self):
        scaler = AdvantageScaler(num_components=K, device=torch.device("cpu"))
        advantages = torch.randn(8, 4, 2, K)
        alive = torch.ones(8, 4, 2, dtype=torch.bool)
        scaler.update(advantages, alive)
        reloaded = AdvantageScaler(num_components=K, device=torch.device("cpu"))
        reloaded.load_state_dict(scaler.state_dict())
        assert torch.allclose(scaler.rms, reloaded.rms)


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
            obs = {
                "pos": torch.rand(B, N, 2),
                "vel": torch.rand(B, N, 2),
                "alive": torch.ones(B, N),
            }
            buf.add(
                obs,
                torch.zeros(B, N, 3),
                torch.zeros(B, N),
                torch.zeros(B, N, Kc),
                torch.zeros(B, N, Kc),
                torch.ones(B, N, dtype=torch.bool),
            )

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
        # rewards are bf16-stored (see _STORAGE_FLOAT), so compare at bf16 precision.
        assert torch.allclose(buf.rewards.float(), expected, atol=5e-3)


class TestStoragePrecision:
    """Reduced-precision storage policy: bf16 floats, fp32 positions, uint8 indices."""

    def _make_typed_buffer(self, T=3, B=2, N=2):
        from boost_and_broadside.env.observation import ObsKey, YemongObservation

        obs_sample = YemongObservation(
            data={
                ObsKey.POS: torch.zeros((B, N, 2)),
                ObsKey.VEL: torch.zeros((B, N, 2)),
                ObsKey.TEAM_ID: torch.zeros((B, N), dtype=torch.int32),
                ObsKey.PREVIOUS_ACTION: torch.zeros((B, N, 3), dtype=torch.int64),
                ObsKey.ALIVE: torch.zeros((B, N), dtype=torch.bool),
            }
        )
        buf = RolloutBuffer(
            num_steps=T,
            num_envs=B,
            num_ships=N,
            num_components=K,
            obs_sample=obs_sample,
            gamma=torch.full((K,), 0.99),
            gae_lambda=torch.full((K,), 0.95),
            device=torch.device("cpu"),
        )
        return buf

    def test_obs_channels_use_per_field_dtypes(self):
        from boost_and_broadside.env.observation import ObsKey

        buf = self._make_typed_buffer()
        # Positions keep full precision (large-map accuracy); other floats drop to bf16.
        assert buf.obs[ObsKey.POS].dtype == torch.float32
        assert buf.obs[ObsKey.VEL].dtype == torch.bfloat16
        # Small non-negative index channels compress to uint8; bool stays bool.
        assert buf.obs[ObsKey.TEAM_ID].dtype == torch.uint8
        assert buf.obs[ObsKey.PREVIOUS_ACTION].dtype == torch.uint8
        assert buf.obs[ObsKey.ALIVE].dtype == torch.bool

    def test_per_component_arrays_are_bf16_never_fp16(self):
        buf = self._make_typed_buffer()
        for arr in (buf.rewards, buf.values, buf.advantages, buf.returns, buf.expert_probs):
            assert arr.dtype == torch.bfloat16
            assert arr.dtype != torch.float16

    def test_accumulators_and_ratio_inputs_stay_fp32(self):
        buf = self._make_typed_buffer()
        # logprobs (PPO ratio) and the fp32 aggregates/accumulators must not be reduced.
        assert buf.logprobs.dtype == torch.float32
        assert buf.adv_agg.dtype == torch.float32
        assert buf.ret_agg.dtype == torch.float32
        assert buf.adv_rms.dtype == torch.float32

    def test_uint8_index_channel_round_trips_through_add(self):
        from boost_and_broadside.env.observation import ObsKey, YemongObservation

        buf = self._make_typed_buffer(T=3, B=2, N=2)
        team = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)  # ships + field id 2
        prev = torch.randint(0, 7, (2, 2, 3), dtype=torch.int64)  # OneHot(3/7/2) indices
        obs = YemongObservation(
            data={
                ObsKey.POS: torch.rand(2, 2, 2),
                ObsKey.VEL: torch.rand(2, 2, 2),
                ObsKey.TEAM_ID: team,
                ObsKey.PREVIOUS_ACTION: prev,
                ObsKey.ALIVE: torch.ones(2, 2, dtype=torch.bool),
            }
        )
        buf.add(
            obs=obs,
            action=torch.zeros(2, 2, 3, dtype=torch.int32),
            logprob=torch.zeros(2, 2),
            reward=torch.zeros(2, 2, K),
            value=torch.zeros(2, 2, K),
            alive=torch.ones(2, 2, dtype=torch.bool),
        )
        # Values survive the uint8 downcast exactly and read back correctly as long.
        assert torch.equal(buf.obs[ObsKey.TEAM_ID][0].long(), team.long())
        assert torch.equal(buf.obs[ObsKey.PREVIOUS_ACTION][0].long(), prev.long())


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
        assert buf.returns.shape == (T, B, N, Kc)

    def test_returns_equals_advantages_plus_values(self):
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components
        buf.compute_gae(
            next_value=torch.zeros(B, N, Kc),
            next_done=torch.zeros(B),
        )
        # returns = advantages + values holds up to bf16 storage rounding.
        assert torch.allclose(
            buf.returns.float(), buf.advantages.float() + buf.values.float(), atol=1e-2
        )

    def test_zero_reward_zero_value_gives_zero_advantage(self):
        """With all-zero rewards and values, advantages should be zero."""
        buf, T, B, N, D = _make_buffer()
        Kc = buf.num_components

        for _ in range(T):
            obs = {
                "pos": torch.rand(B, N, 2),
                "vel": torch.rand(B, N, 2),
                "alive": torch.ones(B, N),
            }
            buf.add(
                obs,
                torch.zeros(B, N, 3),
                torch.zeros(B, N),
                torch.zeros(B, N, Kc),  # reward = 0
                torch.zeros(B, N, Kc),  # value = 0
                torch.ones(B, N, dtype=torch.bool),
            )

        buf.compute_gae(next_value=torch.zeros(B, N, Kc), next_done=torch.zeros(B))

        assert torch.allclose(buf.advantages.float(), torch.zeros(T, B, N, Kc), atol=1e-6)

    def test_per_component_gamma(self):
        """Components with different gammas should produce different advantage decay."""
        T, B, N = 5, 1, 1
        Kc = 2  # component 0: γ=1.0, component 1: γ=0.5
        from boost_and_broadside.env.observation import YemongObservation

        obs_sample = YemongObservation(data={"pos": torch.zeros((B, N, 2))})
        buf = RolloutBuffer(
            num_steps=T,
            num_envs=B,
            num_ships=N,
            num_components=Kc,
            obs_sample=obs_sample,
            gamma=torch.tensor([1.0, 0.5]),
            gae_lambda=torch.tensor([1.0, 1.0]),  # λ=1 isolates gamma effect
            device=torch.device("cpu"),
        )
        # Only the last step has a reward
        for t in range(T):
            reward = torch.zeros(B, N, Kc)
            if t == T - 1:
                reward[..., :] = 1.0
            buf.add(
                {"pos": torch.zeros(B, N, 2)},
                torch.zeros(B, N, 3, dtype=torch.int32),
                torch.zeros(B, N),
                reward,
                torch.zeros(B, N, Kc),  # value = 0
                torch.ones(B, N, dtype=torch.bool),
            )
        buf.compute_gae(next_value=torch.zeros(B, N, Kc), next_done=torch.zeros(B))
        # With λ=1 and zero values, A_t ≈ γ^(T-1-t) * r_{T-1} (symlog(1)=log(2)).
        # Tolerance is set by bf16 advantage storage (~0.4% relative); still far
        # tighter than the γ=1 vs γ=0.5 decay spread the test distinguishes.
        r = math.log(2)
        adv0 = buf.advantages[:, 0, 0, 0].float().tolist()  # γ=1.0: all steps same credit
        adv1 = buf.advantages[:, 0, 0, 1].float().tolist()  # γ=0.5: decays as 0.5^(T-1-t)
        for t in range(T):
            steps_back = T - 1 - t
            assert abs(adv0[t] - r) < 1e-2, f"γ=1 step {t}: {adv0[t]} != {r}"
            assert abs(adv1[t] - r * (0.5**steps_back)) < 1e-2, f"γ=0.5 step {t}"

    def test_done_envs_mask_future_rewards(self):
        """When done=1, bootstrap from next_value should be blocked."""
        T, B, N, Kc = 3, 1, 2, 1
        from boost_and_broadside.env.observation import YemongObservation

        obs_sample = YemongObservation(data={"pos": torch.zeros((B, N, 2))})
        buf = RolloutBuffer(
            num_steps=T,
            num_envs=B,
            num_ships=N,
            num_components=Kc,
            obs_sample=obs_sample,
            gamma=torch.full((Kc,), 1.0),
            gae_lambda=torch.full((Kc,), 1.0),
            device=torch.device("cpu"),
        )
        for t in range(T):
            obs = {"pos": torch.zeros(B, N, 2)}
            buf.add(
                obs,
                torch.zeros(B, N, 3),
                torch.zeros(B, N),
                torch.ones(B, N, Kc),  # reward = 1
                torch.zeros(B, N, Kc),  # value = 0
                torch.ones(B, N, dtype=torch.bool),
                terminated=torch.tensor([t == 1]),
            )

        buf.compute_gae(next_value=torch.full((B, N, Kc), 99.0), next_done=torch.zeros(B))

        # Buffer applies symlog on storage: raw reward=1 → symlog(1)=log(2)≈0.693
        adv_t1 = buf.advantages[1, 0, 0, 0].item()
        assert abs(adv_t1 - math.log(2)) < 0.05

    def test_truncation_cuts_the_trace_like_a_termination(self):
        """A time-limited episode must not bootstrap off the next episode.

        The wrapper auto-resets before returning the observation, so values[t+1]
        after a truncation belongs to a freshly spawned episode. Regression: GAE
        keyed on physics-`dones` alone, so a truncation left non_terminal=1 and
        carried the new episode's value backwards through the whole trace.
        """
        T, B, N, Kc = 3, 1, 1, 1
        from boost_and_broadside.env.observation import YemongObservation

        buf = RolloutBuffer(
            num_steps=T,
            num_envs=B,
            num_ships=N,
            num_components=Kc,
            obs_sample=YemongObservation(data={"pos": torch.zeros((B, N, 2))}),
            gamma=torch.full((Kc,), 1.0),
            gae_lambda=torch.full((Kc,), 1.0),
            device=torch.device("cpu"),
        )
        for t in range(T):
            buf.add(
                {"pos": torch.zeros(B, N, 2)},
                torch.zeros(B, N, 3),
                torch.zeros(B, N),
                torch.zeros(B, N, Kc),  # no reward — any advantage is leaked value
                torch.zeros(B, N, Kc),  # value = 0
                torch.ones(B, N, dtype=torch.bool),
                # Truncated, not physics-done: the distinction the bug turned on.
                terminated=torch.tensor([t == 1]),
            )
        buf.values[2] = 99.0  # the "next episode" the reset spawned

        buf.compute_gae(next_value=torch.zeros(B, N, Kc), next_done=torch.zeros(B))

        # Steps 0 and 1 precede the boundary and must not see step 2's value at all.
        assert abs(buf.advantages[1, 0, 0, 0].item()) < 1e-3
        assert abs(buf.advantages[0, 0, 0, 0].item()) < 1e-3


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
        # Without a microbatch budget each minibatch is a single chunk
        assert all(len(chunks) == 1 for chunks in batches)

    def test_minibatch_obs_shape(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        mb_obs, mb_actions, *_ = next(iter(buf.get_minibatch_iterator(num_minibatches=2)))[0]

        B_mb = B // 2
        assert mb_obs["pos"].shape == (T + 1, B_mb, N, 2)
        assert mb_actions.shape == (T, B_mb, N, 3)

    def test_minibatch_advantage_shape(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        batch = next(iter(buf.get_minibatch_iterator(num_minibatches=2)))[0]
        B_mb = B // 2
        assert batch.advantages.shape == (T, B_mb, N, Kc)
        assert batch.returns.shape == (T, B_mb, N, Kc)

    def test_minibatch_hidden_shape(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        batch = next(iter(buf.get_minibatch_iterator(num_minibatches=2)))[0]
        B_mb = B // 2
        assert batch.hidden.shape == (1, B_mb * N, D)
        assert batch.actor_mask.shape == (T, B_mb, N)
        assert batch.expert_probs.shape == (T, B_mb, N, 12)

    def test_the_env_order_repeats_for_a_seeded_process(self):
        """Minibatch grouping is drawn from the Torch RNG that ``--seed`` sets.

        Paired comparisons between two runs of one configuration depend on this:
        an unseeded permutation groups the same environments differently each
        process, so the runs differ for a reason unrelated to what changed.
        """
        T, B, N, D = 4, 8, 4, 16
        settings = ExecutionSettings(
            device="cpu",
            seed=1234,
            compile_mode=None,
            wandb=False,
            allow_config_drift=False,
        )

        def env_order() -> list[int]:
            buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
            _fill_buffer(buf, T, B, N, D)
            # Stamp each env with its index so the grouping is readable downstream.
            marks = torch.arange(B, dtype=torch.float32).repeat_interleave(N)
            buf.store_initial_hidden(marks.reshape(1, B * N, 1).expand(1, B * N, D).clone())
            buf.compute_gae(torch.zeros(B, N, buf.num_components), torch.zeros(B))
            order = []
            for chunks in buf.get_minibatch_iterator(num_minibatches=2):
                for chunk in chunks:
                    order.extend(chunk.hidden[0, ::N, 0].to(torch.int64).tolist())
            return order

        initialize_execution(settings)
        first = env_order()
        initialize_execution(settings)
        assert env_order() == first
        assert sorted(first) == list(range(B))

    def test_requires_initial_hidden(self):
        T, B, N, D = 4, 4, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        with pytest.raises(AssertionError):
            next(iter(buf.get_minibatch_iterator(num_minibatches=1)))

    def test_microbatch_tokens_splits_minibatch(self):
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        # Minibatch = 4 envs × T × num_tokens = 4 × 4 × 4 = 64 tokens.
        # Budget of 32 tokens → 2 micro-batches of 2 envs each.
        batches = list(buf.get_minibatch_iterator(num_minibatches=2, microbatch_tokens=32))
        assert len(batches) == 2
        for chunks in batches:
            assert len(chunks) == 2
            env_counts = [chunk.actions.shape[1] for chunk in chunks]
            assert sum(env_counts) == B // 2
            assert max(env_counts) - min(env_counts) <= 1
            for chunk in chunks:
                b_mb = chunk.actions.shape[1]
                assert chunk.obs["pos"].shape == (T + 1, b_mb, N, 2)
                assert chunk.alive.shape == (T, b_mb, N)
                assert chunk.hidden.shape == (1, b_mb * N, D)
                # env count respects the token budget
                assert b_mb * T * buf.num_tokens <= 32

    def test_microbatch_chunks_partition_minibatch(self):
        """Micro-batch env columns are disjoint and cover every env exactly once."""
        T, B, N, D = 4, 8, 4, 16
        buf, _, _, _, _ = _make_buffer(T=T, B=B, N=N, D=D)
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components

        # Give each env a unique action fingerprint to track the partition.
        buf.actions[:] = torch.arange(B, dtype=torch.int32).view(1, B, 1, 1)

        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, Kc), torch.zeros(B))

        seen: list[int] = []
        for chunks in buf.get_minibatch_iterator(num_minibatches=2, microbatch_tokens=32):
            for chunk in chunks:
                seen.extend(chunk.actions[0, :, 0, 0].tolist())
        assert sorted(seen) == list(range(B))
