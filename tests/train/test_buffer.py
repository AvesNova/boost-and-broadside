"""Tests for the rollout buffer and GAE computation."""

import pytest
import torch

from boost_and_broadside.execution import ExecutionSettings, initialize_execution
from boost_and_broadside.train.rl.buffer import RolloutBuffer

K = 4  # num_components used across tests (smaller than prod K=12 for speed)


def _make_buffer(
    T=4,
    B=2,
    N=4,
    D=16,
    num_components=K,
    store_expert_probs=True,
) -> tuple[RolloutBuffer, int, int, int, int]:
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
        store_expert_probs=store_expert_probs,
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

    def test_rewards_are_stored_raw(self):
        """GAE runs on raw rewards; only the critic's output is compressed."""
        buf, T, B, N, D = _make_buffer()
        _fill_buffer(buf, T, B, N, D)
        Kc = buf.num_components
        assert torch.equal(buf.rewards, torch.full((T, B, N, Kc), 0.1))

    def test_ppo_only_buffer_reserves_no_bc_payload(self):
        buf, T, B, N, _ = _make_buffer(store_expert_probs=False)

        assert buf.expert_probs.shape == (T, B, N, 0)
        assert buf.expert_probs.numel() == 0
        with pytest.raises(ValueError, match="BC storage off"):
            buf.add(
                obs={
                    "pos": torch.zeros(B, N, 2),
                    "vel": torch.zeros(B, N, 2),
                    "alive": torch.ones(B, N),
                },
                action=torch.zeros(B, N, 3),
                logprob=torch.zeros(B, N),
                reward=torch.zeros(B, N, K),
                value=torch.zeros(B, N, K),
                alive=torch.ones(B, N, dtype=torch.bool),
                expert_probs=torch.zeros(B, N, 12),
            )


class TestStoragePrecision:
    """Reduced-precision storage policy: bf16 floats, fp32 positions, uint8 indices."""

    def _make_typed_buffer(self, T=3, B=2, N=2):
        from boost_and_broadside.env.observation import ObsKey, YemongObservation

        obs_sample = YemongObservation(
            data={
                ObsKey.POS: torch.zeros((B, N, 2)),
                ObsKey.VEL: torch.zeros((B, N, 2)),
                ObsKey.HEALTH: torch.zeros((B, N, 1)),
                ObsKey.COOLDOWN: torch.zeros((B, N, 1)),
                ObsKey.TIME_SINCE_OBSERVATION: torch.zeros((B, N, 1)),
                ObsKey.TEAM_ID: torch.zeros((B, N), dtype=torch.int32),
                ObsKey.PREVIOUS_ACTION: torch.zeros((B, N, 30)),
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
        # The eleven physical ship channels keep full precision: they are the
        # believed state every next-state label steps from, so bf16's ~0.4%
        # resolution would land in the label rather than only in the encoder's
        # input. Everything else that is merely read once drops to bf16.
        for key in (ObsKey.POS, ObsKey.VEL, ObsKey.HEALTH, ObsKey.COOLDOWN):
            assert buf.obs[key].dtype == torch.float32, key
        assert buf.obs[ObsKey.PREVIOUS_ACTION].dtype == torch.bfloat16
        assert buf.obs[ObsKey.TIME_SINCE_OBSERVATION].dtype == torch.bfloat16
        # Small non-negative index channels compress to uint8; bool stays bool.
        assert buf.obs[ObsKey.TEAM_ID].dtype == torch.uint8
        assert buf.obs[ObsKey.ALIVE].dtype == torch.bool

    def test_raw_returns_are_fp32_and_teacher_probabilities_bf16(self):
        """Raw returns reach ~100 win units, where bf16's step is 0.5."""
        buf = self._make_typed_buffer()
        for arr in (buf.rewards, buf.values, buf.advantages, buf.returns):
            assert arr.dtype == torch.float32
        assert buf.expert_probs.dtype == torch.bfloat16

    def test_accumulators_and_ratio_inputs_stay_fp32(self):
        buf = self._make_typed_buffer()
        # logprobs (PPO ratio) and the fp32 aggregates/accumulators must not be reduced.
        assert buf.logprobs.dtype == torch.float32
        assert buf.adv_agg.dtype == torch.float32
        assert buf.ret_agg.dtype == torch.float32
        assert buf.return_scale.dtype == torch.float32

    def test_pending_distribution_round_trips_through_add(self):
        from boost_and_broadside.env.observation import ObsKey, YemongObservation

        buf = self._make_typed_buffer(T=3, B=2, N=2)
        team = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)  # ships + field id 2
        prev = torch.rand(2, 2, 30)
        prev /= prev.sum(-1, keepdim=True)
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
        torch.testing.assert_close(buf.obs[ObsKey.PREVIOUS_ACTION][0], prev.to(torch.bfloat16))


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
        # With λ=1 and zero values, A_t = γ^(T-1-t) * r_{T-1}; rewards are raw.
        r = 1.0
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

        adv_t1 = buf.advantages[1, 0, 0, 0].item()
        assert abs(adv_t1 - 1.0) < 1e-6

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

    def test_respawn_discontinuity_does_not_cut_gae(self):
        """Physical teleport masking must not become an episode boundary."""
        buf, T, B, N, _ = _make_buffer(T=3, B=1, N=1, num_components=1)
        for t in range(T):
            buf.add(
                {
                    "pos": torch.zeros(B, N, 2),
                    "vel": torch.zeros(B, N, 2),
                    "alive": torch.ones(B, N),
                },
                torch.zeros(B, N, 3),
                torch.zeros(B, N),
                torch.full((B, N, 1), float(t == 2)),
                torch.zeros(B, N, 1),
                torch.ones(B, N, dtype=torch.bool),
                terminated=torch.zeros(B, dtype=torch.bool),
                transition_contiguous=torch.tensor([[t != 1]]),
            )

        buf.compute_gae(torch.zeros(B, N, 1), torch.zeros(B))

        assert buf.advantages[0, 0, 0, 0] > 0.0


class TestRespawnContinuitySurvivesTheUpdate:
    """Respawn flags are written during the rollout and read a stage later.

    ``_compute_rollout_gae`` runs between the two, so anything it touches has to
    leave the flags alone. It once did not: the outcome labelling it ran carried
    three lines belonging to ``reset`` -- inserted by 1debbfd -- and wiped every
    teleport mask before the update could read one, silently, from September 16
    2026. Nothing caught it because no test called the two in the order the
    trainer does.
    """

    @staticmethod
    def _add_step(buf, B, N, *, continuity, reward=0.0, terminated=False):
        buf.add(
            {
                "pos": torch.zeros(B, N, 2),
                "vel": torch.zeros(B, N, 2),
                "alive": torch.ones(B, N),
            },
            torch.zeros(B, N, 3),
            torch.zeros(B, N),
            torch.full((B, N, K), reward),
            torch.zeros(B, N, K),
            torch.ones(B, N, dtype=torch.bool),
            terminated=torch.full((B,), terminated, dtype=torch.bool),
            transition_contiguous=continuity,
        )

    def test_outcome_targets_do_not_clear_the_teleport_mask(self):
        buf, T, B, N, _ = _make_buffer(T=1, B=1, N=2)
        teleported = torch.tensor([[True, False]])
        self._add_step(buf, B, N, continuity=teleported, terminated=True)

        buf.compute_outcome_targets(torch.full((B, 4), 0.25), 0.99, 0.95)

        assert torch.equal(buf.transition_contiguous[0], teleported)

    def test_the_mask_reaches_the_minibatch_through_the_trainer_ordering(self):
        """add -> compute_gae -> outcome targets -> iterate, as PPOTrainer runs it."""
        buf, T, B, N, D = _make_buffer(T=1, B=2, N=2)
        teleported = torch.tensor([[True, False], [False, True]])
        self._add_step(buf, B, N, continuity=teleported, terminated=True)
        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, K), torch.zeros(B))
        buf.compute_outcome_targets(torch.full((B, 4), 0.25), 0.99, 0.95)

        batch = next(buf.get_minibatch_iterator(1))[0]

        assert sorted(batch.transition_contiguous[0].tolist()) == sorted(teleported.tolist())

    def test_reset_is_what_clears_the_mask_for_the_next_rollout(self):
        buf, T, B, N, _ = _make_buffer(T=1, B=1, N=2)
        self._add_step(buf, B, N, continuity=torch.tensor([[True, False]]))
        assert not buf.transition_contiguous.all()

        buf.reset()

        assert buf.transition_contiguous.all()

    def test_reset_clears_the_belief_diagnostics_and_targets_do_not(self):
        buf, T, B, N, _ = _make_buffer(T=1, B=1, N=2)
        self._add_step(buf, B, N, continuity=torch.ones(B, N, dtype=torch.bool), terminated=True)
        buf.belief_diagnostics = {"belief/position": (torch.ones(1), torch.ones(1))}

        buf.compute_outcome_targets(torch.full((B, 4), 0.25), 0.99, 0.95)
        assert buf.belief_diagnostics

        buf.reset()
        assert buf.belief_diagnostics == {}


class TestMinibatchIterator:
    def test_carries_per_ship_transition_continuity(self):
        buf, T, B, N, D = _make_buffer(T=1, B=2, N=2)
        continuity = torch.tensor([[True, False], [False, True]])
        obs = {"pos": torch.zeros(B, N, 2), "vel": torch.zeros(B, N, 2), "alive": torch.ones(B, N)}
        buf.add(
            obs,
            torch.zeros(B, N, 3),
            torch.zeros(B, N),
            torch.zeros(B, N, K),
            torch.zeros(B, N, K),
            torch.ones(B, N, dtype=torch.bool),
            transition_contiguous=continuity,
        )
        buf.store_initial_hidden(torch.zeros(1, B * N, D))
        buf.compute_gae(torch.zeros(B, N, K), torch.zeros(B))

        batch = next(buf.get_minibatch_iterator(1))[0]

        assert sorted(batch.transition_contiguous[0].tolist()) == sorted(continuity.tolist())

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
        assert batch.expert_probs.shape == (T, B_mb, N, 10)

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
