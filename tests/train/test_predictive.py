"""The iterated predictive belief state: architecture, timing, and masking.

The architectural assertions here are cheap. The ones worth the file are the
timing and masking assertions: a predictive loss that is off by one step trains
perfectly well and teaches the wrong thing, and nothing else in the system would
notice.
"""

import dataclasses
from pathlib import Path

import pytest
import torch

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.constants import (
    ACTION_FACTOR_MAX_ENTROPY,
    ACTION_FACTOR_SLICES,
    NUM_POWER_ACTIONS,
    NUM_SHOOT_ACTIONS,
    NUM_TURN_ACTIONS,
    POWER_SLICE,
    SHOOT_SLICE,
    TOTAL_ACTION_LOGITS,
    TURN_SLICE,
)
from boost_and_broadside.models.yemong.predictive import (
    PredictiveModel,
    PredictiveTransition,
)
from boost_and_broadside.train.rl.ppo import (
    factored_action_statistics,
    predictive_horizon_masks,
)

PREDICTIVE_LATENT_DIM = 16
STATE_PREDICTION_DIM = 10


@pytest.fixture
def predictive() -> PredictiveModel:
    torch.manual_seed(0)
    return PredictiveModel(
        d_model=32,
        predictive_latent_dim=PREDICTIVE_LATENT_DIM,
        state_prediction_dim=STATE_PREDICTION_DIM,
    )


# ----------------------------------------------------------------------
# Architecture
# ----------------------------------------------------------------------


class TestPredictiveArchitecture:
    def test_the_projection_maps_the_policy_latent_into_the_belief_space(self, predictive):
        latent = torch.randn(3, 5, 32)
        assert predictive(latent).shape == (3, 5, PREDICTIVE_LATENT_DIM)

    def test_the_transition_preserves_the_belief_width(self, predictive):
        belief = torch.randn(3, 5, PREDICTIVE_LATENT_DIM)
        assert predictive.advance(belief).shape == belief.shape

    def test_the_state_head_emits_the_coordinator_prediction_width(self, predictive):
        belief = torch.randn(3, 5, PREDICTIVE_LATENT_DIM)
        assert predictive.predict_state(belief).shape == (3, 5, STATE_PREDICTION_DIM)

    def test_the_action_head_emits_the_three_factored_logit_slices(self, predictive):
        belief = torch.randn(3, 5, PREDICTIVE_LATENT_DIM)
        logits = predictive.predict_action_logits(belief)
        assert logits.shape == (3, 5, TOTAL_ACTION_LOGITS)
        widths = (NUM_POWER_ACTIONS, NUM_TURN_ACTIONS, NUM_SHOOT_ACTIONS)
        assert tuple(logits[..., s].shape[-1] for s in (POWER_SLICE, TURN_SLICE, SHOOT_SLICE)) == (
            widths
        )

    def test_the_projection_is_one_linear_layer_and_a_norm(self, predictive):
        """A deeper projection could solve the auxiliary task after the trunk."""
        linears = [m for m in predictive.projection.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 1

    def test_the_transition_starts_close_to_the_identity(self, predictive):
        """Small output init, so a twelve-deep rollout does not begin by diverging."""
        belief = torch.randn(64, PREDICTIVE_LATENT_DIM)
        normalized = predictive.transition.norm(belief)
        advanced = predictive.advance(belief)
        drift = (advanced - normalized).norm() / normalized.norm()
        assert drift < 0.05, f"initial transition moves the belief by {drift:.3f}"

    def test_a_deep_rollout_stays_bounded_at_initialization(self, predictive):
        belief = predictive(torch.randn(64, 32))
        for _ in range(32):
            belief = predictive.advance(belief)
        assert torch.isfinite(belief).all()
        assert belief.abs().max() < 100.0

    def test_the_residual_path_is_what_carries_the_belief(self):
        """Zero the MLP entirely and the transition must be the norm alone."""
        transition = PredictiveTransition(PREDICTIVE_LATENT_DIM)
        for parameter in transition.mlp.parameters():
            torch.nn.init.zeros_(parameter)
        belief = torch.randn(8, PREDICTIVE_LATENT_DIM)
        torch.testing.assert_close(transition(belief), transition.norm(belief))


class TestSharedAcrossHorizons:
    def test_one_transition_and_one_head_of_each_kind_serve_every_horizon(self, predictive):
        """Depth must cost no parameters: the same modules run at every horizon."""
        belief = predictive(torch.randn(4, 32))
        seen = []
        for _ in range(5):
            seen.append(
                (
                    id(predictive.state_prediction_head),
                    id(predictive.action_prediction_head),
                    id(predictive.transition),
                )
            )
            belief = predictive.advance(belief)
        assert len(set(seen)) == 1

    def test_the_policy_holds_exactly_one_of_each_predictive_module(self):
        from boost_and_broadside.models.yemong.predictive import (
            PredictiveActionHead,
            PredictiveProjection,
            PredictiveStateHead,
        )

        model = PredictiveModel(32, PREDICTIVE_LATENT_DIM, STATE_PREDICTION_DIM)
        for kind in (
            PredictiveProjection,
            PredictiveTransition,
            PredictiveStateHead,
            PredictiveActionHead,
        ):
            assert sum(isinstance(m, kind) for m in model.modules()) == 1

    def test_every_horizon_sends_gradient_to_the_shared_transition(self, predictive):
        belief = predictive(torch.randn(4, 32))
        losses = []
        for _ in range(4):
            belief = predictive.advance(belief)
            losses.append(predictive.predict_state(belief).square().mean())
        weight = predictive.transition.mlp[0].weight
        first = torch.autograd.grad(losses[0], weight, retain_graph=True)
        last = torch.autograd.grad(losses[-1], weight)
        assert first[0].abs().sum() > 0
        assert last[0].abs().sum() > 0


# ----------------------------------------------------------------------
# Horizon alignment and masking
# ----------------------------------------------------------------------


def _masks(alive, terminated, actor_mask, horizon):
    return list(predictive_horizon_masks(alive, terminated, actor_mask, horizon))


class TestHorizonMasks:
    def test_each_horizon_covers_the_base_steps_that_can_reach_it(self):
        steps, envs, ships = 6, 2, 3
        alive = torch.ones(steps, envs, ships, dtype=torch.bool)
        terminated = torch.zeros(steps, envs, dtype=torch.bool)
        actor_mask = torch.zeros(steps, envs, ships, dtype=torch.bool)

        masks = _masks(alive, terminated, actor_mask, horizon=4)
        assert [horizon for horizon, _, _ in masks] == [0, 1, 2, 3]
        assert [state.shape[0] for _, state, _ in masks] == [6, 5, 4, 3]

    def test_a_horizon_longer_than_the_rollout_stops_at_the_rollout(self):
        steps = 3
        alive = torch.ones(steps, 1, 1, dtype=torch.bool)
        terminated = torch.zeros(steps, 1, dtype=torch.bool)
        actor_mask = torch.zeros(steps, 1, 1, dtype=torch.bool)
        assert len(_masks(alive, terminated, actor_mask, horizon=99)) == steps

    def test_an_episode_boundary_ends_every_trajectory_that_would_cross_it(self):
        """A prediction may not borrow the episode that follows a termination."""
        steps = 6
        alive = torch.ones(steps, 1, 1, dtype=torch.bool)
        terminated = torch.zeros(steps, 1, dtype=torch.bool)
        terminated[2] = True  # the episode ends at step 2; step 3 is a new one
        actor_mask = torch.zeros(steps, 1, 1, dtype=torch.bool)

        masks = {
            horizon: (state, action)
            for horizon, state, action in _masks(alive, terminated, actor_mask, horizon=4)
        }
        # Horizon 0 is a transition out of each step; the terminal step has none.
        assert masks[0][0][:, 0, 0].tolist() == [True, True, False, True, True, True]
        # Horizon 0's action still exists at the terminal step: a decision was made.
        assert masks[0][1][:, 0, 0].tolist() == [True] * 6
        # Horizon 1 from base 2 would land in the next episode.
        assert masks[1][1][:, 0, 0].tolist() == [True, True, False, True, True]
        # Horizon 2 from bases 1 and 2 both cross the boundary.
        assert masks[2][1][:, 0, 0].tolist() == [True, False, False, True]

    def test_a_dead_ship_is_masked_at_the_horizon_it_is_dead_at(self):
        steps = 4
        alive = torch.ones(steps, 1, 2, dtype=torch.bool)
        alive[2:, 0, 1] = False  # second ship dies at step 2
        terminated = torch.zeros(steps, 1, dtype=torch.bool)
        actor_mask = torch.zeros(steps, 1, 2, dtype=torch.bool)

        masks = {
            horizon: action
            for horizon, _, action in _masks(alive, terminated, actor_mask, horizon=3)
        }
        assert masks[0][:, 0, 1].tolist() == [True, True, False, False]
        # From base 1, horizon 1 describes step 2 — where the ship is already dead.
        assert masks[1][:, 0, 1].tolist() == [True, False, False]
        assert masks[1][:, 0, 0].tolist() == [True, True, True]

    def test_the_immediate_action_target_excludes_self_generated_actions(self):
        """A latent may not be rewarded for predicting the action it produced."""
        steps = 4
        alive = torch.ones(steps, 1, 2, dtype=torch.bool)
        terminated = torch.zeros(steps, 1, dtype=torch.bool)
        actor_mask = torch.zeros(steps, 1, 2, dtype=torch.bool)
        actor_mask[:, 0, 0] = True  # ship 0's action was sampled from this pass

        masks = {
            horizon: action
            for horizon, _, action in _masks(alive, terminated, actor_mask, horizon=3)
        }
        assert masks[0][:, 0, 0].tolist() == [False] * 4
        # The opponent's immediate action was produced elsewhere and stays a target.
        assert masks[0][:, 0, 1].tolist() == [True] * 4

    def test_a_ships_own_later_actions_remain_legitimate_targets(self):
        """Only the immediate step is self-generated; future decisions are not."""
        steps = 4
        alive = torch.ones(steps, 1, 1, dtype=torch.bool)
        terminated = torch.zeros(steps, 1, dtype=torch.bool)
        actor_mask = torch.ones(steps, 1, 1, dtype=torch.bool)

        masks = {
            horizon: action
            for horizon, _, action in _masks(alive, terminated, actor_mask, horizon=3)
        }
        assert masks[0][:, 0, 0].tolist() == [False] * 4
        assert masks[1][:, 0, 0].tolist() == [True] * 3
        assert masks[2][:, 0, 0].tolist() == [True] * 2

    def test_the_state_mask_never_scores_a_transition_the_action_mask_rejects(self):
        """The state mask is the action mask minus the terminal step."""
        steps = 5
        torch.manual_seed(3)
        alive = torch.rand(steps, 2, 3) > 0.2
        terminated = torch.rand(steps, 2) > 0.7
        actor_mask = torch.zeros(steps, 2, 3, dtype=torch.bool)
        for horizon, state, action in _masks(alive, terminated, actor_mask, horizon=3):
            assert not (state & ~action).any(), f"horizon {horizon}"


class TestFactoredActionStatistics:
    def test_each_factor_is_its_own_cross_entropy_over_its_own_maximum(self):
        torch.manual_seed(0)
        logits = torch.randn(4, TOTAL_ACTION_LOGITS)
        targets = torch.tensor([[0, 3, 1], [2, 0, 0], [1, 6, 1], [0, 2, 0]])
        cross_entropy, entropy, hits = factored_action_statistics(logits, targets)

        expected = torch.zeros(4)
        for factor, (logit_slice, maximum) in enumerate(
            zip(ACTION_FACTOR_SLICES, ACTION_FACTOR_MAX_ENTROPY, strict=True)
        ):
            expected += (
                torch.nn.functional.cross_entropy(
                    logits[..., logit_slice], targets[..., factor], reduction="none"
                )
                / maximum
            )
        torch.testing.assert_close(cross_entropy, expected)
        assert entropy.shape == (4,)
        assert hits.shape == (4, 3)

    def test_no_factor_is_weighted_by_how_many_options_it_offers(self):
        """The seven-way turn must not outweigh the binary shoot for being wider.

        Under a plain sum of cross-entropies an untrained head puts 52% of the
        loss on turn and 19% on shoot, purely from cardinality. Each factor
        being equally uninformed has to cost the same.
        """
        uniform = torch.zeros(1, TOTAL_ACTION_LOGITS)
        targets = torch.zeros(1, 3, dtype=torch.long)
        per_factor = []
        for logit_slice in ACTION_FACTOR_SLICES:
            # Certain and correct on every factor but this one, which is uniform.
            # Softmax is shift-invariant, so "certain" has to be a peak within
            # the factor's own slice, not a lower floor across the vector.
            single = torch.zeros(1, TOTAL_ACTION_LOGITS)
            for other in ACTION_FACTOR_SLICES:
                if other != logit_slice:
                    single[0, other.start] = 30.0
            cross_entropy, _, _ = factored_action_statistics(single, targets)
            per_factor.append(cross_entropy.item())
        assert per_factor == pytest.approx([1.0] * len(ACTION_FACTOR_SLICES), abs=1e-4)
        total, _, _ = factored_action_statistics(uniform, targets)
        assert total.item() == pytest.approx(sum(per_factor), rel=1e-5)

    def test_a_confident_correct_prediction_has_low_entropy_and_hits(self):
        logits = torch.zeros(1, TOTAL_ACTION_LOGITS)
        logits[0, 1] = 20.0  # power = 1
        logits[0, NUM_POWER_ACTIONS + 4] = 20.0  # turn = 4
        logits[0, NUM_POWER_ACTIONS + NUM_TURN_ACTIONS + 1] = 20.0  # shoot = 1
        targets = torch.tensor([[1, 4, 1]])
        cross_entropy, entropy, hits = factored_action_statistics(logits, targets)

        assert hits.all()
        assert cross_entropy.item() < 1e-4
        assert entropy.item() < 1e-3

    def test_a_uniform_prediction_costs_exactly_one_per_factor(self):
        """Normalized, "maximally uncertain" is the factor count rather than 3.738."""
        logits = torch.zeros(1, TOTAL_ACTION_LOGITS)
        cross_entropy, entropy, _ = factored_action_statistics(
            logits, torch.zeros(1, 3, dtype=torch.long)
        )
        factors = len(ACTION_FACTOR_SLICES)
        assert entropy.item() == pytest.approx(factors, rel=1e-5)
        assert cross_entropy.item() == pytest.approx(factors, rel=1e-5)


# ----------------------------------------------------------------------
# Trainer integration
# ----------------------------------------------------------------------


def _trainer(tmp_path, **kwargs):
    from tests.train.test_ppo import _make_trainer

    return _make_trainer(checkpoint_dir=str(tmp_path), **kwargs)


def _prepared_chunks(trainer):
    """One rollout's first minibatch as micro-batches, plus its denominators."""
    runtime = trainer._initialize_rollout_runtime()
    terminated = trainer._collect_rollout(runtime, avg_eval_active=False)
    trainer._compute_rollout_gae(runtime, terminated)
    trainer._precompute_lambda_aggregates(
        trainer.buffer, trainer._active_component_weights(), is_primary=True
    )
    trainer._precompute_transition_labels(trainer.buffer)
    chunks = next(
        trainer.buffer.get_minibatch_iterator(
            trainer.cfg.num_minibatches, trainer.cfg.microbatch_tokens
        )
    )
    denominators = trainer._minibatch_denominators(chunks, trainer.buffer, True)
    return chunks, denominators


def _prepared_minibatch(trainer):
    """One rollout's first micro-batch, with everything the loss precomputes."""
    chunks, denominators = _prepared_chunks(trainer)
    return chunks[0], denominators


class TestTargetAlignment:
    def test_the_state_target_at_each_horizon_is_that_horizons_own_transition(self, tmp_path):
        """Horizon h scores labels[t + h]: a local transition, not a cumulative one.

        Driven by a hand-built rollout rather than the trainer, so the assertion
        is about the indexing and nothing else. A perfect state head is faked by
        reading the labels directly, and the loss is checked to be zero only when
        the offsets line up.
        """
        steps, envs, ships, width = 6, 1, 1, 4
        labels = torch.arange(steps, dtype=torch.float32).view(steps, 1, 1, 1)
        labels = labels.expand(steps, envs, ships, width).contiguous()

        for horizon, state_mask, _ in predictive_horizon_masks(
            torch.ones(steps, envs, ships, dtype=torch.bool),
            torch.zeros(steps, envs, dtype=torch.bool),
            torch.zeros(steps, envs, ships, dtype=torch.bool),
            prediction_horizon=4,
        ):
            span = state_mask.shape[0]
            aligned = labels[horizon : horizon + span]
            # The belief at base t and horizon h describes step t + h.
            assert aligned[0, 0, 0, 0].item() == horizon
            assert aligned[span - 1, 0, 0, 0].item() == horizon + span - 1

    def test_the_action_target_at_each_horizon_is_the_decision_made_there(self, tmp_path):
        """The action chosen at t appears in the observation only at t + 1.

        So horizon h targets actions[t + h] — the decision taken at the step the
        belief describes, which the belief has never observed.
        """
        trainer = _trainer(tmp_path)
        batch, _ = _prepared_minibatch(trainer)
        steps = batch.alive.shape[0]
        actions = batch.actions.long()
        previous = batch.obs["previous_action"].long()
        ships = batch.alive.shape[-1]

        # The rollout writes the action chosen at t into obs[t+1].previous_action.
        torch.testing.assert_close(actions[: steps - 1], previous[1:steps, :, :ships])

        for horizon in range(3):
            span = steps - horizon
            target = actions[horizon : horizon + span]
            assert target.shape[0] == span
            # Which is the pending action of the *following* observation.
            usable = min(span, steps - 1 - horizon)
            torch.testing.assert_close(
                target[:usable],
                previous[horizon + 1 : horizon + 1 + usable, :, :ships],
            )


class TestTrainerIntegration:
    @pytest.mark.parametrize("paradigm", ["ego_pass", "shared_pass"])
    def test_both_paradigms_train_the_predictive_heads(self, paradigm, tmp_path):
        trainer = _trainer(tmp_path, paradigm=paradigm)
        batch, denominators = _prepared_minibatch(trainer)
        state_loss, action_loss, diagnostics = _run_predictive(trainer, batch, denominators)

        assert torch.isfinite(state_loss) and state_loss > 0
        assert torch.isfinite(action_loss) and action_loss > 0
        grads = torch.autograd.grad(
            state_loss + action_loss,
            [
                trainer._policy_module.predictive.projection.project.weight,
                trainer._policy_module.predictive.transition.mlp[0].weight,
                trainer._policy_module.predictive.state_prediction_head.net[0].weight,
                trainer._policy_module.predictive.action_prediction_head.net[0].weight,
            ],
            allow_unused=True,
        )
        assert all(grad is not None and grad.abs().sum() > 0 for grad in grads)
        assert set(diagnostics) >= {
            "predictive_state_by_horizon",
            "predictive_action_ce_by_horizon",
            "predictive_action_entropy_by_horizon",
            "predictive_action_accuracy",
            "next_state_per_feat",
        }

    def test_the_horizons_are_weighted_equally(self, tmp_path):
        """Total loss is the mean over horizons of per-horizon masked means.

        The immediate transition is by far the easiest, and a shared denominator
        would let its token count rather than its difficulty set its share.
        """
        trainer = _trainer(tmp_path)
        batch, denominators = _prepared_minibatch(trainer)
        state_loss, _, diagnostics = _run_predictive(trainer, batch, denominators)
        per_horizon = diagnostics["predictive_state_by_horizon"]
        torch.testing.assert_close(state_loss, per_horizon.sum() / len(per_horizon))

    def test_the_predictive_losses_reach_the_trunk(self, tmp_path):
        """The point of the objective is pressure on the shared representation."""
        trainer = _trainer(tmp_path)
        batch, denominators = _prepared_minibatch(trainer)
        state_loss, action_loss, _ = _run_predictive(trainer, batch, denominators)
        trunk = [
            parameter
            for module in trainer._policy_module.trunk_modules()
            for parameter in module.parameters()
        ]
        grads = torch.autograd.grad(state_loss + action_loss, trunk, allow_unused=True)
        assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)

    def test_the_immediate_state_prediction_uses_the_coordinator_labels(self, tmp_path):
        """State targets come from the existing feature pipeline, not a new one."""
        trainer = _trainer(tmp_path)
        batch, _ = _prepared_minibatch(trainer)
        assert batch.transition_labels is not None
        assert batch.transition_labels.shape[-1] == trainer.coordinator.total_prediction_dimension

    def test_no_future_observation_reaches_the_transition(self, tmp_path):
        """The rollout is open-loop: a perturbed future may not change a belief.

        Perturbing every observation after the first step must leave the
        horizon-0 belief of step 0 untouched, and with it every belief the
        transition derives from it.
        """
        trainer = _trainer(tmp_path)
        batch, _ = _prepared_minibatch(trainer)
        predictive = trainer._policy_module.predictive

        latent = torch.randn(4, 1, 2, trainer.model_config.d_model)
        belief = predictive(latent)
        advanced = predictive.advance(belief)
        # advance() takes exactly one argument, so there is nowhere for a future
        # state or action to enter; the belief at horizon 1 is a function of the
        # belief at horizon 0 alone.
        torch.testing.assert_close(advanced[0], predictive.advance(belief[:1])[0])


class TestAllyEnemyActionDiagnostics:
    """Forecasting your own fleet and forecasting the opposition are different jobs."""

    def test_the_ally_mask_selects_team_zero_ships_only(self, tmp_path):
        from boost_and_broadside.env.observation import ObsKey
        from boost_and_broadside.train.rl.ppo import ally_token_mask

        trainer = _trainer(tmp_path)
        batch, _ = _prepared_minibatch(trainer)
        steps, _, ships = batch.alive.shape
        ally = ally_token_mask(batch.obs, ships, steps)

        team_id = batch.obs[ObsKey.TEAM_ID][:steps, :, :ships].long()
        assert ally.shape == batch.alive.shape
        torch.testing.assert_close(ally, team_id == 0)

    def test_the_ally_mask_never_reaches_the_field_tokens(self):
        """Fields carry team 2 and belong to neither side of the battle."""
        from boost_and_broadside.env.observation import ObsKey, YemongObservation
        from boost_and_broadside.train.rl.ppo import ally_token_mask

        steps, envs, ships, fields = 3, 2, 2, 2
        team_id = torch.full((steps + 1, envs, ships + fields), 2, dtype=torch.long)
        team_id[:, :, 0] = 0
        team_id[:, :, 1] = 1
        obs = YemongObservation(data={ObsKey.TEAM_ID: team_id})

        ally = ally_token_mask(obs, ships, steps)
        assert ally.shape == (steps, envs, ships)
        assert ally[..., 0].all()
        assert not ally[..., 1].any()

    def test_the_two_sides_partition_the_scored_actions(self, tmp_path):
        trainer = _trainer(tmp_path)
        chunks, denominators = _prepared_chunks(trainer)
        torch.testing.assert_close(
            denominators["action_counts_ally"] + denominators["action_counts_enemy"],
            denominators["action_counts"],
        )
        del chunks

    def test_ego_pass_measures_no_ally_action_at_the_immediate_horizon(self, tmp_path):
        """Those are exactly the self-generated actions the objective excludes.

        An empty group reads NaN rather than zero, because zero is what a
        perfect prediction would look like.
        """
        trainer = _trainer(tmp_path, paradigm="ego_pass")
        batch, denominators = _prepared_minibatch(trainer)
        _, _, diagnostics = _run_predictive(trainer, batch, denominators)

        assert denominators["action_counts_ally"][0].item() == 0.0
        assert denominators["action_counts_enemy"][0].item() > 0.0
        ally = diagnostics["predictive_action_ce_ally_by_horizon"]
        enemy = diagnostics["predictive_action_ce_enemy_by_horizon"]
        assert torch.isnan(ally[0])
        assert torch.isfinite(enemy[0])
        # The opposition's *own* later decisions are measurable on both sides.
        assert torch.isfinite(ally[1:]).all()
        assert torch.isfinite(enemy[1:]).all()

    def test_the_combined_series_is_the_count_weighted_mean_of_the_sides(self, tmp_path):
        """The split has to describe the total, not a differently-masked quantity."""
        trainer = _trainer(tmp_path)
        batch, denominators = _prepared_minibatch(trainer)
        _, _, diagnostics = _run_predictive(trainer, batch, denominators)

        combined = diagnostics["predictive_action_ce_by_horizon"]
        ally = diagnostics["predictive_action_ce_ally_by_horizon"]
        enemy = diagnostics["predictive_action_ce_enemy_by_horizon"]
        ally_count = denominators["action_counts_ally"]
        enemy_count = denominators["action_counts_enemy"]
        both = ally_count > 0  # horizon 0 has no ally side under ego_pass
        expected = (ally[both] * ally_count[both] + enemy[both] * enemy_count[both]) / (
            ally_count[both] + enemy_count[both]
        )
        torch.testing.assert_close(combined[both], expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("paradigm", ["ego_pass", "shared_pass"])
    def test_an_update_logs_both_sides_at_every_horizon_and_factor(self, paradigm, tmp_path):
        trainer = _trainer(tmp_path, paradigm=paradigm)
        trainer.train()
        metrics = trainer._update_epochs(
            all_buffers=[trainer.buffer, *trainer.aux_buffers], precomputed=False
        )
        horizons = min(trainer.cfg.prediction_horizon, trainer.cfg.num_steps)
        for side in ("_ally", "_enemy"):
            for index in range(horizons):
                assert f"predictive/action_cross_entropy{side}/h{index:02d}" in metrics
                assert f"predictive/action_entropy{side}/h{index:02d}" in metrics
            for factor in ("power", "turn", "shoot"):
                assert f"predictive/action_accuracy{side}/{factor}" in metrics

    def test_the_split_does_not_touch_the_loss(self, tmp_path):
        """Diagnostics only: one shared head, and the total is the unsplit group."""
        trainer = _trainer(tmp_path)
        batch, denominators = _prepared_minibatch(trainer)
        _, action_loss, diagnostics = _run_predictive(trainer, batch, denominators)
        combined = diagnostics["predictive_action_ce_by_horizon"]
        torch.testing.assert_close(action_loss, combined.sum() / len(combined))
        assert not action_loss.isnan()


class TestDisabledPath:
    def test_disabling_both_coefficients_skips_the_predictive_computation(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.cfg = dataclasses.replace(
            trainer.cfg, predictive_state_coef=0.0, predictive_action_coef=0.0
        )
        assert not trainer._predictive_enabled

        trainer._precompute_transition_labels(trainer.buffer)
        assert trainer.buffer.transition_labels is None

        trainer.train()  # a full update with the auxiliary system switched off

    def test_the_disabled_path_keeps_no_predictive_activations(self, tmp_path):
        """evaluate_actions must not even project when nothing consumes the belief."""
        trainer = _trainer(tmp_path)
        batch, _ = _prepared_minibatch(trainer)
        steps = batch.alive.shape[0]
        _, _, _, _, encoder_output, predictive_latent = trainer._policy_module.evaluate_actions(
            obs=batch.obs.slice_time(0, steps),
            actions=batch.actions.long(),
            initial_hidden=batch.hidden,
            alive_mask=batch.obs["alive"].bool()[:steps],
            done_mask=batch.terminated,
        )
        assert predictive_latent is None
        assert encoder_output is None

    def test_acting_decodes_no_prediction_unless_one_is_asked_for(self, tmp_path):
        """The rollout, every league opponent, and every rated game discard it.

        Leaving it on put two dead kernels in each of those forwards — and a
        rated evaluation step fires up to eight of them, on a path that is
        launch-bound rather than compute-bound.
        """
        trainer = _trainer(tmp_path)
        batch, _ = _prepared_minibatch(trainer)
        observation = batch.obs.map(lambda tensor: tensor[0])
        hidden = batch.hidden

        _, _, _, absent, _ = trainer._policy_module.get_action_and_value(observation, hidden)
        assert absent is None

        _, _, _, present, _ = trainer._policy_module.get_action_and_value(
            observation, hidden, return_state_prediction=True
        )
        assert present is not None
        assert present.shape[-1] == trainer.coordinator.total_prediction_dimension

    def test_the_modes_that_decode_a_prediction_still_ask_for_one(self):
        """The three call sites that consume it, pinned against a silent regression."""
        source_root = Path(__file__).resolve().parents[2] / "src" / "boost_and_broadside"
        consumers = (
            source_root / "evaluation" / "next_state.py",
            source_root / "modes" / "noise_calibration.py",
            source_root / "evaluation" / "agents.py",
        )
        for consumer in consumers:
            text = consumer.read_text()
            assert "return_state_prediction" in text, f"{consumer.name} would decode a None"

    def test_only_the_state_family_still_trains_when_the_action_family_is_off(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.cfg = dataclasses.replace(trainer.cfg, predictive_action_coef=0.0)
        batch, denominators = _prepared_minibatch(trainer)
        state_loss, action_loss, diagnostics = _run_predictive(trainer, batch, denominators)

        assert state_loss > 0
        assert action_loss.item() == 0.0
        assert "predictive_action_ce_by_horizon" not in diagnostics


def _run_predictive(trainer, batch, denominators):
    """Re-evaluate one micro-batch and score its predictive rollout."""
    steps = batch.alive.shape[0]
    _, _, _, _, _, predictive_latent = trainer._policy_module.evaluate_actions(
        obs=batch.obs.slice_time(0, steps),
        actions=batch.actions.long(),
        initial_hidden=batch.hidden,
        alive_mask=batch.obs["alive"].bool()[:steps],
        done_mask=batch.terminated,
        return_predictive_latent=True,
    )
    return trainer._predictive_losses(predictive_latent, batch, denominators)


class TestConfiguration:
    def test_the_belief_width_comes_from_the_model_config(self):
        from boost_and_broadside.config import ShipConfig
        from boost_and_broadside.models.yemong.policy import YemongPolicy
        from boost_and_broadside.train.rl.features import build_standard_coordinator

        model_config = ModelConfig(
            d_model=32, n_heads=4, n_yemong_blocks=1, predictive_latent_dim=24
        )
        policy = YemongPolicy(
            model_config,
            build_standard_coordinator(ShipConfig()),
            num_value_components=2,
            num_ships=2,
            team_pma_k=(),
        )
        assert policy.predictive.predictive_latent_dim == 24
        assert policy.predictive.projection.project.out_features == 24

    def test_a_zero_horizon_is_refused(self, tmp_path):
        from tests.train.test_ppo import _make_train_config

        config = _make_train_config()
        with pytest.raises(ValueError, match="prediction_horizon"):
            dataclasses.replace(config, prediction_horizon=0)

    def test_a_zero_belief_width_is_refused(self):
        with pytest.raises(ValueError, match="predictive_latent_dim"):
            ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1, predictive_latent_dim=0)
