"""The global density auxiliary head: what it reads, what it is graded against.

The grid and the target's exactness are covered by ``test_hex_density.py``. What
is at stake here is the wiring: the head sees the global token and nothing else,
its target is privileged truth the observation does not carry, and the loss
decomposes over micro-batches the way every other masked-mean term does.
"""

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import ObsKey, YemongObservation, observation_from_state
from boost_and_broadside.models.yemong.policy import GlobalDensityHead
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.hex_density import (
    HEX_DENSITY_DIM,
    HexDensityTarget,
    hex_cell_count,
)
from boost_and_broadside.train.rl.policy_io import build_policy
from boost_and_broadside.train.rl.ppo import PPOTrainer

SHIP_CONFIG = PROFILES["rl"].ship_config
NUM_ENVS = 3
NUM_SHIPS = 4
NUM_FIELDS = 2
CPU = torch.device("cpu")


def _env_config() -> EnvConfig:
    profile = PROFILES["rl"]
    return EnvConfig(
        num_ships=NUM_SHIPS,
        num_fields=NUM_FIELDS,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        frontline=profile.frontline,
        vision_range=profile.vision_range,
    )


def _state(seed: int = 11):
    config = _env_config()
    env = TensorEnv(NUM_ENVS, SHIP_CONFIG, config, "cpu")
    env.reset(seed=seed)
    return env.state


def _blind_view(state) -> YemongObservation:
    """Team 0's view with every enemy unseen and never remembered."""

    visible = state.ship_team_id == 0  # (B, N) bool
    return observation_from_state(state, SHIP_CONFIG, ship_visibility=visible, perspective_team=0)


def _target(state, observer_team: int = 0) -> torch.Tensor:
    builder = HexDensityTarget(SHIP_CONFIG.world_size, CPU, None)
    team = torch.full((state.num_envs,), observer_team, dtype=torch.int32)
    return builder(state, team)


def _policy(predict_density: bool = True, global_token: bool = True):
    torch.manual_seed(0)
    policy = build_policy(
        replace(MODEL_CONFIG, global_token=global_token),
        SHIP_CONFIG,
        num_value_components=4,
        num_ships=NUM_SHIPS,
        global_value_k=(0,),
        predict_density=predict_density,
    )
    return policy.eval()


def _sequence(observation: YemongObservation) -> YemongObservation:
    """One timestep of observation, shaped as the update's (T, B, ...) input."""

    return YemongObservation(data={key: value[None] for key, value in observation.items()})


def _evaluate(policy, observation: YemongObservation):
    sequence = _sequence(observation)
    with torch.no_grad():
        return policy.evaluate_actions(
            sequence,
            torch.zeros(1, NUM_ENVS, NUM_SHIPS, 3, dtype=torch.long),
            policy.initial_hidden(NUM_ENVS, policy.num_recurrent_tokens, CPU),
            sequence[ObsKey.BELIEF_VALID],
            return_density=True,
        )


# ---------------------------------------------------------------------------
# What the head reads
# ---------------------------------------------------------------------------


class TestHead:
    def test_the_head_is_as_wide_as_the_grid(self):
        assert GlobalDensityHead(16).out_dim == 2 * hex_cell_count(10) == HEX_DENSITY_DIM

    def test_the_prediction_covers_both_halves_of_one_field_per_step(self):
        density = _evaluate(_policy(), _blind_view(_state()))[-1]
        assert density.shape == (1, NUM_ENVS, HEX_DENSITY_DIM)

    def test_it_reads_a_token_with_no_ship_axis(self):
        """A ship head is handed (T, B, N, D); this one gets the global token
        alone, so its input has no ship axis to average a fleet away in."""
        policy = _policy()
        seen: list[torch.Size] = []
        policy.density_head.register_forward_hook(lambda _m, args, _out: seen.append(args[0].shape))
        _evaluate(policy, _blind_view(_state()))
        assert seen == [torch.Size([1, NUM_ENVS, MODEL_CONFIG.d_model])]

    def test_a_policy_without_the_head_predicts_nothing(self):
        assert _evaluate(_policy(predict_density=False), _blind_view(_state()))[-1] is None

    def test_the_head_cannot_be_built_without_a_global_token(self):
        """Off, the token is K/V-only map memory and no final embedding for it
        leaves the trunk, so there is nothing for this head to read."""
        with pytest.raises(ValueError, match="global_token"):
            _policy(global_token=False)


# ---------------------------------------------------------------------------
# Target and view: privileged target, legal input
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_the_first_half_is_the_observer_s_own_team(self):
        state = _state()
        half = HEX_DENSITY_DIM // 2
        team_0 = _target(state, observer_team=0)
        team_1 = _target(state, observer_team=1)
        assert torch.allclose(team_0[:, :half], team_1[:, half:], atol=1e-6)
        assert torch.allclose(team_0[:, half:], team_1[:, :half], atol=1e-6)

    def test_the_ally_half_peaks_where_the_observer_s_ships_are(self):
        """Alignment between the target's cell order and the world the
        observation describes: the busiest ally cell is nearer to team 0's ships
        than to team 1's."""
        state = _state()
        builder = HexDensityTarget(SHIP_CONFIG.world_size, CPU, None)
        cells = builder.unit_cells * state.playable_boundary_radius[0]
        cells = cells + torch.stack([state.map_center.real[0], state.map_center.imag[0]])
        ally = _target(state)[0, : HEX_DENSITY_DIM // 2]
        peak = cells[int(ally.argmax())]

        position = torch.stack([state.ship_pos[0].real, state.ship_pos[0].imag], dim=-1)
        distance = (position - peak).norm(dim=-1)
        team = state.ship_team_id[0]
        assert distance[team == 0].min() < distance[team == 1].min()

    def test_a_hidden_enemy_counts_in_the_target_and_not_in_the_view(self):
        """The target is privileged and the input is not. Moving an enemy the
        observer cannot see changes what the head is graded against while
        leaving every channel it reads bit-identical."""
        state = _state()
        enemy = int((state.ship_team_id[0] == 1).nonzero()[0])
        before_view = _blind_view(state)
        before_target = _target(state)

        moved = state.ship_pos.clone()
        moved[0, enemy] += complex(900.0, -700.0)
        state.ship_pos = moved
        after_view = _blind_view(state)
        after_target = _target(state)

        for key, value in before_view.items():
            assert torch.equal(value, after_view[key]), key
        half = HEX_DENSITY_DIM // 2
        assert not torch.allclose(before_target[0, half:], after_target[0, half:])
        assert torch.allclose(before_target[0, :half], after_target[0, :half], atol=1e-6)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def _stats(prediction: torch.Tensor, target: torch.Tensor):
    samples = float(prediction.shape[0] * prediction.shape[1])
    return PPOTrainer._global_density_loss(None, prediction, target, samples)


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    stats = _stats(prediction, target)
    return float(stats.loss), float(stats.deviance)


class TestLoss:
    """Poisson negative log likelihood of a soft count field."""

    def test_a_perfect_rate_costs_no_deviance(self):
        """The likelihood itself is not zero there -- it depends on the counts.
        The deviance subtracts that floor, so *it* is zero and the loss is not."""
        target = torch.randint(0, 3, (4, 2, HEX_DENSITY_DIM)).float()
        stats = _stats(target.clamp(min=1e-12).log(), target)
        assert float(stats.deviance) == pytest.approx(0.0, abs=1e-5)
        assert float(stats.loss) != pytest.approx(0.0, abs=1e-3)

    def test_deviance_is_non_negative_for_any_prediction(self):
        """It is a likelihood ratio against the best possible fit, so nothing
        the head can emit drives it below zero."""
        torch.manual_seed(0)
        target = torch.randint(0, 4, (4, 3, HEX_DENSITY_DIM)).float()
        for scale in (-3.0, -1.0, 0.0, 1.0):
            stats = _stats(torch.full_like(target, scale), target)
            assert float(stats.deviance) >= -1e-6

    def test_the_gradient_in_the_logit_is_rate_minus_count(self):
        """The whole reason this is Poisson: the gradient is bounded and it
        vanishes exactly when the rate matches the count, rather than growing
        as the head gets more certain the way a Gaussian's r/sigma^2 does."""
        target = torch.tensor([[[0.0, 1.0, 2.0, 0.5]]])
        logit = torch.tensor([[[-1.0, 0.5, 0.0, -2.0]]], requires_grad=True)
        samples = 1.0
        PPOTrainer._global_density_loss(None, logit, target, samples).loss.backward()
        expected = (logit.exp() - target) / target.shape[-1]
        assert torch.allclose(logit.grad, expected, atol=1e-6)

    def test_an_empty_cell_still_penalises_claimed_rate(self):
        """``y = 0`` contributes ``exp(l)``: the head pays for putting ships
        where there are none, which is what stops it hedging everywhere."""
        target = torch.zeros(1, 1, HEX_DENSITY_DIM)
        quiet = _stats(torch.full_like(target, -5.0), target)
        loud = _stats(torch.full_like(target, 0.0), target)
        assert float(loud.loss) > float(quiet.loss)

    def test_micro_batch_contributions_sum_to_the_whole(self):
        """The denominator is the minibatch's step-environment total, so
        splitting the environments and adding the pieces is exact -- which is
        what makes gradient accumulation equivalent to one large minibatch."""
        target = torch.randint(0, 3, (4, 6, HEX_DENSITY_DIM)).float()
        prediction = torch.randn_like(target)
        samples = float(target.shape[0] * target.shape[1])
        whole = float(PPOTrainer._global_density_loss(None, prediction, target, samples).loss)
        parts = sum(
            float(
                PPOTrainer._global_density_loss(
                    None, prediction[:, start : start + 2], target[:, start : start + 2], samples
                ).loss
            )
            for start in (0, 2, 4)
        )
        assert parts == pytest.approx(whole, rel=1e-6)

    def test_the_gradient_reaches_the_trunk(self):
        """The head exists to shape the trunk; a gradient that stopped at its own
        first Linear would buy nothing."""
        policy = _policy()
        observation = _blind_view(_state())
        sequence = _sequence(observation)
        density = policy.evaluate_actions(
            sequence,
            torch.zeros(1, NUM_ENVS, NUM_SHIPS, 3, dtype=torch.long),
            policy.initial_hidden(NUM_ENVS, policy.num_recurrent_tokens, CPU),
            sequence[ObsKey.BELIEF_VALID],
            return_density=True,
        )[-1]
        target = _target(_state()).unsqueeze(0)
        PPOTrainer._global_density_loss(None, density, target, float(NUM_ENVS)).loss.backward()
        reached = [
            parameter
            for module in policy.trunk_modules()
            for parameter in module.parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0.0
        ]
        assert reached

    def test_the_head_opens_at_the_fleet_s_true_mean_rate(self):
        """A zero bias would claim one ship per cell, two orders of magnitude
        too crowded, and the first updates would go on removing the excess
        rather than on learning where anything is."""
        policy = _policy()
        cells = HEX_DENSITY_DIM / 2
        with torch.no_grad():
            opening = policy.density_head(torch.zeros(1, MODEL_CONFIG.d_model))
        assert float(opening.exp().sum()) == pytest.approx(NUM_SHIPS, rel=0.25)
        assert policy.density_head.init_log_rate == pytest.approx(math.log(NUM_SHIPS / 2.0 / cells))

    def test_a_missing_head_is_an_error_rather_than_a_silent_zero(self):
        with pytest.raises(ValueError, match="density"):
            PPOTrainer._global_density_loss(None, None, torch.zeros(1, 1, 2), 1.0)


class TestHalves:
    """Ally cells and enemy cells are not the same problem.

    The observer sees every ally, so that half of the field is a smoothing of
    what it already knows; the enemy half is mostly belief. One mean over all
    ``2C`` cells lets the easy half carry the hard one, which is exactly the
    reading the aggregate series cannot rule out.
    """

    def test_the_training_term_is_the_mean_of_the_two_halves(self):
        """The split is a decomposition, not a second opinion: whatever is
        logged per half has to add back to the number being optimized."""
        target = torch.randint(0, 3, (4, 3, HEX_DENSITY_DIM)).float()
        stats = _stats(torch.randn_like(target), target)
        assert float(stats.loss) == pytest.approx(
            0.5 * (float(stats.ally_loss) + float(stats.enemy_loss)), rel=1e-6
        )
        assert float(stats.deviance) == pytest.approx(
            0.5 * (float(stats.ally_deviance) + float(stats.enemy_deviance)), rel=1e-6
        )

    def test_each_half_scores_only_its_own_cells(self):
        """Ally cells come first, enemy cells second."""
        cells = HEX_DENSITY_DIM // 2
        target = torch.zeros(2, 2, HEX_DENSITY_DIM)
        prediction = torch.full_like(target, -20.0)  # rate ~0 everywhere: correct
        prediction[..., :cells] = 0.0  # rate 1 on every ally cell: wrong

        stats = _stats(prediction, target)

        assert float(stats.ally_loss) == pytest.approx(1.0, rel=1e-5)
        assert float(stats.enemy_loss) == pytest.approx(0.0, abs=1e-8)

    def test_an_easy_half_cannot_hide_a_hard_one(self):
        """The regression this split exists to catch: a head exact on allies and
        useless on enemies reads as half-deviance in the aggregate and as the
        truth in the halves."""
        cells = HEX_DENSITY_DIM // 2
        target = torch.randint(0, 3, (4, 3, HEX_DENSITY_DIM)).float()
        prediction = target.clamp(min=1e-12).log()  # exact
        prediction[..., cells:] = -20.0  # claims the enemy half is empty

        stats = _stats(prediction, target)

        assert float(stats.ally_deviance) == pytest.approx(0.0, abs=1e-5)
        assert float(stats.enemy_deviance) > 0.5
        assert float(stats.deviance) == pytest.approx(0.5 * float(stats.enemy_deviance), rel=1e-5)

    def test_the_halves_are_additive_across_micro_batches_too(self):
        target = torch.randint(0, 3, (4, 6, HEX_DENSITY_DIM)).float()
        prediction = torch.randn_like(target)
        samples = float(target.shape[0] * target.shape[1])
        whole = PPOTrainer._global_density_loss(None, prediction, target, samples)
        parts = [
            PPOTrainer._global_density_loss(
                None, prediction[:, start : start + 2], target[:, start : start + 2], samples
            )
            for start in (0, 2, 4)
        ]
        for field in ("ally_loss", "enemy_loss", "ally_deviance", "enemy_deviance"):
            assert sum(float(getattr(part, field)) for part in parts) == pytest.approx(
                float(getattr(whole, field)), rel=1e-6
            )
