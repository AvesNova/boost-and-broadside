"""Invariants of the spatial geometry fed to attention: rotary Q/K encoding.

The properties pinned here are the reasons the mechanism exists at all. If the
rotary frequencies stop matching the encoder's Fourier basis, or a rotation
stops being exactly periodic over the toroid, the policy keeps running and
quietly measures something else.
"""

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config.core import EnvConfig, ModelConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG
from boost_and_broadside.models.yemong.rope import (
    RotaryBudgetError,
    SpatialRotary,
    apply_rotary,
    check_rotary_budget,
    rotary_pair_count,
    spatial_rotary_axes,
)
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.checkpoint_schema import (
    ATTITUDE_FOURIER_FREQUENCIES,
    base2_frequencies,
    position_fourier_frequencies,
)
from boost_and_broadside.train.rl.features import AttitudeFourier, Fourier
from boost_and_broadside.train.rl.policy_io import build_policy

FRONTLINE_SHIP_CONFIG = PROFILES["rl"].ship_config
ROPE_MODEL_CONFIG = replace(MODEL_CONFIG, n_spatial_heads=2, spatial_rope=True)


@pytest.fixture
def rotary() -> SpatialRotary:
    return SpatialRotary(FRONTLINE_SHIP_CONFIG, head_dim=ROPE_MODEL_CONFIG.spatial_head_dim)


def _env_config(num_ships: int = 10, num_fields: int = 10) -> EnvConfig:
    profile = PROFILES["rl"]
    return EnvConfig(
        num_ships=num_ships,
        num_fields=num_fields,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        frontline=profile.frontline,
        vision_range=profile.vision_range,
    )


# ---------------------------------------------------------------------------
# The basis is the encoder's basis
# ---------------------------------------------------------------------------


class TestReusedFourierBasis:
    def test_axes_match_the_feature_pipeline_frequency_counts(self):
        width, height = FRONTLINE_SHIP_CONFIG.world_size
        axis_x, axis_y, axis_att = spatial_rotary_axes(FRONTLINE_SHIP_CONFIG)

        assert axis_x.n_freqs == position_fourier_frequencies(width)
        assert axis_y.n_freqs == position_fourier_frequencies(height)
        assert axis_att.n_freqs == ATTITUDE_FOURIER_FREQUENCIES
        assert axis_att.n_freqs == AttitudeFourier().n_freqs
        assert (axis_x.period, axis_y.period) == (width, height)
        assert axis_att.period == pytest.approx(2.0 * math.pi)

    def test_rotary_frequencies_are_the_encoder_input_frequencies(self, rotary):
        """The angles RoPE rotates by are the angles ``Fourier`` already builds.

        Checked against the transform's own device-side construction rather than
        against a literal, so this fails if either definition moves.
        """
        width, height = FRONTLINE_SHIP_CONFIG.world_size
        position = Fourier(position_fourier_frequencies(width), periods=width)
        attitude = AttitudeFourier()
        probe = torch.zeros(1, 1, 1)

        expected = (
            list(position._frequencies(width, probe))
            + list(Fourier(position_fourier_frequencies(height), periods=height)._frequencies(
                height, probe
            ))
            + list(attitude._frequencies(2.0 * math.pi, probe))
        )
        assert torch.allclose(rotary.frequencies, torch.tensor(expected))

    def test_position_frequencies_keep_the_declared_finest_period(self):
        width, _ = FRONTLINE_SHIP_CONFIG.world_size
        frequencies = base2_frequencies(width, position_fourier_frequencies(width))
        finest_period = 2.0 * math.pi / frequencies[-1]
        assert finest_period <= 128.0
        assert finest_period > 64.0  # one fewer frequency would already be too coarse


# ---------------------------------------------------------------------------
# Periodicity and relative geometry
# ---------------------------------------------------------------------------


def _random_qk(pairs: int, head_dim: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, 2, head_dim, generator=generator)
    k = torch.randn(1, 1, 2, head_dim, generator=generator)
    return q, k


def _rotated_score(rotary, q, k, pos_q, pos_k, att_q=None, att_k=None):
    """Dot product of one rotated query against one rotated key."""
    cos_q, sin_q = rotary.tables(pos_q, att_q)
    cos_k, sin_k = rotary.tables(pos_k, att_k)
    return (apply_rotary(q, cos_q, sin_q) * apply_rotary(k, cos_k, sin_k)).sum(-1)


class TestRotaryPeriodicity:
    def test_a_full_world_translation_leaves_the_tables_unchanged(self, rotary):
        width, height = FRONTLINE_SHIP_CONFIG.world_size
        position = torch.tensor([[[137.0, 9001.0], [16000.0, 12.0]]])
        shifted = position + torch.tensor([width, height])

        base = rotary.tables(position, None)
        wrapped = rotary.tables(shifted, None)
        assert torch.allclose(base[0], wrapped[0], atol=1e-4)
        assert torch.allclose(base[1], wrapped[1], atol=1e-4)

    def test_a_full_turn_leaves_the_attitude_rotation_unchanged(self, rotary):
        position = torch.zeros(1, 3, 2)
        angle = torch.tensor([[0.3, -2.1, 3.0]])
        attitude = torch.stack([angle.cos(), angle.sin()], dim=-1)
        turned = angle + 2.0 * math.pi
        attitude_turned = torch.stack([turned.cos(), turned.sin()], dim=-1)

        base = rotary.tables(position, attitude)
        after = rotary.tables(position, attitude_turned)
        assert torch.allclose(base[0], after[0], atol=1e-5)
        assert torch.allclose(base[1], after[1], atol=1e-5)

    def test_a_missing_attitude_is_the_identity_rotation(self, rotary):
        """``ATT = (0, 0)`` and "no attitude" must mean the same thing.

        Map objects carry a zero attitude channel and bullets carry none at all;
        both have to land on the same rotation or a ship's query would compare
        differently against two tokens that are equally heading-less.
        """
        position = torch.tensor([[[400.0, 900.0]]])
        zero_attitude = torch.zeros(1, 1, 2)
        with_zero = rotary.tables(position, zero_attitude)
        without = rotary.tables(position, None)
        assert torch.allclose(with_zero[0], without[0])
        assert torch.allclose(with_zero[1], without[1])

    def test_unrotated_head_dimensions_pass_through(self, rotary):
        q, _ = _random_qk(rotary.pairs, rotary.head_dim)
        cos, sin = rotary.tables(torch.tensor([[[123.0, 456.0]]]), None)
        out = apply_rotary(q, cos, sin)
        assert torch.allclose(out[..., rotary.rotary_dim :], q[..., rotary.rotary_dim :])
        assert not torch.allclose(out[..., : rotary.rotary_dim], q[..., : rotary.rotary_dim])

    def test_the_rotation_preserves_norms(self, rotary):
        q, _ = _random_qk(rotary.pairs, rotary.head_dim)
        cos, sin = rotary.tables(torch.tensor([[[7.0, -3.0]]]), None)
        rotated = apply_rotary(q, cos, sin)
        assert torch.allclose(rotated.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


class TestRelativeGeometry:
    @pytest.mark.parametrize("offset", [0.0, 313.0, 4096.0, -2500.0])
    def test_the_score_depends_only_on_displacement(self, rotary, offset):
        """Translating query and key together must not change their score."""
        q, k = _random_qk(rotary.pairs, rotary.head_dim, seed=1)
        pos_q = torch.tensor([[[1000.0, 2000.0]]])
        pos_k = torch.tensor([[[1450.0, 1700.0]]])
        shift = torch.tensor([offset, -offset])

        base = _rotated_score(rotary, q, k, pos_q, pos_k)
        moved = _rotated_score(rotary, q, k, pos_q + shift, pos_k + shift)
        assert torch.allclose(base, moved, atol=1e-3)

    def test_wraparound_matches_the_minimum_image_displacement(self, rotary):
        """A pair straddling the seam scores as the short way round, not the long."""
        width, _ = FRONTLINE_SHIP_CONFIG.world_size
        q, k = _random_qk(rotary.pairs, rotary.head_dim, seed=2)
        near_seam_q = torch.tensor([[[width - 50.0, 500.0]]])
        near_seam_k = torch.tensor([[[30.0, 500.0]]])  # 80 px away across the seam
        # The same 80 px separation with no seam between them.
        interior_q = torch.tensor([[[8000.0, 500.0]]])
        interior_k = torch.tensor([[[8080.0, 500.0]]])

        assert torch.allclose(
            _rotated_score(rotary, q, k, near_seam_q, near_seam_k),
            _rotated_score(rotary, q, k, interior_q, interior_k),
            atol=1e-3,
        )

    def test_relative_heading_governs_the_attitude_block(self, rotary):
        """Turning both ships by the same angle leaves the score unchanged."""
        q, k = _random_qk(rotary.pairs, rotary.head_dim, seed=3)
        position = torch.tensor([[[10.0, 20.0]]])
        other = torch.tensor([[[110.0, 20.0]]])

        def attitude(angle: float) -> torch.Tensor:
            return torch.tensor([[[math.cos(angle), math.sin(angle)]]])

        base = _rotated_score(rotary, q, k, position, other, attitude(0.4), attitude(1.1))
        turned = _rotated_score(rotary, q, k, position, other, attitude(1.4), attitude(2.1))
        assert torch.allclose(base, turned, atol=1e-4)


# ---------------------------------------------------------------------------
# Dimension budget
# ---------------------------------------------------------------------------


class TestRotaryBudget:
    def test_the_frontline_world_fits_a_64_wide_head_and_not_a_32_wide_one(self):
        assert rotary_pair_count(FRONTLINE_SHIP_CONFIG) == 20  # 8 + 8 + 4
        check_rotary_budget(ROPE_MODEL_CONFIG, FRONTLINE_SHIP_CONFIG)
        with pytest.raises(RotaryBudgetError):
            check_rotary_budget(
                replace(ROPE_MODEL_CONFIG, n_spatial_heads=4), FRONTLINE_SHIP_CONFIG
            )

    def test_an_oversized_world_is_rejected_rather_than_truncated(self):
        """A world needing more than 64 rotary dims must stop, not drop frequencies."""
        huge = replace(FRONTLINE_SHIP_CONFIG, world_size=(2**24, 2**24))
        assert 2 * rotary_pair_count(huge) > ROPE_MODEL_CONFIG.spatial_head_dim
        with pytest.raises(RotaryBudgetError, match="head dimensions"):
            check_rotary_budget(ROPE_MODEL_CONFIG, huge)

    def test_a_policy_cannot_be_built_past_the_budget(self):
        with pytest.raises(RotaryBudgetError):
            build_policy(
                replace(MODEL_CONFIG, spatial_rope=True),  # four 32-wide heads
                FRONTLINE_SHIP_CONFIG,
                num_value_components=4,
                num_ships=10,
                team_pma_k=(),
            )

    def test_rope_requires_a_ship_config(self):
        from boost_and_broadside.models.yemong.policy import YemongPolicy
        from boost_and_broadside.train.rl.features import build_standard_coordinator

        with pytest.raises(ValueError, match="ship_config"):
            YemongPolicy(
                ROPE_MODEL_CONFIG,
                build_standard_coordinator(FRONTLINE_SHIP_CONFIG),
                num_value_components=4,
                num_ships=10,
                team_pma_k=(),
            )


# ---------------------------------------------------------------------------
# Head layout
# ---------------------------------------------------------------------------


class TestSpatialHeadLayout:
    def test_spatial_heads_default_to_the_shared_head_count(self):
        assert MODEL_CONFIG.n_spatial_heads is None
        assert MODEL_CONFIG.spatial_heads == MODEL_CONFIG.n_heads
        assert MODEL_CONFIG.spatial_head_dim == MODEL_CONFIG.d_model // MODEL_CONFIG.n_heads

    def test_two_wide_heads_leave_the_pooling_attention_alone(self):
        config = replace(MODEL_CONFIG, n_spatial_heads=2)
        policy = build_policy(
            config,
            FRONTLINE_SHIP_CONFIG,
            num_value_components=4,
            num_ships=10,
            team_pma_k=(0,),
        )
        spatial = policy.yemong_layers[0].spatial[0]
        assert (spatial.n_heads, spatial.head_dim) == (2, 64)
        assert policy.team_pma.attn.num_heads == config.n_heads == 4

    def test_head_layout_does_not_change_the_parameter_count(self):
        """2x64 and 4x32 are the same weights read differently."""

        def parameters(config: ModelConfig) -> int:
            policy = build_policy(
                config,
                FRONTLINE_SHIP_CONFIG,
                num_value_components=4,
                num_ships=10,
                team_pma_k=(0,),
            )
            return sum(p.numel() for p in policy.parameters())

        assert parameters(replace(MODEL_CONFIG, n_spatial_heads=2)) == parameters(MODEL_CONFIG)

    def test_a_spatial_head_count_must_divide_d_model(self):
        with pytest.raises(ValueError, match="n_spatial_heads"):
            replace(MODEL_CONFIG, n_spatial_heads=5)
        with pytest.raises(ValueError, match="n_spatial_heads"):
            replace(MODEL_CONFIG, n_spatial_heads=0)


# ---------------------------------------------------------------------------
# End-to-end behaviour through the policy
# ---------------------------------------------------------------------------


def _policy(model_config: ModelConfig, num_ships: int = 10):
    torch.manual_seed(0)
    return build_policy(
        model_config,
        FRONTLINE_SHIP_CONFIG,
        num_value_components=4,
        num_ships=num_ships,
        team_pma_k=(0,),
    )


def _observation(num_ships: int = 10, num_fields: int = 10, seed: int = 5):
    from boost_and_broadside.env.env import TensorEnv
    from boost_and_broadside.env.observation import perceived_observation_from_state

    env = TensorEnv(3, FRONTLINE_SHIP_CONFIG, _env_config(num_ships, num_fields), "cpu")
    env.reset(seed=seed)
    observation, _ = perceived_observation_from_state(
        env.state, FRONTLINE_SHIP_CONFIG, _env_config(num_ships, num_fields)
    )
    return observation.for_team(0)


class TestRotaryPolicy:
    def test_rotation_changes_the_policy_output(self):
        """A guard against the tables being built and then ignored."""
        plain = _policy(replace(MODEL_CONFIG, n_spatial_heads=2))
        rotated = _policy(ROPE_MODEL_CONFIG)
        rotated.load_state_dict(plain.state_dict())  # identical weights

        observation = _observation()
        hidden = plain.initial_hidden(3, 10, torch.device("cpu"))
        _, _, plain_value, _, _ = plain.get_action_and_value(observation, hidden)
        _, _, rotated_value, _, _ = rotated.get_action_and_value(observation, hidden)
        assert not torch.allclose(plain_value, rotated_value)

    def test_rotation_adds_no_parameters_or_state_dict_keys(self):
        """So a rotated run's checkpoint stays loadable by shape alone."""
        plain = _policy(replace(MODEL_CONFIG, n_spatial_heads=2))
        rotated = _policy(ROPE_MODEL_CONFIG)
        assert set(plain.state_dict()) == set(rotated.state_dict())

    @pytest.mark.parametrize("num_ships", [2, 10, 40])
    def test_runs_at_unseen_fleet_sizes(self, num_ships):
        policy = _policy(ROPE_MODEL_CONFIG, num_ships=num_ships)
        observation = _observation(num_ships=num_ships)
        hidden = policy.initial_hidden(3, num_ships, torch.device("cpu"))
        action, _, value, _, _ = policy.get_action_and_value(observation, hidden)
        assert action.shape == (3, num_ships, 3)
        assert torch.isfinite(value).all()

    def test_dead_entities_still_cannot_influence_the_living(self):
        policy = _policy(ROPE_MODEL_CONFIG)
        observation = _observation()
        hidden = policy.initial_hidden(3, 10, torch.device("cpu"))

        valid = observation["belief_valid"].clone()
        valid[:, 5:8] = False
        masked = observation.update("belief_valid", valid)
        _, _, base_value, _, _ = policy.get_action_and_value(masked, hidden)

        # Move the masked-out tokens far away; the rotation of a dead key must
        # not reach a living query. Their own outputs are expected to move --
        # a masked token is excluded as a *key*, not deleted -- so the assertion
        # is over the tokens that stayed alive.
        position = masked["pos"].clone()
        position[:, 5:8] += 3333.0
        moved = masked.update("pos", position)
        _, _, moved_value, _, _ = policy.get_action_and_value(moved, hidden)

        living = torch.tensor([0, 1, 2, 3, 4, 8, 9])
        assert torch.allclose(base_value[:, living], moved_value[:, living], atol=1e-5)
        assert not torch.allclose(base_value[:, 5:8], moved_value[:, 5:8])

    def test_step_and_sequence_paths_agree(self):
        """Rollout and PPO re-evaluation must rotate identically."""
        policy = _policy(ROPE_MODEL_CONFIG)
        policy.eval()
        steps = 4
        observations = [_observation(seed=10 + t) for t in range(steps)]

        hidden = policy.initial_hidden(3, 10, torch.device("cpu"))
        step_values = []
        with torch.no_grad():
            for observation in observations:
                _, _, value, _, hidden = policy.get_action_and_value(observation, hidden)
                step_values.append(value)
        step_value = torch.stack(step_values)

        stacked = {
            key: torch.stack([o[key] for o in observations]) for key in observations[0].data
        }
        from boost_and_broadside.env.observation import YemongObservation

        sequence_obs = YemongObservation(data=stacked)
        actions = torch.zeros(steps, 3, 10, 3, dtype=torch.long)
        alive = torch.stack([o["belief_valid"] for o in observations])
        initial = policy.initial_hidden(3, 10, torch.device("cpu"))
        with torch.no_grad():
            _, _, sequence_value, _, _, _, _ = policy.evaluate_actions(
                sequence_obs, actions, initial, alive
            )
        assert torch.allclose(step_value, sequence_value, atol=1e-4)
