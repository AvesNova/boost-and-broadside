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
from boost_and_broadside.models.yemong.relation import (
    RELATION_FEATURES,
    RelationalBias,
    relation_features,
)
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
from boost_and_broadside.train.rl.features import (
    PRESENCE_RADIUS,
    AttitudeFourier,
    Fourier,
    LocalPresenceFeature,
    build_standard_coordinator,
    local_presence,
)
from boost_and_broadside.train.rl.policy_io import build_policy

FRONTLINE_SHIP_CONFIG = PROFILES["rl"].ship_config
# ``MODEL_CONFIG`` ships every spatial feature enabled, so a test that needs the
# unrotated, unwidened, unbiased model has to say so rather than lean on the
# default. Contrasts are taken against this, not against the shipped config.
BASE_MODEL_CONFIG = replace(
    MODEL_CONFIG,
    n_spatial_heads=None,
    spatial_rope=False,
    local_presence=False,
    relational_bias=False,
)
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

    def test_the_encoder_builds_the_written_frequency_definition(self):
        """``Fourier`` constructs on device; ``base2_frequencies`` is the definition.

        They are checked against each other rather than sharing a runtime
        object. A cached device tensor would be a CUDA-graph hazard -- one
        created inside a capture is invalidated by the next replay -- so the
        encoder rebuilds inline and this is what keeps the two honest.
        """
        for n_freqs, period in ((8, 16384.0), (ATTITUDE_FOURIER_FREQUENCIES, 2.0 * math.pi)):
            built = Fourier(n_freqs, period)._frequencies(period, torch.zeros(1))
            assert torch.allclose(built, torch.tensor(base2_frequencies(period, n_freqs)))

    def test_the_encoder_caches_no_device_tensor_across_calls(self):
        """Caching one silently broke every CUDA-graph compile mode."""
        transform = Fourier(4, 1024.0)
        first = transform._frequencies(1024.0, torch.zeros(1))
        second = transform._frequencies(1024.0, torch.zeros(1))
        assert first is not second
        assert torch.allclose(first, second)

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
            + list(
                Fourier(position_fourier_frequencies(height), periods=height)._frequencies(
                    height, probe
                )
            )
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
        # 1e-3, not the 1e-4 the 16384 px world needed. Periodicity is exact in
        # exact arithmetic but limited in practice by the float32 frequency
        # buffer times the period: the finest harmonic's phase argument reaches
        # 2*pi*512 on the 65536 px world, four times what it reached before, and
        # the residual scales with it. Measured 1.265e-04, which is 0.0026 px of
        # implied position against a 10 px collision radius.
        assert torch.allclose(base[0], wrapped[0], atol=1e-3)
        assert torch.allclose(base[1], wrapped[1], atol=1e-3)

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
        assert rotary_pair_count(FRONTLINE_SHIP_CONFIG) == 24  # 10 + 10 + 4
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
                replace(BASE_MODEL_CONFIG, spatial_rope=True),  # four 32-wide heads
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
        """``n_spatial_heads=None`` falls back to ``n_heads``.

        Asserted against an explicitly unset config: the shipped ``MODEL_CONFIG``
        sets the knob, so it can no longer demonstrate the fallback.
        """
        assert BASE_MODEL_CONFIG.n_spatial_heads is None
        assert BASE_MODEL_CONFIG.spatial_heads == BASE_MODEL_CONFIG.n_heads
        assert (
            BASE_MODEL_CONFIG.spatial_head_dim
            == BASE_MODEL_CONFIG.d_model // BASE_MODEL_CONFIG.n_heads
        )

    def test_the_shipped_config_widens_the_spatial_heads_for_rope(self):
        assert MODEL_CONFIG.n_spatial_heads == 2
        assert MODEL_CONFIG.spatial_head_dim == 64
        check_rotary_budget(MODEL_CONFIG, FRONTLINE_SHIP_CONFIG)

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

        assert parameters(replace(BASE_MODEL_CONFIG, n_spatial_heads=2)) == parameters(
            BASE_MODEL_CONFIG
        )

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
        plain = _policy(replace(BASE_MODEL_CONFIG, n_spatial_heads=2))
        rotated = _policy(replace(BASE_MODEL_CONFIG, n_spatial_heads=2, spatial_rope=True))
        rotated.load_state_dict(plain.state_dict())  # identical weights

        observation = _observation()
        hidden = plain.initial_hidden(3, 10, torch.device("cpu"))
        _, _, plain_value, _, _ = plain.get_action_and_value(observation, hidden)
        _, _, rotated_value, _, _ = rotated.get_action_and_value(observation, hidden)
        assert not torch.allclose(plain_value, rotated_value)

    def test_rotation_adds_no_parameters_or_state_dict_keys(self):
        """So a rotated run's checkpoint stays loadable by shape alone."""
        plain = _policy(replace(BASE_MODEL_CONFIG, n_spatial_heads=2))
        rotated = _policy(replace(BASE_MODEL_CONFIG, n_spatial_heads=2, spatial_rope=True))
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

        stacked = {key: torch.stack([o[key] for o in observations]) for key in observations[0].data}
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


# ---------------------------------------------------------------------------
# Ally / enemy presence
# ---------------------------------------------------------------------------

WORLD = tuple(float(side) for side in FRONTLINE_SHIP_CONFIG.world_size)

DENSITY_MODEL_CONFIG = replace(ROPE_MODEL_CONFIG, local_presence=True)
RELATION_MODEL_CONFIG = replace(DENSITY_MODEL_CONFIG, relational_bias=True)


def _fleet(num_per_team: int, spread: float = 400.0, seed: int = 0):
    """A two-team cluster around the world centre, plus one neutral map token."""
    generator = torch.Generator().manual_seed(seed)
    total = 2 * num_per_team
    position = 8192.0 + spread * torch.randn(1, total + 1, 2, generator=generator)
    team_id = torch.cat(
        [
            torch.zeros(num_per_team, dtype=torch.long),
            torch.ones(num_per_team, dtype=torch.long),
            torch.full((1,), 2, dtype=torch.long),
        ]
    ).unsqueeze(0)
    source = team_id < 2
    return position, team_id, source


class TestLocalPresence:
    def test_permutation_invariance(self):
        position, team_id, source = _fleet(5, seed=1)
        base = local_presence(position, team_id, source, WORLD)

        order = torch.randperm(position.shape[1], generator=torch.Generator().manual_seed(7))
        shuffled = local_presence(position[:, order], team_id[:, order], source[:, order], WORLD)
        assert torch.allclose(base[:, order], shuffled, atol=1e-6)

    def test_self_is_excluded_from_ally_presence(self):
        """A lone ship has zero allies, however close it is to itself."""
        position = torch.tensor([[[100.0, 100.0], [9000.0, 9000.0]]])
        team_id = torch.tensor([[0, 1]])
        source = torch.ones(1, 2, dtype=torch.bool)
        presence = local_presence(position, team_id, source, WORLD)
        assert presence[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
        assert presence[0, 1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_enemy_presence_counts_the_other_team_only(self):
        position = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]])
        team_id = torch.tensor([[0, 0, 1]])
        source = torch.ones(1, 3, dtype=torch.bool)
        presence = local_presence(position, team_id, source, WORLD)

        # Ship 0: one ally 10 px away and one enemy 20 px away.
        def expected(distance: float) -> float:
            return math.log1p(math.exp(-(distance**2) / (2 * PRESENCE_RADIUS**2)))

        assert presence[0, 0, 0] == pytest.approx(expected(10.0), abs=1e-5)
        assert presence[0, 0, 1] == pytest.approx(expected(20.0), abs=1e-5)

    def test_toroidal_wraparound(self):
        """Neighbours across the seam count exactly as neighbours inside it."""
        width, height = WORLD
        seam = torch.tensor([[[width - 40.0, 500.0], [10.0, 500.0]]])
        interior = torch.tensor([[[8000.0, 500.0], [8050.0, 500.0]]])
        team_id = torch.tensor([[0, 1]])
        source = torch.ones(1, 2, dtype=torch.bool)
        assert torch.allclose(
            local_presence(seam, team_id, source, WORLD),
            local_presence(interior, team_id, source, WORLD),
            atol=1e-6,
        )
        del height

    def test_monotone_in_proximity_and_in_count(self):
        team_id = torch.tensor([[0, 0]])
        source = torch.ones(1, 2, dtype=torch.bool)

        def ally_of_first(distance: float) -> float:
            position = torch.tensor([[[0.0, 0.0], [distance, 0.0]]])
            return local_presence(position, team_id, source, WORLD)[0, 0, 0].item()

        distances = [10.0, 200.0, 500.0, 2000.0]
        values = [ally_of_first(d) for d in distances]
        assert values == sorted(values, reverse=True)

        # And monotone in how many neighbours are present at a fixed distance.
        previous = 0.0
        for count in (1, 2, 4, 8):
            position = torch.zeros(1, count + 1, 2)
            position[0, 1:, 0] = 100.0
            ids = torch.zeros(1, count + 1, dtype=torch.long)
            here = local_presence(position, ids, torch.ones(1, count + 1, dtype=torch.bool), WORLD)
            value = here[0, 0, 0].item()
            assert value > previous
            previous = value

    def test_masked_sources_do_not_contribute(self):
        position = torch.tensor([[[0.0, 0.0], [50.0, 0.0]]])
        team_id = torch.tensor([[0, 1]])
        seen = local_presence(position, team_id, torch.ones(1, 2, dtype=torch.bool), WORLD)
        hidden = local_presence(position, team_id, torch.tensor([[True, False]]), WORLD)
        assert seen[0, 0, 1] > 0.0
        assert hidden[0, 0, 1] == pytest.approx(0.0, abs=1e-7)

    def test_map_tokens_have_no_neighbourhood(self):
        position, team_id, source = _fleet(3, seed=2)
        presence = local_presence(position, team_id, source, WORLD)
        assert torch.all(presence[:, -1] == 0.0)  # the team-2 token

    @pytest.mark.parametrize("per_team", [1, 5, 25, 50, 200])
    def test_stays_finite_and_bounded_as_the_fleet_grows(self, per_team):
        position, team_id, source = _fleet(per_team, spread=400.0, seed=3)
        presence = local_presence(position, team_id, source, WORLD)
        assert torch.isfinite(presence).all()
        # log1p of a sum bounded by the fleet size; never explosive.
        assert presence.max() <= math.log1p(2.0 * per_team) + 1e-4
        assert presence.min() >= 0.0

    def test_scaling_from_5v5_to_50v50_is_compressed_but_not_flattened(self):
        """The whole point: 10x the fleet must read as *more*, not as saturated."""
        small, small_ids, small_source = _fleet(5, spread=400.0, seed=4)
        large, large_ids, large_source = _fleet(50, spread=400.0, seed=4)
        small_enemy = local_presence(small, small_ids, small_source, WORLD)[0, :10, 1].median()
        large_enemy = local_presence(large, large_ids, large_source, WORLD)[0, :100, 1].median()

        # Ten times the fleet in the same area: more presence, log-compressed.
        assert large_enemy > small_enemy
        assert 1.5 < (large_enemy / small_enemy) < 6.0

    def test_a_dense_swarm_does_not_overflow_in_bfloat16(self):
        position = torch.zeros(1, 400, 2)  # every ship on top of every other
        team_id = torch.cat([torch.zeros(200), torch.ones(200)]).long().unsqueeze(0)
        presence = local_presence(position, team_id, torch.ones(1, 400, dtype=torch.bool), WORLD)
        assert torch.isfinite(presence).all()
        assert presence.max() < 6.0  # log1p(400) = 6.0
        assert torch.isfinite(presence.to(torch.bfloat16)).all()


class TestPresenceFeatureWiring:
    def test_the_feature_is_shipped_on_and_adds_two_channels(self):
        assert MODEL_CONFIG.local_presence is True
        plain = build_standard_coordinator(FRONTLINE_SHIP_CONFIG)
        widened = build_standard_coordinator(FRONTLINE_SHIP_CONFIG, local_presence=True)
        assert widened.total_input_dimension == plain.total_input_dimension + 2

    def test_the_feature_is_input_only(self):
        feature = LocalPresenceFeature(FRONTLINE_SHIP_CONFIG)
        assert feature.predictor is None
        coordinator = build_standard_coordinator(FRONTLINE_SHIP_CONFIG, local_presence=True)
        plain = build_standard_coordinator(FRONTLINE_SHIP_CONFIG)
        assert coordinator.total_prediction_dimension == plain.total_prediction_dimension

    def test_it_reaches_the_encoder_input(self):
        coordinator = build_standard_coordinator(FRONTLINE_SHIP_CONFIG, local_presence=True)
        observation = _observation()
        encoded = coordinator.get_input_vector(observation)
        assert encoded.shape[-1] == coordinator.total_input_dimension
        assert torch.isfinite(encoded).all()
        assert encoded[..., -2:].abs().max() > 0.0

    def test_the_policy_runs_with_presence_at_unseen_fleet_sizes(self):
        policy = _policy(DENSITY_MODEL_CONFIG, num_ships=40)
        observation = _observation(num_ships=40)
        hidden = policy.initial_hidden(3, 40, torch.device("cpu"))
        _, _, value, _, _ = policy.get_action_and_value(observation, hidden)
        assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# Relational attention bias
# ---------------------------------------------------------------------------


def _relation_inputs(position, attitude, velocity):
    return relation_features(position, attitude, velocity, WORLD)


class TestRelationFeatures:
    def test_permutation_equivariance(self):
        generator = torch.Generator().manual_seed(11)
        position = 8192.0 + 500.0 * torch.randn(1, 6, 2, generator=generator)
        angle = torch.randn(1, 6, generator=generator)
        attitude = torch.stack([angle.cos(), angle.sin()], dim=-1)
        velocity = 50.0 * torch.randn(1, 6, 2, generator=generator)

        base = _relation_inputs(position, attitude, velocity)
        order = torch.randperm(6, generator=torch.Generator().manual_seed(12))
        shuffled = _relation_inputs(position[:, order], attitude[:, order], velocity[:, order])
        assert torch.allclose(base[:, order][:, :, order], shuffled, atol=1e-5)

    def test_the_bearing_is_in_the_query_frame(self):
        """A key dead ahead reads forward=1 whichever way the query is pointing."""
        for angle in (0.0, 1.0, -2.5, 3.0):
            heading = torch.tensor([[[math.cos(angle), math.sin(angle)], [1.0, 0.0]]])
            ahead = torch.tensor([[[0.0, 0.0], [300.0 * math.cos(angle), 300.0 * math.sin(angle)]]])
            features = _relation_inputs(ahead, heading, torch.zeros(1, 2, 2))
            assert features[0, 0, 1, 3] == pytest.approx(1.0, abs=1e-4)  # forward
            assert features[0, 0, 1, 4] == pytest.approx(0.0, abs=1e-4)  # lateral

    def test_proximity_is_monotone_and_bounded(self):
        for near, far in ((0.0, 100.0), (100.0, 500.0), (500.0, 5000.0)):
            position = torch.tensor([[[0.0, 0.0], [near, 0.0], [far, 0.0]]])
            features = _relation_inputs(position, torch.zeros(1, 3, 2), torch.zeros(1, 3, 2))
            assert features[0, 0, 1, 0] > features[0, 0, 2, 0]
            assert 0.0 <= features[0, 0, 2, 0] <= 1.0

    def test_toroidal_wraparound(self):
        width, _ = WORLD
        seam = torch.tensor([[[width - 30.0, 0.0], [70.0, 0.0]]])
        interior = torch.tensor([[[8000.0, 0.0], [8100.0, 0.0]]])
        attitude = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        velocity = torch.zeros(1, 2, 2)
        assert torch.allclose(
            _relation_inputs(seam, attitude, velocity),
            _relation_inputs(interior, attitude, velocity),
            atol=1e-5,
        )

    def test_closing_sign_follows_the_range_rate(self):
        position = torch.tensor([[[0.0, 0.0], [300.0, 0.0]]])
        attitude = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        approaching = torch.tensor([[[0.0, 0.0], [-80.0, 0.0]]])  # key moving toward query
        receding = torch.tensor([[[0.0, 0.0], [80.0, 0.0]]])
        closing_index = 5
        near = _relation_inputs(position, attitude, approaching)[0, 0, 1, closing_index]
        away = _relation_inputs(position, attitude, receding)[0, 0, 1, closing_index]
        assert near < 0.0 < away

    def test_a_heading_less_query_falls_back_to_the_world_frame(self):
        position = torch.tensor([[[0.0, 0.0], [250.0, 0.0]]])
        attitude = torch.zeros(1, 2, 2)  # map-object convention
        features = _relation_inputs(position, attitude, torch.zeros(1, 2, 2))
        assert torch.isfinite(features).all()
        assert features[0, 0, 1, 3] == pytest.approx(1.0, abs=1e-4)

    def test_coincident_tokens_stay_finite(self):
        position = torch.zeros(1, 3, 2)
        attitude = torch.tensor([[[1.0, 0.0]] * 3])
        features = _relation_inputs(position, attitude, torch.zeros(1, 3, 2))
        assert torch.isfinite(features).all()


class TestRelationalBias:
    def test_the_projection_is_shared_and_fleet_size_free(self):
        bias = RelationalBias(n_heads=2)
        assert bias.project.weight.shape == (2, RELATION_FEATURES)
        assert sum(p.numel() for p in bias.parameters()) == 2 * RELATION_FEATURES

    def test_output_is_permutation_equivariant(self):
        torch.manual_seed(0)
        bias = RelationalBias(n_heads=2)
        with torch.no_grad():
            bias.project.weight.copy_(torch.randn(2, RELATION_FEATURES) * 0.1)
        relation = torch.randn(1, 5, 5, RELATION_FEATURES)
        base = bias(relation)
        order = torch.randperm(5, generator=torch.Generator().manual_seed(3))
        shuffled = bias(relation[:, order][:, :, order])
        assert torch.allclose(base[:, :, order][:, :, :, order], shuffled, atol=1e-6)

    def test_zero_init_is_exactly_a_no_op(self):
        bias = RelationalBias(n_heads=2)
        relation = torch.randn(2, 4, 4, RELATION_FEATURES)
        assert torch.all(bias(relation) == 0.0)

    def test_scales_to_fifty_versus_fifty_without_new_parameters(self):
        small = _policy(RELATION_MODEL_CONFIG, num_ships=10)
        large = _policy(RELATION_MODEL_CONFIG, num_ships=100)
        assert sum(p.numel() for p in small.parameters()) == sum(
            p.numel() for p in large.parameters()
        )

    def test_masked_keys_stay_masked_under_a_trained_bias(self):
        """A finite relational term must not resurrect an infinitely-masked key."""
        policy = _policy(RELATION_MODEL_CONFIG)
        with torch.no_grad():
            for block in policy.yemong_layers:
                for sublayer in block.spatial:
                    sublayer.relational.project.weight.normal_(0.0, 5.0)

        observation = _observation()
        valid = observation["belief_valid"].clone()
        valid[:, 6:] = False
        masked = observation.update("belief_valid", valid)
        hidden = policy.initial_hidden(3, 10, torch.device("cpu"))
        _, _, base_value, _, _ = policy.get_action_and_value(masked, hidden)

        position = masked["pos"].clone()
        position[:, 6:] += 1234.0
        moved = masked.update("pos", position)
        _, _, moved_value, _, _ = policy.get_action_and_value(moved, hidden)
        assert torch.allclose(base_value[:, :6], moved_value[:, :6], atol=1e-5)

    def test_the_bias_changes_the_output_once_trained(self):
        policy = _policy(RELATION_MODEL_CONFIG)
        observation = _observation()
        hidden = policy.initial_hidden(3, 10, torch.device("cpu"))
        _, _, before, _, _ = policy.get_action_and_value(observation, hidden)
        with torch.no_grad():
            for block in policy.yemong_layers:
                for sublayer in block.spatial:
                    sublayer.relational.project.weight.normal_(0.0, 1.0)
        _, _, after, _, _ = policy.get_action_and_value(observation, hidden)
        assert not torch.allclose(before, after)

    def test_step_and_sequence_paths_agree_with_the_full_stack(self):
        policy = _policy(RELATION_MODEL_CONFIG)
        with torch.no_grad():
            for block in policy.yemong_layers:
                for sublayer in block.spatial:
                    sublayer.relational.project.weight.normal_(0.0, 0.5)
        policy.eval()
        steps = 3
        observations = [_observation(seed=20 + t) for t in range(steps)]

        hidden = policy.initial_hidden(3, 10, torch.device("cpu"))
        step_values = []
        with torch.no_grad():
            for observation in observations:
                _, _, value, _, hidden = policy.get_action_and_value(observation, hidden)
                step_values.append(value)

        from boost_and_broadside.env.observation import YemongObservation

        stacked = {key: torch.stack([o[key] for o in observations]) for key in observations[0].data}
        sequence_obs = YemongObservation(data=stacked)
        actions = torch.zeros(steps, 3, 10, 3, dtype=torch.long)
        alive = torch.stack([o["belief_valid"] for o in observations])
        with torch.no_grad():
            _, _, sequence_value, _, _, _, _ = policy.evaluate_actions(
                sequence_obs,
                actions,
                policy.initial_hidden(3, 10, torch.device("cpu")),
                alive,
            )
        assert torch.allclose(torch.stack(step_values), sequence_value, atol=1e-4)
