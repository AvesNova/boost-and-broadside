"""
Tests for StochasticAgentConfig and StochasticScriptedAgent.

Covers:
  - Config instantiation and field structure
  - from_vector / default_vector round-trip
  - _linear_ramp behaviour (prob_lo/prob_hi, no invert flag)
  - get_actions_and_probs output shapes and validity
"""
import pytest
import torch
import numpy as np

from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.env2.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.env2.env import TensorEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ship_config():
    return ShipConfig()


@pytest.fixture
def default_cfg():
    return StochasticAgentConfig()


@pytest.fixture
def agent(ship_config, default_cfg):
    return StochasticScriptedAgent(ship_config, default_cfg)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def simple_state(ship_config, device):
    """1 env, 4 ships (2v2)."""
    env = TensorEnv(num_envs=1, config=ship_config, device=device, max_ships=4)
    env.reset(options={"team_sizes": (2, 2)})
    return env.state


# ---------------------------------------------------------------------------
# StochasticAgentConfig: structure
# ---------------------------------------------------------------------------

class TestStochasticAgentConfig:

    def test_default_fields_present(self, default_cfg):
        assert hasattr(default_cfg, "boost_speed_ramp")
        assert hasattr(default_cfg, "boost_speed_prob")
        assert hasattr(default_cfg, "close_range_ramp")
        assert hasattr(default_cfg, "close_range_prob")
        assert hasattr(default_cfg, "turn_angle_ramp")
        assert hasattr(default_cfg, "turn_angle_prob")
        assert hasattr(default_cfg, "sharp_turn_angle_ramp")
        assert hasattr(default_cfg, "sharp_turn_angle_prob")
        assert hasattr(default_cfg, "shoot_angle_ramp")
        assert hasattr(default_cfg, "shoot_angle_prob")
        assert hasattr(default_cfg, "shoot_distance_ramp")
        assert hasattr(default_cfg, "shoot_distance_prob")

    def test_removed_fields_absent(self, default_cfg):
        assert not hasattr(default_cfg, "max_shooting_range")
        assert not hasattr(default_cfg, "radius_multiplier")
        assert not hasattr(default_cfg, "target_radius")
        assert not hasattr(default_cfg, "shoot_angle_multiplier_ramp")

    def test_all_ramps_are_ordered(self, default_cfg):
        """low <= high for every ramp (not a hard requirement but a sanity check)."""
        for attr in ["boost_speed_ramp", "close_range_ramp", "turn_angle_ramp",
                     "sharp_turn_angle_ramp", "shoot_angle_ramp", "shoot_distance_ramp"]:
            lo, hi = getattr(default_cfg, attr)
            assert lo <= hi, f"{attr}: {lo} > {hi}"

    def test_all_probs_in_unit_interval(self, default_cfg):
        for attr in ["boost_speed_prob", "close_range_prob", "turn_angle_prob",
                     "sharp_turn_angle_prob", "shoot_angle_prob", "shoot_distance_prob"]:
            for val in getattr(default_cfg, attr):
                assert 0.0 <= val <= 1.0, f"{attr} value {val} out of [0, 1]"


# ---------------------------------------------------------------------------
# from_vector / default_vector
# ---------------------------------------------------------------------------

class TestFromVector:

    def test_vector_length(self):
        v = StochasticAgentConfig.default_vector()
        assert len(v) == 2 * len(StochasticAgentConfig.PARAM_BOUNDS)

    def test_default_vector_in_unit_interval(self):
        v = StochasticAgentConfig.default_vector()
        assert all(0.0 <= x <= 1.0 for x in v), f"Out-of-range values: {[x for x in v if not 0<=x<=1]}"

    def test_round_trip(self):
        cfg = StochasticAgentConfig()
        v = StochasticAgentConfig.default_vector()
        cfg2 = StochasticAgentConfig.from_vector(v)
        for attr in ["boost_speed_ramp", "boost_speed_prob",
                     "close_range_ramp", "close_range_prob",
                     "turn_angle_ramp", "turn_angle_prob",
                     "sharp_turn_angle_ramp", "sharp_turn_angle_prob",
                     "shoot_angle_ramp", "shoot_angle_prob",
                     "shoot_distance_ramp", "shoot_distance_prob"]:
            orig = getattr(cfg, attr)
            rt = getattr(cfg2, attr)
            assert abs(orig[0] - rt[0]) < 1e-9, f"{attr}[0]: {orig[0]} vs {rt[0]}"
            assert abs(orig[1] - rt[1]) < 1e-9, f"{attr}[1]: {orig[1]} vs {rt[1]}"

    def test_zeros_vector(self):
        """Zero vector should produce the lower bounds of each field."""
        v = [0.0] * 24
        cfg = StochasticAgentConfig.from_vector(v)
        for field_idx, (lo, _) in enumerate(StochasticAgentConfig.PARAM_BOUNDS):
            attr_name = [
                "boost_speed_ramp", "boost_speed_prob",
                "close_range_ramp", "close_range_prob",
                "turn_angle_ramp", "turn_angle_prob",
                "sharp_turn_angle_ramp", "sharp_turn_angle_prob",
                "shoot_angle_ramp", "shoot_angle_prob",
                "shoot_distance_ramp", "shoot_distance_prob",
            ][field_idx]
            val = getattr(cfg, attr_name)
            assert abs(val[0] - lo) < 1e-9
            assert abs(val[1] - lo) < 1e-9

    def test_ones_vector(self):
        """Ones vector should produce the upper bounds of each field."""
        v = [1.0] * 24
        cfg = StochasticAgentConfig.from_vector(v)
        for field_idx, (_, hi) in enumerate(StochasticAgentConfig.PARAM_BOUNDS):
            attr_name = [
                "boost_speed_ramp", "boost_speed_prob",
                "close_range_ramp", "close_range_prob",
                "turn_angle_ramp", "turn_angle_prob",
                "sharp_turn_angle_ramp", "sharp_turn_angle_prob",
                "shoot_angle_ramp", "shoot_angle_prob",
                "shoot_distance_ramp", "shoot_distance_prob",
            ][field_idx]
            val = getattr(cfg, attr_name)
            assert abs(val[0] - hi) < 1e-9
            assert abs(val[1] - hi) < 1e-9

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            StochasticAgentConfig.from_vector([0.5] * 10)


# ---------------------------------------------------------------------------
# _linear_ramp
# ---------------------------------------------------------------------------

class TestLinearRamp:

    def test_normal_ramp(self, agent):
        x = torch.tensor([0.0, 0.5, 1.0])
        # prob_lo=0.0, prob_hi=1.0 → identity mapping on [0,1]
        result = agent._linear_ramp(x, 0.0, 1.0, 0.0, 1.0)
        assert torch.allclose(result, x)

    def test_inverted_ramp(self, agent):
        x = torch.tensor([0.0, 0.5, 1.0])
        result = agent._linear_ramp(x, 0.0, 1.0, 1.0, 0.0)
        expected = torch.tensor([1.0, 0.5, 0.0])
        assert torch.allclose(result, expected)

    def test_clamping_below(self, agent):
        x = torch.tensor([-5.0])
        result = agent._linear_ramp(x, 0.0, 1.0, 0.2, 0.8)
        assert torch.allclose(result, torch.tensor([0.2]))

    def test_clamping_above(self, agent):
        x = torch.tensor([5.0])
        result = agent._linear_ramp(x, 0.0, 1.0, 0.2, 0.8)
        assert torch.allclose(result, torch.tensor([0.8]))

    def test_equal_low_high(self, agent):
        """When low==high, should return prob_lo everywhere."""
        x = torch.tensor([0.0, 1.0, 2.0])
        result = agent._linear_ramp(x, 0.5, 0.5, 0.3, 0.9)
        assert torch.allclose(result, torch.full_like(x, 0.3))

    def test_custom_prob_range(self, agent):
        x = torch.tensor([0.0, 0.5, 1.0])
        result = agent._linear_ramp(x, 0.0, 1.0, 0.1, 0.9)
        expected = torch.tensor([0.1, 0.5, 0.9])
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# get_actions_and_probs: shapes and validity
# ---------------------------------------------------------------------------

class TestAgentOutputs:

    def test_action_shapes_marginal(self, agent, simple_state):
        actions, probs = agent.get_actions_and_probs(simple_state)
        B, N = simple_state.ship_pos.shape
        assert actions.shape == (B, N, 3)
        assert probs.shape == (B, N, 12)  # 3+7+2

    def test_action_shapes_flat(self, ship_config, simple_state):
        cfg = StochasticAgentConfig(flat_action_sampling=True)
        flat_agent = StochasticScriptedAgent(ship_config, cfg)
        actions, probs = flat_agent.get_actions_and_probs(simple_state)
        B, N = simple_state.ship_pos.shape
        assert actions.shape == (B, N, 3)
        assert probs.shape == (B, N, 42)  # 3*7*2

    def test_action_values_in_range(self, agent, simple_state):
        actions, _ = agent.get_actions_and_probs(simple_state)
        power, turn, shoot = actions[..., 0], actions[..., 1], actions[..., 2]
        assert power.min() >= 0 and power.max() <= 2
        assert turn.min() >= 0 and turn.max() <= 6
        assert shoot.min() >= 0 and shoot.max() <= 1

    def test_probs_sum_to_one(self, agent, simple_state):
        _, probs = agent.get_actions_and_probs(simple_state)
        power_sum = probs[..., :3].sum(dim=-1)
        turn_sum  = probs[..., 3:10].sum(dim=-1)
        shoot_sum = probs[..., 10:].sum(dim=-1)
        assert torch.allclose(power_sum, torch.ones_like(power_sum), atol=1e-5)
        assert torch.allclose(turn_sum,  torch.ones_like(turn_sum),  atol=1e-5)
        assert torch.allclose(shoot_sum, torch.ones_like(shoot_sum), atol=1e-5)

    def test_probs_non_negative(self, agent, simple_state):
        _, probs = agent.get_actions_and_probs(simple_state)
        assert (probs >= 0).all()

    def test_from_vector_agent_runs(self, ship_config, simple_state):
        """Agent built from a random vector should not crash."""
        rng = np.random.default_rng(42)
        v = rng.uniform(0.0, 1.0, 24).tolist()
        cfg = StochasticAgentConfig.from_vector(v)
        rand_agent = StochasticScriptedAgent(ship_config, cfg)
        actions, probs = rand_agent.get_actions_and_probs(simple_state)
        assert actions.shape[-1] == 3
        assert not torch.isnan(probs).any()
