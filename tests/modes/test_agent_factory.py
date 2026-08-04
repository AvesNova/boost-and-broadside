"""Tests for watch-mode helpers in modes/agent_factory.py."""

import math

import pytest
import torch

from boost_and_broadside.agents.semi_random_scripted import SemiRandomScriptedAgent
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.modes.agent_factory import _decode_targets_to_obs, resolve_agent_spec
from boost_and_broadside.train.rl.features import build_standard_coordinator


def _make_prev_obs(B: int, N: int) -> YemongObservation:
    """Minimal ship-only observation with every key _decode_targets_to_obs touches."""
    return YemongObservation(
        data={
            ObsKey.POS: torch.zeros(B, N, 2),
            ObsKey.VEL: torch.zeros(B, N, 2),
            ObsKey.ATT: torch.zeros(B, N, 2),
            ObsKey.ANG_VEL: torch.zeros(B, N, 1),
            ObsKey.HEALTH: torch.zeros(B, N, 1),
            ObsKey.POWER: torch.zeros(B, N, 1),
            ObsKey.COOLDOWN: torch.zeros(B, N, 1),
            ObsKey.LOCAL_LOG_INDEX: torch.zeros(B, N, 1),
            ObsKey.LOCAL_INDEX_GRADIENT: torch.zeros(B, N, 2),
            ObsKey.ALIVE: torch.ones(B, N, dtype=torch.bool),
            ObsKey.PREVIOUS_ACTION: torch.zeros(B, N, 3, dtype=torch.long),
        }
    )


class TestDecodeTargetsToObs:
    def test_position_decodes_each_axis_with_its_own_world_extent(self):
        """Regression (audit §1.4): pos_y must decode with world height, not width.

        Uses a rectangular world (W != H) so decoding y with W produces the
        wrong coordinate; encode (x, y) exactly as the feature pipeline does
        (Fourier(1, period) → (sin, cos)) and expect the round-trip identity.
        """
        ship_config = ShipConfig(
            world_size=(1024.0, 512.0),
            field_radius_max=200.0,
        )
        coordinator = build_standard_coordinator(ship_config)
        target_slices = coordinator.target_slices()
        W, H = ship_config.world_size
        x, y = 700.0, 300.0

        targets = torch.zeros(1, 1, coordinator.total_target_dimension)
        targets[0, 0, target_slices["position_x"]] = torch.tensor(
            [math.sin(2 * math.pi * x / W), math.cos(2 * math.pi * x / W)]
        )
        targets[0, 0, target_slices["position_y"]] = torch.tensor(
            [math.sin(2 * math.pi * y / H), math.cos(2 * math.pi * y / H)]
        )
        targets[0, 0, target_slices["attitude"]] = torch.tensor([0.0, 1.0])

        obs = _decode_targets_to_obs(
            targets,
            prev_obs=_make_prev_obs(B=1, N=1),
            action=torch.zeros(1, 1, 3, dtype=torch.long),
            N=1,
            coordinator=coordinator,
        )

        decoded = obs[ObsKey.POS][0, 0]
        assert torch.allclose(decoded, torch.tensor([x, y]), atol=1e-3)


class TestBulletReadingAgents:
    """Modes must rebuild — and then feed — a bullet-reading checkpoint.

    A policy trained with bullet cross-attention accepts a bullet-free
    observation without complaint and plays blind to every shot in flight, so
    both halves are silent when wrong: the encoder must be rebuilt from the
    weights, and the observation must carry the bullet axis.
    """

    @staticmethod
    def _bullet_checkpoint(tmp_path):
        from boost_and_broadside.config import ModelConfig
        from tests.train.test_ppo import _make_trainer

        model_config = ModelConfig(
            d_model=32, n_heads=4, n_yemong_blocks=1, n_bullet_cross_per_block=1
        )
        trainer = _make_trainer(checkpoint_dir=str(tmp_path), model_config=model_config)
        return str(trainer._save_ladder_snapshot()), model_config, trainer.wrapper.num_ships

    def test_resolved_checkpoint_agent_keeps_its_bullet_encoder(self, tmp_path):
        from boost_and_broadside.modes.agent_factory import ResolvedAgent, agents_read_bullets

        path, model_config, num_ships = self._bullet_checkpoint(tmp_path)

        resolved = resolve_agent_spec(path, ShipConfig(), model_config, "cpu", num_ships=num_ships)

        assert resolved.kind == "policy"
        assert resolved.agent.bullet_encoder is not None
        assert agents_read_bullets(resolved)
        assert not agents_read_bullets(ResolvedAgent("scripted", None), None)

    def test_matchup_feeds_bullets_to_a_bullet_reading_policy(self, tmp_path):
        """End-to-end through a mode: the observation must reach the policy with
        its bullet axis attached, on both the ego and the team-flipped side."""
        from boost_and_broadside.config import EnvConfig
        from boost_and_broadside.modes.collect import evaluate_matchup

        path, model_config, num_ships = self._bullet_checkpoint(tmp_path)
        agent0 = resolve_agent_spec(path, ShipConfig(), model_config, "cpu", num_ships=num_ships)
        agent1 = resolve_agent_spec(path, ShipConfig(), model_config, "cpu", num_ships=num_ships)

        seen: dict[str, list[bool]] = {"ego": [], "flipped": []}

        def watch(agent, side: str) -> None:
            encoder = agent.agent.bullet_encoder
            original = encoder.forward

            def record(obs):
                seen[side].append(obs.bullets is not None)
                return original(obs)

            encoder.forward = record

        watch(agent0, "ego")
        watch(agent1, "flipped")

        evaluate_matchup(
            agent0,
            agent1,
            2,
            2,
            2,
            ShipConfig(),
            EnvConfig(num_ships=num_ships, max_bullets=8, max_episode_steps=8),
            "cpu",
        )

        assert seen["ego"] and all(seen["ego"])
        assert seen["flipped"] and all(seen["flipped"])


def test_semi_scripted_agent_spec_resolves_probability() -> None:
    resolved = resolve_agent_spec(
        "semi_scripted:0.35",
        ShipConfig(),
        None,  # type: ignore[arg-type]
        "cpu",
    )

    assert resolved.kind == "semi_random"
    assert isinstance(resolved.agent, SemiRandomScriptedAgent)
    assert resolved.agent.p_scripted == pytest.approx(0.35)


@pytest.mark.parametrize("spec", ["semi_scripted:nope", "semi_scripted:-0.1", "semi_scripted:1.1"])
def test_semi_scripted_agent_spec_rejects_invalid_probability(spec: str) -> None:
    with pytest.raises(ValueError):
        resolve_agent_spec(spec, ShipConfig(), None, "cpu")  # type: ignore[arg-type]
