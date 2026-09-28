"""The global/game token: a permanent query/recurrent token that no head reads.

Pinned against observations a real environment builds and against the
production model configuration, because the properties that matter -- where the
token sits, which path it takes through the trunk, what the heads consume --
are all consequences of the layout and the config together.
"""

import dataclasses

import pytest
import torch

from boost_and_broadside.config import EnvConfig, FrontlineConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG, SHIP_CONFIG
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import FRONTLINE_WORLD_SIZE
from boost_and_broadside.env.observation import (
    GameMode,
    ObjectType,
    ObsKey,
    YemongObservation,
    observation_from_state,
)
from boost_and_broadside.train.rl.policy_io import build_policy

NUM_SHIPS = 4
NUM_FIELDS = 2
K = 3
SHIP = dataclasses.replace(SHIP_CONFIG, world_size=FRONTLINE_WORLD_SIZE)


def _frontline() -> FrontlineConfig:
    return FrontlineConfig(
        zone_radius=330.0,
        zone_ring_radius=1200.0,
        playable_radius=2600.0,
        capture_seconds=20.0,
        respawn_health=25.0,
        respawn_power=20.0,
        respawn_speed=30.0,
        shield_recharge_delay=4.0,
        shield_recharge_per_second=20.0,
        boundary_damage_per_second=5.0,
        boundary_damage_per_pixel_second=0.05,
        front_win_threshold=5,
    )


def _env(frontline: bool, envs: int = 2) -> TensorEnv:
    env_config = EnvConfig(
        num_ships=NUM_SHIPS,
        max_bullets=0,
        max_episode_steps=600,
        num_fields=NUM_FIELDS,
        frontline=_frontline() if frontline else None,
    )
    env = TensorEnv(envs, SHIP, env_config, torch.device("cpu"))
    env.reset(seed=3)
    return env


def _observations(steps: int, frontline: bool = True) -> list[YemongObservation]:
    env = _env(frontline)
    generator = torch.Generator().manual_seed(0)
    observations = []
    for _ in range(steps):
        observations.append(observation_from_state(env.state, SHIP))
        power = torch.randint(0, 3, (2, NUM_SHIPS, 1), generator=generator)
        turn = torch.randint(0, 7, (2, NUM_SHIPS, 1), generator=generator)
        shoot = torch.randint(0, 2, (2, NUM_SHIPS, 1), generator=generator)
        env.step(torch.cat([power, turn, shoot], dim=-1))
    return observations


def _policy(**overrides):
    torch.manual_seed(0)
    config = dataclasses.replace(MODEL_CONFIG, **overrides)
    return build_policy(
        config, SHIP, num_value_components=K, num_ships=NUM_SHIPS, team_pma_k=(0,)
    ).eval()


def _capture_inputs(module: torch.nn.Module, store: list[torch.Tensor]):
    return module.register_forward_hook(lambda _m, args, _out: store.append(args[0]))


def _capture_trunk_output(policy, store: list[torch.Tensor]) -> None:
    """Record the last block's rollout output; ``step`` is not a hooked ``forward``."""

    block = policy.yemong_layers[-1]
    step = block.step

    def recording(*args, **kwargs):
        out = step(*args, **kwargs)
        store.append(out[0])
        return out

    block.step = recording


class TestObservationLayout:
    @pytest.mark.parametrize("frontline", [True, False])
    def test_every_mode_has_one_global_token_right_after_the_ships(self, frontline):
        obs = _observations(1, frontline)[0]
        types = obs[ObsKey.OBJECT_TYPE][0].tolist()
        assert types.count(int(ObjectType.GLOBAL)) == 1
        assert types[NUM_SHIPS] == int(ObjectType.GLOBAL)
        assert types[:NUM_SHIPS] == [int(ObjectType.SHIP)] * NUM_SHIPS

    @pytest.mark.parametrize(
        ("frontline", "mode"), [(True, GameMode.FRONTLINE), (False, GameMode.ELIMINATION)]
    )
    def test_game_mode_is_one_hot_on_the_global_token_only(self, frontline, mode):
        game_mode = _observations(1, frontline)[0][ObsKey.GAME_MODE]
        expected = torch.zeros(len(GameMode))
        expected[int(mode)] = 1.0
        assert torch.equal(game_mode[:, NUM_SHIPS], expected.expand(2, -1))
        others = torch.cat([game_mode[:, :NUM_SHIPS], game_mode[:, NUM_SHIPS + 1 :]], dim=1)
        assert not others.any()


class TestGlobalTokenInTheTrunk:
    def test_global_token_carries_recurrent_state(self):
        policy = _policy()
        obs = _observations(1)[0]
        queries = NUM_SHIPS + 1
        assert policy.num_recurrent_tokens == queries

        hidden = policy.initial_hidden(2, policy.num_recurrent_tokens, torch.device("cpu"))
        *_, new_hidden = policy.get_action_and_value(obs, hidden)
        per_token = new_hidden.reshape(new_hidden.shape[0], 2, queries, -1)
        assert per_token[:, :, NUM_SHIPS].abs().sum() > 0

        # State held only in the global token's slot reaches the ships.
        perturbed = hidden.reshape(hidden.shape[0], 2, queries, -1).clone()
        perturbed[:, :, NUM_SHIPS] = torch.randn_like(perturbed[:, :, NUM_SHIPS])
        torch.manual_seed(1)
        base = policy.get_action_and_value(obs, hidden)[2]
        torch.manual_seed(1)
        moved = policy.get_action_and_value(obs, perturbed.reshape_as(hidden))[2]
        assert not torch.allclose(base, moved)

    def test_global_token_takes_trunk_updates(self):
        policy = _policy()
        obs = _observations(1)[0]
        encoded, trunk_out = [], []
        hooks = [policy.encoder.register_forward_hook(lambda _m, _a, out: encoded.append(out))]
        _capture_trunk_output(policy, trunk_out)
        hidden = policy.initial_hidden(2, policy.num_recurrent_tokens, torch.device("cpu"))
        policy.get_action_and_value(obs, hidden)
        for hook in hooks:
            hook.remove()

        assert trunk_out[0].shape[1] == NUM_SHIPS + 1
        before = encoded[0][:, NUM_SHIPS]
        after = trunk_out[0][:, NUM_SHIPS]
        assert not torch.allclose(before, after)

    def test_map_objects_stay_key_value_only(self):
        policy = _policy()
        obs = _observations(1)[0]
        tokens = obs[ObsKey.OBJECT_TYPE].shape[1]
        memory_in, trunk_out = [], []
        hooks = [_capture_inputs(policy.map_memory_proj, memory_in)]
        _capture_trunk_output(policy, trunk_out)
        hidden = policy.initial_hidden(2, policy.num_recurrent_tokens, torch.device("cpu"))
        *_, new_hidden = policy.get_action_and_value(obs, hidden)
        for hook in hooks:
            hook.remove()

        # Every token after the query prefix is memory, and only memory.
        assert memory_in[0].shape[1] == tokens - (NUM_SHIPS + 1)
        assert trunk_out[0].shape[1] == NUM_SHIPS + 1
        assert new_hidden.shape[1] == 2 * (NUM_SHIPS + 1)

    def test_heads_read_ship_tokens_only(self):
        policy = _policy()
        obs = _observations(1)[0]
        head_inputs: list[torch.Tensor] = []
        hooks = [
            _capture_inputs(module, head_inputs)
            for module in (
                policy.action_head,
                policy.enemy_action_head,
                policy.next_state_head,
                policy.value_head_local,
                policy.team_pma,
            )
        ]
        hidden = policy.initial_hidden(2, policy.num_recurrent_tokens, torch.device("cpu"))
        policy.get_action_and_value(obs, hidden, return_enemy_action=True)
        for hook in hooks:
            hook.remove()

        assert len(head_inputs) == 5
        for tensor in head_inputs:
            assert tensor.shape[1] == NUM_SHIPS

    def test_rollout_and_sequence_reevaluation_agree(self):
        policy = _policy()
        T = 5
        observations = _observations(T)
        initial = policy.initial_hidden(2, policy.num_recurrent_tokens, torch.device("cpu"))
        # A non-zero start state, so the global token's carried state is exercised.
        initial = initial + 0.1 * torch.randn_like(initial)
        done = torch.zeros(T, 2, dtype=torch.bool)
        done[2, 0] = True

        hidden = initial
        actions, logprobs, values = [], [], []
        for t, obs in enumerate(observations):
            action, logprob, value, _pred, hidden = policy.get_action_and_value(obs, hidden)
            actions.append(action)
            logprobs.append(logprob)
            values.append(value)
            hidden = policy.reset_hidden_for_envs(hidden, done[t], policy.num_recurrent_tokens)

        stacked = YemongObservation(
            data={
                key: torch.stack([obs[key] for obs in observations]) for key in observations[0].data
            }
        )
        alive = stacked[ObsKey.ALIVE]
        logprob, _entropy, value, *_ = policy.evaluate_actions(
            stacked, torch.stack(actions), initial, alive, done_mask=done
        )
        torch.testing.assert_close(logprob, torch.stack(logprobs), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(value, torch.stack(values), atol=1e-4, rtol=1e-4)


class TestGlobalTokenSwitch:
    def test_switch_off_demotes_the_token_to_map_memory(self):
        policy = _policy(global_token=False)
        obs = _observations(1)[0]
        tokens = obs[ObsKey.OBJECT_TYPE].shape[1]
        assert policy.num_recurrent_tokens == NUM_SHIPS

        memory_in: list[torch.Tensor] = []
        hook = _capture_inputs(policy.map_memory_proj, memory_in)
        hidden = policy.initial_hidden(2, policy.num_recurrent_tokens, torch.device("cpu"))
        *_, new_hidden = policy.get_action_and_value(obs, hidden)
        hook.remove()
        assert memory_in[0].shape[1] == tokens - NUM_SHIPS
        assert new_hidden.shape[1] == 2 * NUM_SHIPS

    def test_both_settings_share_one_parameter_layout(self):
        on = {name: p.shape for name, p in _policy().named_parameters()}
        off = {name: p.shape for name, p in _policy(global_token=False).named_parameters()}
        assert on == off

    def test_an_observation_without_the_token_is_refused(self):
        policy = _policy()
        obs = _observations(1)[0]
        without = YemongObservation(
            data={
                key: torch.cat([value[:, :NUM_SHIPS], value[:, NUM_SHIPS + 1 :]], dim=1)
                for key, value in obs.items()
            }
        )
        hidden = policy.initial_hidden(2, policy.num_recurrent_tokens, torch.device("cpu"))
        with pytest.raises(ValueError, match="global_token"):
            policy.get_action_and_value(without, hidden)

    def test_a_ship_sized_hidden_state_is_refused(self):
        policy = _policy()
        obs = _observations(1)[0]
        hidden = policy.initial_hidden(2, NUM_SHIPS, torch.device("cpu"))
        with pytest.raises(ValueError, match="recurrent_token_count"):
            policy.get_action_and_value(obs, hidden)
