"""The permanent global/game token: layout, trunk participation, and head isolation.

Built from real environment observations so the token under test is the one the
environment actually places, not a fixture's idea of it.
"""

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import (
    NUM_GAME_MODES,
    GameMode,
    ObjectType,
    ObsKey,
    YemongObservation,
    perceived_observation_from_state,
)
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.policy_io import build_policy

SHIP_CONFIG = PROFILES["rl"].ship_config
NUM_ENVS = 3
NUM_SHIPS = 4
NUM_FIELDS = 3
CPU = torch.device("cpu")


def _env_config(frontline: bool = True) -> EnvConfig:
    profile = PROFILES["rl"]
    return EnvConfig(
        num_ships=NUM_SHIPS,
        num_fields=NUM_FIELDS,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        frontline=profile.frontline if frontline else None,
        vision_range=profile.vision_range,
    )


def _observation(seed: int = 5, frontline: bool = True) -> YemongObservation:
    config = _env_config(frontline)
    env = TensorEnv(NUM_ENVS, SHIP_CONFIG, config, "cpu")
    env.reset(seed=seed)
    observation, _ = perceived_observation_from_state(env.state, SHIP_CONFIG, config)
    return observation.for_team(0)


def _policy(global_token: bool = True, global_value_k=(0,), predict_density: bool = False):
    torch.manual_seed(0)
    policy = build_policy(
        replace(MODEL_CONFIG, global_token=global_token),
        SHIP_CONFIG,
        num_value_components=4,
        num_ships=NUM_SHIPS,
        global_value_k=global_value_k,
        predict_density=predict_density,
    )
    return policy.eval()


def _global_slot(hidden: torch.Tensor) -> torch.Tensor:
    """The global token's recurrent state, ``(n_layers, B, CONV_KERNEL*D)``."""
    n_layers, _, width = hidden.shape
    return hidden.reshape(n_layers, NUM_ENVS, NUM_SHIPS + 1, width)[:, :, NUM_SHIPS]


# ---------------------------------------------------------------------------
# Layout: every mode presents the token directly after the ships
# ---------------------------------------------------------------------------


class TestLayout:
    @pytest.mark.parametrize("frontline", [True, False])
    def test_the_slot_after_the_ships_is_the_global_token(self, frontline):
        object_type = _observation(frontline=frontline)[ObsKey.OBJECT_TYPE]
        assert torch.all(object_type[:, NUM_SHIPS] == int(ObjectType.GLOBAL))

    @pytest.mark.parametrize(
        ("frontline", "mode"), [(True, GameMode.FRONTLINE), (False, GameMode.COMBAT)]
    )
    def test_game_mode_is_a_one_hot_on_the_global_token(self, frontline, mode):
        game_mode = _observation(frontline=frontline)[ObsKey.GAME_MODE]
        expected = torch.nn.functional.one_hot(torch.tensor(int(mode)), NUM_GAME_MODES).float()
        assert torch.equal(game_mode[:, NUM_SHIPS], expected.expand(NUM_ENVS, -1))

    def test_game_mode_is_zero_on_every_other_token(self):
        game_mode = _observation()[ObsKey.GAME_MODE].clone()
        game_mode[:, NUM_SHIPS] = 0.0
        assert torch.all(game_mode == 0.0)


# ---------------------------------------------------------------------------
# Trunk participation
# ---------------------------------------------------------------------------


class TestRecurrence:
    def test_the_global_token_carries_recurrent_state(self):
        assert _policy().num_recurrent_tokens == NUM_SHIPS + 1

    def test_its_recurrent_state_is_carried_forward(self):
        """The global slot's incoming state reaches its outgoing state."""
        policy = _policy()
        observation = _observation()
        zero = policy.initial_hidden(NUM_ENVS, policy.num_recurrent_tokens, CPU)
        seeded = zero.clone()
        torch.manual_seed(1)
        _global_slot(seeded).copy_(torch.randn_like(_global_slot(seeded)))
        with torch.no_grad():
            *_, from_zero = policy.get_action_and_value(observation, zero)
            *_, from_seeded = policy.get_action_and_value(observation, seeded)
        assert not torch.allclose(_global_slot(from_zero), _global_slot(from_seeded), atol=1e-5)

    def test_its_recurrent_state_reaches_the_ships(self):
        """Through a later block's attention, the global token's memory is read."""
        policy = _policy()
        observation = _observation()
        zero = policy.initial_hidden(NUM_ENVS, policy.num_recurrent_tokens, CPU)
        seeded = zero.clone()
        torch.manual_seed(1)
        _global_slot(seeded).copy_(torch.randn_like(_global_slot(seeded)))
        with torch.no_grad():
            _, _, value_zero_critic, _, _ = policy.get_action_and_value(observation, zero)
            value_zero = value_zero_critic.value
            _, _, value_seeded_critic, _, _ = policy.get_action_and_value(observation, seeded)
            value_seeded = value_seeded_critic.value
        assert not torch.allclose(value_zero, value_seeded, atol=1e-5)

    def test_it_is_updated_by_attention_over_the_ships(self):
        """Moving a ship changes the global token's next state: it queried them."""
        policy = _policy()
        observation = _observation()
        position = observation[ObsKey.POS].clone()
        position[:, 0] += 700.0
        moved = observation.update(ObsKey.POS, position)
        hidden = policy.initial_hidden(NUM_ENVS, policy.num_recurrent_tokens, CPU)
        with torch.no_grad():
            *_, base = policy.get_action_and_value(observation, hidden)
            *_, shifted = policy.get_action_and_value(moved, hidden)
        assert not torch.allclose(_global_slot(base), _global_slot(shifted), atol=1e-5)

    def test_a_ship_only_hidden_state_is_rejected(self):
        policy = _policy()
        hidden = policy.initial_hidden(NUM_ENVS, NUM_SHIPS, CPU)
        with pytest.raises(ValueError, match="query set"):
            policy.get_action_and_value(_observation(), hidden)


# ---------------------------------------------------------------------------
# Map objects stay key/value-only; heads read ships only
# ---------------------------------------------------------------------------


def _record_inputs(modules: dict[str, torch.nn.Module]) -> dict[str, torch.Size]:
    seen: dict[str, torch.Size] = {}
    for name, module in modules.items():
        module.register_forward_hook(
            lambda _module, args, _out, name=name: seen.__setitem__(name, args[0].shape)
        )
    return seen


def _run_step(policy) -> None:
    observation = _observation()
    hidden = policy.initial_hidden(NUM_ENVS, policy.num_recurrent_tokens, CPU)
    with torch.no_grad():
        policy.get_action_and_value(observation, hidden, return_enemy_action=True)


class TestMapObjects:
    @pytest.mark.parametrize(("global_token", "queries"), [(True, 5), (False, 4)])
    def test_only_ships_and_the_global_token_enter_the_trunk(self, global_token, queries):
        policy = _policy(global_token)
        seen = _record_inputs({"spatial": policy.yemong_layers[0].spatial[0]})
        _run_step(policy)
        assert seen["spatial"][1] == queries

    @pytest.mark.parametrize(("global_token", "memory"), [(True, 8), (False, 9)])
    def test_fields_and_zones_are_the_key_value_memory(self, global_token, memory):
        """Off, the global token falls back into the memory it was promoted out of."""
        policy = _policy(global_token)
        seen = _record_inputs({"memory": policy.map_memory_proj})
        _run_step(policy)
        # NUM_FIELDS fields and the Frontline zones; the global token only when off.
        assert seen["memory"][1] == memory

    def test_the_switch_off_keeps_ships_as_the_only_recurrent_tokens(self):
        assert _policy(global_token=False).num_recurrent_tokens == NUM_SHIPS


class TestHeads:
    def test_every_per_ship_head_still_reads_ships_only(self):
        """The global token shapes these through the trunk; none of them decodes
        it. Only the heads whose *subject* is the game read it directly."""
        policy = _policy()
        heads = {
            "action": policy.action_head,
            "enemy_action": policy.enemy_action_head,
            "value": policy.value_head_local,
            "next_state": policy.next_state_head,
        }
        seen = _record_inputs(heads)
        _run_step(policy)
        assert {name: shape[-2] for name, shape in seen.items()} == dict.fromkeys(heads, NUM_SHIPS)

    def test_the_team_level_value_head_reads_the_global_token(self):
        """One estimate per environment, not one per ship: the outcome pays
        every ship on a side the same number, so its return is the same too."""
        policy = _policy(global_value_k=(0,))
        seen = _record_inputs({"global_value": policy.value_head_global.net})
        _run_step(policy)
        # The global token is one token, so the head sees the embedding itself
        # with no token axis -- not a length-N slice of one.
        assert seen["global_value"][-1] == MODEL_CONFIG.d_model
        assert len(seen["global_value"]) == 2  # (B, D)

    def test_the_team_level_value_falls_back_to_ships_when_the_token_is_off(self):
        """Phase 8's variant A drops the promotion. The density head cannot
        survive that and says so; this one can, because the per-ship categorical
        head can value the outcome like any other level -- so the ablation
        degrades to N redundant estimates rather than failing to build."""
        policy = _policy(global_token=False, global_value_k=(0,))
        assert policy.value_head_global is None
        assert policy._global_value_k == ()

    def test_the_outcome_head_returns_four_classes_per_environment(self):
        policy = _policy(global_value_k=(0,))
        observation = _observation()
        sequence = YemongObservation(data={key: value[None] for key, value in observation.items()})
        with torch.no_grad():
            critic = policy.evaluate_actions(
                sequence,
                torch.zeros(1, NUM_ENVS, NUM_SHIPS, 3, dtype=torch.long),
                policy.initial_hidden(NUM_ENVS, policy.num_recurrent_tokens, CPU),
                sequence[ObsKey.BELIEF_VALID],
            )[2]
        assert critic.outcome_logits.shape == (1, NUM_ENVS, 4)


# ---------------------------------------------------------------------------
# Rollout and sequence re-evaluation agree
# ---------------------------------------------------------------------------


class TestEquivalence:
    @pytest.mark.parametrize("global_token", [True, False])
    def test_rollout_and_sequence_agree_across_an_episode_boundary(self, global_token):
        policy = _policy(global_token)
        steps = 4
        observations = [_observation(seed=30 + t) for t in range(steps)]
        done = torch.zeros(steps, NUM_ENVS, dtype=torch.bool)
        done[1, 0] = True  # env 0's episode ends at t=1; t=2 starts fresh

        initial = policy.initial_hidden(NUM_ENVS, policy.num_recurrent_tokens, CPU)
        hidden = initial
        step_values, actions = [], []
        with torch.no_grad():
            for t, observation in enumerate(observations):
                action, _, value_critic, _, hidden = policy.get_action_and_value(
                    observation, hidden
                )
                value = value_critic.value
                hidden = policy.reset_hidden_for_envs(hidden, done[t], policy.num_recurrent_tokens)
                step_values.append(value)
                actions.append(action)

        sequence = YemongObservation(
            data={key: torch.stack([o[key] for o in observations]) for key in observations[0].data}
        )
        with torch.no_grad():
            _, _, sequence_value_critic, _, _, _ = policy.evaluate_actions(
                sequence,
                torch.stack(actions),
                initial,
                sequence[ObsKey.BELIEF_VALID],
                done_mask=done,
            )
            sequence_value = sequence_value_critic.value
        assert torch.allclose(torch.stack(step_values), sequence_value, atol=1e-4)
