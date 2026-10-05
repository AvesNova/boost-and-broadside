"""Contracts for the physical belief plane and the legal views composed from it.

Every test drives the production path: a real ``TensorState``, the tracker's own
``observe``, and ``observation_from_state`` with the source it returns. There is
no separate "compose onto an already-built observation" path left to test, which
is the point of Phase 3.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.constants import NUM_JOINT_ACTIONS
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.train.rl.belief import BeliefTracker, DualBeliefTracker
from boost_and_broadside.train.rl.physical_belief import (
    CERTAIN_LOG_SIGMA,
    HEALTH,
    LOG_SIGMA_COLUMNS,
    NEXT_STATE_OUTPUT_DIM,
    PHYSICAL_MEAN_DIM,
    POSITION_X,
    SHIELD_DELAY,
    UNKNOWN_LOG_SIGMA,
    PhysicalNextState,
)

_SHIP = ShipConfig()
# 300 px of sight on a 1024 px torus: the enemy is visible at x=300 (200 px
# away), hidden at x=600 (500 px, and 524 px the other way round the seam).
_CONFIG = EnvConfig(
    num_ships=2, num_fields=1, max_bullets=0, max_episode_steps=256, vision_range=300.0
)
_VISIBLE_X = 300.0
_HIDDEN_X = 600.0
_HOLD = torch.zeros((1, 2, NEXT_STATE_OUTPUT_DIM))


def _env(*, frontline: bool = False) -> TensorEnv:
    """A two-ship environment whose ships sit on team 0 and team 1."""

    env = TensorEnv(1, _SHIP, _CONFIG, "cpu")
    env.reset(seed=5)
    env.state.ship_team_id[:] = torch.tensor([[0, 1]], dtype=torch.int32)
    env.state.ship_alive[:] = True
    env.state.ship_health[:] = torch.tensor([[80.0, 70.0]])
    env.state.ship_power[:] = torch.tensor([[60.0, 50.0]])
    env.state.ship_cooldown[:] = torch.tensor([[0.2, 0.3]])
    env.state.ship_shield_delay[:] = 0.0
    env.state.ship_spawned[:] = False
    env.state.prev_action[:] = torch.tensor([[[0, 2, 1], [1, 3, 0]]], dtype=torch.int32)
    _place(env, enemy_x=_VISIBLE_X)
    if frontline:
        # Instant respawn is what makes a remembered ship always alive.
        env.state.zone_pos = torch.zeros((1, 5), dtype=torch.complex64)
        env.state.zone_radius = torch.full((1, 5), 10.0)
        env.state.zone_roles = torch.arange(5, dtype=torch.int32).unsqueeze(0)
        env.state.zone_capture_progress = torch.zeros((1, 5))
        env.state.zone_capture_direction = torch.zeros((1, 5), dtype=torch.int32)
    return env


def _place(env: TensorEnv, *, enemy_x: float) -> None:
    """Ally at the origin; enemy at ``enemy_x`` -- in sight below 100 px."""

    env.state.ship_pos[:] = torch.tensor([[complex(100.0, 200.0), complex(enemy_x, 200.0)]])
    env.state.ship_vel[:] = torch.tensor([[complex(1.0, 2.0), complex(3.0, 4.0)]])
    env.state.ship_attitude[:] = torch.tensor([[complex(1.0, 0.0), complex(0.0, 1.0)]])
    env.state.ship_ang_vel[:] = torch.tensor([[0.1, 0.2]])


def _view(env: TensorEnv, tracker: BeliefTracker, team: int = 0):
    """The legal observation this tracker's observer sees, composed as production does."""

    visibility = team_visibility_from_state(env.state, _SHIP, _CONFIG, False)
    source = tracker.observe(env.state, visibility.ship[:, team])
    return observation_from_state(
        env.state,
        _SHIP,
        ship_visibility=visibility.ship[:, team],
        perspective_team=team,
        belief=source,
    )


def _tracker(**kwargs) -> BeliefTracker:
    return BeliefTracker(1, 2, 0.1, _SHIP, "cpu", **kwargs)


class TestComposition:
    def test_never_seen_enemy_stays_absent(self):
        env = _env()
        _place(env, enemy_x=_HIDDEN_X)
        view = _view(env, _tracker())

        assert not view[ObsKey.BELIEF_VALID][0, 1]
        assert not view[ObsKey.VISIBLE][0, 1]
        assert view[ObsKey.POS][0, 1].equal(torch.zeros(2))
        assert view[ObsKey.TEAM_ID][0, 1] == 0, "identity is not known either"
        assert view[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0
        # Maximal doubt, not the zero a masked channel used to leave behind.
        spreads = view[ObsKey.BELIEF_UNCERTAINTY][0, 1][list(LOG_SIGMA_COLUMNS)]
        assert (spreads == UNKNOWN_LOG_SIGMA).all()

    def test_a_visible_ship_is_truth_to_the_bit(self):
        env = _env()
        view = _view(env, _tracker())

        assert torch.equal(view[ObsKey.POS][0, 1], torch.tensor([_VISIBLE_X, 200.0]))
        assert torch.equal(view[ObsKey.VEL][0, 1], torch.tensor([3.0, 4.0]))
        assert view[ObsKey.HEALTH][0, 1, 0] == 70.0
        spreads = view[ObsKey.BELIEF_UNCERTAINTY][0, :2][:, list(LOG_SIGMA_COLUMNS)]
        assert (spreads == CERTAIN_LOG_SIGMA).all()

    def test_seen_then_hidden_uses_the_forecast_and_ages(self):
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        tracker.advance(_HOLD)

        _place(env, enemy_x=_HIDDEN_X)
        hidden = _view(env, tracker)

        assert hidden[ObsKey.BELIEF_VALID][0, 1]
        assert not hidden[ObsKey.VISIBLE][0, 1]
        assert torch.allclose(hidden[ObsKey.POS][0, 1], torch.tensor([_VISIBLE_X, 200.0]))
        assert hidden[ObsKey.TEAM_ID][0, 1] == 1, "identity is remembered"
        assert hidden[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == pytest.approx(0.1)
        # grad(n) is not forecast, so a remembered ship reads zero for it.
        assert hidden[ObsKey.LOCAL_INDEX_GRADIENT][0, 1].equal(torch.zeros(2))

        tracker.advance(_HOLD)
        again = _view(env, tracker)
        assert again[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == pytest.approx(0.2)

    def test_the_forecast_moves_the_remembered_position(self):
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        prediction = torch.zeros((1, 2, NEXT_STATE_OUTPUT_DIM))
        prediction[0, 1, POSITION_X] = 4.0  # 4 * 2.5 px
        tracker.advance(prediction)

        _place(env, enemy_x=_HIDDEN_X)
        hidden = _view(env, tracker)
        assert hidden[ObsKey.POS][0, 1, 0] == pytest.approx(_VISIBLE_X + 10.0)


class TestInformationFlow:
    def test_hidden_truth_cannot_reach_an_opponent_slot(self):
        """Plant distinctive truth on a hidden enemy and look for it everywhere."""
        env = _env()
        tracker = _tracker()
        _view(env, tracker)  # acquire, so the slot is valid and forecast-driven
        tracker.advance(_HOLD)

        _place(env, enemy_x=_HIDDEN_X)
        env.state.ship_vel[0, 1] = complex(432.25, 567.75)
        env.state.ship_health[0, 1] = 13.5
        env.state.ship_power[0, 1] = 91.5
        env.state.ship_cooldown[0, 1] = 0.07
        env.state.ship_ang_vel[0, 1] = 12.25
        env.state.prev_action[0, 1] = torch.tensor([2, 3, 1], dtype=torch.int32)

        hidden = _view(env, tracker)
        assert not hidden[ObsKey.VISIBLE][0, 1]
        planted = {432.25, 567.75, 13.5, 91.5, 0.07, 12.25, _HIDDEN_X}
        for key, value in hidden.items():
            flat = value[0, 1].flatten().float()
            for planted_value in planted:
                assert not torch.isclose(flat, torch.tensor(planted_value), atol=1e-4).any(), (
                    f"{key} leaked a hidden truth value"
                )
        # And the committed enemy command is not recoverable either: slot 41 is
        # (2, 3, 1) as a joint id, and a belief distribution is not a one-hot.
        pending = hidden[ObsKey.PREVIOUS_ACTION][0, 1]
        assert pending.argmax().item() != 41 or pending.max().item() < 1.0

    def test_two_observers_on_one_seat_remember_different_things(self):
        """What a league slot and the trainee are: same seat, separate memories.

        Both views are team-0 legal, so sharing one would leak nothing -- it
        would attribute one player's memory to another, which makes its
        behaviour stop being a function of its own weights.
        """
        env = _env()
        trainee, opponent = _tracker(), _tracker()
        visibility = team_visibility_from_state(env.state, _SHIP, _CONFIG, False)
        trainee.observe(env.state, visibility.ship[:, 0])
        opponent.observe(env.state, visibility.ship[:, 0])

        fast = torch.zeros((1, 2, NEXT_STATE_OUTPUT_DIM))
        fast[0, 1, POSITION_X] = 8.0  # 8 * 2.5 px
        trainee.advance(fast)
        opponent.advance(torch.zeros((1, 2, NEXT_STATE_OUTPUT_DIM)))

        _place(env, enemy_x=_HIDDEN_X)
        visibility = team_visibility_from_state(env.state, _SHIP, _CONFIG, False)
        assert trainee.observe(env.state, visibility.ship[:, 0]).pos[0, 1, 0] == pytest.approx(
            _VISIBLE_X + 20.0
        )
        assert opponent.observe(env.state, visibility.ship[:, 0]).pos[0, 1, 0] == pytest.approx(
            _VISIBLE_X
        )


class TestActionBelief:
    def test_the_prediction_from_t_appears_at_t_plus_one(self):
        env = _env()
        tracker = _tracker()
        current = _view(env, tracker)
        torch.testing.assert_close(
            current[ObsKey.PREVIOUS_ACTION][0, 1], torch.full((30,), 1.0 / 30)
        )

        logits = torch.full((1, 2, 30), -8.0)
        logits[0, 1, 17] = 8.0
        tracker.advance(_HOLD, logits)

        _place(env, enemy_x=_HIDDEN_X)
        later = _view(env, tracker)
        torch.testing.assert_close(later[ObsKey.PREVIOUS_ACTION][0, 1], logits[0, 1].softmax(-1))
        # The allied command stays environment truth, exactly one-hot.
        ally = later[ObsKey.PREVIOUS_ACTION][0, 0]
        assert ally.sum().item() == pytest.approx(1.0)
        assert ally.max().item() == 1.0

    def test_a_visible_enemy_still_shows_the_prediction_not_its_command(self):
        env = _env()
        tracker = _tracker()
        logits = torch.full((1, 2, 30), -8.0)
        logits[0, 1, 3] = 8.0
        _view(env, tracker)
        tracker.advance(_HOLD, logits)
        view = _view(env, tracker)  # enemy is in sight the whole time
        assert view[ObsKey.VISIBLE][0, 1]
        torch.testing.assert_close(view[ObsKey.PREVIOUS_ACTION][0, 1], logits[0, 1].softmax(-1))

    def test_the_spawn_null_overrides_a_stale_prediction(self):
        env = _env()
        tracker = _tracker()
        logits = torch.full((1, 2, 30), -8.0)
        logits[0, 1, 17] = 8.0
        _view(env, tracker)
        tracker.advance(_HOLD, logits)

        # A spawned ship's queue is null by construction -- the scheduler
        # neutralizes a command across a death/respawn -- and that null is public.
        env.state.ship_spawned[:] = True
        env.state.prev_action[:] = 0
        view = _view(env, tracker)
        expected = F.one_hot(torch.tensor(0), NUM_JOINT_ACTIONS).float()
        torch.testing.assert_close(view[ObsKey.PREVIOUS_ACTION][0, 1], expected)

    def test_physical_indices_are_never_remapped_by_canonicalization(self):
        from boost_and_broadside.train.rl.opponents import flip_team_obs

        env = _env()
        trackers = DualBeliefTracker(1, 2, 0.1, _SHIP, "cpu")
        visibility = team_visibility_from_state(env.state, _SHIP, _CONFIG, False)
        trackers.observe(env.state, visibility.ship)
        logits0 = torch.full((1, 2, 30), -8.0)
        logits1 = torch.full((1, 2, 30), -8.0)
        logits0[0, 1, 6] = 8.0  # Team 0 predicts physical ship 1.
        logits1[0, 0, 15] = 8.0  # Team 1 predicts physical ship 0.
        trackers.advance(_HOLD, _HOLD, logits0, logits1)

        visibility = team_visibility_from_state(env.state, _SHIP, _CONFIG, False)
        source0, source1 = trackers.observe(env.state, visibility.ship)
        view0 = observation_from_state(
            env.state,
            _SHIP,
            ship_visibility=visibility.ship[:, 0],
            perspective_team=0,
            belief=source0,
        )
        view1 = observation_from_state(
            env.state,
            _SHIP,
            ship_visibility=visibility.ship[:, 1],
            perspective_team=1,
            belief=source1,
        )
        assert view0[ObsKey.PREVIOUS_ACTION][0, 1].argmax().item() == 6
        assert view1[ObsKey.PREVIOUS_ACTION][0, 0].argmax().item() == 15
        canonical = flip_team_obs(view1, 2)
        assert canonical[ObsKey.PREVIOUS_ACTION][0, 0].argmax().item() == 15
        assert canonical[ObsKey.TEAM_ID][0, 0].item() == 1
        assert canonical[ObsKey.TEAM_ID][0, 1].item() == 0


class TestUncertainty:
    def test_the_head_states_the_next_spread_rather_than_accumulating(self):
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        prediction = torch.zeros((1, 2, NEXT_STATE_OUTPUT_DIM))
        prediction[0, 1, PHYSICAL_MEAN_DIM] = 2.0  # position log sigma x
        tracker.advance(prediction)
        _place(env, enemy_x=_HIDDEN_X)
        first = _view(env, tracker)
        assert first[ObsKey.BELIEF_UNCERTAINTY][0, 1, 0] == pytest.approx(2.0)

        tracker.advance(prediction)
        second = _view(env, tracker)
        assert second[ObsKey.BELIEF_UNCERTAINTY][0, 1, 0] == pytest.approx(2.0), (
            "a restated spread must not compound into 4.0"
        )

    def test_seeing_a_ship_settles_it(self):
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        prediction = torch.zeros((1, 2, NEXT_STATE_OUTPUT_DIM))
        prediction[0, 1, PHYSICAL_MEAN_DIM] = 5.0
        tracker.advance(prediction)
        _place(env, enemy_x=_HIDDEN_X)
        assert _view(env, tracker)[ObsKey.BELIEF_UNCERTAINTY][0, 1, 0] == pytest.approx(5.0)

        tracker.advance(prediction)
        _place(env, enemy_x=_VISIBLE_X)
        assert _view(env, tracker)[ObsKey.BELIEF_UNCERTAINTY][0, 1, 0] == pytest.approx(
            CERTAIN_LOG_SIGMA
        )


class TestLifecycle:
    def test_reacquisition_overwrites_the_forecast_and_reset_forgets(self):
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        tracker.advance(_HOLD)
        _place(env, enemy_x=_HIDDEN_X)
        _view(env, tracker)

        _place(env, enemy_x=177.0)
        reacquired = _view(env, tracker)
        assert reacquired[ObsKey.POS][0, 1, 0] == 177.0
        assert reacquired[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0.0

        tracker.reset(torch.tensor([True]))
        _place(env, enemy_x=_HIDDEN_X)
        assert not _view(env, tracker)[ObsKey.BELIEF_VALID][0, 1]

    def test_a_revealed_respawn_corrects_a_stale_belief(self):
        """The reveal ends a lifecycle discontinuity the tracker cannot see.

        The belief advances a hidden ship by the policy's own forecast and is
        never told it died, so without the reveal an unobserved respawn leaves it
        tracking a corpse's trajectory: the policy acts on a phantom at the old
        position, and the label carries a teleport no head could have predicted,
        on every step until the ship is next seen.
        """
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        for _ in range(3):
            tracker.advance(_HOLD)
            _place(env, enemy_x=_HIDDEN_X)
            stale = _view(env, tracker)
        assert stale[ObsKey.POS][0, 1, 0] == pytest.approx(_VISIBLE_X)

        tracker.advance(_HOLD)
        env.state.ship_spawned[0, 1] = True
        _place(env, enemy_x=900.0)
        revealed = _view(env, tracker)
        assert revealed[ObsKey.POS][0, 1, 0] == pytest.approx(900.0)
        assert revealed[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0

    def test_a_spawn_without_a_reveal_voids_the_belief(self):
        """A teleport invalidates whatever was remembered, reveal or no reveal."""
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        tracker.advance(_HOLD)
        env.state.ship_spawned[0, 1] = True
        _place(env, enemy_x=_HIDDEN_X)  # spawned far away and unseen
        view = _view(env, tracker)
        assert not view[ObsKey.BELIEF_VALID][0, 1]
        assert view[ObsKey.POS][0, 1].equal(torch.zeros(2))

    def test_the_opening_reveal_makes_validity_a_constant(self):
        """One revealed decision marks every ship valid for the rest of the episode.

        This is the property the attention key-mask removal rests on.
        """
        config = EnvConfig(
            num_ships=4,
            max_bullets=2,
            max_episode_steps=64,
            vision_range=100.0,
            spawn_reveal=True,
        )
        wrapper = YemongEnvWrapper(2, _SHIP, config, REWARDS, "cpu")
        wrapper.reset(seed=11)
        trackers = DualBeliefTracker(2, 4, 0.1, _SHIP, "cpu")
        prediction = torch.zeros((2, 4, NEXT_STATE_OUTPUT_DIM))

        view = wrapper.observe(trackers.observe(wrapper.env.state, wrapper.last_visibility.ship))
        assert view[ObsKey.BELIEF_VALID][:, :4].all(), "the opening decision reveals every ship"

        for _ in range(8):
            trackers.advance(prediction, prediction)
            wrapper.step(torch.zeros((2, 4, 3), dtype=torch.long), observe=False)
            view = wrapper.observe(
                trackers.observe(wrapper.env.state, wrapper.last_visibility.ship)
            )
            assert view[ObsKey.BELIEF_VALID][:, :4].all()


class TestNumericalSafety:
    def test_the_recursion_cannot_leave_its_bounded_set(self):
        """A hidden ship's belief is an autoregressive rollout with nothing else
        bounding it, so a small bias compounds for as long as it stays unseen.

        Run 734 died that way: velocity error in the 30s+ hidden bucket went
        99 -> 1178 px/s over ten updates and then overflowed, and the non-finite
        logits asserted inside ``multinomial``. Physical means make that
        impossible rather than merely counted -- every channel either wraps or
        clamps -- so what is checked here is the bound itself, not a repair.
        """
        env = _env()
        tracker = _tracker()
        _view(env, tracker)

        prediction = torch.full((1, 2, NEXT_STATE_OUTPUT_DIM), 1e9)
        prediction[0, 0, 0] = float("nan")
        prediction[0, 1, 0] = float("inf")
        for _ in range(10):
            tracker.advance(prediction)

        means = tracker.means
        assert torch.isfinite(means).all()
        assert torch.isfinite(means.double().square()).all()
        spec = PhysicalNextState.from_ship_config(_SHIP)
        upper = spec.upper_vector(means.device)
        lower = spec.lower_vector(means.device)
        finite = torch.isfinite(upper) & torch.isfinite(lower)
        assert (means[..., finite] <= upper[finite] + 1e-3).all()
        assert (means[..., finite] >= lower[finite] - 1e-3).all()
        assert (means[..., POSITION_X].abs() <= _SHIP.world_size[0]).all()
        assert int(tracker.clamp_events) > 0, "the non-finite outputs were counted"

    def test_the_counter_does_not_bind_on_ordinary_predictions(self):
        env = _env()
        tracker = _tracker()
        _view(env, tracker)
        tracker.advance(torch.full((1, 2, NEXT_STATE_OUTPUT_DIM), 0.5))
        assert torch.isfinite(tracker.means).all()
        assert int(tracker.clamp_events) == 0


class TestRespawnMode:
    def test_a_hidden_frontline_enemy_with_no_shields_remains_alive(self):
        env = _env(frontline=True)
        tracker = _tracker()
        _view(env, tracker)
        tracker.advance(_HOLD)
        tracker.means[0, 1, HEALTH] = 0.0
        tracker.means[0, 1, SHIELD_DELAY] = 0.0
        _place(env, enemy_x=_HIDDEN_X)
        view = _view(env, tracker)
        assert view[ObsKey.HEALTH][0, 1, 0] == 0
        assert view[ObsKey.ALIVE][0, 1], "instant respawn means a remembered ship is alive"
        assert view[ObsKey.BELIEF_VALID][0, 1]
        assert view[ObsKey.SHIELD_DELAY][0, 1, 0] == 0

    def test_an_imagined_frontline_ship_with_no_shields_remains_alive(self):
        from boost_and_broadside.evaluation.next_state import means_to_observation

        env = _env(frontline=True)
        view = _view(env, _tracker())
        means = torch.zeros(1, 2, PHYSICAL_MEAN_DIM)
        imagined = means_to_observation(
            means,
            view,
            torch.zeros(1, 2, 3, dtype=torch.long),
            2,
            2.0 * math.log(_SHIP.field_index_step),
        )
        assert imagined[ObsKey.ALIVE][0, :2].all()
        assert imagined[ObsKey.HEALTH][0, :2].eq(0).all()
