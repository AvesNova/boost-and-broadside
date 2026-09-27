"""Neither side may be able to tell which seat it is in.

Team 0 and Team 1 play the same game. A model handed the canonical team-0 view
must not be able to recover which physical side produced it, and both sides must
receive the same information about their own ships and about the enemy.

The regression these pin down: the rollout passed the *observation's* ``TEAM_ID``
channel to the pending-action privacy mask. Hidden ships are zeroed there and
zero is also Team 0's real id, so once every enemy was out of sight the channel
read as all zeros -- Team 0 then privatised nothing and saw every enemy's pending
command, while Team 1 privatised everything and could not see its own. Only
``state.ship_team_id`` is a correct source for that mask.

``spawn_reveal`` is on in every one of these, matching the training profiles.
With it off, a ship can be permanently unseen by one side, the belief tracker
never promotes it, and its channels stay at their masked zeros -- which produces
apparent asymmetries that are a property of the test config, not the code.
"""

from __future__ import annotations

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.observation import ObsKey
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.runtime.actions import PendingActionState
from boost_and_broadside.train.rl.belief import DualBeliefTracker
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.train.rl.opponents import flip_team_obs

NUM_SHIPS = 6
NUM_ENVS = 4
# Far enough in that ships have scattered and most enemies are out of sight. At
# reset every ship is revealed, which hides exactly this class of bug.
WARMUP_STEPS = 12


def _env_config() -> EnvConfig:
    return EnvConfig(
        num_ships=NUM_SHIPS,
        max_bullets=4,
        max_episode_steps=400,
        vision_range=300.0,
        spawn_reveal=True,
    )


def _wrapper() -> tuple[YemongEnvWrapper, ShipConfig, EnvConfig]:
    ship_config = ShipConfig()
    env_config = _env_config()
    wrapper = YemongEnvWrapper(
        num_envs=NUM_ENVS,
        ship_config=ship_config,
        env_config=env_config,
        rewards=REWARDS,
        device=torch.device("cpu"),
    )
    return wrapper, ship_config, env_config


def _is_exact_one_hot(distribution: torch.Tensor) -> torch.Tensor:
    return (
        distribution.sum(-1).eq(1)
        & distribution.eq(1).sum(-1).eq(1)
        & (distribution.eq(0) | distribution.eq(1)).all(-1)
    )


def _run_to_midgame(wrapper: YemongEnvWrapper, ship_config: ShipConfig):
    """Step with the scripted controller until enemies are genuinely hidden."""
    scripted = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    action_state = PendingActionState.allocate(NUM_ENVS, NUM_SHIPS, torch.device("cpu"))
    obs = wrapper.reset(seed=17)
    for _ in range(WARMUP_STEPS):
        selected = scripted.get_actions(wrapper.env.state, wrapper.last_visibility.ship).int()
        obs, _, dones, truncated, info = wrapper.step(action_state.applied_action())
        action_state.commit(selected, info["actuator_contiguous"], dones | truncated)
        action_state.write_observation(
            obs,
            wrapper.env.state.ship_team_id,
            wrapper.env.state.ship_spawned,
            NUM_SHIPS,
        )
    return obs


def test_observation_team_id_is_not_authoritative() -> None:
    """The masked TEAM_ID channel is not authoritative physical state."""
    wrapper, ship_config, _ = _wrapper()
    obs = _run_to_midgame(wrapper, ship_config)

    true_team = wrapper.env.state.ship_team_id
    observed_team = obs["team_id"][:, :NUM_SHIPS].long()
    hidden = ~wrapper.last_visibility.ship[:, 0].bool()

    assert hidden.any(), "warmup hid nothing; the test cannot bind"
    assert torch.all(observed_team[hidden] == 0)
    assert not torch.equal(observed_team, true_team.long())


def test_pending_action_privacy_hides_enemies_and_spares_own_ships() -> None:
    """Own ships are exact one-hots; raw enemy slots contain no command."""
    wrapper, ship_config, _ = _wrapper()
    obs = _run_to_midgame(wrapper, ship_config)

    true_team = wrapper.env.state.ship_team_id
    for observer, view in ((0, obs.data), (1, obs.team1_data)):
        pending = view[ObsKey.PREVIOUS_ACTION][:, :NUM_SHIPS]
        own = true_team == observer
        enemy = true_team != observer
        assert own.any() and enemy.any()
        assert _is_exact_one_hot(pending)[own].all()
        assert pending[enemy].eq(0).all(), f"observer {observer} can read enemy commands"


def test_pending_action_privacy_survives_the_team_flip() -> None:
    """Team flipping relabels teams without moving physical action slots."""
    wrapper, ship_config, _ = _wrapper()
    obs = _run_to_midgame(wrapper, ship_config)

    flipped = flip_team_obs(obs.for_team(1), NUM_SHIPS)
    pending = flipped.data[ObsKey.PREVIOUS_ACTION][:, :NUM_SHIPS]
    true_team = wrapper.env.state.ship_team_id
    own = true_team == 1
    enemy = true_team == 0

    assert _is_exact_one_hot(pending)[own].all()
    assert pending[enemy].eq(0).all()

def _composed_views(wrapper, ship_config, env_config, steps: int):
    """Drive both sides with one controller and return the composed seat views."""
    scripted = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    coordinator = build_standard_coordinator(ship_config)
    beliefs = DualBeliefTracker(
        NUM_ENVS,
        NUM_SHIPS,
        ship_config.dt * env_config.action_repeat,
        coordinator,
        torch.device("cpu"),
    )
    action_state = PendingActionState.allocate(NUM_ENVS, NUM_SHIPS, torch.device("cpu"))
    raw = wrapper.reset(seed=23)
    for _ in range(steps):
        beliefs.compose(raw)
        selected = scripted.get_actions(wrapper.env.state, wrapper.last_visibility.ship).int()
        raw, _, dones, truncated, info = wrapper.step(action_state.applied_action())
        action_state.commit(selected, info["actuator_contiguous"], dones | truncated)
        action_state.write_observation(
            raw,
            wrapper.env.state.ship_team_id,
            wrapper.env.state.ship_spawned,
            NUM_SHIPS,
        )
        beliefs.reset(dones | truncated)
    composed = beliefs.compose(raw)
    return composed.for_team(0), flip_team_obs(composed.for_team(1), NUM_SHIPS)


def test_composed_seat_labels_are_exactly_self_relative() -> None:
    """Each seat's composed view must label its own ships 0 and the enemy 1.

    Exact rather than distributional: a seat that mislabels even one slot has a
    different picture of the board than its opponent does.
    """
    wrapper, ship_config, env_config = _wrapper()
    view0, view1 = _composed_views(wrapper, ship_config, env_config, WARMUP_STEPS)
    true_team = wrapper.env.state.ship_team_id.long()

    assert torch.equal(view0.data[ObsKey.TEAM_ID][:, :NUM_SHIPS].long(), true_team)
    assert torch.equal(view1.data[ObsKey.TEAM_ID][:, :NUM_SHIPS].long(), 1 - true_team)


def test_composed_views_agree_on_ships_both_sides_can_see() -> None:
    """A ship visible to both seats must look identical from either one."""
    wrapper, ship_config, env_config = _wrapper()
    view0, view1 = _composed_views(wrapper, ship_config, env_config, WARMUP_STEPS)

    visibility = wrapper.last_visibility.ship
    both = visibility[:, 0].bool() & visibility[:, 1].bool()
    assert both.any(), "no ship was visible to both sides; the test cannot bind"

    for key in (ObsKey.HEALTH, ObsKey.POS, ObsKey.VEL, ObsKey.POWER, ObsKey.ALIVE):
        a = view0.data[key][:, :NUM_SHIPS]
        b = view1.data[key][:, :NUM_SHIPS]
        mask = both
        while mask.dim() < a.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(a)
        torch.testing.assert_close(a[mask], b[mask], msg=f"{key.name} differs between seats")


def _fixed_action_script(steps: int) -> list[torch.Tensor]:
    """One command sequence, indexed by slot, shared by a game and its mirror.

    A controller would do, but only if it were provably side-blind and consumed
    its randomness identically in both runs. A recorded script removes the
    question: the two games then differ in exactly one thing, which is the point.
    """
    generator = torch.Generator().manual_seed(4242)
    return [
        torch.stack(
            [
                torch.randint(0, 3, (NUM_ENVS, NUM_SHIPS), generator=generator),
                torch.randint(0, 7, (NUM_ENVS, NUM_SHIPS), generator=generator),
                torch.randint(0, 2, (NUM_ENVS, NUM_SHIPS), generator=generator),
            ],
            dim=-1,
        ).int()
        for _ in range(steps)
    ]


def _play_script(script: list[torch.Tensor], *, mirrored: bool):
    """Play the script, optionally with the two sides' labels exchanged.

    The mirror relabels ``ship_team_id`` once, immediately after reset, and then
    plays the identical commands per slot. Nothing physical changes: positions,
    velocities, damage and the fog geometry are all side-blind, and the win
    condition is symmetric, so the two runs stay in lockstep with only the labels
    different. Non-Frontline on purpose -- a zone carries the side that owns it,
    so mirroring the ships without mirroring the map is not a mirrored game.

    Returns the composed seat views, both canonicalized to "my team is 0".
    """
    wrapper, ship_config, env_config = _wrapper()
    coordinator = build_standard_coordinator(ship_config)
    beliefs = DualBeliefTracker(
        NUM_ENVS,
        NUM_SHIPS,
        ship_config.dt * env_config.action_repeat,
        coordinator,
        torch.device("cpu"),
    )
    action_state = PendingActionState.allocate(NUM_ENVS, NUM_SHIPS, torch.device("cpu"))
    raw = wrapper.reset(seed=17)
    if mirrored:
        state = wrapper.env.state
        state.ship_team_id = 1 - state.ship_team_id
        raw = wrapper._get_obs()
    for selected in script:
        beliefs.compose(raw)
        raw, _, dones, truncated, info = wrapper.step(action_state.applied_action())
        assert not (dones | truncated).any(), "an episode ended; the two runs may have diverged"
        action_state.commit(selected, info["actuator_contiguous"], dones | truncated)
        action_state.write_observation(
            raw,
            wrapper.env.state.ship_team_id,
            wrapper.env.state.ship_spawned,
            NUM_SHIPS,
        )
    composed = beliefs.compose(raw)
    return composed.for_team(0), flip_team_obs(composed.for_team(1), NUM_SHIPS), wrapper


def _mirror_pair():
    """Seat 1 of a game, and seat 0 of the mirror of that same game."""
    script = _fixed_action_script(WARMUP_STEPS)
    _, seat_one, wrapper = _play_script(script, mirrored=False)
    mirrored_seat_zero, _, mirror_wrapper = _play_script(script, mirrored=True)
    # The relabelling is the only difference: pin that the physics agreed, or a
    # failure below could be divergence rather than asymmetry.
    torch.testing.assert_close(
        wrapper.env.state.ship_pos, mirror_wrapper.env.state.ship_pos, msg="the runs diverged"
    )
    assert torch.equal(wrapper.env.state.ship_team_id, 1 - mirror_wrapper.env.state.ship_team_id), (
        "the mirror did not stay mirrored"
    )
    visibility = wrapper.last_visibility.ship
    assert not visibility[:, 0].all(), "nothing is hidden; the test cannot bind"
    return seat_one, mirrored_seat_zero


def test_the_mirrored_game_hands_each_seat_the_other_seat_s_view() -> None:
    """Relabelling the teams must exchange the two seats' canonical views exactly.

    This is the whole seat-symmetry contract in one assertion, and it covers
    every channel rather than the handful a "both sides can see it" test can
    reach: if any of them carried physical seat identity, the two sides here
    would differ.
    """
    seat_one, mirrored_seat_zero = _mirror_pair()

    assert set(seat_one.data) == set(mirrored_seat_zero.data)
    for key in seat_one.data:
        torch.testing.assert_close(
            seat_one.data[key],
            mirrored_seat_zero.data[key],
            msg=lambda message, key=key: f"{key} is not seat-symmetric\n{message}",
        )
    if seat_one.bullets is not None:
        for key in seat_one.bullets:
            torch.testing.assert_close(
                seat_one.bullets[key],
                mirrored_seat_zero.bullets[key],
                msg=lambda message, key=key: f"bullet {key} is not seat-symmetric\n{message}",
            )


def test_the_mirrored_game_gives_the_policy_the_same_answers() -> None:
    """And nothing downstream of the observation reintroduces seat identity.

    Redundant with the view comparison only as long as the model reads nothing
    but the view. That is what this pins: it fails if a head, the rotary
    geometry or the team pooling ever reaches around the observation for the
    physical side, or if a channel this test cannot name turns out to matter.
    """
    from boost_and_broadside.config import ModelConfig
    from boost_and_broadside.train.rl.policy_io import build_policy

    seat_one, mirrored_seat_zero = _mirror_pair()
    policy = build_policy(
        ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
        ShipConfig(),
        num_value_components=3,
        num_ships=NUM_SHIPS,
        team_pma_k=(2,),
    ).eval()

    outputs = []
    for view in (seat_one, mirrored_seat_zero):
        # Seeded per call: the action is sampled, so an unseeded comparison would
        # fail on the draw rather than on the distribution behind it.
        torch.manual_seed(0)
        with torch.no_grad():
            hidden = policy.initial_hidden(NUM_ENVS, NUM_SHIPS, torch.device("cpu"))
            outputs.append(policy.get_action_and_value(view, hidden))

    names = ("action", "logprob", "value", "pred_next", "hidden")
    for name, left, right in zip(names, *outputs):
        torch.testing.assert_close(left, right, msg=f"{name} is not seat-symmetric")
