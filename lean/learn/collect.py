"""Rollout collection: the learner on one team per environment, an opponent on the other."""

import dataclasses
import importlib
from dataclasses import dataclass

import torch
from torch import Tensor

from lean.learn.config import LearnConfig
from lean.learn.returns import value_centres
from lean.model import draw_position_offset, next_state_target
from lean.model.value import decode_value
from lean.learn.rewards import rewards

POWERS_TIMES_FIRES, FIRES = 6, 2


def split_joint(joint: Tensor) -> Tensor:
    """Joint action (...,) long -> (..., 3) long [stick, power, fire]; joint = stick * 6 + power * 2 + fire."""
    return torch.stack([joint // POWERS_TIMES_FIRES, (joint % POWERS_TIMES_FIRES) // FIRES, joint % FIRES], dim=-1)


def join_action(action: Tensor) -> Tensor:
    """(..., 3) long [stick, power, fire] -> joint action (...,) long."""
    return action[..., 0] * POWERS_TIMES_FIRES + action[..., 1] * FIRES + action[..., 2]


def map_fields(record, function):
    """A copy of a dataclass with `function` applied to every tensor field."""
    changes = {
        field.name: function(getattr(record, field.name))
        for field in dataclasses.fields(record)
        if isinstance(getattr(record, field.name), Tensor)
    }
    return dataclasses.replace(record, **changes)


def select_environments(chosen: Tensor, when_chosen: Tensor, otherwise: Tensor) -> Tensor:
    """Per-environment choice between two tensors with a leading E; `chosen` is (E,) bool."""
    return torch.where(chosen.view(-1, *[1] * (otherwise.dim() - 1)), when_chosen, otherwise)


@dataclass
class Opponent:
    """Who plays the other team.

    The first round(scripted_share * E) environments always face the scripted agent. The
    rest face `snapshot` (a frozen Policy) when one is given, otherwise the scripted
    agent too.
    """

    scripted_config: object
    snapshot: object = None
    scripted_share: float = 0.5
    persona_spread: float = 0.0


@dataclass
class CollectCarry:
    """What persists between rollouts besides the game state."""

    policy: Tensor  # the learner's recurrent carry, leading E
    snapshot: Tensor  # the snapshot opponent's carry, leading E
    learner_team: Tensor  # (E,) long, the team the learner plays this episode
    persona: dict  # the scripted agent's per-ship parameters, tensors with leading E
    reset: Tensor  # (E,) bool, the environment was re-dealt before the next tick
    position_offset: Tensor  # (E, 2) the model's position offset of the current match


@dataclass
class Rollout:
    """T ticks of E environments. Time first, as Policy.sequence takes it.

    observation: the learner's view, every field with a leading T; `own` marks the valued
    and trained rows, `key_mask` the attendable ones.
    action, teacher: (T, E, N) long joint actions (sampled; the scripted agent's choice).
    log_probability: (T, E, N). value: (T, E, N, LEVELS), zero outside `own`.
    reward: (T, E, N, LEVELS). done: (T, E) the episode ended on this tick.
    reset: (T, E) the environment was re-dealt just before this tick.
    position_offset: (T, E, 2) the model's position offset of the match played at that tick.
    next_state: (T, E, N, NEXT_STATE) lean.model.next_state_target of the game step;
    next_valid: (T, E, N) that row is a target (own now, alive after the step, and the
    episode did not end on the tick).
    bootstrap_value: (E, N, LEVELS) value after the last tick, zero outside `own`.
    initial_carry: the learner's carry before the first tick.
    state, carry: where to continue collecting.
    scripted_games, scripted_wins, episodes: 0-d tensors counted over this rollout;
    a win is the learner's team winning an episode against the scripted agent.
    """

    observation: object
    action: Tensor
    teacher: Tensor
    log_probability: Tensor
    value: Tensor
    reward: Tensor
    done: Tensor
    reset: Tensor
    position_offset: Tensor
    next_state: Tensor
    next_valid: Tensor
    bootstrap_value: Tensor
    initial_carry: Tensor
    state: object
    carry: CollectCarry
    scripted_games: Tensor
    scripted_wins: Tensor
    episodes: Tensor


def carry_axis(carry: Tensor) -> int:
    """The environment axis of a recurrent carry: 0 for (E, N, width), 1 for
    lean.model's (blocks, E, 1 + N, width)."""
    return 1 if carry.dim() == 4 else 0


def reset_carry(carry: Tensor, done: Tensor) -> Tensor:
    """Zero the carry of the environments in `done` (E,) bool."""
    shape = [1] * carry.dim()
    shape[carry_axis(carry)] = -1
    return carry * (~done).view(shape)


def index_carry(carry: Tensor, index: Tensor) -> Tensor:
    """The carry of the environments in `index`."""
    return carry.index_select(carry_axis(carry), index)


def generator_device(generator):
    """The device a generator draws on; the CPU when there is none."""
    return generator.device if generator is not None else torch.device("cpu")


def draw_team_sizes(config: LearnConfig, envs: int, generator) -> Tensor:
    """Ships per team, (E, 2) long, uniform over the configured range; the two teams of
    an environment are equal when `equal_teams`, else drawn independently."""
    span = config.team_size_maximum - config.team_size_minimum + 1
    sizes = config.team_size_minimum + torch.randint(0, span, (envs, 2), generator=generator, device=generator_device(generator))
    if config.equal_teams:
        sizes[:, 1] = sizes[:, 0]
    return sizes


def scripted_uniforms(envs: int, slots: int, generator) -> Tensor:
    """The uniform draws lean.scripted.act consumes for one tick. Assumed (E, N, 3)."""
    return torch.rand(envs, slots, 3, generator=generator, device=generator_device(generator))


def team_view(game, game_config, state, team: Tensor):
    """The observation of team `team[e]` in each environment e, merged from the two fixed-team views."""
    first = game.observe(game_config, state, 0)
    second = game.observe(game_config, state, 1)
    chosen = team == 0
    changes = {
        field.name: select_environments(chosen, getattr(first, field.name), getattr(second, field.name))
        for field in dataclasses.fields(first)
        if isinstance(getattr(first, field.name), Tensor)
    }
    return dataclasses.replace(first, **changes)


def scripted_actions(scripted, scripted_config, state, persona, generator) -> Tensor:
    """The scripted agent's action for every ship of both teams, (E, N, 3) long."""
    envs, slots = state.exists.shape
    uniforms = scripted_uniforms(envs, slots, generator)
    return scripted.act(scripted_config, state, 0, persona, uniforms) + scripted.act(scripted_config, state, 1, persona, uniforms)


def sample_joint(logits: Tensor, generator) -> Tensor:
    """One sample per row from categorical logits (..., JOINT), by the Gumbel-max trick."""
    uniform = torch.rand(logits.shape, generator=generator, device=logits.device).clamp(1e-10, 1.0 - 1e-7)
    return (logits - torch.log(-torch.log(uniform))).argmax(-1)


def start(config: LearnConfig, game_config, policy, opponent: Opponent, envs: int, generator, game=None, scripted=None):
    """Deal E environments and build the carry. Returns (state, CollectCarry)."""
    game = game or importlib.import_module("lean.game")
    scripted = scripted or importlib.import_module("lean.scripted")
    device = generator_device(generator)
    sizes = draw_team_sizes(config, envs, generator)
    # The game sizes its slot axis from the largest total dealt at reset.
    sizes[0] = config.team_size_maximum
    state = game.reset(game_config, envs, sizes, generator, device)
    slots = state.exists.shape[1]
    carry = CollectCarry(
        policy=policy.initial_carry(envs, slots, device),
        snapshot=policy.initial_carry(envs, slots, device),
        learner_team=torch.randint(0, 2, (envs,), generator=generator, device=device),
        persona=scripted.personas(opponent.scripted_config, envs, slots, generator, device, opponent.persona_spread),
        reset=torch.ones(envs, dtype=torch.bool, device=device),
        position_offset=draw_position_offset(policy.config, envs, generator, device),
    )
    return state, carry


@torch.no_grad()
def collect(
    config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, ticks: int, generator,
    *, shaping: float = 1.0, game=None, scripted=None,
) -> Rollout:
    """Step E environments for `ticks` ticks and record the learner's side.

    The learner's policy acts for team `carry.learner_team[e]` in environment e; the
    opponent acts for the other team. Finished environments are re-dealt at once with
    fresh team sizes, a fresh learner team and fresh personas, and both recurrent carries
    are zeroed there. All random draws come from `generator`, whose device is the device
    of the run. `shaping` is the taper passed to `rewards`. `game` and `scripted` default
    to the lean.game and lean.scripted modules. Nothing is read back to the host inside
    the loop.
    """
    game = game or importlib.import_module("lean.game")
    scripted = scripted or importlib.import_module("lean.scripted")
    envs, slots = state.exists.shape
    device = state.exists.device
    faces_scripted = torch.arange(envs, device=device) < round(opponent.scripted_share * envs)
    if opponent.snapshot is None:
        faces_scripted = torch.ones_like(faces_scripted)

    policy_carry, snapshot_carry = carry.policy, carry.snapshot
    learner_team, persona, reset = carry.learner_team, carry.persona, carry.reset
    position_offset = carry.position_offset
    initial_carry = policy_carry
    centres = None
    records = {name: [] for name in ("observation", "action", "teacher", "log_probability", "value", "reward", "done", "reset", "position_offset", "next_state", "next_valid")}
    scripted_games = torch.zeros((), dtype=torch.long, device=device)
    scripted_wins = torch.zeros((), dtype=torch.long, device=device)
    episodes = torch.zeros((), dtype=torch.long, device=device)

    for _ in range(ticks):
        observation = team_view(game, game_config, state, learner_team)
        logits, value_logits, policy_carry = policy.step(observation, policy_carry, position_offset)
        if centres is None:
            centres = value_centres(config, value_logits.shape[-1], device)
        joint = sample_joint(logits, generator)
        log_probability = torch.log_softmax(logits, -1).gather(-1, joint[..., None]).squeeze(-1)
        value = decode_value(value_logits, centres) * observation.own[..., None]

        teacher = scripted_actions(scripted, opponent.scripted_config, state, persona, generator)
        opponent_action = teacher
        if opponent.snapshot is not None:
            opponent_view = team_view(game, game_config, state, 1 - learner_team)
            opponent_logits, _, snapshot_carry = opponent.snapshot.step(opponent_view, snapshot_carry, position_offset)
            snapshot_action = split_joint(sample_joint(opponent_logits, generator))
            opponent_action = select_environments(faces_scripted, teacher, snapshot_action)
        learner_ship = state.team == learner_team[:, None]
        action = torch.where(learner_ship[..., None], split_joint(joint), opponent_action)

        noise = game.draw_noise(game_config, envs, slots, generator, device)
        next_state, step_out = game.step(game_config, state, action, noise)
        done = step_out.done

        for name, item in (
            ("observation", observation), ("action", joint), ("teacher", join_action(teacher)),
            ("log_probability", log_probability), ("value", value),
            ("reward", rewards(step_out, state, config, shaping)), ("done", done), ("reset", reset),
            ("position_offset", position_offset),
            ("next_state", next_state_target(policy.config, state, next_state)),
            ("next_valid", observation.own & next_state.alive & ~done[:, None]),
        ):
            records[name].append(item)

        against_scripted = done & faces_scripted
        scripted_games += against_scripted.sum()
        scripted_wins += (against_scripted & (step_out.result == learner_team)).sum()
        episodes += done.sum()

        state = game.reset_done(game_config, next_state, done, draw_team_sizes(config, envs, generator), generator)
        learner_team = torch.where(done, torch.randint(0, 2, (envs,), generator=generator, device=device), learner_team)
        fresh = scripted.personas(opponent.scripted_config, envs, slots, generator, device, opponent.persona_spread)
        persona = {name: select_environments(done, fresh[name], persona[name]) for name in persona}
        fresh_offset = draw_position_offset(policy.config, envs, generator, device)
        position_offset = torch.where(done[:, None], fresh_offset, position_offset)
        policy_carry = reset_carry(policy_carry, done)
        snapshot_carry = reset_carry(snapshot_carry, done)
        reset = done

    final_observation = team_view(game, game_config, state, learner_team)
    _, final_logits, _ = policy.step(final_observation, policy_carry, position_offset)
    bootstrap_value = decode_value(final_logits, centres) * final_observation.own[..., None]
    # After a reset the final observation belongs to a new episode; GAE never reads the
    # bootstrap there because `done` on the last tick cuts it.

    first = records["observation"][0]
    stacked = {
        field.name: torch.stack([getattr(item, field.name) for item in records["observation"]])
        for field in dataclasses.fields(first)
        if isinstance(getattr(first, field.name), Tensor)
    }
    observation = dataclasses.replace(first, **stacked)
    done = torch.stack(records["done"])
    return Rollout(
        observation=observation,
        action=torch.stack(records["action"]),
        teacher=torch.stack(records["teacher"]),
        log_probability=torch.stack(records["log_probability"]),
        value=torch.stack(records["value"]),
        reward=torch.stack(records["reward"]),
        done=done,
        reset=torch.stack(records["reset"]),
        position_offset=torch.stack(records["position_offset"]),
        next_state=torch.stack(records["next_state"]),
        next_valid=torch.stack(records["next_valid"]),
        bootstrap_value=bootstrap_value,
        initial_carry=initial_carry,
        state=state,
        carry=CollectCarry(policy_carry, snapshot_carry, learner_team, persona, reset, position_offset),
        scripted_games=scripted_games,
        scripted_wins=scripted_wins,
        episodes=episodes,
    )
