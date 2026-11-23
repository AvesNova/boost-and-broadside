"""Rollout collection: the learner on one team per environment, an opponent on the other."""

import contextlib
import dataclasses
import importlib
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.func import functional_call, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

from lean.contract import random_keywords
from lean.learn.config import LearnConfig, policy_precision
from lean.learn.returns import value_centres
from lean.learn.rewards import Ledger, blank_ledger, clear_ledger, reward_step, tick_events
from lean.model import draw_position_offset, next_state_target
from lean.model.value import decode_value

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


def whole(tensor: Tensor) -> Tensor:
    """`tensor` with contiguous strides (a copy only when some stride is 0, a broadcast)."""
    return tensor.clone(memory_format=torch.contiguous_format) if 0 in tensor.stride() else tensor


def select_environments(chosen: Tensor, when_chosen: Tensor, otherwise: Tensor) -> Tensor:
    """Per-environment choice between two tensors with a leading E; `chosen` is (E,) bool."""
    return torch.where(chosen.view(-1, *[1] * (otherwise.dim() - 1)), when_chosen, otherwise)


@dataclass
class Opponent:
    """Who plays whom.

    The first round(self_play_share * E) environments are self-play: the live policy
    flies both teams and both teams' rows train. The rest are league games: the live
    policy against one member of `league` (lean.learn.pool.League), drawn per environment
    at every reset; only the live side trains. Member 0 is the scripted agent. Without a
    league every league game is against the scripted agent and nothing is tallied.
    """

    scripted_config: object
    league: object = None
    self_play_share: float = 0.5
    persona_spread: float = 0.0


@dataclass
class NextEpisodes:
    """Deals for the episodes that start during one segment, drawn together at the segment's
    start (agreed fixes 7 and 9): a finished environment takes the next unused one in order,
    so the deal, the team sizes, the personas, the learner team and the position offset are
    drawn once per segment instead of for every environment on every tick. P deals (P = E);
    `taken` counts those used. Past P a deal is used again (counted in
    Rollout.repeated_deals): only when more than E games end within one segment.
    """

    state: object  # a game deal of P environments at the state's N (lean.game.deal)
    persona: dict  # the scripted agent's per-ship parameters, tensors with leading P
    learner_team: Tensor  # (P,) long
    position_offset: Tensor  # (P, 2)
    taken: Tensor  # () long


@dataclass
class CollectCarry:
    """What persists between rollouts besides the game state.

    A seat is one team of one environment played by the live policy: seat s < E is the
    team `learner_team[s]` of environment s; seat E + e is the other team of self-play
    environment e. League environments are e >= E_s (E_s self-play environments).
    """

    policy: Tensor  # the live policy's recurrent carry over seats (S = E + E_s)
    opponent: Tensor  # the league opponents' carry over the league environments (L = E - E_s)
    member: Tensor  # (L,) long, the league member each league environment plays this episode
    learner_team: Tensor  # (E,) long
    persona: dict  # the scripted agent's per-ship parameters, tensors with leading E
    reset: Tensor  # (E,) bool, the environment was re-dealt before the next tick
    position_offset: Tensor  # (E, 2) the model's position offset of the current match
    ledger: Ledger  # the rewards' outstanding damage per environment (lean.learn.rewards)
    # (E,) long, each environment's row in the segment's deal table (Rollout.deal_table):
    # r < E the environment r as the segment started, E + k the segment's deal k.
    deal_row: Tensor
    next_episodes: NextEpisodes | None = None  # drawn by collect at each segment's start


@dataclass
class Rollout:
    """T ticks over S seats (see CollectCarry). Time first, as Policy.sequence takes it;
    the seat axis plays the part of Policy.sequence's environment axis.

    observation: each seat's view, every field (T, S, ...); `own` marks the valued and
    trained rows (that seat's living ships), `key_mask` the attendable ones.
    deal_row: (T, S) long, the seat's row in `deal_table`, {name: (2E, ...)}: the
    observation fields constant within an episode (lean.learn.store.EPISODE_CONSTANT, those
    the game's Obs has) and the position offset, once per deal (rows r < E: environment r as
    the segment started; E + k: the segment's deal k). A RolloutStore keeps these instead of
    the per-tick copies.
    action, teacher: (T, S, N) long joint actions (sampled; the scripted agent's choice).
    log_probability: (T, S, N). value: (T, S, N, LEVELS), zero outside `own`.
    reward: (T, S, N, LEVELS). done: (T, S) the seat's episode ended on this tick.
    reset: (T, S) the seat's environment was re-dealt just before this tick.
    position_offset: (T, S, 2).
    next_state: (T, S, N, NEXT_STATE) lean.model.next_state_target of the game step;
    next_valid: (T, S, N) that row is a target (own now, alive after the step, and the
    episode did not end on the tick).
    hit_label: (T, S, N) long, the outcome of the bullet the ship emitted on that tick
    (0 missed, 1 hit an enemy, 2 hit a team-mate; -1 unlabelled: no bullet, or it had not
    ended by the segment's end). Labels every slot of the seat's environment; the loss
    takes the `own` rows. Streamed into a store, the bullets fired in the previous segment
    that end in this one label that segment's rows (RolloutStore.end_segment); only the
    update's last segment keeps an unlabelled tail.
    bootstrap_value: (S, N, LEVELS) value after the last tick, zero outside `own`.
    value_centres: (LEVELS, BINS) the critic's support the values were decoded with.
    seat_environment, seat_team: (S,) the environment of each seat and its team at the
    last tick. self_play: (S,) bool, the seat's environment is a self-play one.
    initial_carry: the live policy's carry before the first tick.
    state, carry: where to continue collecting.
    Streamed into a RolloutStore (collect's `store`), the per-tick fields (observation to
    hit_label, deal_row) are None: the store holds them.
    league_games, league_wins, scripted_games, scripted_wins, episodes: 0-d tensors
    counted over this rollout (episodes over environments; a win is the live team
    winning a league game). repeated_deals: 0-d, episodes of this segment that started
    from a deal already used (NextEpisodes; zero unless more than E games ended).
    """

    observation: object
    action: Tensor | None
    teacher: Tensor | None
    log_probability: Tensor | None
    value: Tensor | None
    reward: Tensor | None
    done: Tensor | None
    reset: Tensor | None
    position_offset: Tensor | None
    next_state: Tensor | None
    next_valid: Tensor | None
    hit_label: Tensor | None
    deal_row: Tensor | None
    deal_table: dict
    bootstrap_value: Tensor
    value_centres: Tensor
    seat_environment: Tensor
    seat_team: Tensor
    self_play: Tensor
    initial_carry: Tensor
    state: object
    carry: CollectCarry
    league_games: Tensor
    league_wins: Tensor
    scripted_games: Tensor
    scripted_wins: Tensor
    episodes: Tensor
    repeated_deals: Tensor

    # The per-tick fields stacked from each tick's records (all but observation and hit_label).
    TICK_FIELDS = (
        "action", "teacher", "log_probability", "value", "reward", "done", "reset", "position_offset", "next_state",
        "next_valid", "deal_row",
    )


def carry_axis(carry: Tensor) -> int:
    """The environment axis of a recurrent carry: 0 for (E, N, width), 1 for
    lean.model's (blocks, E, 1 + N, 4, width)."""
    return 1 if carry.dim() >= 4 else 0


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


def draw_team_sizes(config: LearnConfig, envs: int, generator, device=None) -> Tensor:
    """Ships per team, (E, 2) long, uniform over the configured range; the two teams of
    an environment are equal when `equal_teams`, else drawn independently. `device`
    defaults to the generator's."""
    span = config.team_size_maximum - config.team_size_minimum + 1
    device = generator_device(generator) if device is None else device
    sizes = config.team_size_minimum + torch.randint(0, span, (envs, 2), **random_keywords(generator, device))
    if config.equal_teams:
        sizes[:, 1] = sizes[:, 0]
    return sizes


def scripted_uniforms(envs: int, slots: int, generator, device=None) -> Tensor:
    """The uniform draws lean.scripted.act consumes for one tick. Assumed (E, N, 3)."""
    device = generator_device(generator) if device is None else device
    return torch.rand(envs, slots, 3, **random_keywords(generator, device))


def scripted_actions(scripted, scripted_config, state, persona, generator) -> Tensor:
    """The scripted agent's action for every ship of both teams, (E, N, 3) long, in one call."""
    envs, slots = state.exists.shape
    uniforms = scripted_uniforms(envs, slots, generator, state.exists.device)
    return scripted.act(scripted_config, state, None, persona, uniforms)


def sample_joint(logits: Tensor, generator) -> Tensor:
    """One sample per row from categorical logits (..., JOINT), by the Gumbel-max trick."""
    uniform = torch.rand(logits.shape, **random_keywords(generator, logits.device)).clamp(1e-10, 1.0 - 1e-7)
    return (logits - torch.log(-torch.log(uniform))).argmax(-1)


def self_play_count(opponent: Opponent, envs: int) -> int:
    """E_s, the number of self-play environments (the first E_s indices)."""
    return round(opponent.self_play_share * envs)


def seats(learner_team: Tensor, self_play: int) -> tuple[Tensor, Tensor]:
    """(seat_environment, seat_team), each (S,) long: every environment's learner team,
    then the other team of the first `self_play` environments."""
    envs = learner_team.shape[0]
    environment = torch.cat([torch.arange(envs, device=learner_team.device), torch.arange(self_play, device=learner_team.device)])
    team = torch.cat([learner_team, 1 - learner_team[:self_play]])
    return environment, team


def seat_view(views, environment: Tensor, team: Tensor):
    """The view of team `team[s]` in environment `environment[s]`, merged from the two
    fixed-team views `views` = (team 0's, team 1's), each with a leading E."""
    first, second = views
    chosen = team == 0
    changes = {
        field.name: select_environments(
            chosen, getattr(first, field.name).index_select(0, environment), getattr(second, field.name).index_select(0, environment)
        )
        for field in dataclasses.fields(first)
        if isinstance(getattr(first, field.name), Tensor)
    }
    return dataclasses.replace(first, **changes)


def draw_members(opponent: Opponent, count: int, generator, device) -> Tensor:
    """League members for `count` league environments: drawn from the league, or all the
    scripted agent (member 0) without one."""
    if opponent.league is None or count == 0:
        return torch.zeros(count, dtype=torch.long, device=device)
    return opponent.league.draw(count, generator)


def snapshot_actions(config: LearnConfig, template, parameters: dict, buffers: dict, views, carry: CollectCarry,
                     scripted_action: Tensor, generator):
    """The league side's actions (L, N, 3) and opponent carry when the league holds
    snapshots: every snapshot slot runs on every league environment in one batched call
    (torch.func.vmap over the stacked weights, League.weights, through the weightless
    `template`), and each environment keeps the output of the member it plays. Scripted
    environments take `scripted_action`'s rows and keep their carry. A fixed amount of
    work (capacity x L rows), no read-back and no loop over members."""
    league_envs = carry.member.shape[0]
    envs = carry.learner_team.shape[0]
    first = envs - league_envs
    device = carry.member.device
    view = seat_view(views, torch.arange(first, envs, device=device), 1 - carry.learner_team[first:])
    offset = carry.position_offset[first:]
    slots = next(iter((parameters or buffers).values())).shape[0]

    def one_member(member_parameters, member_buffers, _):
        return functional_call(template, (member_parameters, member_buffers), (view, carry.opponent, offset))

    # Eagerly, vmap hands the fused attention kernels a mask whose batch stride they reject
    # on the GPU ("attn_bias is not correctly aligned"); the math kernel takes any layout.
    # Compiled, the attention is decomposed and keeps the fused path.
    eager_attention = contextlib.nullcontext() if torch.compiler.is_compiling() else sdpa_kernel(SDPBackend.MATH)
    with policy_precision(config, device), eager_attention:
        # The slot index is a batched input so that vmap has one even for a weightless policy.
        logits, _, new_carry = vmap(one_member)(parameters, buffers, torch.arange(slots, device=device))
    # Each environment's own member: slot member - 1 (scripted environments read slot 0 and
    # are overwritten below).
    slot = (carry.member - 1).clamp(min=0)
    rows = torch.arange(league_envs, device=device)
    snapshot = carry.member > 0
    axis = carry_axis(carry.opponent)
    chosen_carry = new_carry.movedim(axis + 1, 1)[slot, rows].movedim(0, axis)
    sampled = split_joint(sample_joint(logits[slot, rows].float(), generator))
    action = torch.where(snapshot[:, None, None], sampled, scripted_action[first:])
    shape = [1] * carry.opponent.dim()
    shape[axis] = -1
    # Contiguous, so the carry keeps one layout from tick to tick (a compiled graph is
    # specialised on strides).
    chosen_carry = chosen_carry.to(carry.opponent.dtype).contiguous()
    return action, torch.where(snapshot.view(shape), chosen_carry, carry.opponent).contiguous()


def league_actions(config: LearnConfig, opponent: Opponent, views, carry: CollectCarry, scripted_action: Tensor,
                   generator, snapshots=snapshot_actions):
    """Actions of the league side of every league environment, (L, N, 3), and the new
    opponent carry: the scripted agent's (already computed as the BC target) until the
    league holds a snapshot, then `snapshots` (snapshot_actions or its compiled form)."""
    league = opponent.league
    league_envs = carry.member.shape[0]
    if league is None or league.snapshots == 0 or league_envs == 0:
        return scripted_action[scripted_action.shape[0] - league_envs:], carry.opponent
    parameters, buffers = league.weights
    return snapshots(config, league.template, parameters, buffers, views, carry, scripted_action, generator)


def draw_offset(policy, envs: int, generator, device) -> Tensor:
    """lean.model.draw_position_offset drawn directly on `device` (that one draws on the
    generator's device, the CPU without one, which a compiled tick must not)."""
    uniform = torch.rand(envs, 2, **random_keywords(generator, device))
    return (uniform - 0.5) * policy.config.offset_span


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
    self_play = self_play_count(opponent, envs)
    carry = CollectCarry(
        policy=policy.initial_carry(envs + self_play, slots, device),
        opponent=policy.initial_carry(envs - self_play, slots, device),
        member=draw_members(opponent, envs - self_play, generator, device),
        learner_team=torch.randint(0, 2, (envs,), **random_keywords(generator, device)),
        persona=scripted.personas(opponent.scripted_config, envs, slots, generator, device, opponent.persona_spread),
        reset=torch.ones(envs, dtype=torch.bool, device=device),
        position_offset=draw_position_offset(policy.config, envs, generator, device),
        ledger=blank_ledger(envs, slots, device),
        deal_row=torch.arange(envs, device=device),
    )
    return state, carry


def draw_next_episodes(config: LearnConfig, game_config, policy, opponent: Opponent, envs: int, slots: int,
                       generator, device, game, scripted) -> NextEpisodes:
    """E deals for the episodes that start during the next segment (NextEpisodes)."""
    sizes = draw_team_sizes(config, envs, generator, device)
    return NextEpisodes(
        state=game.deal(game_config, envs, slots, sizes, generator, device),
        persona=scripted.personas(opponent.scripted_config, envs, slots, generator, device, opponent.persona_spread),
        learner_team=torch.randint(0, 2, (envs,), **random_keywords(generator, device)),
        position_offset=draw_offset(policy, envs, generator, device),
        taken=torch.zeros((), dtype=torch.long, device=device),
    )


def redeal(opponent: Opponent, state, carry: CollectCarry, done: Tensor, generator, game):
    """Deal the environments in `done` (E,) bool afresh from carry.next_episodes, in order
    (new team sizes, learner team, personas and position offset), and draw their league
    members (from the league's tallies as they stand); both carries zeroed there. Nothing is
    read back."""
    envs = done.shape[0]
    device = done.device
    self_play = envs - carry.member.shape[0]
    pool = carry.next_episodes
    # The k-th environment to finish in this segment takes deal k (modulo P).
    index = (pool.taken + torch.cumsum(done.long(), 0) - 1) % pool.learner_team.shape[0]
    state = game.take_deal(state, done, pool.state, index)

    def take(fresh: Tensor, old: Tensor) -> Tensor:
        return select_environments(done, fresh.index_select(0, index), old)

    seat_environment, _ = seats(carry.learner_team, self_play)
    league_done = done[self_play:]
    carry = CollectCarry(
        policy=reset_carry(carry.policy, done.index_select(0, seat_environment)),
        opponent=reset_carry(carry.opponent, league_done),
        member=torch.where(league_done, draw_members(opponent, league_done.shape[0], generator, device), carry.member),
        learner_team=take(pool.learner_team, carry.learner_team),
        persona={name: take(pool.persona[name], carry.persona[name]) for name in carry.persona},
        reset=done,
        position_offset=take(pool.position_offset, carry.position_offset),
        ledger=clear_ledger(carry.ledger, done),
        deal_row=torch.where(done, envs + index, carry.deal_row),
        next_episodes=dataclasses.replace(pool, taken=pool.taken + done.sum()),
    )
    return state, carry


def abandon(config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, member: int,
            generator, *, game=None, scripted=None):
    """Re-deal the league environments playing `member` (whose weights the league just
    replaced) and draw them new opponents. Their unfinished games are not tallied. The
    previous rollout bootstrapped those seats from the old state, as at a segment end.
    Returns (state, carry); carry.reset marks the re-dealt environments."""
    game = game or importlib.import_module("lean.game")
    scripted = scripted or importlib.import_module("lean.scripted")
    envs, slots = state.exists.shape
    self_play = envs - carry.member.shape[0]
    chosen = torch.cat([torch.zeros(self_play, dtype=torch.bool, device=carry.member.device), carry.member == member])
    # Its own deals (rare, between segments); the next segment draws afresh.
    pool = draw_next_episodes(config, game_config, policy, opponent, envs, slots, generator, carry.member.device, game, scripted)
    state, fresh = redeal(opponent, state, dataclasses.replace(carry, next_episodes=pool), chosen, generator, game)
    return state, dataclasses.replace(fresh, reset=carry.reset | chosen)


def hit_label_entries(step_out, state, label_row: Tensor | int, slot_index: Tensor, environment_index: Tensor, spare: int):
    """(index, outcome) for scattering this tick's ended bullets into the flat hit labels
    (row, environment, slot) of `collect`.

    `label_row` is this tick's row in the labels. A bullet that ended on this tick left the
    gun on episode tick step_out.shot_tick; its row is label_row - (state.tick - shot_tick),
    since episode ticks and rows advance together within an episode (a bullet of the
    current episode can never reach a row before the episode's first). A bullet whose row
    would be negative goes to the spare last entry `spare`.
    """
    envs, slots = state.exists.shape
    row = label_row - (state.tick[:, None, None] - step_out.shot_tick)
    landed = (step_out.shot_outcome >= 0) & (row >= 0)
    index = (row * envs + environment_index) * slots + slot_index
    index = torch.where(landed, index, spare)
    return index.flatten(), step_out.shot_outcome.flatten()


def act(config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, generator, game, scripted):
    """The first half of a tick: both teams' views, the live policy on every seat (its
    sampled joint action, log-probability and value) and the scripted agent's action for
    every ship. Returns (views, observation, seat_offset, joint, log_probability, value,
    teacher, policy_carry, centres), centres the critic's support (LEVELS, BINS)."""
    envs = state.exists.shape[0]
    self_play = envs - carry.member.shape[0]
    seat_environment, seat_team = seats(carry.learner_team, self_play)
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    observation = seat_view(views, seat_environment, seat_team)
    seat_offset = carry.position_offset.index_select(0, seat_environment)
    with policy_precision(config, seat_offset.device):
        logits, value_logits, policy_carry = policy.step(observation, carry.policy, seat_offset)
    logits, value_logits = logits.float(), value_logits.float()
    joint = sample_joint(logits, generator)
    log_probability = torch.log_softmax(logits, -1).gather(-1, joint[..., None]).squeeze(-1)
    centres = value_centres(config, value_logits.shape[-1], value_logits.device)
    value = decode_value(value_logits, centres) * observation.own[..., None]
    teacher = scripted_actions(scripted, opponent.scripted_config, state, carry.persona, generator)
    return views, observation, seat_offset, joint, log_probability, value, teacher, policy_carry, centres


def advance(
    config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, observation,
    joint: Tensor, teacher: Tensor, league_action: Tensor, opponent_carry: Tensor, policy_carry: Tensor,
    label_row: Tensor, hit_label: Tensor, counters: dict, generator, game, scripted,
):
    """The second half of a tick: every ship's action, the game step, rewards, hit labels
    (written into `hit_label` in place, `label_row` this tick's row there, see
    hit_label_entries), the counters and league tallies, then the re-deal of finished
    environments. `label_row` is a 0-d long tensor (an int would be a new constant for a
    compiled graph every tick). Returns (state, carry, this tick's
    records by Rollout field name, counters)."""
    envs, slots = state.exists.shape
    device = state.exists.device
    self_play = envs - carry.member.shape[0]
    seat_environment, _ = seats(carry.learner_team, self_play)
    live_action = split_joint(joint)
    other_side = torch.cat([live_action[envs:], league_action])
    learner_ship = state.team == carry.learner_team[:, None]
    action = torch.where(learner_ship[..., None], live_action[:envs], other_side)

    noise = game.draw_noise(game_config, envs, slots, generator, device)
    next_state, step_out = game.step(game_config, state, action, noise)
    done = step_out.done
    seat_done = done.index_select(0, seat_environment)
    reward, ledger = reward_step(config, carry.ledger, tick_events(step_out, state, next_state, game_config), state.team)
    slot_index = torch.arange(slots, device=device)[None, :, None]
    environment_index = torch.arange(envs, device=device)[:, None, None]
    spare = hit_label.shape[0] - 1
    hit_label.scatter_(0, *hit_label_entries(step_out, state, label_row, slot_index, environment_index, spare))
    records = {
        "teacher": join_action(teacher).index_select(0, seat_environment),
        "reward": reward.index_select(0, seat_environment),
        "done": seat_done,
        "reset": carry.reset.index_select(0, seat_environment),
        "next_state": next_state_target(policy.config, state, next_state).index_select(0, seat_environment),
        "next_valid": observation.own & next_state.alive.index_select(0, seat_environment) & ~seat_done[:, None],
    }

    finished = done[self_play:]
    won = step_out.result[self_play:] == carry.learner_team[self_play:]
    drawn = step_out.result[self_play:] == 2
    against_scripted = finished & (carry.member == 0)
    counters = {
        "league_games": counters["league_games"] + finished.sum(),
        "league_wins": counters["league_wins"] + (finished & won).sum(),
        "scripted_games": counters["scripted_games"] + against_scripted.sum(),
        "scripted_wins": counters["scripted_wins"] + (against_scripted & won).sum(),
        "episodes": counters["episodes"] + done.sum(),
    }
    if opponent.league is not None:
        opponent.league.record(carry.member, finished, won, drawn)

    carry = dataclasses.replace(carry, policy=policy_carry, opponent=opponent_carry, ledger=ledger)
    state, carry = redeal(opponent, next_state, carry, done, generator, game)
    return state, carry, records, counters


def deal_table(game_config, state, carry: CollectCarry, game) -> dict[str, Tensor]:
    """The segment's deal table (Rollout.deal_table): the episode-constant observation fields
    the game's Obs has and the position offset, for the environments as the segment starts
    and then for carry.next_episodes' deals. Two observe calls per segment, eagerly."""
    from lean.learn.store import EPISODE_CONSTANT

    pool = carry.next_episodes
    now, dealt = game.observe(game_config, state, 0), game.observe(game_config, pool.state, 0)
    table = {name: torch.cat([getattr(now, name), getattr(dealt, name)]) for name in EPISODE_CONSTANT if hasattr(now, name)}
    table["position_offset"] = torch.cat([carry.position_offset, pool.position_offset])
    return table


@torch.no_grad()
def collect(
    config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, ticks: int, generator,
    *, game=None, scripted=None, compiled: bool = False, store=None, segment: int = 0,
) -> Rollout:
    """Step E environments for `ticks` ticks and record every seat the live policy plays.

    Self-play environments (the first E_s) have two live seats, league environments one;
    the league side plays the member `carry.member` (the scripted agent or a frozen
    snapshot). The scripted agent's action is computed for every ship every tick: it is
    the BC target and the scripted member's move. Finished environments are re-dealt at
    once from deals drawn together at the segment's start (NextEpisodes: team sizes, the
    game's deal, personas, learner team, position offset) with a league member drawn as they
    finish, and the carries are zeroed there; league games that end are added to the
    league's tallies.

    `store` (a lean.learn.store.RolloutStore): each tick's records are streamed into its
    segment `segment` as they are made (asynchronous copies to pinned memory on a GPU) and
    the returned Rollout holds no per-tick fields. Without it they are stacked on the device.

    All random draws come from `generator`; with `compiled` (a tick's two halves, the
    snapshots' moves run as lean.learn.compiled's graphs) they come from
    torch's default generator of the device instead, since a compiled graph cannot take a
    generator object. Nothing is read back to the host in the loop.
    """
    game = game or importlib.import_module("lean.game")
    scripted = scripted or importlib.import_module("lean.scripted")
    envs, slots = state.exists.shape
    device = state.exists.device
    self_play = envs - carry.member.shape[0]
    if compiled:
        from lean.learn import compiled as graphs

        act_part, advance_part, snapshots = graphs.act, graphs.advance, graphs.snapshot_actions
        generator = None
        # A fresh state can hold broadcast (stride 0) tensors where a re-dealt one holds
        # whole ones (even empty ones: fields with F = 0); the graphs are specialised on
        # strides, so start from whole tensors.
        state = map_fields(state, whole)
        if dataclasses.is_dataclass(getattr(state, "constants", None)):
            state = dataclasses.replace(state, constants=map_fields(state.constants, whole))
    else:
        act_part, advance_part, snapshots = act, advance, snapshot_actions
    carry = dataclasses.replace(
        carry,
        next_episodes=draw_next_episodes(config, game_config, policy, opponent, envs, slots, generator, device, game, scripted),
        deal_row=torch.arange(envs, device=device),
    )
    table = deal_table(game_config, state, carry, game)
    initial_carry = carry.policy
    if store is not None:
        store.begin_segment(segment, ticks, initial_carry, table)
    records = {}
    counters = {name: torch.zeros((), dtype=torch.long, device=device) for name in ("league_games", "league_wins", "scripted_games", "scripted_wins", "episodes")}
    # Hit labels over (row, environment, slot), flat: rows 0 to T - 1 are the previous
    # segment's ticks (its bullets that end in this segment), rows T to 2T - 1 this
    # segment's; one spare entry at the end takes every bullet fired earlier.
    hit_label = torch.full((2 * ticks * envs * slots + 1,), -1, dtype=torch.long, device=device)
    label_row = torch.full((), ticks, dtype=torch.long, device=device)
    seat_environment, _ = seats(carry.learner_team, self_play)

    for index in range(ticks):
        views, observation, seat_offset, joint, log_probability, value, teacher, policy_carry, _ = act_part(
            config, game_config, policy, opponent, state, carry, generator, game, scripted
        )
        deal_row = carry.deal_row.index_select(0, seat_environment)
        league_action, opponent_carry = league_actions(config, opponent, views, carry, teacher, generator, snapshots)
        state, carry, tick_records, counters = advance_part(
            config, game_config, policy, opponent, state, carry, observation, joint, teacher, league_action,
            opponent_carry, policy_carry, label_row, hit_label, counters, generator, game, scripted,
        )
        tick_records = {
            "observation": observation, "action": joint, "log_probability": log_probability, "value": value,
            "position_offset": seat_offset, "deal_row": deal_row, **tick_records,
        }
        if store is not None:
            store.write_tick(segment, index, tick_records)
        else:
            for name, item in tick_records.items():
                records.setdefault(name, []).append(item)
        label_row = label_row + 1

    seat_environment, seat_team = seats(carry.learner_team, self_play)
    # The value after the last tick from the tick's first half (one graph fewer to compile;
    # its samples and the scripted actions are discarded).
    final = act_part(config, game_config, policy, opponent, state, carry, generator, game, scripted)
    bootstrap_value, centres = final[5], final[8]
    # After a reset the final observation belongs to a new episode; GAE never reads the
    # bootstrap there because `done` on the last tick cuts it.
    # A seat is the same environment's same slots in every segment (`seats`), so the
    # previous segment's rows go to the same seats there.
    previous_hit_label, seat_hit_label = hit_label[:-1].view(2, ticks, envs, slots).index_select(2, seat_environment).unbind()

    if store is not None:
        # The first segment's earlier rows belong to the previous update (already trained on).
        store.end_segment(segment, seat_hit_label, bootstrap_value, centres, previous_hit_label if segment > 0 else None)
        observation, stacked, seat_hit_label = None, dict.fromkeys(Rollout.TICK_FIELDS), None
    else:
        first = records["observation"][0]
        observation = dataclasses.replace(first, **{
            field.name: torch.stack([getattr(item, field.name) for item in records["observation"]])
            for field in dataclasses.fields(first)
            if isinstance(getattr(first, field.name), Tensor)
        })
        stacked = {name: torch.stack(records[name]) for name in Rollout.TICK_FIELDS}
    return Rollout(
        observation=observation,
        **stacked,
        hit_label=seat_hit_label,
        deal_table=table,
        bootstrap_value=bootstrap_value,
        value_centres=centres,
        seat_environment=seat_environment,
        seat_team=seat_team,
        self_play=seat_environment < self_play,
        initial_carry=initial_carry,
        state=state,
        carry=carry,
        repeated_deals=(carry.next_episodes.taken - envs).clamp(min=0),
        **counters,
    )
