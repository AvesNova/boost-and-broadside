"""Rollout collection: the live policy on its seats, the league on the other side."""

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.func import functional_call

from lean import game, scripted
from lean.contract import POWERS, STICKS, Obs, random_keywords, split_joint
from lean.learn.config import LearnConfig, policy_precision
from lean.learn.returns import value_centres
from lean.learn.rewards import Ledger, blank_ledger, clear_ledger, reward_step, tick_events
from lean.model import draw_position_offset, next_state_target
from lean.model.value import decode_value
from lean.scripted import ScriptedConfig

# The seat (environment) axis of lean.model's recurrent carry, (blocks, E, 1 + N, 4, width).
SEAT_AXIS = 1
# The teacher's distribution as recorded: the stick's 7 probabilities, the power's 3, then
# the probability of firing (lean.scripted.Marginals; the three are independent).
TEACHER_WIDTH = STICKS + POWERS + 1


def teacher_marginals(parts) -> Tensor:
    """A lean.scripted.Marginals as one (E, N, TEACHER_WIDTH) float32 record."""
    return torch.cat([parts.stick, parts.power, parts.fire[..., None]], -1).float()


def teacher_joint(marginals: Tensor) -> Tensor:
    """The teacher's distribution over the 42 joint actions, (..., JOINT), from its
    recorded marginals (..., TEACHER_WIDTH): the product of the three, indexed as
    lean.contract.join_action (stick * 6 + power * 2 + fire)."""
    stick, power, fire = marginals.split((STICKS, POWERS, 1), -1)
    fire = torch.cat([1.0 - fire, fire], -1)
    joint = stick[..., :, None, None] * power[..., None, :, None] * fire[..., None, None, :]
    return joint.flatten(-3)


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

    The first round(LearnConfig.self_play_share * E) environments are self-play: the live
    policy flies both teams and both teams' rows train. The rest are league games: the live
    policy against one member of `league` (lean.learn.pool.League), drawn per environment at
    every reset within the members' row capacities; only the live side trains. Member 0 is
    the scripted agent (personas at LearnConfig.persona_spread). Without a league every
    league game is against the scripted agent and nothing is tallied.
    """

    scripted_config: ScriptedConfig
    league: object = None


@dataclass
class NextEpisodes:
    """Deals for the episodes that start during one segment, drawn together at the segment's
    start (agreed fixes 7 and 9): a finished environment takes the next unused one in order,
    so the deal, the team sizes, the personas, the learner team and the position offset are
    drawn once per segment instead of for every environment on every tick. P deals (P = E);
    `taken` counts those used. Past P a deal is used again: only when more than E games end
    within one segment.
    """

    state: object  # a game deal of P environments at the state's N (lean.game.deal)
    persona: dict | None  # the scripted agent's per-ship parameters, leading P (None: the yardstick)
    scripted_memory: dict  # the scripted agent's memory at a match's start (lean.scripted.fresh_memory)
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
    # (L,) long, the league member each league environment plays this episode (pool.RETIRED:
    # one that has left the league, its game finishing in a retiring slot, not tallied).
    member: Tensor
    slot: Tensor  # (L,) long, the network slot playing it (League.slot_weights), -1 the scripted agent
    learner_team: Tensor  # (E,) long
    persona: dict | None  # the scripted agent's per-ship parameters, leading E (None: the yardstick)
    # The scripted agent's memory (kept target, head-on side), leading E: one memory for every
    # ship, as the one scripted call decides for both teams (teacher and league opponent).
    scripted_memory: dict
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
    observation fields constant within an episode (lean.learn.store.EPISODE_CONSTANT) and
    the position offset, once per deal (rows r < E: environment r as the segment started;
    E + k: the segment's deal k). A RolloutStore keeps these instead of the per-tick copies.
    action: (T, S, N) long joint actions sampled on that tick's observation. teacher: (T, S,
    N, TEACHER_WIDTH) the scripted agent's full distribution there (teacher_marginals; the
    BC target, teacher_joint). The game delays every action by one tick: the
    action of tick t is applied on tick t + 1, and the action applied on tick t is the
    observation's pending one (observation.pending, for own living ships). PPO trains on the
    sampled action with log_probability: (T, S, N), its log-probability under the policy on
    tick t's observation. value: (T, S, N, LEVELS), zero outside `own`.
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
    """

    observation: Obs | None
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

    # The per-tick fields stacked from each tick's records (all but observation and hit_label).
    TICK_FIELDS = (
        "action", "teacher", "log_probability", "value", "reward", "done", "reset", "position_offset", "next_state",
        "next_valid", "deal_row",
    )


def reset_carry(carry: Tensor, done: Tensor) -> Tensor:
    """Zero lean.model's carry of the environments (seats) in `done` (E,) bool."""
    return carry * (~done).view(1, -1, 1, 1, 1)


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


def scripted_actions(scripted_config: ScriptedConfig, state, persona, memory: dict, generator) -> tuple[Tensor, dict, Tensor]:
    """The scripted agent's action for every ship of both teams, (E, N, 3) long, in one call,
    its new memory (the caller carries it to the next tick) and the distribution the actions
    were drawn from, (E, N, TEACHER_WIDTH) (teacher_marginals)."""
    envs, slots = state.exists.shape
    uniforms = scripted.draw_uniforms(envs, slots, generator, state.exists.device)
    action, memory, parts = scripted.decide(scripted_config, state, None, persona, uniforms, memory)
    return action, memory, teacher_marginals(parts)


def sample_joint(logits: Tensor, generator) -> Tensor:
    """One sample per row from categorical logits (..., JOINT), by the Gumbel-max trick."""
    uniform = torch.rand(logits.shape, **random_keywords(generator, logits.device)).clamp(1e-10, 1.0 - 1e-7)
    return (logits - torch.log(-torch.log(uniform))).argmax(-1)


def self_play_count(config: LearnConfig, envs: int) -> int:
    """E_s, the number of self-play environments (the first E_s indices)."""
    return round(config.self_play_share * envs)


def seats(learner_team: Tensor, self_play: int) -> tuple[Tensor, Tensor]:
    """(seat_environment, seat_team), each (S,) long: every environment's learner team,
    then the other team of the first `self_play` environments."""
    envs = learner_team.shape[0]
    environment = torch.cat([torch.arange(envs, device=learner_team.device), torch.arange(self_play, device=learner_team.device)])
    team = torch.cat([learner_team, 1 - learner_team[:self_play]])
    return environment, team


def seat_view(views: tuple[Obs, Obs], environment: Tensor, team: Tensor) -> Obs:
    """The view of team `team[s]` in environment `environment[s]`, merged from the two
    fixed-team views `views` = (team 0's, team 1's), each with a leading E."""
    first, second = views
    chosen = team == 0
    return Obs(**{
        field.name: select_environments(
            chosen, getattr(first, field.name).index_select(0, environment), getattr(second, field.name).index_select(0, environment)
        )
        for field in dataclasses.fields(first)
    })


def draw_members(opponent: Opponent, member: Tensor, done: Tensor, generator) -> tuple[Tensor, Tensor]:
    """(member, slot) of the league environments, new draws where `done` (L,) bool (League.draw);
    without a league every one plays the scripted agent (member 0, slot -1)."""
    if opponent.league is None or member.shape[0] == 0:
        return torch.zeros_like(member), torch.full_like(member, -1)
    return opponent.league.draw(member, done, generator)


def slot_actions(config: LearnConfig, template, parameters: dict, buffers: dict, slot: Tensor, positions: Tensor,
                 views: tuple[Obs, Obs], carry: CollectCarry, action: Tensor, opponent_carry: Tensor, generator):
    """One network slot's moves (capacity routing): the league environments it plays
    (carry.slot == `slot`, a 0-d long) take the rows 0, 1, ... of a W-row batch in order
    (W = positions.shape[0], League.prepare's table width: never fewer than the slot's
    environments), the slot's weights (through the weightless `template`) run once on those
    W rows, and each of its environments reads its row back into `action` (L, N, 3) and
    `opponent_carry`; the other environments keep theirs. Padding rows (fewer environments
    than W) run on environment 0's input and are never read. No host read, and one compiled
    graph for every slot, W and K."""
    league_envs = carry.member.shape[0]
    envs = carry.learner_team.shape[0]
    first = envs - league_envs
    device = carry.member.device
    width = positions.shape[0]
    mine = carry.slot == slot
    rank = mine.long().cumsum(0) - 1  # each of the slot's environments: its row
    row = torch.where(mine, rank, width)  # others to a spare row past the batch
    environment = torch.zeros(width + 1, dtype=torch.long, device=device)
    environment = environment.scatter(0, row, torch.arange(league_envs, device=device))[:-1]
    view = seat_view(views, first + environment, 1 - carry.learner_team.index_select(0, first + environment))
    rows_carry = carry.opponent.index_select(SEAT_AXIS, environment)
    offset = carry.position_offset[first:].index_select(0, environment)
    with policy_precision(config, device):
        logits, _, new_carry = functional_call(template, (parameters, buffers), (view, rows_carry, offset))
    read = rank.clamp(min=0, max=width - 1)
    sampled = split_joint(sample_joint(logits.float(), generator)).index_select(0, read)
    action = torch.where(mine[:, None, None], sampled, action)
    new_carry = new_carry.to(opponent_carry.dtype).index_select(SEAT_AXIS, read)
    # Contiguous, so the carry keeps one layout from tick to tick (a compiled graph is
    # specialised on strides).
    return action, torch.where(mine.view(1, -1, 1, 1, 1), new_carry, opponent_carry).contiguous()


def network_actions(config: LearnConfig, league, views: tuple[Obs, Obs], carry: CollectCarry, scripted_action: Tensor,
                    generator, step=slot_actions):
    """The league side's actions (L, N, 3) and opponent carry: the scripted agent's
    `scripted_action` rows, then each slot of the league's network table (League.prepare)
    moves its own environments (`step`: slot_actions or its compiled form), one call per
    slot holding weights."""
    league_envs = carry.member.shape[0]
    # A copy of the carry to write into: an input that aliases another (carry.opponent)
    # would be a second compiled graph.
    action, opponent_carry = scripted_action[scripted_action.shape[0] - league_envs:], carry.opponent.clone()
    for slot, parameters, buffers in league.table.slots:
        action, opponent_carry = step(
            config, league.template, parameters, buffers, slot, league.table.positions, views, carry, action,
            opponent_carry, generator,
        )
    return action, opponent_carry


def league_actions(config: LearnConfig, opponent: Opponent, views: tuple[Obs, Obs], carry: CollectCarry,
                   scripted_action: Tensor, generator, step=slot_actions):
    """Actions of the league side of every league environment, (L, N, 3), and the new
    opponent carry: the scripted agent's (already computed as the BC target) until the
    league has a network table (League.prepare), then network_actions (`step` the per-slot
    function, slot_actions or its compiled form)."""
    league = opponent.league
    league_envs = carry.member.shape[0]
    if league is None or league.table is None or league_envs == 0:
        return scripted_action[scripted_action.shape[0] - league_envs:], carry.opponent
    return network_actions(config, league, views, carry, scripted_action, generator, step)


def start(config: LearnConfig, game_config, policy, opponent: Opponent, envs: int, generator):
    """Deal E environments and build the carry. Returns (state, CollectCarry)."""
    device = generator_device(generator)
    sizes = draw_team_sizes(config, envs, generator)
    # The game sizes its slot axis from the largest total dealt at reset.
    sizes[0] = config.team_size_maximum
    state = game.reset(game_config, envs, sizes, generator, device)
    slots = state.exists.shape[1]
    self_play = self_play_count(config, envs)
    league_envs = envs - self_play
    unset = torch.zeros(league_envs, dtype=torch.long, device=device)
    if opponent.league is not None:
        opponent.league.prepare(league_envs, unset - 1)
    member, slot = draw_members(opponent, unset, torch.ones_like(unset, dtype=torch.bool), generator)
    carry = CollectCarry(
        policy=policy.initial_carry(envs + self_play, slots, device),
        opponent=policy.initial_carry(league_envs, slots, device),
        member=member,
        slot=slot,
        learner_team=torch.randint(0, 2, (envs,), **random_keywords(generator, device)),
        persona=scripted.personas(opponent.scripted_config, envs, slots, generator, device, config.persona_spread),
        scripted_memory=scripted.fresh_memory(envs, slots, generator, device),
        reset=torch.ones(envs, dtype=torch.bool, device=device),
        position_offset=draw_position_offset(policy.config, envs, generator, device),
        ledger=blank_ledger(envs, slots, device),
        deal_row=torch.arange(envs, device=device),
    )
    return state, carry


def draw_next_episodes(config: LearnConfig, game_config, policy, opponent: Opponent, envs: int, slots: int,
                       generator, device) -> NextEpisodes:
    """E deals for the episodes that start during the next segment (NextEpisodes)."""
    sizes = draw_team_sizes(config, envs, generator, device)
    return NextEpisodes(
        state=game.deal(game_config, envs, slots, sizes, generator, device),
        persona=scripted.personas(opponent.scripted_config, envs, slots, generator, device, config.persona_spread),
        scripted_memory=scripted.fresh_memory(envs, slots, generator, device),
        learner_team=torch.randint(0, 2, (envs,), **random_keywords(generator, device)),
        position_offset=draw_position_offset(policy.config, envs, generator, device),
        taken=torch.zeros((), dtype=torch.long, device=device),
    )


def redeal(opponent: Opponent, state, carry: CollectCarry, done: Tensor, generator):
    """Deal the environments in `done` (E,) bool afresh from carry.next_episodes, in order
    (new team sizes, learner team, personas, the scripted agent's fresh memory and position
    offset), and draw their league members (from the league's tallies as they stand); both
    carries zeroed there (within the members' row capacities, League.draw). Nothing is
    read back."""
    envs = done.shape[0]
    self_play = envs - carry.member.shape[0]
    pool = carry.next_episodes
    # The k-th environment to finish in this segment takes deal k (modulo P).
    index = (pool.taken + torch.cumsum(done.long(), 0) - 1) % pool.learner_team.shape[0]
    state = game.take_deal(state, done, pool.state, index)

    def take(fresh: Tensor, old: Tensor) -> Tensor:
        return select_environments(done, fresh.index_select(0, index), old)

    seat_environment, _ = seats(carry.learner_team, self_play)
    league_done = done[self_play:]
    member, slot = draw_members(opponent, carry.member, league_done, generator)
    carry = CollectCarry(
        policy=reset_carry(carry.policy, done.index_select(0, seat_environment)),
        opponent=reset_carry(carry.opponent, league_done),
        member=member,
        slot=slot,
        learner_team=take(pool.learner_team, carry.learner_team),
        persona=None if carry.persona is None else {name: take(pool.persona[name], carry.persona[name]) for name in carry.persona},
        scripted_memory={name: take(pool.scripted_memory[name], carry.scripted_memory[name]) for name in carry.scripted_memory},
        reset=done,
        position_offset=take(pool.position_offset, carry.position_offset),
        ledger=clear_ledger(carry.ledger, done),
        deal_row=torch.where(done, envs + index, carry.deal_row),
        next_episodes=dataclasses.replace(pool, taken=pool.taken + done.sum()),
    )
    return state, carry


def abandon(config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, slot: int,
            generator):
    """Re-deal the league environments played by network slot `slot` and draw them new
    opponents: only when the league had no slot free for a new snapshot (League.after_update;
    normally a dropped snapshot's games finish in its retiring slot). Their unfinished games
    are not tallied. The previous rollout bootstrapped those seats from the old state, as at
    a segment end. Returns (state, carry); carry.reset marks the re-dealt environments."""
    envs, slots = state.exists.shape
    self_play = envs - carry.member.shape[0]
    device = carry.member.device
    chosen = torch.cat([torch.zeros(self_play, dtype=torch.bool, device=device), carry.slot == slot])
    # Its own deals (rare, between segments); the next segment draws afresh.
    pool = draw_next_episodes(config, game_config, policy, opponent, envs, slots, generator, device)
    state, fresh = redeal(opponent, state, dataclasses.replace(carry, next_episodes=pool), chosen, generator)
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


def act(config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, generator):
    """The first half of a tick: both teams' views, the live policy on every seat (its
    sampled joint action, log-probability and value) and the scripted agent's action and
    distribution for every ship. Returns (views, observation, seat_offset, joint,
    log_probability, value, scripted_action, teacher, policy_carry, centres,
    scripted_memory), teacher the distribution (E, N, TEACHER_WIDTH), centres the critic's
    support (LEVELS, BINS), scripted_memory the scripted agent's memory after this
    decision."""
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
    scripted_action, scripted_memory, teacher = scripted_actions(
        opponent.scripted_config, state, carry.persona, carry.scripted_memory, generator
    )
    return (
        views, observation, seat_offset, joint, log_probability, value, scripted_action, teacher, policy_carry, centres,
        scripted_memory,
    )


def advance(
    config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, observation: Obs,
    joint: Tensor, teacher: Tensor, league_action: Tensor, opponent_carry: Tensor, policy_carry: Tensor,
    scripted_memory: dict, label_row: Tensor, hit_label: Tensor, generator,
):
    """The second half of a tick: every ship's action, the game step, rewards, hit labels
    (written into `hit_label` in place, `label_row` this tick's row there, see
    hit_label_entries), the league tallies, then the re-deal of finished environments.
    `label_row` is a 0-d long tensor (an int would be a new constant for a compiled graph
    every tick). Returns (state, carry, this tick's records by Rollout field name)."""
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
    reward, ledger = reward_step(config, carry.ledger, tick_events(step_out, state), state.team)
    slot_index = torch.arange(slots, device=device)[None, :, None]
    environment_index = torch.arange(envs, device=device)[:, None, None]
    spare = hit_label.shape[0] - 1
    hit_label.scatter_(0, *hit_label_entries(step_out, state, label_row, slot_index, environment_index, spare))
    records = {
        "teacher": teacher.index_select(0, seat_environment),
        "reward": reward.index_select(0, seat_environment),
        "done": seat_done,
        "reset": carry.reset.index_select(0, seat_environment),
        "next_state": next_state_target(policy.config, state, next_state, step_out, game_config.tick_length).index_select(
            0, seat_environment
        ),
        "next_valid": observation.own & next_state.alive.index_select(0, seat_environment) & ~seat_done[:, None],
    }
    if opponent.league is not None:
        result = step_out.result[self_play:]
        opponent.league.record(carry.member, done[self_play:], result == carry.learner_team[self_play:], result == 2)

    carry = dataclasses.replace(
        carry, policy=policy_carry, opponent=opponent_carry, scripted_memory=scripted_memory, ledger=ledger
    )
    state, carry = redeal(opponent, next_state, carry, done, generator)
    return state, carry, records


def deal_table(game_config, state, carry: CollectCarry) -> dict[str, Tensor]:
    """The segment's deal table (Rollout.deal_table): the episode-constant observation fields
    and the position offset, for the environments as the segment starts and then for
    carry.next_episodes' deals. Two observe calls per segment, eagerly."""
    from lean.learn.store import EPISODE_CONSTANT

    pool = carry.next_episodes
    now, dealt = game.observe(game_config, state, 0), game.observe(game_config, pool.state, 0)
    table = {name: torch.cat([getattr(now, name), getattr(dealt, name)]) for name in EPISODE_CONSTANT}
    table["position_offset"] = torch.cat([carry.position_offset, pool.position_offset])
    return table


@torch.no_grad()
def collect(
    config: LearnConfig, game_config, policy, opponent: Opponent, state, carry: CollectCarry, ticks: int, generator,
    *, compiled: bool = False, store=None, segment: int = 0,
) -> Rollout:
    """Step E environments for `ticks` ticks and record every seat the live policy plays.

    Self-play environments (the first E_s) have two live seats, league environments one;
    the league side plays the member `carry.member` (the scripted agent, an average or a
    snapshot) through its slot `carry.slot` (network_actions; League.prepare builds the
    segment's network table first, with one host read before the loop). The scripted agent's action is computed for every ship every tick: it is
    the BC target and the scripted member's move. Finished environments are re-dealt at
    once from deals drawn together at the segment's start (NextEpisodes: team sizes, the
    game's deal, personas, learner team, position offset) with a league member drawn as they
    finish, and the carries are zeroed there; league games that end are added to the
    league's tallies.

    `store` (a lean.learn.store.RolloutStore): each tick's records are streamed into its
    segment `segment` as they are made (asynchronous copies to pinned memory on a GPU) and
    the returned Rollout holds no per-tick fields. Without it they are stacked on the device.

    All random draws come from `generator`; with `compiled` (a tick's two halves, the
    league networks' moves run as lean.learn.compiled's graphs) they come from
    torch's default generator of the device instead, since a compiled graph cannot take a
    generator object. Nothing is read back to the host in the loop.
    """
    envs, slots = state.exists.shape
    device = state.exists.device
    self_play = envs - carry.member.shape[0]
    if compiled:
        from lean.learn.compiled import graphs

        act_part, advance_part, networks = graphs().act, graphs().advance, graphs().slot_actions
        generator = None
        # A fresh state can hold broadcast (stride 0) tensors where a re-dealt one holds
        # whole ones (even empty ones: fields with F = 0); the graphs are specialised on
        # strides, so start from whole tensors.
        state = map_fields(state, whole)
        state = dataclasses.replace(state, constants=map_fields(state.constants, whole))
    else:
        act_part, advance_part, networks = act, advance, slot_actions
    if opponent.league is not None:
        opponent.league.prepare(carry.member.shape[0], carry.slot)
    carry = dataclasses.replace(
        carry,
        next_episodes=draw_next_episodes(config, game_config, policy, opponent, envs, slots, generator, device),
        deal_row=torch.arange(envs, device=device),
    )
    table = deal_table(game_config, state, carry)
    initial_carry = carry.policy
    if store is not None:
        store.begin_segment(segment, ticks, initial_carry, table)
    records = {}
    # Hit labels over (row, environment, slot), flat: rows 0 to T - 1 are the previous
    # segment's ticks (its bullets that end in this segment), rows T to 2T - 1 this
    # segment's; one spare entry at the end takes every bullet fired earlier.
    hit_label = torch.full((2 * ticks * envs * slots + 1,), -1, dtype=torch.long, device=device)
    label_row = torch.full((), ticks, dtype=torch.long, device=device)
    seat_environment, _ = seats(carry.learner_team, self_play)

    for index in range(ticks):
        (views, observation, seat_offset, joint, log_probability, value, scripted_action, teacher, policy_carry, _,
         scripted_memory) = act_part(config, game_config, policy, opponent, state, carry, generator)
        deal_row = carry.deal_row.index_select(0, seat_environment)
        league_action, opponent_carry = league_actions(config, opponent, views, carry, scripted_action, generator, networks)
        state, carry, tick_records = advance_part(
            config, game_config, policy, opponent, state, carry, observation, joint, teacher, league_action,
            opponent_carry, policy_carry, scripted_memory, label_row, hit_label, generator,
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
    final = act_part(config, game_config, policy, opponent, state, carry, generator)
    bootstrap_value, centres = final[5], final[9]
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
        observation = Obs(**{
            field.name: torch.stack([getattr(item, field.name) for item in records["observation"]])
            for field in dataclasses.fields(Obs)
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
    )
