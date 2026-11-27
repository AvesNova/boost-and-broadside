"""Rollout collection: the live policy on its seats, the league on the other side."""

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from lean import game, scripted
from lean.contract import (
    EPISODE_CONSTANT,
    JOINT,
    Obs,
    random_keywords,
    split_joint,
    whole_state,
)
from lean.learn.config import LearnConfig
from lean.learn.returns import ValueSupport, value_support
from lean.learn.rewards import Ledger, blank_ledger, clear_ledger, reward_step, tick_events
from lean.model import draw_position_offset, next_state_target
from lean.model.agent import (
    Network,
    agent_step,
    merge_routed,
    routed_step,
    seat_view,
    select_environments,
)
from lean.model.value import decode_value
from lean.scripted import ScriptedConfig

# The teacher's distribution as recorded: its probabilities of the 42 joint actions, indexed as
# lean.contract.join_action (lean.scripted.Choice.joint).
TEACHER_WIDTH = JOINT


class ScriptedMove(NamedTuple):
    """The scripted agent's decision for every ship of both teams (scripted_actions): `action`
    (E, N, 3) long and `teacher` the distribution the actions were drawn from, (E, N,
    TEACHER_WIDTH) (lean.scripted.Choice.joint). The agent keeps no memory."""

    action: Tensor
    teacher: Tensor


class LeagueMove(NamedTuple):
    """The league side of every league environment: `action` (L, N, 3) long and `carry` the
    league opponents' new recurrent carry."""

    action: Tensor
    carry: Tensor


class Decision(NamedTuple):
    """The first half of a tick (act): both teams' `views`, the live policy's `observation` on
    every seat with each seat's `position_offset` (S, 2), its sampled `joint` actions (S, N),
    their `log_probability` and the decoded `value` (S, N, LEVELS) (zero outside `own`) on the
    critic's support `value_centres` (LEVELS, BINS), the policy's new carry `policy_carry`, and
    the scripted agent's move for every ship (`scripted`)."""

    views: tuple[Obs, Obs]
    observation: Obs
    position_offset: Tensor
    joint: Tensor
    log_probability: Tensor
    value: Tensor
    value_centres: Tensor
    policy_carry: Tensor
    scripted: ScriptedMove


@dataclass
class Opponent:
    """Who plays whom.

    The first round(LearnConfig.self_play_share * E) environments are self-play: the live
    policy flies both teams and both teams' rows train. The rest are league games: the live
    policy against the member held by one slot of `league` (lean.learn.pool.League), the slot
    drawn per environment at every reset (level 2); only the live side trains. Member 0 is
    the yardstick scripted agent (personas at LearnConfig.persona_spread); the league's
    scripted personas play through the same scripted call with per-row parameters
    (scripted_rows). Without a league every league game is against the scripted agent and
    nothing is tallied.
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
    persona: (
        dict | None
    )  # the scripted agent's per-ship parameters, leading P (None: the yardstick)
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
    # one that has left the league, its game finishing in its draining slot, not tallied).
    member: Tensor
    # (L,) long, the league slot playing it (lean.learn.pool: a network's weights or a
    # scripted agent's parameters; League.overflow the overflow slot); -1 without a league
    slot: Tensor
    learner_team: Tensor  # (E,) long
    persona: (
        dict | None
    )  # the scripted agent's per-ship parameters, leading E (None: the yardstick)
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
    observation fields constant within an episode (lean.contract.EPISODE_CONSTANT) and
    the position offset, once per deal (rows r < E: environment r as the segment started;
    E + k: the segment's deal k). A RolloutStore keeps these instead of the per-tick copies.
    action: (T, S, N) long joint actions sampled on that tick's observation. teacher: (T, S,
    N, TEACHER_WIDTH) the scripted agent's full distribution there over the 42 joint actions
    (the BC target). The game delays every action by one tick: the
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
        "action",
        "teacher",
        "log_probability",
        "value",
        "reward",
        "done",
        "reset",
        "position_offset",
        "next_state",
        "next_valid",
        "deal_row",
    )


def generator_device(generator):
    """The device a generator draws on; the CPU when there is none."""
    return generator.device if generator is not None else torch.device("cpu")


def draw_team_sizes(config: LearnConfig, envs: int, generator, device=None) -> Tensor:
    """Ships per team, (E, 2) long, uniform over the configured range; the two teams of
    an environment are equal when `equal_teams`, else drawn independently. `device`
    defaults to the generator's."""
    span = config.team_size_maximum - config.team_size_minimum + 1
    device = generator_device(generator) if device is None else device
    sizes = config.team_size_minimum + torch.randint(
        0, span, (envs, 2), **random_keywords(generator, device)
    )
    if config.equal_teams:
        sizes[:, 1] = sizes[:, 0]
    return sizes


def scripted_actions(scripted_config: ScriptedConfig, state, persona, generator) -> ScriptedMove:
    """The scripted agent's move for every ship of both teams in one call (ScriptedMove)."""
    envs, slots = state.exists.shape
    uniforms = scripted.draw_uniforms(envs, slots, generator, state.exists.device)
    action, choice = scripted.decide(scripted_config, state, None, persona, uniforms)
    return ScriptedMove(action, choice.joint.float())


def self_play_count(config: LearnConfig, envs: int) -> int:
    """E_s, the number of self-play environments (the first E_s indices)."""
    return round(config.self_play_share * envs)


def seats(learner_team: Tensor, self_play: int) -> tuple[Tensor, Tensor]:
    """(seat_environment, seat_team), each (S,) long: every environment's learner team,
    then the other team of the first `self_play` environments."""
    envs = learner_team.shape[0]
    environment = torch.cat(
        [
            torch.arange(envs, device=learner_team.device),
            torch.arange(self_play, device=learner_team.device),
        ]
    )
    team = torch.cat([learner_team, 1 - learner_team[:self_play]])
    return environment, team


def draw_members(
    opponent: Opponent, member: Tensor, slot: Tensor, done: Tensor, generator
) -> tuple[Tensor, Tensor]:
    """(member, slot) of the league environments, new draws where `done` (L,) bool, the
    others keeping `member` and `slot` (League.draw, level 2); without a league every one
    plays the scripted agent (member 0, slot -1)."""
    if opponent.league is None or member.shape[0] == 0:
        return torch.zeros_like(member), torch.full_like(member, -1)
    return opponent.league.draw(member, slot, done, generator)


def scripted_rows(opponent: Opponent, state, carry: CollectCarry) -> dict | None:
    """The scripted agent's per-ship parameters for the one scripted call over every ship
    (E, N): the ships of the league side of a league environment whose slot holds a scripted
    member take that member's row of the league's `persona_table` (row 0 the yardstick, then
    the personas); every other ship (the live policy's, whose teacher is the yardstick, and a
    network's) the yardstick's row, or its persona of carry.persona when one is drawn
    (LearnConfig.persona_spread). The same in both stages (in BC every slot is the
    yardstick's), so both run one scripted graph; without a league, carry.persona itself. No
    host read: the slots' rows are device tensors, so a load or a new persona is new data for
    the compiled tick, never a new graph."""
    league = opponent.league
    if league is None:
        return carry.persona
    envs = state.exists.shape[0]
    league_envs = carry.member.shape[0]
    row = league.slot_persona.index_select(0, carry.slot.clamp(min=0))
    row = torch.cat([torch.zeros(envs - league_envs, dtype=row.dtype, device=row.device), row])
    other_side = state.team != carry.learner_team[:, None]
    agent = torch.where(other_side, row[:, None], 0)
    values = league.persona_table.index_select(0, agent.flatten()).view(*agent.shape, -1)
    rows = dict(zip(league.parameter_rows, values.unbind(-1), strict=True))
    if carry.persona is not None:
        yardstick = agent == 0
        for name, drawn in carry.persona.items():
            rows[name] = torch.where(yardstick, drawn, rows[name]) if name in rows else drawn
    return rows


def slot_actions(
    network_: Network,
    slot: Tensor,
    positions: Tensor,
    views: tuple[Obs, Obs],
    carry: CollectCarry,
    generator,
) -> tuple[Tensor, Tensor]:
    """One network slot's moves (lean.model.agent.routed_step, the routing the evaluation
    runner shares): the league environments it plays (carry.slot == `slot`, a 0-d long) run
    on a W-row batch (W = positions.shape[0], League.prepare's table width, fixed for the run:
    never fewer than the slot's environments) from their league side's view. Returns the
    batch's actions (W, N, 3) and carry (merge_league puts them back). No host read, and one
    compiled graph for every slot (an empty slot runs on no rows)."""
    league_envs = carry.member.shape[0]
    envs = carry.learner_team.shape[0]
    first = envs - league_envs

    def view_of(environment: Tensor) -> Obs:
        return seat_view(
            views, first + environment, 1 - carry.learner_team.index_select(0, first + environment)
        )

    return routed_step(
        network_,
        carry.slot == slot,
        positions,
        view_of,
        carry.opponent,
        carry.position_offset[first:],
        generator,
        carry.reset[first:],
    )


def merge_league(
    carry: CollectCarry,
    scripted_action: Tensor,
    actions: list[Tensor],
    carries: list[Tensor],
    network: Tensor,
) -> LeagueMove:
    """The league side's move from the slots' outputs (slot_actions, all K + R in slot order,
    a skipped slot's a filler no row reads, NetworkTable): each league environment whose slot
    plays a network (`network` (K + R + 1,) bool, League.slot_network) takes its slot's action
    and carry (lean.model.agent.merge_routed), the others (scripted slots: the yardstick and
    the personas) the scripted agent's action, computed with their rows' parameters
    (scripted_rows), and their carry. One pass over the league environments."""
    league_envs = carry.member.shape[0]
    routed = torch.where(network.index_select(0, carry.slot.clamp(min=0)), carry.slot, -1)
    action, new_carry = merge_routed(
        routed,
        actions,
        carries,
        scripted_action[scripted_action.shape[0] - league_envs :],
        carry.opponent,
    )
    return LeagueMove(action, new_carry)


def league_rows(scripted_action: Tensor, league_envs: int) -> Tensor:
    """The scripted agent's actions of the league environments (the last `league_envs` rows),
    as a tensor of their own: a slice is a view, and a compiled graph guards a view's base,
    so a view here and a graph's output there (no base) would be two graphs (`advance`
    when the first network member joins)."""
    return scripted_action[scripted_action.shape[0] - league_envs :].clone()


def network_actions(
    config: LearnConfig,
    league,
    views: tuple[Obs, Obs],
    carry: CollectCarry,
    scripted_action: Tensor,
    generator,
    step=slot_actions,
    merge=merge_league,
) -> LeagueMove:
    """The league side's move: each slot of the league's network table (League.prepare: the
    slots holding weights) runs on its own environments (`step`: slot_actions or its compiled
    form), one call per listed slot, the same calls every tick of a segment; then `merge`
    (merge_league or its compiled form) puts the moves back from K entries, the unlisted
    slots' the table's zero fillers (made at the segment's first tick, shaped as a slot's
    output), so the merge's shapes never change."""
    table = league.table
    outputs = {}
    for index, slot, parameters, buffers in table.slots:
        network_ = Network(league.template, parameters, buffers, config.bf16)
        outputs[index] = step(network_, slot, table.positions, views, carry, generator)
    if table.fillers is None:
        action, new_carry = next(iter(outputs.values()))
        table.fillers = {
            index: (torch.zeros_like(action), torch.zeros_like(new_carry))
            for index in range(table.count)
            if index not in outputs
        }
    entries = [outputs.get(index) or table.fillers[index] for index in range(table.count)]
    return merge(
        carry,
        scripted_action,
        [entry[0] for entry in entries],
        [entry[1] for entry in entries],
        league.slot_network,
    )


def league_actions(
    config: LearnConfig,
    opponent: Opponent,
    views: tuple[Obs, Obs],
    carry: CollectCarry,
    scripted_action: Tensor,
    generator,
    step=slot_actions,
    merge=merge_league,
) -> LeagueMove:
    """The league side's move in every league environment: the scripted agent's actions
    (already computed as the BC target) and the unchanged opponent carry while the league has
    no network table (no network slot holds weights: the BC stage),
    then network_actions (`step` the per-slot function and `merge`, or their compiled
    forms)."""
    league = opponent.league
    league_envs = carry.member.shape[0]
    if league is None or league.table is None or league_envs == 0:
        return LeagueMove(league_rows(scripted_action, league_envs), carry.opponent)
    return network_actions(config, league, views, carry, scripted_action, generator, step, merge)


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
    member, slot = draw_members(
        opponent, unset, unset - 1, torch.ones_like(unset, dtype=torch.bool), generator
    )
    carry = CollectCarry(
        policy=policy.initial_carry(envs + self_play, slots, device),
        opponent=policy.initial_carry(league_envs, slots, device),
        member=member,
        slot=slot,
        learner_team=torch.randint(0, 2, (envs,), **random_keywords(generator, device)),
        persona=scripted.personas(
            opponent.scripted_config, envs, slots, generator, device, config.persona_spread
        ),
        reset=torch.ones(envs, dtype=torch.bool, device=device),
        position_offset=draw_position_offset(policy.config, envs, generator, device),
        ledger=blank_ledger(envs, slots, device),
        deal_row=torch.arange(envs, device=device),
    )
    return state, carry


def draw_next_episodes(
    config: LearnConfig,
    game_config,
    policy,
    opponent: Opponent,
    envs: int,
    slots: int,
    generator,
    device,
) -> NextEpisodes:
    """E deals for the episodes that start during the next segment (NextEpisodes)."""
    sizes = draw_team_sizes(config, envs, generator, device)
    return NextEpisodes(
        state=game.deal(game_config, envs, slots, sizes, generator, device),
        persona=scripted.personas(
            opponent.scripted_config, envs, slots, generator, device, config.persona_spread
        ),
        learner_team=torch.randint(0, 2, (envs,), **random_keywords(generator, device)),
        position_offset=draw_position_offset(policy.config, envs, generator, device),
        taken=torch.zeros((), dtype=torch.long, device=device),
    )


def redeal(opponent: Opponent, state, carry: CollectCarry, done: Tensor, generator):
    """Deal the environments in `done` (E,) bool afresh from carry.next_episodes, in order
    (new team sizes, learner team, personas and position offset), and draw their league
    members (from the league's tallies as they stand, within the members' row capacities,
    League.draw). The carries are left as they are: `reset` marks the environments, and the
    next step (Policy.step's reset, as the update's sequence reads it) treats their carry as
    zero. Nothing is read back."""
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
    member, slot = draw_members(opponent, carry.member, carry.slot, league_done, generator)
    carry = CollectCarry(
        policy=carry.policy,
        opponent=carry.opponent,
        member=member,
        slot=slot,
        learner_team=take(pool.learner_team, carry.learner_team),
        persona=None
        if carry.persona is None
        else {name: take(pool.persona[name], carry.persona[name]) for name in carry.persona},
        reset=done,
        position_offset=take(pool.position_offset, carry.position_offset),
        ledger=clear_ledger(carry.ledger, done),
        deal_row=torch.where(done, envs + index, carry.deal_row),
        next_episodes=dataclasses.replace(pool, taken=pool.taken + done.sum()),
    )
    return state, carry


def hit_outcomes_by_age(step_out, state) -> Tensor:
    """(E, N, K) int8: this tick's outcome (step_out.shot_outcome: -1 none, 0 missed, 1 hit an
    enemy, 2 a team-mate) of each ship's bullet of age a (left the gun a ticks before this
    tick, on episode tick state.tick - a), indexed by a: the ring slot of age a is
    (state.tick - a) mod K (lean.game.core.bullet_ages). One tick's record for
    assemble_hit_labels; nothing is written in place in the tick."""
    slots = step_out.shot_outcome.shape[-1]
    by_age = (state.tick[:, None] - torch.arange(slots, device=state.tick.device)) % slots
    index = by_age[:, None, :].expand_as(step_out.shot_outcome)
    return step_out.shot_outcome.gather(-1, index).to(torch.int8)


def assemble_hit_labels(outcomes: Tensor) -> Tensor:
    """The segment's hit labels (2, T, E, N) long from its ticks' hit_outcomes_by_age
    (T, E, N, K): the outcome of the bullet each ship emitted on each row's tick, rows 0 to
    T - 1 the previous segment's ticks (its bullets that end in this segment), rows T to
    2T - 1 this segment's; -1 where it has not ended in this segment. A bullet of age a that
    ends on tick t left the gun on row T + t - a (episode ticks and rows advance together
    within an episode: a bullet of the current episode never reaches a row before the
    episode's first); one fired before the previous segment is dropped. Each bullet ends
    once, so a row's entries are -1 but one: the maximum over the ages assembles them."""
    ticks, envs, slots, ages = outcomes.shape
    labels = torch.full((2 * ticks, envs, slots), -1, dtype=torch.int8, device=outcomes.device)
    for age in range(ages):
        first = ticks - age  # the row of tick 0's bullet of this age
        skip = max(0, -first)  # ticks whose bullet of this age predates the previous segment
        if skip >= ticks:  # a segment shorter than half a bullet's life
            continue
        rows = labels[first + skip : first + ticks]
        torch.maximum(rows, outcomes[skip:, ..., age], out=rows)
    return labels.long().view(2, ticks, envs, slots)


def act(
    config: LearnConfig,
    game_config,
    policy,
    opponent: Opponent,
    state,
    carry: CollectCarry,
    generator,
    support: ValueSupport | None = None,
) -> Decision:
    """The first half of a tick (Decision): both teams' views, the live policy on every seat
    (lean.model.agent.agent_step, the agent tick the league and evaluation share; its sampled
    joint action, log-probability and value) and the scripted agent's move for every ship.
    `support` (returns.value_support) the critic's support on the device, else built here."""
    envs = state.exists.shape[0]
    self_play = envs - carry.member.shape[0]
    seat_environment, seat_team = seats(carry.learner_team, self_play)
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    observation = seat_view(views, seat_environment, seat_team)
    seat_offset = carry.position_offset.index_select(0, seat_environment)
    step = agent_step(
        policy.step,
        config.bf16,
        observation,
        carry.policy,
        seat_offset,
        generator,
        carry.reset.index_select(0, seat_environment),
    )
    log_probability = torch.log_softmax(step.logits, -1).gather(-1, step.joint[..., None])
    value_logits = step.value_logits.float()
    if support is None:
        support = value_support(config, value_logits.shape[-1], value_logits.device)
    value = decode_value(value_logits, support.centres, support.knees) * observation.own[..., None]
    move = scripted_actions(
        opponent.scripted_config, state, scripted_rows(opponent, state, carry), generator
    )
    return Decision(
        views=views,
        observation=observation,
        position_offset=seat_offset,
        joint=step.joint,
        log_probability=log_probability.squeeze(-1),
        value=value,
        value_centres=support.centres,
        policy_carry=step.carry,
        scripted=move,
    )


def advance(
    config: LearnConfig,
    game_config,
    policy,
    opponent: Opponent,
    state,
    carry: CollectCarry,
    decision: Decision,
    league_move: LeagueMove,
    generator,
):
    """The second half of a tick, after `decision` (act) and `league_move` (league_actions):
    every ship's action, the game step, rewards, the tick's bullet outcomes for the hit labels
    (hit_outcomes_by_age), the league tallies, then the re-deal of finished environments.
    Returns (state, carry, this tick's records by Rollout field name, and "hit_outcome")."""
    envs, slots = state.exists.shape
    self_play = envs - carry.member.shape[0]
    seat_environment, _ = seats(carry.learner_team, self_play)
    observation = decision.observation
    live_action = split_joint(decision.joint)
    other_side = torch.cat([live_action[envs:], league_move.action])
    learner_ship = state.team == carry.learner_team[:, None]
    action = torch.where(learner_ship[..., None], live_action[:envs], other_side)

    next_state, step_out = game.tick(game_config, state, action, generator)
    done = step_out.done
    seat_done = done.index_select(0, seat_environment)
    reward, ledger = reward_step(config, carry.ledger, tick_events(step_out, state), state.team)
    records = {
        "hit_outcome": hit_outcomes_by_age(step_out, state),
        "teacher": decision.scripted.teacher.index_select(0, seat_environment),
        "reward": reward.index_select(0, seat_environment),
        "done": seat_done,
        "reset": carry.reset.index_select(0, seat_environment),
        "next_state": next_state_target(
            policy.config, state, next_state, step_out, game_config.tick_length
        ).index_select(0, seat_environment),
        "next_valid": observation.own
        & next_state.alive.index_select(0, seat_environment)
        & ~seat_done[:, None],
    }
    if opponent.league is not None:
        result = step_out.result[self_play:]
        opponent.league.record(
            carry.member, done[self_play:], result == carry.learner_team[self_play:], result == 2
        )

    carry = dataclasses.replace(
        carry,
        policy=decision.policy_carry,
        opponent=league_move.carry,
        ledger=ledger,
    )
    state, carry = redeal(opponent, next_state, carry, done, generator)
    return state, carry, records


def deal_table(game_config, state, carry: CollectCarry) -> dict[str, Tensor]:
    """The segment's deal table (Rollout.deal_table): the episode-constant observation fields
    and the position offset, for the environments as the segment starts and then for
    carry.next_episodes' deals. Two observe calls per segment, eagerly."""
    pool = carry.next_episodes
    now, dealt = game.observe(game_config, state, 0), game.observe(game_config, pool.state, 0)
    table = {
        name: torch.cat([getattr(now, name), getattr(dealt, name)]) for name in EPISODE_CONSTANT
    }
    table["position_offset"] = torch.cat([carry.position_offset, pool.position_offset])
    return table


@torch.no_grad()
def collect(
    config: LearnConfig,
    game_config,
    policy,
    opponent: Opponent,
    state,
    carry: CollectCarry,
    ticks: int,
    generator,
    *,
    compiled: bool = False,
    store=None,
    segment: int = 0,
) -> Rollout:
    """Step E environments for `ticks` ticks and record every seat the live policy plays.

    Self-play environments (the first E_s) have two live seats, league environments one;
    the league side plays the member `carry.member` (the scripted agent, an average or a
    snapshot) through its slot `carry.slot` (network_actions; League.prepare builds the
    segment's network table first, with one host read before the loop). The scripted agent's action
    is computed for every ship every tick: it is
    the BC target and the scripted member's move. Finished environments are re-dealt at
    once from deals drawn together at the segment's start (NextEpisodes: team sizes, the
    game's deal, personas, learner team, position offset) with a league member drawn as they
    finish, their carries treated as zero by the next step (`reset`); league games that end
    are added to the league's tallies.

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

        compiled_graphs = graphs(config.compile_shapes)
        act_part, advance_part = compiled_graphs.act, compiled_graphs.advance
        networks, merge = compiled_graphs.slot_actions, compiled_graphs.merge_league
        generator = None
        # A fresh state can hold broadcast (stride 0) tensors where a re-dealt one holds
        # whole ones (even empty ones: fields with F = 0).
        state = whole_state(state)
    else:
        act_part, advance_part, networks, merge = act, advance, slot_actions, merge_league
    if opponent.league is not None:
        opponent.league.prepare(carry.member.shape[0], carry.slot)
    carry = dataclasses.replace(
        carry,
        next_episodes=draw_next_episodes(
            config, game_config, policy, opponent, envs, slots, generator, device
        ),
        deal_row=torch.arange(envs, device=device),
    )
    table = deal_table(game_config, state, carry)
    support = value_support(config, policy.config.bins, device)
    initial_carry = carry.policy
    if store is not None:
        store.begin_segment(segment, ticks, initial_carry, table)
    records = {}
    hit_outcomes = []  # each tick's hit_outcomes_by_age, assembled at the segment's end
    seat_environment, _ = seats(carry.learner_team, self_play)

    for index in range(ticks):
        decision = act_part(config, game_config, policy, opponent, state, carry, generator, support)
        deal_row = carry.deal_row.index_select(0, seat_environment)
        league_move = league_actions(
            config,
            opponent,
            decision.views,
            carry,
            decision.scripted.action,
            generator,
            networks,
            merge,
        )
        state, carry, tick_records = advance_part(
            config,
            game_config,
            policy,
            opponent,
            state,
            carry,
            decision,
            league_move,
            generator,
        )
        hit_outcomes.append(tick_records.pop("hit_outcome"))
        tick_records = {
            "observation": decision.observation,
            "action": decision.joint,
            "log_probability": decision.log_probability,
            "value": decision.value,
            "position_offset": decision.position_offset,
            "deal_row": deal_row,
            **tick_records,
        }
        if store is not None:
            store.write_tick(segment, index, tick_records)
        else:
            for name, item in tick_records.items():
                records.setdefault(name, []).append(item)

    seat_environment, seat_team = seats(carry.learner_team, self_play)
    # The value after the last tick from the tick's first half (one graph fewer to compile;
    # its samples and the scripted actions are discarded).
    final = act_part(config, game_config, policy, opponent, state, carry, generator, support)
    bootstrap_value, centres = final.value, final.value_centres
    # After a reset the final observation belongs to a new episode; GAE never reads the
    # bootstrap there because `done` on the last tick cuts it.
    # A seat is the same environment's same slots in every segment (`seats`), so the
    # previous segment's rows go to the same seats there.
    previous_hit_label, seat_hit_label = (
        assemble_hit_labels(torch.stack(hit_outcomes)).index_select(2, seat_environment).unbind()
    )

    if store is not None:
        # The first segment's earlier rows belong to the previous update (already trained on).
        store.end_segment(
            segment,
            seat_hit_label,
            bootstrap_value,
            centres,
            previous_hit_label if segment > 0 else None,
        )
        observation, stacked, seat_hit_label = None, dict.fromkeys(Rollout.TICK_FIELDS), None
    else:
        observation = Obs(
            **{
                field.name: torch.stack(
                    [getattr(item, field.name) for item in records["observation"]]
                )
                for field in dataclasses.fields(Obs)
            }
        )
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
