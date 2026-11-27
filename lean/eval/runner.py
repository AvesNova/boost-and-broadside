"""The evaluation runner: a queue of games played on E environments at once.

Every environment is a slot. A game is (agent A, agent B, the team A plays, A's ships,
B's ships). The first E games of the queue are dealt at the start; when a slot's game ends
it takes the next game of the queue at once (rolling). At the cutoff (the queue's end, or
`seconds` of play) no game starts and every started game plays to its end (the drain), so
the results are those of a prefix of the queue, each game played whole.

Seats: each game's seat A and seat B is either a scripted agent (an index into `scripted`, a
table of full parameter vectors, lean.scripted.parameters.vector) or a network (an index
into the call's networks, all of one architecture). One queue may hold games of many
pairings: scripted against scripted, networks against scripted agents and against each
other. The scripted agent runs once per tick for every ship with every parameter row as a
per-ship tensor (rows_of), so new vectors never compile a new graph. The 2E seats (seat A of
every environment, then seat B) are routed to their networks as the training league routes
its environments to its members (lean.model.agent.routed_step, mixture-of-experts style):
each network runs once per tick on its own seats only, padded to a width W the host sets per
segment (the network's seats in play plus those of the games the segment may start), from
the seat's team view through the agent tick training uses (lean.model.agent.network_step:
the weights are inputs, so another network of the same architecture is new data, and the
precision is the network's own). Each seat draws its own position offset per game (as in
training). The game half of the tick is lean.game.tick, also training's and the sandbox's.

Deals are drawn once per segment (`segment` ticks) for the next games of the queue (at most
`deal_rows` start in a segment), and the host reads 2 + K numbers per segment (games taken,
slots busy, each network's seats in play); nothing is read in a tick.

Each slot sums its game's style counters (lean.eval.style) per seat inside the tick; a finished
game's sums go to its record and the slot's sums start at zero with the next deal.
"""

import dataclasses
import functools
import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from lean import game, scripted
from lean.contract import SEAT_AXIS, Obs, compile_dynamic, split_joint, whole_state
from lean.eval import style as style_counters
from lean.model import draw_position_offset
from lean.model.agent import (
    Network,
    network,
    network_step,
    reset_carry,
    route_rows,
    seat_view,
)
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import rows_of

# A game's result from seat A's side.
A_WON, B_WON, DRAW, NOT_PLAYED = 0, 1, 2, -1
# A seat the scripted agent plays (Queue.net_a, net_b).
SCRIPTED = -1
# The scripted agent's config beside full per-ship rows: never read for a value.
FIXED_CONFIG = ScriptedConfig()
# The smallest batch width a network runs on: a compiled graph specialises sizes 0 and 1.
MINIMUM_WIDTH = 2


@dataclass
class Queue:
    """Q games in order, (Q,) long each: `a`, `b` the scripted index of seats A and B (into
    the call's table; ignored for a seat a network plays), `side` the team seat A plays (0 or
    1), `size_a`, `size_b` the seats' ships, `net_a`, `net_b` the network index of seats A
    and B (into the call's networks; SCRIPTED, the default, for a scripted seat)."""

    a: Tensor
    b: Tensor
    side: Tensor
    size_a: Tensor
    size_b: Tensor
    net_a: Tensor | None = None
    net_b: Tensor | None = None

    def __post_init__(self):
        if self.net_a is None:
            self.net_a = torch.full_like(self.a, SCRIPTED)
        if self.net_b is None:
            self.net_b = torch.full_like(self.b, SCRIPTED)

    def __len__(self) -> int:
        return self.a.shape[0]

    def team_sizes(self) -> Tensor:
        """(Q, 2) ships of team 0 and team 1."""
        first = torch.where(self.side == 0, self.size_a, self.size_b)
        return torch.stack([first, self.size_a + self.size_b - first], -1)

    def to(self, device) -> Queue:
        return Queue(*(getattr(self, field.name).to(device) for field in dataclasses.fields(self)))


@dataclass
class Outcome:
    """Per game of the queue, (Q,) long: `result` (A_WON, B_WON, DRAW, NOT_PLAYED: past the
    cutoff) and `length` in ticks (0 when not played); `style` (Q, 2, len(style.SUMS))
    float32 the style counters' sums of seat A and seat B (zero when not played)."""

    result: Tensor
    length: Tensor
    style: Tensor


@dataclass
class Deals:
    """The next P games of the queue dealt at once (one segment's starts): the game state and
    both seats' position offsets, leading P; `first` the queue index of
    row 0 (a 0-d tensor: a new value is data for a compiled graph)."""

    state: object
    offset_a: Tensor
    offset_b: Tensor
    first: Tensor  # () long


@dataclass
class Slots:
    """The E environments: `game` (E,) long, the queue index each plays (-1 idle), and
    `style` (E, 2, len(style.SUMS)) the game's style sums so far. With
    networks, per seat (seat A of every environment, then seat B, then one spare row that
    padding rows write to): `carry` (blocks, 2E + 1, ...) the networks' recurrent carry and
    `seat_action` (2E + 1, N, 3) their last moves, both written in place by seat_moves only
    at the network's own seats (None without networks); and `offset` (2E, 2) the seats'
    position offsets."""

    game: Tensor
    carry: Tensor | None
    seat_action: Tensor | None
    offset: Tensor
    style: Tensor


class Decision(NamedTuple):
    """One tick's scripted half and the seats' routing (decide): `action` (E, N, 3) the
    scripted agent's moves (zero without a scripted seat), `side` (E,) the
    team seat A plays, `seat_a` (E, N) the ship plays for seat A, `playing` (E,); with
    networks, the two fixed-team `views` and per seat (2E,) its network (`seat_net`,
    SCRIPTED for a scripted or idle seat), environment, team and whether its game starts at
    this tick (`seat_fresh`: the carry is zeroed)."""

    action: Tensor
    side: Tensor
    seat_a: Tensor
    playing: Tensor
    views: tuple[Obs, Obs] | None
    seat_net: Tensor | None
    seat_environment: Tensor | None
    seat_team: Tensor | None
    seat_fresh: Tensor | None


def draw_deals(game_config, queue: Queue, first: int, count: int, slots: int, model_config, device):
    """Deals for the queue's games first .. first + count - 1 (rows past the queue's end
    repeat its last game; they are never started). `model_config`: the networks' config (None
    without networks: zero offsets)."""
    index = (first + torch.arange(count, device=device)).clamp(max=len(queue) - 1)
    sizes = queue.team_sizes().index_select(0, index)
    state = game.deal(game_config, count, slots, sizes, None, device)
    offsets = [
        draw_position_offset(model_config, count, None, device)
        if model_config is not None
        else torch.zeros(count, 2, device=device)
        for _ in range(2)
    ]
    first_tensor = torch.tensor(first, dtype=torch.long, device=device)
    return Deals(state, *offsets, first_tensor)


def decide(game_config, state, slots: Slots, queue: Queue, table: Tensor, scripted_seats: bool):
    """The scripted agent's moves for every ship (when the queue has a scripted seat) and,
    with networks (slots.carry), the views and each seat's network. Returns a Decision."""
    envs, ships = state.exists.shape
    device = state.exists.device
    playing = slots.game >= 0
    current = slots.game.clamp(min=0)
    side = queue.side.index_select(0, current)
    seat_a = state.team == side[:, None]  # (E, N) the ship plays for seat A
    action = torch.zeros((envs, ships, 3), dtype=torch.long, device=device)
    if scripted_seats:
        a, b = queue.a.index_select(0, current), queue.b.index_select(0, current)
        agent = torch.where(seat_a, a[:, None], b[:, None])
        uniforms = scripted.draw_uniforms(envs, ships, None, device)
        action, _ = scripted.decide(FIXED_CONFIG, state, None, rows_of(table, agent), uniforms)
    if slots.carry is None:
        return Decision(action, side, seat_a, playing, *[None] * 5)
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    nets = torch.cat([queue.net_a.index_select(0, current), queue.net_b.index_select(0, current)])
    seat_net = torch.where(torch.cat([playing, playing]), nets, SCRIPTED)
    environments = torch.arange(envs, device=device)
    fresh = state.tick == 0
    return Decision(
        action,
        side,
        seat_a,
        playing,
        views,
        seat_net,
        torch.cat([environments, environments]),
        torch.cat([side, 1 - side]),
        torch.cat([fresh, fresh]),
    )


def seat_moves(
    network_: Network,
    net: Tensor,
    positions: Tensor,
    decision: Decision,
    carry: Tensor,
    offset: Tensor,
    seat_action: Tensor,
) -> None:
    """One network's moves on the seats it plays (decision.seat_net == `net`, a 0-d long),
    routed as the league routes its environments (lean.model.agent.route_rows, W =
    positions.shape[0]): the network runs once on its W rows (padding past its seats) from
    each seat's team view, its carry zeroed where the seat's game starts, and writes its
    seats' moves and carry into `seat_action` and `carry` in place (padding rows to the spare
    row 2E). Only the network's own rows are read and written."""
    mine = decision.seat_net == net
    source, _ = route_rows(mine, positions.shape[0])
    view = seat_view(
        decision.views,
        decision.seat_environment.index_select(0, source),
        decision.seat_team.index_select(0, source),
    )
    rows_carry = reset_carry(
        carry.index_select(SEAT_AXIS, source), decision.seat_fresh.index_select(0, source)
    )
    step = network_step(network_, view, rows_carry, offset.index_select(0, source), None)
    spare = mine.shape[0]
    target = torch.where(positions < mine.sum(), source, spare)
    seat_action.index_copy_(0, target, split_joint(step.joint))
    carry.index_copy_(SEAT_AXIS, target, step.carry)


def advance(
    game_config,
    state,
    slots: Slots,
    decision: Decision,
    deals: Deals,
    taken: Tensor,
    limit: Tensor,
    result: Tensor,
    length: Tensor,
    style: Tensor,
):
    """The tick's game half and the starts, from the scripted moves and the networks' seat
    moves (slots.seat_action). `taken` (0-d long) counts the rows of `deals` used, `limit`
    (0-d long) is the first queue index that may not start; `result` and `length` (Q + 1,)
    and `style` (Q + 1, 2, len(style.SUMS)) take each finished game's record in place (entry
    Q is spare). Returns (state, slots, taken)."""
    envs = state.exists.shape[0]
    seat_a, playing = decision.seat_a, decision.playing
    action = decision.action
    if slots.seat_action is not None:
        network_a = (decision.seat_net[:envs] >= 0)[:, None] & seat_a
        network_b = (decision.seat_net[envs:] >= 0)[:, None] & ~seat_a
        action = torch.where(network_a[..., None], slots.seat_action[:envs], action)
        action = torch.where(network_b[..., None], slots.seat_action[envs : 2 * envs], action)

    start_state = state
    state, out = game.tick(game_config, state, action, None)
    sums = slots.style + style_counters.tick_sums(
        start_state, seat_a, playing, start_state.pending[..., 1], state.fired, out.hit_damage
    )
    side = decision.side
    finished = out.done & playing
    record = torch.where(finished, slots.game, result.shape[0] - 1)
    seat_result = torch.where(out.result == 2, DRAW, torch.where(out.result == side, A_WON, B_WON))
    result.scatter_(0, record, seat_result)
    length.scatter_(0, record, state.tick)
    style.index_copy_(0, record, sums)

    # Starts: every slot whose game ended or that is idle takes the next deal, in slot order,
    # while the segment's deals and the queue up to the limit last.
    want = out.done | ~playing
    row = taken + torch.cumsum(want.long(), 0) - 1
    rows = deals.offset_a.shape[0]
    next_game = deals.first + row
    start = want & (row < rows) & (next_game < limit)
    pick = row.clamp(0, rows - 1)
    state = game.take_deal(state, start, deals.state, pick)

    seats_start = torch.cat([start, start])
    fresh_offset = torch.cat(
        [deals.offset_a.index_select(0, pick), deals.offset_b.index_select(0, pick)]
    )
    slots = Slots(
        game=torch.where(start, next_game, torch.where(want, -1, slots.game)),
        carry=slots.carry,
        seat_action=slots.seat_action,
        offset=torch.where(seats_start[:, None], fresh_offset, slots.offset),
        style=torch.where(want[:, None, None], 0.0, sums),
    )
    return state, slots, taken + start.sum()


def tick(game_config, state, slots: Slots, queue: Queue, table: Tensor, *rest):
    """One tick without networks (every seat scripted): decide and advance in one graph.
    `rest`: advance's deals, taken, limit, result, length, style."""
    decision = decide(game_config, state, slots, queue, table, True)
    return advance(game_config, state, slots, decision, *rest)


def advance_decide(game_config, state, slots: Slots, decision: Decision, *rest):
    """With networks: advance, then the next tick's decide, in one graph. `rest`: advance's
    deals, taken, limit, result, length, style, then decide's queue, table, scripted_seats.
    Returns (state, slots, taken, the next Decision)."""
    state, slots, taken = advance(game_config, state, slots, decision, *rest[:6])
    return state, slots, taken, decide(game_config, state, slots, *rest[6:])


@functools.cache
def compiled():
    """The runner's functions under torch.compile (lean.contract.compile_dynamic, as
    lean.learn.compiled): tick (no networks), and with them decide (the first tick only),
    seat_moves (once per network and tick) and advance_decide."""
    return {
        name: compile_dynamic(function)
        for name, function in (
            ("tick", tick),
            ("decide", decide),
            ("seat_moves", seat_moves),
            ("advance_decide", advance_decide),
        )
    }


def eager() -> dict:
    return {
        "tick": tick,
        "decide": decide,
        "seat_moves": seat_moves,
        "advance_decide": advance_decide,
    }


def seat_counts(queue: Queue, networks: int) -> Tensor:
    """(Q + 1, K) the cumulative seats of each network over the queue's games (row i: games
    0 .. i - 1), on the host."""
    seats = torch.zeros(len(queue), networks + 1, dtype=torch.long)
    for nets in (queue.net_a.cpu(), queue.net_b.cpu()):
        seats.scatter_add_(1, (nets + 1)[:, None], torch.ones(len(queue), 1, dtype=torch.long))
    return torch.cat([torch.zeros(1, networks, dtype=torch.long), seats[:, 1:].cumsum(0)])


def seats_in_play(slots: Slots, queue: Queue, networks: int) -> Tensor:
    """(K,) each network's seats in play (busy slots only)."""
    current = slots.game.clamp(min=0)
    busy = slots.game >= 0
    nets = torch.cat(
        [
            torch.where(busy, queue.net_a.index_select(0, current), SCRIPTED),
            torch.where(busy, queue.net_b.index_select(0, current), SCRIPTED),
        ]
    )
    return torch.bincount(nets + 1, minlength=networks + 1)[1:]


@torch.no_grad()
def play(
    game_config,
    queue: Queue,
    scripted_table: Tensor,
    policies=(),
    *,
    bf16=None,
    envs: int = 4096,
    segment: int = 32,
    deal_share: float = 0.125,
    seconds: float | None = None,
    compiled: bool = False,
    seed: int = 0,
    progress=None,
    slot_games: int = 16,
    minimum_envs: int = 256,
) -> Outcome:
    """Play the queue's games (see the module's docstring) on at most `envs` environments on
    the table's device (fewer for a short queue: Q / `slot_games`, at least `minimum_envs`).
    `scripted_table` (S, len(PARAMETERS)) float32 holds the scripted agents' vectors (one row
    at least, even when every seat is a network); `policies` the lean.model.Policy objects
    the queue's network indices name (one architecture), `bf16` each one's precision, the
    setting it was trained with (lean.learn.LearnConfig.bf16; default True for each). At most
    ceil(`deal_share` x E) games start in a segment (the networks' batch width follows it).
    `seconds`: the cutoff in seconds of play, after which no game starts (None: the queue's
    end). Every draw comes from torch's default generators (a compiled graph cannot take a
    generator object), seeded by `seed` inside a fork of their states: the caller's are
    restored on return. `progress(games_started, busy_slots, seconds)` is called after each
    segment.
    """
    policies = list(policies)
    bf16 = [True] * len(policies) if bf16 is None else list(bf16)
    if len({repr(policy.config) for policy in policies}) > 1:
        raise ValueError("one runner call takes networks of one architecture")
    devices = list(range(torch.cuda.device_count())) if scripted_table.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        return _play(
            game_config,
            queue,
            scripted_table,
            policies,
            bf16,
            envs,
            segment,
            deal_share,
            seconds,
            compiled,
            progress,
            slot_games,
            minimum_envs,
        )


def _play(
    game_config,
    queue: Queue,
    scripted_table: Tensor,
    policies: list,
    bf16: list,
    envs: int,
    segment: int,
    deal_share: float,
    seconds: float | None,
    compiled_: bool,
    progress,
    slot_games: int,
    minimum_envs: int,
) -> Outcome:
    device = scripted_table.device
    queue = queue.to(device)
    total = len(queue)
    # The drain lasts up to a whole game while slots fall idle, so a short queue takes fewer
    # slots: at least `slot_games` games per slot, never fewer than `minimum_envs` slots.
    envs = min(envs, total, max(minimum_envs, -(-total // slot_games)))
    deal_rows = max(1, math.ceil(deal_share * envs))
    ships = int(queue.team_sizes().sum(-1).max())
    count = len(policies)
    networks = [
        (network(policy, precision), torch.tensor(index, device=device))
        for index, (policy, precision) in enumerate(zip(policies, bf16, strict=True))
    ]
    scripted_seats = bool(((queue.net_a < 0) | (queue.net_b < 0)).any())
    cumulative = seat_counts(queue, count)
    # The most seats a network can hold: E for each seat (A, B) it plays in the queue.
    ceiling = [
        envs * (int((queue.net_a == index).any()) + int((queue.net_b == index).any()))
        for index in range(count)
    ]
    functions = compiled() if compiled_ else eager()
    model_config = policies[0].config if policies else None

    def deals_from(first: int, rows: int) -> Deals:
        return draw_deals(game_config, queue, first, rows, ships, model_config, device)

    opening = deals_from(0, envs)
    # The state is the runner's: its own copies of the deal's shared fixed rows, every tensor
    # whole (a compiled graph is specialised on strides).
    shared = ("bullets", "bullet_state", "tick", "pending")
    state = dataclasses.replace(
        opening.state, **{name: getattr(opening.state, name).clone() for name in shared}
    )
    state = whole_state(state)
    width = len(style_counters.SUMS)
    slots = Slots(
        torch.arange(envs, device=device),
        # The seats' carry and moves, one spare row past the 2E seats (seat_moves).
        policies[0].initial_carry(2 * envs + 1, ships, device) if policies else None,
        torch.zeros((2 * envs + 1, ships, 3), dtype=torch.long, device=device)
        if policies
        else None,
        torch.cat([opening.offset_a, opening.offset_b]),
        torch.zeros(envs, 2, width, device=device),
    )
    result = torch.full((total + 1,), NOT_PLAYED, dtype=torch.long, device=device)
    length = torch.zeros(total + 1, dtype=torch.long, device=device)
    style = torch.zeros(total + 1, 2, width, device=device)
    first, limit = envs, total
    deals = deals_from(first, deal_rows)
    taken = torch.zeros((), dtype=torch.long, device=device)
    limit_tensor = torch.tensor(limit, device=device)
    in_play = seats_in_play(slots, queue, count).tolist() if count else []
    if networks:
        decision = functions["decide"](
            game_config, state, slots, queue, scripted_table, scripted_seats
        )
    started = time.perf_counter()
    while True:
        # Each network's batch width this segment: its seats in play and those of the games
        # the segment may start, at most its ceiling (a bound: no row past it is routed).
        end = min(first + deal_rows, limit)
        upcoming = (cumulative[max(end, first)] - cumulative[first]).tolist()
        positions = [
            torch.arange(max(MINIMUM_WIDTH, min(held + new, most)), device=device)
            for held, new, most in zip(in_play, upcoming, ceiling, strict=True)
        ]
        active = [index for index in range(count) if in_play[index] + upcoming[index] > 0]
        for _ in range(segment):
            if not networks:
                state, slots, taken = functions["tick"](
                    game_config, state, slots, queue, scripted_table,
                    deals, taken, limit_tensor, result, length, style,
                )  # fmt: skip
                continue
            for index in active:
                network_, net = networks[index]
                functions["seat_moves"](
                    network_, net, positions[index], decision, slots.carry, slots.offset,
                    slots.seat_action,
                )  # fmt: skip
            state, slots, taken, decision = functions["advance_decide"](
                game_config, state, slots, decision,
                deals, taken, limit_tensor, result, length, style,
                queue, scripted_table, scripted_seats,
            )  # fmt: skip
        read = [taken, (slots.game >= 0).sum()]
        if count:
            read.append(seats_in_play(slots, queue, count))
        values = torch.cat([value.reshape(-1) for value in read]).tolist()
        used, busy, in_play = values[0], values[1], values[2:]
        first += used
        elapsed = time.perf_counter() - started
        if progress is not None:
            progress(first, busy, elapsed)
        if seconds is not None and elapsed > seconds and limit > first:
            limit = first  # the cutoff: no game starts from here
            limit_tensor = torch.tensor(limit, device=device)
        if first >= limit and busy == 0:
            break
        # The drain needs no deals: a row's queue index past the limit never starts.
        if first < limit:
            deals = deals_from(first, deal_rows)
        else:
            deals = dataclasses.replace(deals, first=torch.tensor(first, device=device))
        taken = torch.zeros((), dtype=torch.long, device=device)
    return Outcome(result[:total].cpu(), length[:total].cpu(), style[:total].cpu())
