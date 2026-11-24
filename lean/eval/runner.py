"""The evaluation runner: a queue of games played on E environments at once.

Every environment is a slot. A game is (agent A, agent B, the team A plays, A's ships,
B's ships). The first E games of the queue are dealt at the start; when a slot's game ends
it takes the next game of the queue at once (rolling). At the cutoff (the queue's end, or
`seconds` of play) no game starts and every started game plays to its end (the drain), so
the results are those of a prefix of the queue, each game played whole.

Seats: in one call, seat A is either one network for every game or a scripted agent per game
(an index into `scripted`, a table of full parameter vectors, lean.scripted.parameters.vector),
and seat B the same. The scripted agent runs once per tick for every ship with every
parameter row as a per-ship tensor (rows_of), so new vectors never compile a new graph. A
network runs on all E environments from its seat's team view as in training
(lean.model.agent.network_step: the weights are inputs, so another network of the same
architecture is new data too, and the precision is the network's own); each game draws its
own position offset (as in training).

Deals are drawn once per segment (`segment` ticks) for the next E games of the queue, and
the host reads two numbers per segment (games taken, slots busy); nothing is read in a tick.

Each slot sums its game's style counters (lean.eval.style) per seat inside the tick; a finished
game's sums go to its record and the slot's sums start at zero with the next deal.
"""

import dataclasses
import functools
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from lean import game, scripted
from lean.contract import compile_dynamic, split_joint, whole_state
from lean.eval import style as style_counters
from lean.model import draw_position_offset
from lean.model.agent import Network, network, network_step, reset_carry, seat_view
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import rows_of

# A game's result from seat A's side.
A_WON, B_WON, DRAW, NOT_PLAYED = 0, 1, 2, -1
# The scripted agent's config beside full per-ship rows: never read for a value.
FIXED_CONFIG = ScriptedConfig()


@dataclass
class Queue:
    """Q games in order, (Q,) long each: `a`, `b` the scripted index of seats A and B (into
    the call's table; ignored for a seat a network plays), `side` the team seat A plays (0 or
    1), `size_a`, `size_b` the seats' ships."""

    a: Tensor
    b: Tensor
    side: Tensor
    size_a: Tensor
    size_b: Tensor

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
    """The next P games of the queue dealt at once (one segment's starts): the game state, the
    scripted memory and both seats' position offsets, leading P; `first` the queue index of
    row 0 (a 0-d tensor: a new value is data for a compiled graph)."""

    state: object
    memory: dict
    offset_a: Tensor
    offset_b: Tensor
    first: Tensor  # () long


@dataclass
class Slots:
    """The E environments: `game` (E,) long, the queue index each plays (-1 idle), the
    scripted memory, both seats' network carries and position offsets (None without a
    network there), and `style` (E, 2, len(style.SUMS)) the game's style sums so far."""

    game: Tensor
    memory: dict
    carry_a: Tensor | None
    carry_b: Tensor | None
    offset_a: Tensor
    offset_b: Tensor
    style: Tensor


def draw_deals(
    game_config, queue: Queue, first: int, count: int, slots: int, model_configs, device
) -> Deals:
    """Deals for the queue's games first .. first + count - 1 (rows past the queue's end
    repeat its last game; they are never started). `model_configs`: seat A's and seat B's
    network configs (None for a scripted seat: a zero offset)."""
    index = (first + torch.arange(count, device=device)).clamp(max=len(queue) - 1)
    sizes = queue.team_sizes().index_select(0, index)
    state = game.deal(game_config, count, slots, sizes, None, device)
    offsets = [
        draw_position_offset(config, count, None, device)
        if config is not None
        else torch.zeros(count, 2, device=device)
        for config in model_configs
    ]
    first_tensor = torch.tensor(first, dtype=torch.long, device=device)
    return Deals(state, scripted.fresh_memory(count, slots, None, device), *offsets, first_tensor)


def network_moves(
    network_: Network, seat_team: Tensor, views, carry: Tensor, offset: Tensor
) -> tuple[Tensor, Tensor]:
    """A network's sampled actions (E, N, 3) from team `seat_team[e]`'s view of every
    environment, and its new carry."""
    envs = seat_team.shape[0]
    view = seat_view(views, torch.arange(envs, device=seat_team.device), seat_team)
    joint, carry = network_step(network_, view, carry, offset, None)
    return split_joint(joint), carry


def tick(
    game_config,
    state,
    slots: Slots,
    queue: Queue,
    table: Tensor,
    network_a,
    network_b,
    deals: Deals,
    taken: Tensor,
    limit: Tensor,
    result: Tensor,
    length: Tensor,
    style: Tensor,
):
    """One tick of every environment, then the starts. `taken` (0-d long) counts the rows of
    `deals` used, `limit` (0-d long) is the first queue index that may not start; `result`
    and `length` (Q + 1,) and `style` (Q + 1, 2, len(style.SUMS)) take each finished game's
    record in place (entry Q is spare). Returns (state, slots, taken)."""
    envs, ships = state.exists.shape
    device = state.exists.device
    playing = slots.game >= 0
    current = slots.game.clamp(min=0)
    a, b, side = (
        queue.a.index_select(0, current),
        queue.b.index_select(0, current),
        queue.side.index_select(0, current),
    )
    seat_a = state.team == side[:, None]  # (E, N) the ship plays for seat A
    action = torch.zeros((envs, ships, 3), dtype=torch.long, device=device)
    memory = slots.memory
    if network_a is None or network_b is None:
        agent = torch.where(seat_a, a[:, None], b[:, None])
        uniforms = scripted.draw_uniforms(envs, ships, None, device)
        action, memory, _ = scripted.decide(
            FIXED_CONFIG, state, None, rows_of(table, agent), uniforms, memory
        )
    carry_a, carry_b = slots.carry_a, slots.carry_b
    if network_a is not None or network_b is not None:
        views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
        if network_a is not None:
            moves, carry_a = network_moves(network_a, side, views, carry_a, slots.offset_a)
            action = torch.where(seat_a[..., None], moves, action)
        if network_b is not None:
            moves, carry_b = network_moves(network_b, 1 - side, views, carry_b, slots.offset_b)
            action = torch.where(seat_a[..., None], action, moves)

    noise = game.draw_noise(game_config, envs, ships, None, device)
    start_state = state
    state, out = game.step(game_config, state, action, noise)
    sums = slots.style + style_counters.tick_sums(
        start_state, seat_a, playing, start_state.pending[..., 1], state.fired, out.hit_damage
    )
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

    def take(fresh: Tensor, old: Tensor) -> Tensor:
        return torch.where(start.view(-1, *[1] * (old.dim() - 1)), fresh.index_select(0, pick), old)

    slots = Slots(
        game=torch.where(start, next_game, torch.where(want, -1, slots.game)),
        memory={name: take(deals.memory[name], memory[name]) for name in memory},
        carry_a=None if carry_a is None else reset_carry(carry_a, start),
        carry_b=None if carry_b is None else reset_carry(carry_b, start),
        offset_a=take(deals.offset_a, slots.offset_a),
        offset_b=take(deals.offset_b, slots.offset_b),
        style=torch.where(want[:, None, None], 0.0, sums),
    )
    return state, slots, taken + start.sum()


@functools.cache
def compiled_tick():
    """`tick` under torch.compile (lean.contract.compile_dynamic, as lean.learn.compiled)."""
    return compile_dynamic(tick)


@torch.no_grad()
def play(
    game_config,
    queue: Queue,
    scripted_table: Tensor,
    policy_a=None,
    policy_b=None,
    *,
    envs: int = 4096,
    segment: int = 32,
    seconds: float | None = None,
    compiled: bool = False,
    seed: int = 0,
    progress=None,
    slot_games: int = 16,
    minimum_envs: int = 256,
    bf16: tuple[bool, bool] = (True, True),
) -> Outcome:
    """Play the queue's games (see the module's docstring) on at most `envs` environments on
    the table's device (fewer for a short queue: Q / `slot_games`, at least `minimum_envs`).
    `scripted_table` (S, len(PARAMETERS)) float32 holds the scripted
    agents' vectors (one row at least, even when both seats are networks); `policy_a`,
    `policy_b` a lean.model.Policy playing every game's seat A or B, or None for the
    scripted agent of the queue's index; `bf16` each seat's network precision, the setting
    it was trained with (lean.learn.LearnConfig.bf16). `seconds`: the cutoff in seconds of
    play, after which no game starts (None: the queue's end). Every draw comes from torch's
    default generators (a compiled graph cannot take a generator object), seeded by `seed`
    inside a fork of their states: the caller's are restored on return.
    `progress(games_started, busy_slots, seconds)` is called after each segment.
    """
    devices = list(range(torch.cuda.device_count())) if scripted_table.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        return _play(
            game_config,
            queue,
            scripted_table,
            policy_a,
            policy_b,
            envs,
            segment,
            seconds,
            compiled,
            progress,
            slot_games,
            minimum_envs,
            bf16,
        )


def _play(
    game_config,
    queue: Queue,
    scripted_table: Tensor,
    policy_a,
    policy_b,
    envs: int,
    segment: int,
    seconds: float | None,
    compiled: bool,
    progress,
    slot_games: int,
    minimum_envs: int,
    bf16: tuple[bool, bool],
) -> Outcome:
    device = scripted_table.device
    queue = queue.to(device)
    total = len(queue)
    # The drain lasts up to a whole game while slots fall idle, so a short queue takes fewer
    # slots: at least `slot_games` games per slot, never fewer than `minimum_envs` slots.
    envs = min(envs, total, max(minimum_envs, -(-total // slot_games)))
    ships = int(queue.team_sizes().sum(-1).max())
    network_a = None if policy_a is None else network(policy_a, bf16[0])
    network_b = None if policy_b is None else network(policy_b, bf16[1])
    configs = [None if policy is None else policy.config for policy in (policy_a, policy_b)]
    function = compiled_tick() if compiled else tick

    def deals_from(first: int) -> Deals:
        return draw_deals(game_config, queue, first, envs, ships, configs, device)

    opening = deals_from(0)
    # The state is the runner's: its own copies of the deal's shared fixed rows, every tensor
    # whole (a compiled graph is specialised on strides).
    shared = ("bullets", "bullet_counts", "bullet_real", "tick", "pending")
    state = dataclasses.replace(
        opening.state, **{name: getattr(opening.state, name).clone() for name in shared}
    )
    state = whole_state(state)
    carries = [
        None if policy is None else policy.initial_carry(envs, ships, device)
        for policy in (policy_a, policy_b)
    ]
    width = len(style_counters.SUMS)
    slots = Slots(
        torch.arange(envs, device=device),
        opening.memory,
        *carries,
        opening.offset_a,
        opening.offset_b,
        torch.zeros(envs, 2, width, device=device),
    )
    result = torch.full((total + 1,), NOT_PLAYED, dtype=torch.long, device=device)
    length = torch.zeros(total + 1, dtype=torch.long, device=device)
    style = torch.zeros(total + 1, 2, width, device=device)
    first, limit = envs, total
    deals = deals_from(first)
    taken = torch.zeros((), dtype=torch.long, device=device)
    limit_tensor = torch.tensor(limit, device=device)
    started = time.perf_counter()
    while True:
        for _ in range(segment):
            state, slots, taken = function(
                game_config,
                state,
                slots,
                queue,
                scripted_table,
                network_a,
                network_b,
                deals,
                taken,
                limit_tensor,
                result,
                length,
                style,
            )
        used, busy = torch.stack([taken, (slots.game >= 0).sum()]).tolist()
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
            deals = deals_from(first)
        else:
            deals = dataclasses.replace(deals, first=torch.tensor(first, device=device))
        taken = torch.zeros((), dtype=torch.long, device=device)
    return Outcome(result[:total].cpu(), length[:total].cpu(), style[:total].cpu())
