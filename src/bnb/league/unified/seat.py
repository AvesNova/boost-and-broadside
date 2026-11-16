"""The unified evaluator's seat: every row's player as data, so an environment may change
players between its games with nothing recompiled.

Each row (an environment's role) holds the index of the player it plays (`RefillCarry.
player`); at the first tick of a new game (the observation's reset) it takes the player the
host wrote for its next game (`RefillSeat.pending`, written in place between graph
replays). The players are fixed for the run:

- A scripted player (Imugi at a temperature, share and floor) plays through one scripted
  pass over every row, each row at its player's play (read from the player tables by
  index, so the pass is the same graph whoever plays).
- A network player has a lane of its group (one group per build; the lanes' weights
  stacked once, `bnb.runtime.stacked`). Each lane has a fixed capacity of rows: every tick
  the rows playing each lane's player are compacted into the lane's table (a sort by
  player, static shapes), each group's lanes run as one stacked forward over their tables,
  and the actions and network carry are written back to the rows. A lane's unused
  capacity runs on padding and writes nothing. The host keeps every player's rows within
  its capacity (the planner's `fits`), counting both a row's current and its next game.

The seat also keeps each row's game log (`count_events`): the deaths of both teams and the
ticks so far, and at a game's end its result, deaths and ticks as the row's last game
(`last_*`), with the count of games the row has finished, for the host to read after a
replay. A row's results are its own side's; the engine reads its first-role rows.
"""

import dataclasses

import torch
from torch.utils import _pytree

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.tree import contract
from bnb.controllers.imugi import Imugi
from bnb.controllers.scripted import floored_actions
from bnb.runtime import stacked
from bnb.runtime.opponents import OUTCOMES


@dataclasses.dataclass(frozen=True)
class SeatPlayer:
    """A player as the seat plays it: group, its network group (None: scripted); for a
    scripted player its temperature, share and floor; for a network its sampling
    temperature."""

    name: str
    group: int | None = None
    temperature: float = 1.0
    share: float = 1.0
    floor: float = 0.0


@contract
class RefillCarry:
    """Per row (R leading): player (R,) int64, the player it plays now; model, the network
    groups' carry ({group: carry}); scripted, Imugi's; deaths (R, 2) float, its own team's
    and the enemy's deaths in the game so far; ticks (R,) float; the last finished game's
    last_result (R,) int32 (+1 won, 0 drew, -1 lost), last_deaths (R, 2) and last_ticks (R,);
    finished (R,) int64 the games ended; stepped (R,) bool, the row has decided once (the
    run's first observation's reset begins the first game, which the carry's player
    already plays)."""

    player: torch.Tensor
    model: object
    scripted: object
    deaths: torch.Tensor
    ticks: torch.Tensor
    last_result: torch.Tensor
    last_deaths: torch.Tensor
    last_ticks: torch.Tensor
    finished: torch.Tensor
    stepped: torch.Tensor


class _Decision(torch.nn.Module):
    """A network controller's decision as a module call, for `functional_call`."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.model = controller.model
        self.controller = controller

    def forward(self, carry, obs: Obs, reset, decision_key, temperature):
        action, carry, _ = self.controller.step(
            carry, obs, reset, decision_key, temperature=temperature
        )
        return action.packed(), carry


class RefillSeat:
    """controllers: one network controller per group (over a template model of the group's
    build); players: every player of the run; runs: (group, its players, rows per lane),
    the network players cut into runs of like demand, each run one stacked forward; rows:
    the seat's rows (2 per environment); characteristics and ships_per_team: Imugi's.
    states: each network player's model state dict, loaded into its lane once."""

    def __init__(
        self,
        controllers: list,
        players: list[SeatPlayer],
        states: dict[int, dict[str, torch.Tensor]],
        runs: list[tuple[int, list[int], int]],
        rows: int,
        characteristics,
        ships_per_team: int,
        device: torch.device,
    ) -> None:
        self.rows, self.ships_per_team, self.players = rows, ships_per_team, players
        self.columns, self.mode_count = 1, 1
        count = len(players)
        self.scripted_player = torch.tensor(
            [one.group is None for one in players], dtype=torch.bool, device=device
        )
        self.player_temperature = torch.tensor(
            [one.temperature if one.group is None else 1.0 for one in players], device=device
        )
        self.player_share = torch.tensor([one.share for one in players], device=device)
        self.player_floor = torch.tensor([one.floor for one in players], device=device)
        self.player_network_temperature = torch.tensor(
            [one.temperature if one.group is not None else 1.0 for one in players], device=device
        )
        # the scripted pass reads these per row, refreshed from the tables each tick
        self.temperature = torch.ones(rows, device=device)
        self.share = torch.ones(rows, device=device)
        self.agent = Imugi(
            characteristics, ships_per_team, temperature=self.temperature, share=self.share
        )
        self.decisions = [_Decision(controller) for controller in controllers]
        for decision in self.decisions:
            decision.requires_grad_(False)
        # the runs: (group, its players, capacity); every network player in exactly one
        self.run_group = [group for group, _, _ in runs]
        self.lanes = [list(members) for _, members, _ in runs]
        self.capacity = [min(int(capacity), rows) for _, _, capacity in runs]
        placed = sorted(index for members in self.lanes for index in members)
        networks = [index for index, one in enumerate(players) if one.group is not None]
        if placed != networks:
            raise ValueError(f"runs {runs}: each network player in exactly one run")
        lane_of = torch.full((count,), -1, dtype=torch.int64)
        run_of = torch.full((count,), -1, dtype=torch.int64)
        for run, members in enumerate(self.lanes):
            for lane, index in enumerate(members):
                if players[index].group != self.run_group[run]:
                    raise ValueError(f"run {run}: player {index} is of another group")
                lane_of[index], run_of[index] = lane, run
        self.lane_of, self.run_of = lane_of.to(device), run_of.to(device)
        self.parameters: list[dict[str, torch.Tensor]] = []
        reads = Imugi.reads
        for decision in self.decisions:
            reads = decision.controller.reads + reads
        for run, members in enumerate(self.lanes):
            template = self.decisions[self.run_group[run]].state_dict()
            stacked_state = {
                name: value.detach()[None].repeat((len(members),) + (1,) * value.dim()).clone()
                for name, value in template.items()
            }
            for lane, index in enumerate(members):
                for name, value in states[index].items():
                    stacked_state[f"model.{name}"][lane].copy_(value)
            self.parameters.append(stacked_state)
        self.reads = tuple(dict.fromkeys(reads))
        # the host's: each row's next game's player (written in place between replays)
        self.pending = torch.zeros(rows, dtype=torch.int64, device=device)
        # rows past a lane's capacity this tick (never, when the host keeps to it)
        self.overflow = torch.zeros(1, dtype=torch.int64, device=device)
        self._row = torch.arange(rows, device=device)

    # ---- the controller

    def init_carry(self, views: int, device: torch.device) -> RefillCarry:
        """Every row's carry at the run's start: each row's first player from the pending buffer."""
        model = {}
        for group, decision in enumerate(self.decisions):
            one = decision.controller.init_carry(views, device)
            if one is not None:
                model[str(group)] = one
        zeros = torch.zeros(views, device=device)
        return RefillCarry(
            player=self.pending.clone()
            if views == self.rows
            else torch.zeros(views, dtype=torch.int64, device=device),
            model=model,
            scripted=self.agent.init_carry(views, device),
            deaths=torch.zeros((views, 2), device=device),
            ticks=zeros.clone(),
            last_result=torch.zeros(views, dtype=torch.int32, device=device),
            last_deaths=torch.zeros((views, 2), device=device),
            last_ticks=zeros.clone(),
            finished=torch.zeros(views, dtype=torch.int64, device=device),
            stepped=torch.zeros(views, dtype=torch.bool, device=device),
        )

    def _table(self, run: int, player: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(lanes, capacity) the rows each lane's player plays now, as (the rows to read,
        padding reading row 0; the rows to write, padding writing the spare row `rows`): the
        rows sorted by (lane, row), a static compaction."""
        lanes, capacity = len(self.lanes[run]), self.capacity[run]
        in_run = self.run_of[player] == run
        lane = torch.where(in_run, self.lane_of[player], lanes)  # outside: past the last
        order = torch.argsort(lane * self.rows + self._row)
        counts = (lane[None, :] == torch.arange(lanes, device=lane.device)[:, None]).sum(1)
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(capacity, device=lane.device)
        position = (starts[:, None] + slot[None, :]).clamp_max(self.rows - 1)
        real = slot[None, :] < counts[:, None]
        self.overflow.add_((counts - capacity).clamp_min(0).sum())
        rows = order[position]
        return torch.where(real, rows, 0), torch.where(real, rows, self.rows)

    def step(self, carry: RefillCarry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """One decision for every row; see `Controller.step`. A view count other than the
        seat's rows (the rollout's probe) gets idle actions."""
        views = reset.shape[0]
        if views != self.rows:
            idle = torch.zeros((views, self.ships_per_team, 3), dtype=torch.int32)
            return Action.unpack(idle.to(reset.device)), carry, Outputs()
        player = torch.where(reset & carry.stepped, self.pending, carry.player)
        self.temperature.copy_(self.player_temperature[player])
        self.share.copy_(self.player_share[player])
        scripted, scripted_carry, _ = self.agent.step(carry.scripted, obs, reset, decision_key)
        action = floored_actions(scripted.packed(), self.player_floor[player], decision_key)
        model_carry = dict(carry.model)
        if self.decisions:
            network = torch.zeros_like(action)
            temperature = self.player_network_temperature[player]
            for run, group in enumerate(self.run_group):
                decision = self.decisions[group]
                read, write = self._table(run, player)
                flat, target = read.reshape(-1), write.reshape(-1)
                arguments = (model_carry.get(str(group)), obs, reset, decision_key, temperature)
                inputs = _pytree.tree_map_only(
                    torch.Tensor,
                    lambda x: x.index_select(0, flat).unflatten(0, read.shape),
                    arguments,
                )
                packed, lane_carry = stacked.call(decision, self.parameters[run], inputs)

                def written(full, part, target=target):
                    if full is None:
                        return None
                    # padding writes the spare row past the end, then dropped
                    spare = torch.cat([full, full[:1]])
                    return spare.index_copy(0, target, part.flatten(0, 1).to(full.dtype))[:-1]

                network = written(network, packed)
                if lane_carry is not None:
                    model_carry[str(group)] = _pytree.tree_map(
                        written, model_carry[str(group)], lane_carry
                    )
            is_scripted = self.scripted_player[player]
            action = torch.where(is_scripted[:, None, None], action, network)
        carry = dataclasses.replace(
            carry,
            player=player,
            model=model_carry,
            scripted=scripted_carry,
            stepped=torch.ones_like(carry.stepped),
        )
        return Action.unpack(action), carry, Outputs()

    def tally(self, carry: RefillCarry, done: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """Nothing (the game log is `count_events`'), as the rollout's tally reads it."""
        return torch.zeros(len(OUTCOMES), dtype=torch.int64, device=done.device)

    def count_events(self, carry: RefillCarry, events, environments, team) -> None:
        """Add a tick to each row's game log, in place, and at a game's end move it to the
        row's last game (see the module docstring)."""
        S = self.ships_per_team
        ship_team = torch.arange(2 * S, device=team.device) // S
        own = ship_team[None, :] == team[:, None]
        died = events.died[environments]
        deaths = torch.stack([(died & own).sum(-1), (died & ~own).sum(-1)], -1).float()
        total = carry.deaths + deaths
        ticks = carry.ticks + 1.0
        done = events.done[environments]
        result = torch.where(team == 0, events.result[environments], -events.result[environments])
        carry.last_result.copy_(torch.where(done, result.to(torch.int32), carry.last_result))
        carry.last_deaths.copy_(torch.where(done[:, None], total, carry.last_deaths))
        carry.last_ticks.copy_(torch.where(done, ticks, carry.last_ticks))
        carry.finished.add_(done.to(torch.int64))
        carry.deaths.copy_(torch.where(done[:, None], 0.0, total))
        carry.ticks.copy_(torch.where(done, 0.0, ticks))
