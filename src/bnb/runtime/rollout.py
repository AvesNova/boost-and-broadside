"""The batched rollout: controllers act, the game ticks, the learner's views are recorded.

A tick is four stages over one carry (a dict of persistent device buffers):
  act       controllers choose actions; the tick's random draws; record learner views
  advance   physics phases 1-3 (fire, ships fly, bullets fly); record the shot capture
  resolve   phases 4-8 (hits through episode end), rewards and their ledger, seat redraw
            at resets
  perceive  visibility, the observation and the truth of the new state
Each stage is a pure function of the carry whose outputs are copied back in.

Every random draw is keyed (`bnb.streams`), never a position in a shared sequence, so two
rollouts of one seed that differ elsewhere (a model option, map-size sampling, a policy)
draw alike wherever they are alike. The carry counts each environment's episodes
(`episode`) and the ticks of its current one (`clock`):
  episode draws   keyed by (seed, environment, episode): the map, the arena scale, the
                  formations and the first spawns of an environment's k-th episode
                  (`api.resolve`'s next_episode_uniforms), and the seat it is played from
  tick draws      keyed by (seed, environment, episode, clock): the tick uniforms a running
                  episode reads (respawns) and the bullet spread
  decision keys   per controller row, the tick's key and the row's role: each controller
                  draws its decision's randomness from it (`Controller.step`)
  stagger         keyed by (seed, environment): each environment's starting clock
`restart`'s batch number keys a restart's episodes apart from every earlier batch's. Stages are
compiled separately (the cut list is this tuple), and on a GPU a block of ticks is captured
as one CUDA graph, so a rollout is a few graph replays.

The learner's records go into a staging ring of two blocks on the device, and after each
block into the segment store (`store.SegmentStore`: pinned host memory, or the device), so
the device holds two blocks of records, not a whole segment (learner spec 2).

After an update's last shard the rollout plays the lookahead (`run_lookahead`): more ticks
of the same games, every environment alike, of which the store keeps only what the
learner's targets read there (`LOOKAHEAD_RETURNS` for the whole lookahead, `LOOKAHEAD_SHOTS`
for a bullet's flight). The next update continues from its end.
"""

import torch
from torch.utils import _pytree

from bnb.contracts import REWARD_LEVELS, Action, Controller, GameConfig, Obs, Pool, Trajectory
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game import api
from bnb.learn import rewards
from bnb.learn.shots import ShotCapture
from bnb.streams import (
    MASK,
    keyed_hash,
    keyed_normal,
    keyed_uniform,
    purpose_key,
    stream_key,
)

from . import analytics
from .analytics import EPISODE_COLUMNS
from .lineup import Lineup
from .opponents import OUTCOMES
from .store import SegmentStore

# The keyed draws' uses (`bnb.streams.purpose_key`): each of the tick's and episode's keys
# feeds several.
GAME_DRAWS = purpose_key("game uniforms")
SPREAD_DRAWS = purpose_key("bullet spread")
SEAT_DRAWS = purpose_key("seat")
DECISION_DRAWS = purpose_key("decision")

PARTS = ("ships", "fields", "globals", "zones")
COUNTERS = ("shots", "hits", "deaths", "resets")
# What the store keeps of the lookahead: the returns' records over all of it, and the shot
# re-simulation's over the first bullet flight (the observation for its fields).
LOOKAHEAD_RETURNS = ("reward", "values", "outcome", "done", "win_loss")
LOOKAHEAD_SHOTS = ("capture", "obs", "done")


def _assign(target, source) -> None:
    """Copy a pytree of tensors into the matching carry buffers. A leaf a stage passed
    through unchanged (the buffer itself) is not copied onto itself."""
    for target_leaf, source_leaf in zip(_tensors(target), _tensors(source), strict=True):
        if target_leaf is not source_leaf:
            target_leaf.copy_(source_leaf)


def _tensors(tree) -> list[torch.Tensor]:
    return [x for x in _pytree.tree_leaves(tree) if isinstance(x, torch.Tensor)]


def _clone(tree):
    return _pytree.tree_map_only(torch.Tensor, lambda x: x.contiguous().clone(), tree)


class Rollout:
    """types: the ship card the games are played with (the carry keeps its own copy).
    shards: segments the store holds (an update consumes them); host_store: keep them in
    pinned host memory (on a CUDA device) rather than on the device; staggered: start each
    environment's clock at a random tick of its episode (else every game starts fresh);
    extra_reads: observation parts the learner's records keep beyond what its controller
    reads (BC's teacher labels the stored observations); lookahead: ticks `run_lookahead`
    plays (a multiple of the graph's block); ship_type: (environments, N) int32, each ship's
    type per environment (`api.reset`; None: the game's lineup), held in place as
    `self.ship_type`, which a `restart` reads (the match engine writes it between
    batches); benching: the games take each team's active ship count as an input
    (`bnb.game.api`; the carry's `active`, (B, 2) int32, every ship until `restart` is
    given counts), for the crossover's unequal teams; off, the game is called without it.
    seed: the run's seed, which keys every draw (the module docstring)."""

    def __init__(
        self,
        config: GameConfig,
        types: api.ShipTypes,
        pools: tuple[Pool, ...],
        controllers: dict[str, Controller],
        learner: str,
        environments: int,
        length: int,
        device: torch.device,
        compiled: bool = True,
        graph_ticks: int = 32,
        reward_config: rewards.RewardConfig | None = None,
        shards: int = 1,
        host_store: bool = False,
        staggered: bool = True,
        extra_reads: tuple[str, ...] = (),
        lookahead: int = 0,
        ship_type: torch.Tensor | None = None,
        benching: bool = False,
        seed: int = 0,
    ) -> None:
        self.reward_config = reward_config or rewards.RewardConfig()
        self.rules = api.with_deathmatch_outcome(
            api.rules_of(config, pools, environments, types=types),
            self.reward_config.deathmatch_outcome,
        )
        self.ship_type = None if ship_type is None else ship_type.to(device, torch.int32, copy=True)
        self.benching = benching
        self.teams, self.ships_per_team, self.ship_count = (
            api.TEAMS,
            config.ships_per_team,
            config.num_ships,
        )
        self.environments, self.length, self.device = environments, length, device
        self.lineup = Lineup(pools, environments, device)
        missing = set(self.lineup.names) - set(controllers)
        if missing:
            raise ValueError(f"pools name controllers {missing} that were not given")
        self.controllers = {k: controllers[k] for k in self.lineup.names}
        self.learner = learner
        # controllers that keep a record of their games' results (the league's seat)
        self.tallied = tuple(
            name for name, controller in self.controllers.items() if hasattr(controller, "tally")
        )
        self.reads = tuple(
            part for part in PARTS if part in controllers[learner].reads + extra_reads
        )
        self.block = max(d for d in range(1, graph_ticks + 1) if length % d == 0)
        if lookahead % self.block:
            raise ValueError(
                f"the lookahead's {lookahead} ticks are not whole {self.block}-tick blocks"
            )
        self.lookahead = lookahead
        self.ring = min(length, 2 * self.block)  # the staging ring: two blocks of records
        self.episode_ticks = api.episode_ticks(self.rules, environments, device).float()
        self.tick_stream = stream_key(seed, "tick")
        self.episode_stream = stream_key(seed, "episode")
        self.stagger_stream = stream_key(seed, "stagger")
        self.environment_index = torch.arange(environments, device=device)
        self.slot_index = torch.arange(self.rules.tick_uniform_count, device=device)
        self.spread_index = (  # (ship, component) of the (B, N, 2) spread
            torch.arange(self.ship_count, device=device)[:, None],
            torch.arange(2, device=device),
        )
        self.carry = self._initial_carry(types, staggered)
        records = self.carry["records"]
        ticks = _pytree.tree_map_only(torch.Tensor, lambda x: x[0], records)
        kept = {name: lookahead for name in LOOKAHEAD_RETURNS if name in records}
        for name in LOOKAHEAD_SHOTS:
            kept[name] = max(kept.get(name, 0), min(lookahead, self.rules.longest_bullet_moves))
        kept = {name: count for name, count in kept.items() if count > 0}
        self.store = SegmentStore(
            ticks, self.carry["segment"], shards, length, device, host_store, kept
        )
        stages = (self._act, self._advance, self._resolve, self._perceive)
        self.stages = (
            tuple(torch.compile(f, fullgraph=True, dynamic=False) for f in stages)
            if compiled
            else stages
        )
        # The bootstrap runs once per update: uncompiled, it costs a forward (milliseconds);
        # compiled, it cost a whole second compile of the learner's step (65 s cold).
        self._evaluate = self.controllers[learner].step
        self.graph: torch.cuda.CUDAGraph | None = None
        self._written: dict = {}  # ring slot or segment buffer -> the copy that reads it

    # ---- carry ------------------------------------------------------------------------

    # ---- keyed draws ------------------------------------------------------------------

    def _episode_keys(self, episode: torch.Tensor) -> torch.Tensor:
        """(B,) each environment's key for its episode `episode` (B,) int64 (a batch number
        in the high 32 bits, `restart`)."""
        return keyed_hash(
            self.episode_stream, self.environment_index, episode & MASK, episode >> 32
        )

    def _episode_uniforms(self, episode: torch.Tensor) -> torch.Tensor:
        """(B, U) the tick uniforms that start each environment's episode `episode`."""
        return keyed_uniform(GAME_DRAWS, self._episode_keys(episode)[:, None], self.slot_index)

    def _seats(self, episode: torch.Tensor) -> torch.Tensor:
        """(B,) int64 the seat each environment's episode `episode` is played from."""
        uniform = keyed_uniform(SEAT_DRAWS, self._episode_keys(episode))
        return (uniform * self.teams).long().clamp_max(self.teams - 1)

    def _tick_keys(self, carry: dict) -> torch.Tensor:
        """(B,) each environment's key for this tick: its episode and its clock."""
        episode = carry["episode"]
        return keyed_hash(
            self.tick_stream, self.environment_index, episode & MASK, episode >> 32, carry["clock"]
        )

    # ---- carry ------------------------------------------------------------------------

    def _initial_carry(
        self,
        types: api.ShipTypes,
        staggered: bool,
        active: torch.Tensor | None = None,
        batch: int = 0,
    ) -> dict:
        rules, B, device = self.rules, self.environments, self.device
        types = _pytree.tree_map_only(torch.Tensor, lambda x: x.to(device, copy=True), types)
        if self.benching and active is None:
            active = torch.full((B, self.teams), self.ships_per_team, dtype=torch.int32)
        if active is not None:
            active = active.to(device=device, dtype=torch.int32)
        episode = torch.full((B,), batch << 32, dtype=torch.int64, device=device)
        state = api.reset(
            rules,
            types,
            self._episode_uniforms(episode),
            self.lineup.fogged,
            ship_type=self.ship_type,
            active=active,
        )
        clock = torch.zeros(B, dtype=torch.int64, device=device)
        if staggered:
            episode_ticks = api.episode_ticks(rules, B, device)
            uniform = keyed_uniform(self.stagger_stream, self.environment_index)
            clock = torch.minimum((uniform * episode_ticks).long(), episode_ticks.long() - 1)
            state = api.start_clocks(state, clock)
        state, obs, truth = api.perceive(rules, state, active)
        views, ring, S, N = (
            self.lineup.rows(self.learner),
            self.ring,
            self.ships_per_team,
            self.ship_count,
        )
        view_shape = {part: getattr(obs, part).shape[2:] for part in PARTS}
        launch = api.launch_capture(state, api.sweep_like(state))[0]
        flags = dict(dtype=torch.bool, device=device)
        records = dict(
            obs=Obs(
                **{
                    part: torch.zeros((ring, views) + view_shape[part], device=device)
                    if part in self.reads
                    else None
                    for part in PARTS
                }
            ),
            action=Action.unpack(
                torch.zeros((ring, views, N, 3), dtype=torch.int32, device=device)
            ),
            log_probability=torch.zeros((ring, views, S), device=device),
            value=torch.zeros((ring, views, S), device=device),
            reward=torch.zeros((ring, views, S, len(REWARD_LEVELS)), device=device),
            done=torch.zeros((ring, views), **flags),
            result=torch.zeros((ring, views), dtype=torch.int32, device=device),
            win_loss=torch.zeros((ring, views, 2), device=device),
            alive=torch.zeros((ring, views, N), **flags),
            valued=torch.zeros((ring, views, S), **flags),
            truth=torch.zeros((ring, views, N - S, truth.ships.shape[-1]), device=device),
            capture=ShotCapture(
                launch=_pytree.tree_map_only(
                    torch.Tensor,
                    lambda x: torch.zeros((ring, views, N), dtype=x.dtype, device=device),
                    launch,
                ),
                displacement_x=torch.zeros((ring, views, N), device=device),
                displacement_y=torch.zeros((ring, views, N), device=device),
                alive=torch.zeros((ring, views, N), **flags),
            ),
        )
        controller_carry = {
            name: c.init_carry(self.lineup.rows(name), device)
            for name, c in self.controllers.items()
        }
        records.update(self._learner_buffers(obs, controller_carry[self.learner]))
        # the outcome ledger's seconds of elapsed time: the longest episode's, and one more
        blocks = int(float(self.episode_ticks.max()) * rules.tick_length) + 2
        segment = dict(final_truth=torch.zeros((views, N, truth.ships.shape[-1]), device=device))
        if controller_carry[self.learner] is not None:
            segment["initial_carry"] = controller_carry[self.learner]
        carry = dict(
            state=state,
            types=types,
            obs={part: getattr(obs, part) for part in PARTS},
            truth=truth.ships,
            seat=self._seats(episode),
            episode=episode,
            clock=clock,
            actions=torch.zeros((B, self.ship_count, 3), dtype=torch.int32, device=device),
            spread=torch.zeros((B, self.ship_count, 2), device=device),
            uniforms=torch.zeros((B, rules.tick_uniform_count), device=device),
            sweep=api.sweep_like(state),
            ledger=rewards.blank_ledger(B, self.ship_count, device),
            controllers=controller_carry,
            records=records,
            segment=segment,
            segment_tick=torch.zeros(1, dtype=torch.long, device=device),
            tallies={
                name: torch.zeros(
                    self.controllers[name].columns
                    * self.controllers[name].mode_count
                    * len(OUTCOMES),
                    dtype=torch.int64,
                    device=device,
                )
                for name in self.tallied
            },
            counters=torch.zeros(len(COUNTERS), dtype=torch.int64, device=device),
            episodes=torch.zeros((B, len(EPISODE_COLUMNS)), device=device),
            game=analytics.blank_game(B, self.ship_count, device),
            game_sums=analytics.blank_game_sums(B, device),
        )
        if active is not None:
            carry["active"] = active
        if "outcome" in records:  # the learner's critic predicts the match's outcome
            carry["outcome"] = analytics.blank_outcome(views, blocks, device)
            carry["outcome_tables"] = analytics.blank_outcome_tables(device)
        return _clone(carry)

    @torch.no_grad()
    def _learner_buffers(self, obs: Obs, learner_carry) -> dict:
        """Ring buffers for what the learner's controller gives beyond log_probability and value:
        values, outcome, replay and probabilities, shaped (ring, R, ...). Their shapes come
        from one probe decision on the learner's first row only."""
        ring, views, device = self.ring, self.lineup.rows(self.learner), self.device
        environments = self.lineup.environments[self.learner][:1]
        team = self.lineup.role[self.learner][:1]
        view = Obs(**{part: getattr(obs, part)[environments, team] for part in PARTS})
        reset = view.globals[..., GLOBAL["reset"]] > 0.5
        one_row = _pytree.tree_map_only(torch.Tensor, lambda x: x[:1].clone(), learner_carry)
        _, _, outputs = self.controllers[self.learner].step(one_row, view, reset)
        outputs = dict(
            values=outputs.values,
            outcome=outputs.outcome,
            replay=outputs.replay,
            probabilities=outputs.probabilities,
        )
        return {
            name: _pytree.tree_map_only(
                torch.Tensor,
                lambda x: torch.zeros((ring, views) + x.shape[1:], dtype=x.dtype, device=device),
                value,
            )
            for name, value in outputs.items()
            if value is not None
        }

    def _team(self, name: str, seat: torch.Tensor) -> torch.Tensor:
        return (self.lineup.role[name] + seat[self.lineup.environments[name]]) % self.teams

    def _view(self, carry: dict, name: str, parts: tuple[str, ...]) -> Obs:
        environments, team = self.lineup.environments[name], self._team(name, carry["seat"])
        return Obs(
            **{
                part: carry["obs"][part][environments, team] if part in parts else None
                for part in PARTS
            }
        )

    def _rows(self, carry: dict, values: torch.Tensor) -> torch.Tensor:
        """Per-ship values (B, N, ...) in sim order, as the learner's views' rows (R, N, ...)."""
        name = self.learner
        order = api.view_order(self.ships_per_team, self.device)[
            self._team(name, carry["seat"])
        ]  # (R, N)
        rows = values[self.lineup.environments[name]]
        return rows.gather(
            1, order.reshape(order.shape + (1,) * (rows.dim() - 2)).expand(rows.shape)
        )

    # ---- stages -----------------------------------------------------------------------

    def _slot(self, carry: dict) -> torch.Tensor:
        return carry["segment_tick"] % self.ring

    def _act(self, carry: dict) -> None:
        B, T, S, device = self.environments, self.teams, self.ships_per_team, self.device
        actions = torch.zeros((B, T, S, 3), dtype=torch.int32, device=device)
        slot, records = self._slot(carry), carry["records"]
        for name, controller in self.controllers.items():
            reads = self.reads if name == self.learner else controller.reads
            view = self._view(carry, name, reads + ("globals",))
            reset = view.globals[..., GLOBAL["reset"]] > 0.5
            action, new_carry, outputs = controller.step(carry["controllers"][name], view, reset)
            if new_carry is not None:
                _assign(carry["controllers"][name], new_carry)
            packed = action.packed()
            actions = actions.index_put(
                (self.lineup.environments[name], self._team(name, carry["seat"])), packed
            )
            if name == self.learner:
                for part in self.reads:
                    getattr(records["obs"], part).index_copy_(0, slot, getattr(view, part)[None])
                ships = carry["obs"]["ships"][
                    self.lineup.environments[name], self._team(name, carry["seat"])
                ]
                alive = ships[..., SHIP["alive"]] > 0.5
                records["alive"].index_copy_(0, slot, alive[None])
                waiting = ships[:, :S, SHIP["respawn_wait"]] > 0.0
                records["valued"].index_copy_(0, slot, (alive[:, :S] | waiting)[None])
                if (
                    outputs.log_probability is not None
                ):  # a scripted "learner" (baselines) records no policy
                    records["log_probability"].index_copy_(0, slot, outputs.log_probability[None])
                    records["value"].index_copy_(0, slot, outputs.value[None])
                if "outcome" in carry:
                    elapsed = (
                        view.globals[..., GLOBAL["episode_progress"]]
                        * self.episode_ticks[self.lineup.environments[name]]
                        * self.rules.tick_length
                    )
                    analytics.outcome_record(carry["outcome"], outputs.outcome, elapsed)
                for part in ("values", "outcome", "replay", "probabilities"):
                    if part in records:
                        for stored, given in zip(
                            _tensors(records[part]), _tensors(getattr(outputs, part)), strict=True
                        ):
                            stored.index_copy_(0, slot, given[None])
        flat = actions.reshape(B, self.ship_count, 3)
        carry["actions"].copy_(flat)
        environments, team = (
            self.lineup.environments[self.learner],
            self._team(self.learner, carry["seat"]),
        )
        for stored, given in zip(
            _tensors(records["action"]),
            _tensors(Action.unpack(self._rows(carry, flat))),
            strict=True,
        ):
            stored.index_copy_(0, slot, given[None])
        records["truth"].index_copy_(0, slot, carry["truth"][environments, team][None, :, S:])
        ticks = self._tick_keys(carry)
        ship, component = self.spread_index
        carry["spread"].copy_(keyed_normal(SPREAD_DRAWS, ticks[:, None, None], ship, component))
        carry["uniforms"].copy_(keyed_uniform(GAME_DRAWS, ticks[:, None], self.slot_index))

    def _advance(self, carry: dict) -> None:
        state = carry["state"]
        moved, sweep = api.move(
            self.rules, carry["types"], state, Action.unpack(carry["actions"]), carry["spread"]
        )
        launch, displacement_x, displacement_y, alive = api.launch_capture(state, sweep)
        capture = ShotCapture(
            launch=_pytree.tree_map_only(torch.Tensor, lambda x: self._rows(carry, x), launch),
            displacement_x=self._rows(carry, displacement_x),
            displacement_y=self._rows(carry, displacement_y),
            alive=self._rows(carry, alive),
        )
        slot = self._slot(carry)
        for stored, given in zip(
            _tensors(carry["records"]["capture"]), _tensors(capture), strict=True
        ):
            stored.index_copy_(0, slot, given[None])
        _assign(carry["state"], moved)
        _assign(carry["sweep"], sweep)

    def _resolve(self, carry: dict) -> None:
        state, events = api.resolve(
            self.rules,
            carry["types"],
            carry["state"],
            carry["sweep"],
            Action.unpack(carry["actions"]),
            carry["uniforms"],
            carry.get("active"),
            next_episode_uniforms=self._episode_uniforms(carry["episode"] + 1),
        )
        _assign(carry["state"], state)
        team = api.ship_team(self.ships_per_team, self.device)
        reward, ledger, event_rewards = rewards.reward_step_events(
            self.reward_config, carry["ledger"], events, team
        )
        _assign(carry["ledger"], ledger)
        analytics.resolve_tick(
            carry["game"],
            carry["game_sums"],
            events,
            reward,
            event_rewards,
            carry["truth"][:, 0],
            self.rules.tick_length,
        )
        reward = reward.reshape(
            self.environments, self.teams, self.ships_per_team, len(REWARD_LEVELS)
        )
        name, records, slot = self.learner, carry["records"], self._slot(carry)
        environments, team = self.lineup.environments[name], self._team(name, carry["seat"])
        records["reward"].index_copy_(0, slot, reward[environments, team][None])
        records["done"].index_copy_(0, slot, events.done[environments][None])
        # Events.result is team 0's; the other team's is its negation (two teams).
        result = torch.where(team == 0, events.result[environments], -events.result[environments])
        records["result"].index_copy_(0, slot, result[None])
        win_loss = events.win_loss[environments]  # team 0's; team 1's is flipped
        win_loss = torch.where((team == 0)[:, None], win_loss, win_loss.flip(-1))
        records["win_loss"].index_copy_(0, slot, win_loss[None])
        for name in self.tallied:  # each ended game's result, from the controller's side
            environments, team = self.lineup.environments[name], self._team(name, carry["seat"])
            own = torch.where(team == 0, events.result[environments], -events.result[environments])
            carry["tallies"][name].add_(
                self.controllers[name].tally(
                    carry["controllers"][name], events.done[environments], own
                )
            )
            count = getattr(self.controllers[name], "count_events", None)
            if count is not None:  # a seat that keeps its games' statistics (the league's)
                count(carry["controllers"][name], events, environments, team)
        # the ended episodes' sums (EPISODE_COLUMNS); the observation is the step's, so its
        # progress is the last tick's, and the episode lasted one tick more
        progress = carry["obs"]["globals"][:, 0, GLOBAL["episode_progress"]]
        ticks = progress * self.episode_ticks + 1.0
        if "outcome" in carry:
            views = self.lineup.environments[self.learner]
            analytics.outcome_score(
                carry["outcome"],
                carry["outcome_tables"],
                events.done[views],
                result > 0,
                ticks[views] * self.rules.tick_length,
            )
        ended = events.done.float()
        timeout = ticks >= self.episode_ticks - 0.5
        carry["episodes"].add_(torch.stack([ended, ended * timeout, ended * ticks], -1))
        episode = carry["episode"] + events.done.long()
        carry["episode"].copy_(episode)
        carry["clock"].copy_(torch.where(events.done, 0, carry["clock"] + 1))
        carry["seat"].copy_(torch.where(events.done, self._seats(episode), carry["seat"]))
        carry["counters"].add_(
            torch.stack(
                [
                    events.fired.sum(),
                    (events.damage > 0).sum(),
                    events.died.sum(),
                    events.done.sum(),
                ]
            )
        )

    def _perceive(self, carry: dict) -> None:
        state, obs, truth = api.perceive(self.rules, carry["state"], carry.get("active"))
        analytics.perceive_tick(
            carry["game_sums"],
            carry["obs"]["ships"],
            obs.ships,
            truth.ships,
            obs.globals,
            carry["obs"]["globals"],
            self.ships_per_team,
            self.rules.tick_length,
            api.MAP_SCALE_RADIUS,
        )
        _assign(api.perceived(carry["state"]), api.perceived(state))
        _assign(carry["obs"], {part: getattr(obs, part) for part in PARTS})
        carry["truth"].copy_(truth.ships)
        carry["segment_tick"].add_(1)

    def take_episodes(self) -> torch.Tensor:
        """The ended episodes' sums since the last take, (B, EPISODE_COLUMNS) on the device
        (no host read), and the sums zeroed."""
        sums = self.carry["episodes"].clone()
        self.carry["episodes"].zero_()
        return sums

    def take_games(self) -> tuple[dict, dict | None]:
        """The games' sums since the last take (`analytics.blank_game_sums`) and the outcome
        tables (None without an outcome head), on the device, and both reset."""
        sums = _clone(self.carry["game_sums"])
        analytics.reset_game_sums(self.carry["game_sums"])
        tables = self.carry.get("outcome_tables")
        if tables is not None:
            tables = _clone(tables)
            for table in self.carry["outcome_tables"].values():
                table.zero_()
        return sums, tables

    def take_tally(self, name: str) -> torch.Tensor:
        """The results the controller `name` has tallied since the last take, on the host
        (a host read), and the tally zeroed."""
        tally = self.carry["tallies"][name]
        taken = tally.to("cpu", copy=True)
        tally.zero_()
        return taken

    @torch.no_grad()
    def restart(self, active: torch.Tensor | None = None, batch: int = 0) -> None:
        """Every environment starts a fresh game from its clock's start, and the whole carry
        is a new rollout's (every controller's carry, the seats, the tallies, the sums),
        copied in place so a captured graph stays valid: the offline match engine's batch
        (`bnb.league.matches.Arena`). active (B, 2) int32: each team's active ships from
        now on, for a rollout built with benching (None: every ship). batch: the batch's
        number, which keys its episodes (the same number plays the same maps and draws)."""
        if active is not None and not self.benching:
            raise ValueError("active ship counts need a rollout built with benching")
        if not 0 <= batch < 1 << 31:
            raise ValueError(f"batch {batch} must be in [0, 2^31)")
        _assign(self.carry, self._initial_carry(self.carry["types"], False, active, batch))

    # ---- checkpoints --------------------------------------------------------------------

    def saved_carry(self) -> dict:
        """The carry a resume needs (runs spec 3): the game state, the observation, every
        controller's carry, the seats, the episode counts and clocks that key the draws, the
        reward ledger, the counters and the games' running state (`game`, the outcome
        ledger). Not the staging ring or the segment buffers, which every segment writes
        before it reads them, nor the update's sums (the episodes', the
        games', the outcome tables), which every update takes, nor the active ship counts
        (`benching`: the offline match engine's, which no training run resumes)."""
        excluded = ("records", "segment", "episodes", "game_sums", "outcome_tables", "active")
        return {name: tree for name, tree in self.carry.items() if name not in excluded}

    def load_carry(self, carry: dict) -> None:
        """Copy a saved carry into the carry's buffers (a captured graph keeps reading
        them)."""
        _assign(self.saved_carry(), carry)

    # ---- running ----------------------------------------------------------------------

    def _tick(self) -> None:
        for stage in self.stages:
            stage(self.carry)

    def _capture(self) -> None:
        """Compile, warm up and capture `block` ticks, leaving the carry and the device's
        default generator as they were: the capture draws nothing a replay would, so a run
        resumed in a fresh process (which captures again) draws what the first did."""
        saved = _clone(self.carry)
        generator = torch.cuda.default_generators[
            self.device.index if self.device.index is not None else torch.cuda.current_device()
        ]
        generator_state = generator.get_state()
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._tick()
        torch.cuda.current_stream(self.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(self.block):
                self._tick()
        _assign(self.carry, saved)
        generator.set_state(generator_state)

    def _reusable(self, key) -> None:
        """Before the compute stream writes a ring slot or segment buffer again, wait for
        the copy that reads it."""
        written = self._written.pop(key, None)
        if written is not None:
            torch.cuda.current_stream(self.device).wait_event(written)

    def _write(self, key, source: dict, shard: int | None, start: int | None = None) -> None:
        """Copy `source` into the store's shard `shard`, or the lookahead's ticks from `start`
        (shard None)."""
        if shard is None:
            written = self.store.write_lookahead(source, start)
        else:
            written = self.store.write(source, shard, start)
        if written is not None:
            self._written[key] = written

    def _play(self, ticks: int, shard: int | None) -> None:
        """Advance `ticks` ticks block by block, each block's records into the store's shard
        `shard` (None: the lookahead's)."""
        carry, records = self.carry, self.carry["records"]
        cuda = self.device.type == "cuda"
        if cuda and self.graph is None:
            self._capture()
            carry["segment_tick"].zero_()
        for start in range(0, ticks, self.block):
            slots = slice(start % self.ring, start % self.ring + self.block)
            self._reusable(slots.start)
            if cuda:
                self.graph.replay()
            else:
                for _ in range(self.block):
                    self._tick()
            self._write(
                slots.start,
                {
                    name: _pytree.tree_map_only(torch.Tensor, lambda x: x[slots], tree)
                    for name, tree in records.items()
                },
                shard,
                start,
            )

    @torch.no_grad()
    def run(self, shard: int = 0) -> Trajectory:
        """Advance `length` ticks into the store's shard `shard`; returns that shard as a
        Trajectory (views of the store, valid until the shard is written again)."""
        carry, segment = self.carry, self.carry["segment"]
        carry["segment_tick"].zero_()
        self._reusable("segment")
        if "initial_carry" in segment:
            _assign(segment["initial_carry"], carry["controllers"][self.learner])
        self._play(self.length, shard)
        segment["final_truth"].copy_(self._final_truth(carry))
        self._write("segment", segment, shard)
        return self.trajectory(shard)

    @torch.no_grad()
    def run_lookahead(self) -> None:
        """Advance the lookahead's ticks after the update's last shard: the same games go on,
        and the store keeps the lookahead's records (`SegmentStore.lookahead`)."""
        self.carry["segment_tick"].zero_()
        self._play(self.lookahead, None)

    def _final_truth(self, carry: dict) -> torch.Tensor:
        """The truth of the observation after the last step, in the learner's views' rows."""
        name = self.learner
        return carry["truth"][self.lineup.environments[name], self._team(name, carry["seat"])]

    def trajectory(self, shard: int) -> Trajectory:
        """Shard `shard` of the store as a Trajectory, (L, R, ...)."""
        self.store.synchronize()
        count = self.lineup.rows(self.learner)
        views = slice(shard * count, (shard + 1) * count)
        fields = {}
        for name in self.store.names:
            leaves = self.store.data[name]
            picked = [x[:, views] if name in self.store.per_tick else x[views] for x in leaves]
            fields[name] = _pytree.tree_unflatten(picked, self.store.specs[name])
        return Trajectory(**fields)

    @torch.no_grad()
    @torch.no_grad()
    def bootstrap(self):
        """The learner's controller on its current views without acting (learner spec 2):
        the Outputs (values, outcome) of the next segment's first observation."""
        view = self._view(self.carry, self.learner, self.reads + ("globals",))
        reset = view.globals[..., GLOBAL["reset"]] > 0.5
        _, _, outputs = self._evaluate(self.carry["controllers"][self.learner], view, reset)
        return outputs

    def counters(self) -> dict[str, int]:
        """The carry's running totals by COUNTERS name: shots fired, damage events, deaths and
        environment resets."""
        return dict(zip(COUNTERS, self.carry["counters"].tolist()))
