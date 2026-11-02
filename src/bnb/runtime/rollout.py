"""The batched rollout: controllers act, the game ticks, the learner's views are recorded.

A tick is four stages over one carry (a dict of persistent device buffers):
  act       controllers choose actions; the tick's random draws; record learner views
  advance   physics phases 1-3 (fire, ships fly, bullets fly); record the shot capture
  resolve   phases 4-8 (hits through episode end), rewards and their ledger, seat redraw
            at resets
  perceive  visibility, the observation and the truth of the new state
Each stage is a pure function of the carry whose outputs are copied back in. Stages are
compiled separately (the cut list is this tuple), and on a GPU a block of ticks is captured
as one CUDA graph, so a rollout is a few graph replays.

The learner's records go into a staging ring of two blocks on the device, and after each
block into the segment store (`store.SegmentStore`: pinned host memory, or the device), so
the device holds two blocks of records, not a whole segment (learner spec 2).
"""

import torch
from torch.utils import _pytree

from bnb.contracts import REWARD_LEVELS, Action, Controller, GameConfig, Obs, Pool, Trajectory
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game import api
from bnb.game.layout import environment
from bnb.game.perception.observe import view_order
from bnb.game.physics.bullets import launch_state
from bnb.learn import rewards
from bnb.learn.shots import ShotCapture

from .lineup import Lineup
from .store import SegmentStore

PARTS = ("ships", "fields", "globals", "zones")
COUNTERS = ("shots", "hits", "deaths", "resets")


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
    """shards: segments the store holds (an update consumes them); host_store: keep them in
    pinned host memory (on a CUDA device) rather than on the device; staggered: start each
    environment's clock at a random tick of its episode (else every game starts fresh);
    extra_reads: observation parts the learner's records keep beyond what its controller
    reads (BC's teacher labels the stored observations)."""

    def __init__(
        self,
        config: GameConfig,
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
    ) -> None:
        self.rules = api.rules_of(config, pools, environments)
        self.reward_config = reward_config or rewards.RewardConfig()
        self.teams, self.ships_per_team, self.ships = (
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
        self.reads = tuple(
            part for part in PARTS if part in controllers[learner].reads + extra_reads
        )
        self.block = max(d for d in range(1, graph_ticks + 1) if length % d == 0)
        self.ring = min(length, 2 * self.block)  # the staging ring: two blocks of records
        self.carry = self._initial_carry(staggered)
        ticks = _pytree.tree_map_only(torch.Tensor, lambda x: x[0], self.carry["records"])
        self.store = SegmentStore(ticks, self.carry["segment"], shards, length, device, host_store)
        stages = (self._act, self._advance, self._resolve, self._perceive)
        self.stages = (
            tuple(torch.compile(f, fullgraph=True, dynamic=False) for f in stages)
            if compiled
            else stages
        )
        learner_step = self.controllers[learner].step
        self._evaluate = (
            torch.compile(learner_step, fullgraph=True, dynamic=False) if compiled else learner_step
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self._written: dict = {}  # ring slot or segment buffer -> the copy that reads it

    # ---- carry ------------------------------------------------------------------------

    def _initial_carry(self, staggered: bool) -> dict:
        rules, B, device = self.rules, self.environments, self.device
        types = api.ship_types(device)
        state = api.reset(rules, types, api.uniforms(rules, B, device), self.lineup.fogged)
        if staggered:
            episode_ticks = environment(rules, B, device).episode_ticks
            state = api.start_clocks(
                state, torch.randint(0, 1 << 30, (B,), device=device) % episode_ticks
            )
        state, obs, truth = api.perceive(rules, state)
        views, ring, S, N = (
            self.lineup.rows(self.learner),
            self.ring,
            self.ships_per_team,
            self.ships,
        )
        view_shape = {part: getattr(obs, part).shape[2:] for part in PARTS}
        launch = launch_state(state.ships)
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
        segment = dict(final_truth=torch.zeros((views, N, truth.ships.shape[-1]), device=device))
        if controller_carry[self.learner] is not None:
            segment["initial_carry"] = controller_carry[self.learner]
        carry = dict(
            state=state,
            types=types,
            obs={part: getattr(obs, part) for part in PARTS},
            truth=truth.ships,
            seat=torch.randint(0, self.teams, (B,), device=device),
            actions=torch.zeros((B, self.ships, 3), dtype=torch.int32, device=device),
            spread=torch.zeros((B, self.ships, 2), device=device),
            uniforms=torch.zeros((B, rules.tick_uniform_count), device=device),
            sweep=api.sweep_like(state),
            ledger=rewards.blank_ledger(B, self.ships, device),
            controllers=controller_carry,
            records=records,
            segment=segment,
            segment_tick=torch.zeros(1, dtype=torch.long, device=device),
            counters=torch.zeros(len(COUNTERS), dtype=torch.int64, device=device),
        )
        return _clone(carry)

    @torch.no_grad()
    def _learner_buffers(self, obs: Obs, learner_carry) -> dict:
        """Ring buffers for what the learner's controller gives beyond log_probability and value:
        values, outcome and replay, shaped (ring, R, ...). Their shapes come from one probe
        decision on the learner's first row only."""
        ring, views, device = self.ring, self.lineup.rows(self.learner), self.device
        environments = self.lineup.environments[self.learner][:1]
        team = self.lineup.role[self.learner][:1]
        view = Obs(**{part: getattr(obs, part)[environments, team] for part in PARTS})
        reset = view.globals[..., GLOBAL["reset"]] > 0.5
        one_row = _pytree.tree_map_only(torch.Tensor, lambda x: x[:1].clone(), learner_carry)
        _, _, outputs = self.controllers[self.learner].step(one_row, view, reset)
        outputs = dict(values=outputs.values, outcome=outputs.outcome, replay=outputs.replay)
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
        order = view_order(self.ships_per_team, self.device)[
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
                for part in ("values", "outcome", "replay"):
                    if part in records:
                        for stored, given in zip(
                            _tensors(records[part]), _tensors(getattr(outputs, part)), strict=True
                        ):
                            stored.index_copy_(0, slot, given[None])
        flat = actions.reshape(B, self.ships, 3)
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
        carry["spread"].copy_(api.spread(self.rules, B, device))
        carry["uniforms"].copy_(api.uniforms(self.rules, B, device))

    def _advance(self, carry: dict) -> None:
        state = carry["state"]
        moved, sweep = api.move(
            self.rules, carry["types"], state, Action.unpack(carry["actions"]), carry["spread"]
        )
        capture = ShotCapture(
            launch=_pytree.tree_map_only(
                torch.Tensor, lambda x: self._rows(carry, x), launch_state(state.ships)
            ),
            displacement_x=self._rows(carry, sweep.ship_displacement_x),
            displacement_y=self._rows(carry, sweep.ship_displacement_y),
            alive=self._rows(carry, state.ships.alive),
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
            self.rules, carry["types"], carry["state"], carry["sweep"], carry["uniforms"]
        )
        _assign(carry["state"], state)
        team = torch.arange(self.ships, device=self.device) // self.ships_per_team
        reward, ledger = rewards.reward_step(self.reward_config, carry["ledger"], events, team)
        _assign(carry["ledger"], ledger)
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
        redraw = torch.randint(0, self.teams, (self.environments,), device=self.device)
        carry["seat"].copy_(torch.where(events.done, redraw, carry["seat"]))
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
        state, obs, truth = api.perceive(self.rules, carry["state"])
        carry["state"].match.ticks_since_seen.copy_(state.match.ticks_since_seen)
        _assign(carry["obs"], {part: getattr(obs, part) for part in PARTS})
        carry["truth"].copy_(truth.ships)
        carry["segment_tick"].add_(1)

    # ---- checkpoints --------------------------------------------------------------------

    def saved_carry(self) -> dict:
        """The carry a resume needs (runs spec 3): the game state, the observation, every
        controller's carry, the seats, the reward ledger and the counters. Not the staging
        ring or the segment buffers, which every segment writes before it reads them."""
        return {
            name: tree for name, tree in self.carry.items() if name not in ("records", "segment")
        }

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

    def _write(self, key, source: dict, shard: int, start: int | None = None) -> None:
        written = self.store.write(source, shard, start)
        if written is not None:
            self._written[key] = written

    @torch.no_grad()
    def run(self, shard: int = 0) -> Trajectory:
        """Advance `length` ticks into the store's shard `shard`; returns that shard as a
        Trajectory (views of the store, valid until the shard is written again)."""
        carry, records, segment = self.carry, self.carry["records"], self.carry["segment"]
        carry["segment_tick"].zero_()
        self._reusable("segment")
        if "initial_carry" in segment:
            _assign(segment["initial_carry"], carry["controllers"][self.learner])
        cuda = self.device.type == "cuda"
        if cuda and self.graph is None:
            self._capture()
            carry["segment_tick"].zero_()
        for start in range(0, self.length, self.block):
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
        segment["final_truth"].copy_(self._final_truth(carry))
        self._write("segment", segment, shard)
        return self.trajectory(shard)

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
    def bootstrap(self):
        """The learner's controller on its current views without acting (learner spec 2):
        the Outputs (values, outcome) of the next segment's first observation."""
        view = self._view(self.carry, self.learner, self.reads + ("globals",))
        reset = view.globals[..., GLOBAL["reset"]] > 0.5
        _, _, outputs = self._evaluate(self.carry["controllers"][self.learner], view, reset)
        return outputs

    def counters(self) -> dict[str, int]:
        return dict(zip(COUNTERS, self.carry["counters"].tolist()))
