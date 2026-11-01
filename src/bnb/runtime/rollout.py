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


def _assign(dst, src) -> None:
    """Copy a pytree of tensors into the matching carry buffers. A leaf a stage passed
    through unchanged (the buffer itself) is not copied onto itself."""
    for d, s in zip(_tensors(dst), _tensors(src), strict=True):
        if d is not s:
            d.copy_(s)


def _tensors(tree) -> list[torch.Tensor]:
    return [x for x in _pytree.tree_leaves(tree) if isinstance(x, torch.Tensor)]


def _clone(tree):
    return _pytree.tree_map_only(torch.Tensor, lambda x: x.contiguous().clone(), tree)


class Rollout:
    """shards: segments the store holds (an update consumes them); host_store: keep them in
    pinned host memory (on a CUDA device) rather than on the device; staggered: start each
    environment's clock at a random tick of its episode (else every game starts fresh)."""

    def __init__(self, cfg: GameConfig, pools: tuple[Pool, ...],
                 controllers: dict[str, Controller], learner: str, num_envs: int,
                 length: int, device: torch.device, compile: bool = True,
                 graph_ticks: int = 32, reward_config: rewards.RewardConfig | None = None,
                 shards: int = 1, host_store: bool = False, staggered: bool = True) -> None:
        self.rules = api.rules_of(cfg, pools, num_envs)
        self.reward_config = reward_config or rewards.RewardConfig()
        self.T, self.S, self.N = cfg.num_teams, cfg.ships_per_team, cfg.num_ships
        self.B, self.length, self.device = num_envs, length, device
        self.lineup = Lineup(pools, num_envs, device)
        missing = set(self.lineup.names) - set(controllers)
        if missing:
            raise ValueError(f"pools name controllers {missing} that were not given")
        self.controllers = {k: controllers[k] for k in self.lineup.names}
        self.learner = learner
        self.reads = controllers[learner].reads
        self.block = max(d for d in range(1, graph_ticks + 1) if length % d == 0)
        self.ring = min(length, 2 * self.block)  # the staging ring: two blocks of records
        self.carry = self._initial_carry(staggered)
        ticks = _pytree.tree_map_only(torch.Tensor, lambda x: x[0], self.carry["buf"])
        self.store = SegmentStore(ticks, self.carry["segment"], shards, length, device,
                                  host_store)
        stages = (self._act, self._advance, self._resolve, self._perceive)
        self.stages = tuple(torch.compile(f, fullgraph=True, dynamic=False) for f in stages) \
            if compile else stages
        learner_step = self.controllers[learner].step
        self._evaluate = torch.compile(learner_step, fullgraph=True, dynamic=False) \
            if compile else learner_step
        self.graph: torch.cuda.CUDAGraph | None = None
        self._written: dict = {}  # ring slot or segment buffer -> the copy that reads it

    # ---- carry ------------------------------------------------------------------------

    def _initial_carry(self, staggered: bool) -> dict:
        rules, B, dev = self.rules, self.B, self.device
        types = api.ship_types(dev)
        st = api.reset(rules, types, api.uniforms(rules, B, dev), self.lineup.fogged)
        if staggered:
            episode_ticks = environment(rules, B, dev).episode_ticks
            st = api.start_clocks(st, torch.randint(0, 1 << 30, (B,), device=dev)
                                  % episode_ticks)
        st, obs, truth = api.perceive(rules, st)
        R, ring, S, N = self.lineup.rows(self.learner), self.ring, self.S, self.N
        view_shape = {part: getattr(obs, part).shape[2:] for part in PARTS}
        launch = launch_state(st.ships)
        flags = dict(dtype=torch.bool, device=dev)
        buf = dict(
            obs=Obs(**{part: torch.zeros((ring, R) + view_shape[part], device=dev)
                       if part in self.reads else None for part in PARTS}),
            action=Action.unpack(torch.zeros((ring, R, N, 3), dtype=torch.int32, device=dev)),
            logp=torch.zeros((ring, R, S), device=dev), value=torch.zeros((ring, R, S), device=dev),
            reward=torch.zeros((ring, R, S, len(REWARD_LEVELS)), device=dev),
            done=torch.zeros((ring, R), **flags),
            alive=torch.zeros((ring, R, N), **flags), valued=torch.zeros((ring, R, S), **flags),
            truth=torch.zeros((ring, R, N - S, truth.ships.shape[-1]), device=dev),
            capture=ShotCapture(
                launch=_pytree.tree_map_only(
                    torch.Tensor, lambda x: torch.zeros((ring, R, N), dtype=x.dtype, device=dev),
                    launch),
                displacement_x=torch.zeros((ring, R, N), device=dev),
                displacement_y=torch.zeros((ring, R, N), device=dev),
                alive=torch.zeros((ring, R, N), **flags)))
        controller_carry = {name: c.init_carry(self.lineup.rows(name), dev)
                            for name, c in self.controllers.items()}
        buf.update(self._learner_buffers(obs, controller_carry[self.learner]))
        segment = dict(final_truth=torch.zeros((R, N, truth.ships.shape[-1]), device=dev))
        if controller_carry[self.learner] is not None:
            segment["initial_carry"] = controller_carry[self.learner]
        carry = dict(
            state=st, types=types, obs={part: getattr(obs, part) for part in PARTS},
            truth=truth.ships, seat=torch.randint(0, self.T, (B,), device=dev),
            actions=torch.zeros((B, self.N, 3), dtype=torch.int32, device=dev),
            spread=torch.zeros((B, self.N, 2), device=dev),
            uniforms=torch.zeros((B, rules.tick_uniform_count), device=dev),
            sweep=api.sweep_like(st),
            ledger=rewards.blank_ledger(B, self.N, dev),
            ctrl=controller_carry,
            buf=buf, segment=segment, t_idx=torch.zeros(1, dtype=torch.long, device=dev),
            counters=torch.zeros(len(COUNTERS), dtype=torch.int64, device=dev))
        return _clone(carry)

    @torch.no_grad()
    def _learner_buffers(self, obs: Obs, learner_carry) -> dict:
        """Ring buffers for what the learner's controller gives beyond logp and value:
        values, outcome and replay, shaped (ring, R, ...). Their shapes come from one probe
        decision on the learner's first row only."""
        ring, R, dev = self.ring, self.lineup.rows(self.learner), self.device
        e, team = self.lineup.env[self.learner][:1], self.lineup.role[self.learner][:1]
        view = Obs(**{part: getattr(obs, part)[e, team] for part in PARTS})
        done = view.globals[..., GLOBAL["reset"]] > 0.5
        one_row = _pytree.tree_map_only(torch.Tensor, lambda x: x[:1].clone(), learner_carry)
        _, _, out = self.controllers[self.learner].step(one_row, view, done)
        out = dict(values=out.values, outcome=out.outcome, replay=out.replay)
        return {name: _pytree.tree_map_only(
            torch.Tensor,
            lambda x: torch.zeros((ring, R) + x.shape[1:], dtype=x.dtype, device=dev), value)
            for name, value in out.items() if value is not None}

    def _team(self, name: str, seat: torch.Tensor) -> torch.Tensor:
        return (self.lineup.role[name] + seat[self.lineup.env[name]]) % self.T

    def _view(self, carry: dict, name: str, parts: tuple[str, ...]) -> Obs:
        e, team = self.lineup.env[name], self._team(name, carry["seat"])
        return Obs(**{part: carry["obs"][part][e, team] if part in parts else None
                      for part in PARTS})

    def _rows(self, carry: dict, values: torch.Tensor) -> torch.Tensor:
        """Per-ship values (B, N, ...) in sim order, as the learner's views' rows (R, N, ...)."""
        name = self.learner
        order = view_order(self.S, self.device)[self._team(name, carry["seat"])]  # (R, N)
        rows = values[self.lineup.env[name]]
        return rows.gather(1, order.reshape(order.shape + (1,) * (rows.dim() - 2))
                           .expand(rows.shape))

    # ---- stages -----------------------------------------------------------------------

    def _slot(self, carry: dict) -> torch.Tensor:
        return carry["t_idx"] % self.ring

    def _act(self, carry: dict) -> None:
        B, T, S, dev = self.B, self.T, self.S, self.device
        actions = torch.zeros((B, T, S, 3), dtype=torch.int32, device=dev)
        slot, buf = self._slot(carry), carry["buf"]
        for name, ctrl in self.controllers.items():
            view = self._view(carry, name, ctrl.reads + ("globals",))
            done = view.globals[..., GLOBAL["reset"]] > 0.5
            action, new_carry, out = ctrl.step(carry["ctrl"][name], view, done)
            if new_carry is not None:
                _assign(carry["ctrl"][name], new_carry)
            packed = action.packed()
            actions = actions.index_put((self.lineup.env[name], self._team(name, carry["seat"])),
                                        packed)
            if name == self.learner:
                for part in self.reads:
                    getattr(buf["obs"], part).index_copy_(0, slot, getattr(view, part)[None])
                ships = carry["obs"]["ships"][self.lineup.env[name],
                                              self._team(name, carry["seat"])]
                alive = ships[..., SHIP["alive"]] > 0.5
                buf["alive"].index_copy_(0, slot, alive[None])
                waiting = ships[:, :S, SHIP["respawn_wait"]] > 0.0
                buf["valued"].index_copy_(0, slot, (alive[:, :S] | waiting)[None])
                if out.logp is not None:  # a scripted "learner" (baselines) records no policy
                    buf["logp"].index_copy_(0, slot, out.logp[None])
                    buf["value"].index_copy_(0, slot, out.value[None])
                for part in ("values", "outcome", "replay"):
                    if part in buf:
                        for stored, given in zip(_tensors(buf[part]), _tensors(getattr(out, part)),
                                                 strict=True):
                            stored.index_copy_(0, slot, given[None])
        flat = actions.reshape(B, self.N, 3)
        carry["actions"].copy_(flat)
        e, team = self.lineup.env[self.learner], self._team(self.learner, carry["seat"])
        for stored, given in zip(_tensors(buf["action"]), _tensors(Action.unpack(
                self._rows(carry, flat))), strict=True):
            stored.index_copy_(0, slot, given[None])
        buf["truth"].index_copy_(0, slot, carry["truth"][e, team][None, :, S:])
        carry["spread"].copy_(api.spread(self.rules, B, dev))
        carry["uniforms"].copy_(api.uniforms(self.rules, B, dev))

    def _advance(self, carry: dict) -> None:
        state = carry["state"]
        moved, sweep = api.move(self.rules, carry["types"], state,
                                Action.unpack(carry["actions"]), carry["spread"])
        capture = ShotCapture(
            launch=_pytree.tree_map_only(torch.Tensor, lambda x: self._rows(carry, x),
                                         launch_state(state.ships)),
            displacement_x=self._rows(carry, sweep.ship_displacement_x),
            displacement_y=self._rows(carry, sweep.ship_displacement_y),
            alive=self._rows(carry, state.ships.alive))
        slot = self._slot(carry)
        for stored, given in zip(_tensors(carry["buf"]["capture"]), _tensors(capture),
                                 strict=True):
            stored.index_copy_(0, slot, given[None])
        _assign(carry["state"], moved)
        _assign(carry["sweep"], sweep)

    def _resolve(self, carry: dict) -> None:
        st, ev = api.resolve(self.rules, carry["types"], carry["state"], carry["sweep"],
                             carry["uniforms"])
        _assign(carry["state"], st)
        team = torch.arange(self.N, device=self.device) // self.S
        reward, ledger = rewards.reward_step(self.reward_config, carry["ledger"], ev, team)
        _assign(carry["ledger"], ledger)
        reward = reward.reshape(self.B, self.T, self.S, len(REWARD_LEVELS))
        name, buf, slot = self.learner, carry["buf"], self._slot(carry)
        e = self.lineup.env[name]
        buf["reward"].index_copy_(0, slot, reward[e, self._team(name, carry["seat"])][None])
        buf["done"].index_copy_(0, slot, ev.done[e][None])
        redraw = torch.randint(0, self.T, (self.B,), device=self.device)
        carry["seat"].copy_(torch.where(ev.done, redraw, carry["seat"]))
        carry["counters"].add_(torch.stack([ev.fired.sum(), (ev.damage > 0).sum(), ev.died.sum(),
                                            ev.done.sum()]))

    def _perceive(self, carry: dict) -> None:
        st, obs, truth = api.perceive(self.rules, carry["state"])
        carry["state"].match.ticks_since_seen.copy_(st.match.ticks_since_seen)
        _assign(carry["obs"], {part: getattr(obs, part) for part in PARTS})
        carry["truth"].copy_(truth.ships)
        carry["t_idx"].add_(1)

    # ---- running ----------------------------------------------------------------------

    def _tick(self) -> None:
        for stage in self.stages:
            stage(self.carry)

    def _capture(self) -> None:
        """Compile, warm up and capture `block` ticks, leaving the carry as it was."""
        saved = _clone(self.carry)
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
        carry, buf, segment = self.carry, self.carry["buf"], self.carry["segment"]
        carry["t_idx"].zero_()
        self._reusable("segment")
        if "initial_carry" in segment:
            _assign(segment["initial_carry"], carry["ctrl"][self.learner])
        cuda = self.device.type == "cuda"
        if cuda and self.graph is None:
            self._capture()
            carry["t_idx"].zero_()
        for start in range(0, self.length, self.block):
            slots = slice(start % self.ring, start % self.ring + self.block)
            self._reusable(slots.start)
            if cuda:
                self.graph.replay()
            else:
                for _ in range(self.block):
                    self._tick()
            self._write(slots.start, {name: _pytree.tree_map_only(
                torch.Tensor, lambda x: x[slots], tree) for name, tree in buf.items()},
                shard, start)
        segment["final_truth"].copy_(self._final_truth(carry))
        self._write("segment", segment, shard)
        return self.trajectory(shard)

    def _final_truth(self, carry: dict) -> torch.Tensor:
        """The truth of the observation after the last step, in the learner's views' rows."""
        name = self.learner
        return carry["truth"][self.lineup.env[name], self._team(name, carry["seat"])]

    def trajectory(self, shard: int) -> Trajectory:
        """Shard `shard` of the store as a Trajectory, (L, R, ...)."""
        self.store.synchronize()
        R = self.lineup.rows(self.learner)
        views = slice(shard * R, (shard + 1) * R)
        fields = {}
        for name, leaves in self.store.data.items():
            picked = [x[:, views] if name in self.store.per_tick else x[views] for x in leaves]
            fields[name] = _pytree.tree_unflatten(picked, self.store.specs[name])
        return Trajectory(**fields)

    @torch.no_grad()
    def bootstrap(self):
        """The learner's controller on its current views without acting (learner spec 2):
        the Outputs (values, outcome) of the next segment's first observation."""
        view = self._view(self.carry, self.learner, self.reads + ("globals",))
        done = view.globals[..., GLOBAL["reset"]] > 0.5
        _, _, out = self._evaluate(self.carry["ctrl"][self.learner], view, done)
        return out

    @torch.no_grad()
    def learner_view(self) -> Obs:
        """The learner's current views."""
        return self._view(self.carry, self.learner, self.reads)

    def counters(self) -> dict[str, int]:
        return dict(zip(COUNTERS, self.carry["counters"].tolist()))
