"""The batched rollout: controllers act, the game ticks, the learner's views are recorded.

A tick is four stages over one carry (a dict of persistent device buffers):
  act       controllers choose actions; the tick's random draws; record learner views
  advance   physics phases 1-3 (fire, ships fly, bullets fly)
  resolve   phases 4-8 (hits through episode end), rewards, seat redraw at resets
  perceive  visibility and the observation of the new state
Each stage is a pure function of the carry whose outputs are copied back in. Stages are
compiled separately (the cut list is this tuple), and on a GPU a block of ticks is captured
as one CUDA graph, so a rollout is a few graph replays.
"""

import torch
from torch.utils import _pytree

from bnb.contracts import Action, Controller, GameConfig, Obs, Pool, Trajectory
from bnb.contracts.obs import GLOBAL
from bnb.game import match
from bnb.game.modes import deathmatch

from .lineup import Lineup

PARTS = ("ships", "fields", "globals")
COUNTERS = ("shots", "hits", "deaths", "resets")


def _assign(dst, src) -> None:
    """Copy a pytree of tensors into the matching carry buffers."""
    for d, s in zip(_tensors(dst), _tensors(src), strict=True):
        d.copy_(s)


def _tensors(tree) -> list[torch.Tensor]:
    return [x for x in _pytree.tree_leaves(tree) if isinstance(x, torch.Tensor)]


def _clone(tree):
    return _pytree.tree_map_only(torch.Tensor, lambda x: x.contiguous().clone(), tree)


class Rollout:
    def __init__(self, cfg: GameConfig, pools: tuple[Pool, ...],
                 controllers: dict[str, Controller], learner: str, num_envs: int,
                 length: int, device: torch.device, compile: bool = True,
                 graph_ticks: int = 32) -> None:
        self.p = match.rules_of(cfg)
        self.T, self.S, self.N = cfg.num_teams, cfg.ships_per_team, cfg.num_ships
        self.B, self.length, self.device = num_envs, length, device
        self.lineup = Lineup(pools, num_envs, device)
        missing = set(self.lineup.names) - set(controllers)
        if missing:
            raise ValueError(f"pools name controllers {missing} that were not given")
        self.controllers = {k: controllers[k] for k in self.lineup.names}
        self.learner = learner
        self.reads = controllers[learner].reads
        self.carry = self._initial_carry()
        stages = (self._act, self._advance, self._resolve, self._perceive)
        self.stages = tuple(torch.compile(f, fullgraph=True, dynamic=False) for f in stages) \
            if compile else stages
        self.block = max(d for d in range(1, graph_ticks + 1) if length % d == 0)
        self.graph: torch.cuda.CUDAGraph | None = None

    # ---- carry ------------------------------------------------------------------------

    def _initial_carry(self) -> dict:
        p, B, dev = self.p, self.B, self.device
        types = match.ship_types(dev)
        st = match.reset(p, match.uniforms(p, B, dev), types)
        st = match.start_clocks(st, torch.randint(0, p.episode_ticks, (B,), device=dev))
        st, obs, _ = match.perceive(p, st)
        R, L, S = self.lineup.rows(self.learner), self.length, self.S
        view_shape = {part: getattr(obs, part).shape[2:] for part in PARTS}
        buf = dict(
            obs={part: torch.zeros((L, R) + view_shape[part], device=dev) for part in self.reads},
            action=torch.zeros((L, R, S, 3), dtype=torch.int32, device=dev),
            logp=torch.zeros((L, R, S), device=dev), value=torch.zeros((L, R, S), device=dev),
            reward=torch.zeros((L, R, S), device=dev),
            done=torch.zeros((L, R), dtype=torch.bool, device=dev))
        carry = dict(
            state=st, types=types, obs={part: getattr(obs, part) for part in PARTS},
            seat=torch.randint(0, self.T, (B,), device=dev),
            actions=torch.zeros((B, self.N, 3), dtype=torch.int32, device=dev),
            spread=torch.zeros((B, self.N, 2), device=dev),
            uniforms=torch.zeros((B, p.num_uniforms), device=dev),
            sweep=match.sweep_like(st),
            ctrl={name: c.init_carry(self.lineup.rows(name), dev)
                  for name, c in self.controllers.items()},
            buf=buf, t_idx=torch.zeros(1, dtype=torch.long, device=dev),
            counters=torch.zeros(len(COUNTERS), dtype=torch.int64, device=dev))
        return _clone(carry)

    def _team(self, name: str, seat: torch.Tensor) -> torch.Tensor:
        return (self.lineup.role[name] + seat[self.lineup.env[name]]) % self.T

    def _view(self, carry: dict, name: str, parts: tuple[str, ...]) -> Obs:
        e, team = self.lineup.env[name], self._team(name, carry["seat"])
        return Obs(**{part: carry["obs"][part][e, team] if part in parts else None
                      for part in PARTS})

    # ---- stages -----------------------------------------------------------------------

    def _act(self, carry: dict) -> None:
        B, T, S, dev = self.B, self.T, self.S, self.device
        actions = torch.zeros((B, T, S, 3), dtype=torch.int32, device=dev)
        t = carry["t_idx"]
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
                buf = carry["buf"]
                for part in self.reads:
                    buf["obs"][part].index_copy_(0, t, getattr(view, part)[None])
                buf["action"].index_copy_(0, t, packed[None])
                if out.logp is not None:  # a scripted "learner" (baselines) records no policy
                    buf["logp"].index_copy_(0, t, out.logp[None])
                    buf["value"].index_copy_(0, t, out.value[None])
        carry["actions"].copy_(actions.reshape(B, self.N, 3))
        carry["spread"].copy_(match.spread(self.p, B, dev))
        carry["uniforms"].copy_(match.uniforms(self.p, B, dev))

    def _advance(self, carry: dict) -> None:
        moved, sweep = match.move(self.p, carry["state"], Action.unpack(carry["actions"]),
                                   carry["spread"], carry["types"])
        _assign(carry["state"], moved)
        _assign(carry["sweep"], sweep)

    def _resolve(self, carry: dict) -> None:
        st, ev = match.resolve(self.p, carry["state"], carry["sweep"], carry["uniforms"],
                               carry["types"])
        _assign(carry["state"], st)
        reward = deathmatch.reward(ev).reshape(self.B, self.T, self.S)
        name, buf, t = self.learner, carry["buf"], carry["t_idx"]
        e = self.lineup.env[name]
        buf["reward"].index_copy_(0, t, reward[e, self._team(name, carry["seat"])][None])
        buf["done"].index_copy_(0, t, ev.done[e][None])
        redraw = torch.randint(0, self.T, (self.B,), device=self.device)
        carry["seat"].copy_(torch.where(ev.done, redraw, carry["seat"]))
        carry["counters"].add_(torch.stack([ev.fired.sum(), (ev.damage > 0).sum(), ev.died.sum(),
                                            ev.done.sum()]))

    def _perceive(self, carry: dict) -> None:
        st, obs, _ = match.perceive(self.p, carry["state"])
        carry["state"].match.last_seen.copy_(st.match.last_seen)
        _assign(carry["obs"], {part: getattr(obs, part) for part in PARTS})
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

    @torch.no_grad()
    def run(self) -> Trajectory:
        """Advance `length` ticks; the returned trajectory views the carry's buffers (valid
        until the next run)."""
        self.carry["t_idx"].zero_()
        if self.device.type == "cuda":
            if self.graph is None:
                self._capture()
                self.carry["t_idx"].zero_()
            for _ in range(self.length // self.block):
                self.graph.replay()
        else:
            for _ in range(self.length):
                self._tick()
        buf = self.carry["buf"]
        obs = Obs(**{part: buf["obs"].get(part) for part in PARTS})
        return Trajectory(obs=obs, action=Action.unpack(buf["action"]), logp=buf["logp"],
                          value=buf["value"], reward=buf["reward"], done=buf["done"])

    @torch.no_grad()
    def learner_view(self) -> Obs:
        """The learner's current views (for the value bootstrap after a rollout)."""
        return self._view(self.carry, self.learner, self.reads)

    def counters(self) -> dict[str, int]:
        return dict(zip(COUNTERS, self.carry["counters"].tolist()))
