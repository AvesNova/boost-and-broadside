"""One match on the CPU for interactive play: the same game functions at B = 1.

The tick is one compiled library (AOTInductor's C++), shared by every match of the same
rules in the process. It is built in a child process and kept on disk (`library_path`, in
`bnb.cache`: keyed by the game's code, torch's version and the rules), then loaded in
milliseconds. Building it in this process would trace and compile on the interpreter lock
for tens of seconds and freeze the window's thread however the work was threaded; a child
has its own. With `background=True` the match runs the uncompiled tick until the library is
ready, so a new match starts at once. The library agrees with the uncompiled tick to
float32 fusion rounding; a replay repeats each tick as it ran (`uses_compiled`), so it
reproduces the match either way. The match draws its randomness from its own
`torch.Generator`, outside the tick, so a seed, a keyframe and the actions decide every
state. Process-wide settings (thread count) belong to the entry point.

The tick takes the rules' tunable values as tensors (`api.tuning.live_of`): one library
serves every value of them, keyed on the rules with those values blanked
(`api.tuning.static_of`), and the ship card is an input too. So a live change
(`Realtime.retune`: new tunable values or a new card, of the same sizes) is new data on the
next tick, never a rebuild.
"""

import dataclasses
import multiprocessing
import os
import pathlib
import threading
from collections.abc import Callable

import torch
from torch.utils import _pytree

from bnb import cache
from bnb.contracts import Action, Events, GameConfig, Obs
from bnb.game import api

TickFunction = Callable[..., tuple[api.State, Obs, Events]]

_TICK_CODE = ("game", "contracts", "runtime/realtime.py")  # the code the built tick holds
_BUILT: dict[api.Rules, Library] = {}  # this process's loaded libraries, by static rules
_LOCK = threading.Lock()


def _tick_of(static: api.Rules) -> TickFunction:
    """The tick for these static rules: move, resolve and perceive, with the rules' live
    values, the card and the draws as inputs."""

    def tick(
        state: api.State,
        types: api.ShipTypes,
        live: dict[str, torch.Tensor],
        actions: torch.Tensor,
        spread: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> tuple[api.State, Obs, Events]:
        rules = api.tuning.with_live(static, live)
        moved, sweep = api.move(rules, types, state, Action.unpack(actions[None]), spread)
        stepped, events = api.resolve(rules, types, moved, sweep, uniforms)
        stepped, obs, _ = api.perceive(rules, stepped)
        return stepped, obs, events

    return tick


def clone(tree):
    """A deep copy of a tensor tree (a state, an observation)."""
    return _pytree.tree_map(torch.clone, tree)


def _throwaway_inputs(rules: api.Rules) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Actions and draws to build or try a tick on (its result is dropped)."""
    generator = torch.Generator().manual_seed(0)
    return (
        torch.zeros(rules.num_ships, 3, dtype=torch.int32),
        api.spread(rules, 1, generator=generator),
        api.uniforms(rules, 1, generator=generator),
    )


def library_path(static: api.Rules) -> pathlib.Path:
    """Where the built tick for these static rules is kept (`bnb.cache`)."""
    return cache.path("ticks", static, packages=_TICK_CODE, suffix=".pt2")


def _tensor_positions(leaves: list) -> list[int]:
    return [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]


class _Flat(torch.nn.Module):
    """The tick on its inputs' tensor leaves, returning its outputs' tensor leaves (what
    export packages: the trees themselves are rebuilt by `Library`)."""

    def __init__(self, static: api.Rules, arguments: tuple) -> None:
        super().__init__()
        self.tick = _tick_of(static)
        self.leaves, self.spec = _pytree.tree_flatten(arguments)
        self.positions = _tensor_positions(self.leaves)

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        leaves = list(self.leaves)
        for position, tensor in zip(self.positions, tensors, strict=True):
            leaves[position] = tensor
        out = self.tick(*_pytree.tree_unflatten(leaves, self.spec))
        return tuple(leaf for leaf in _pytree.tree_leaves(out) if isinstance(leaf, torch.Tensor))


def _build(path: pathlib.Path, rules: api.Rules, static: api.Rules, state, types, live) -> None:
    """In a child process: export the tick for these static rules and package it at `path`
    (written aside, then moved in, so a reader never meets half a file)."""
    torch.set_num_threads(1)
    arguments = (state, types, live, *_throwaway_inputs(rules))
    module = _Flat(static, arguments)
    tensors = tuple(module.leaves[position] for position in module.positions)
    with torch.no_grad():
        program = torch.export.export(module, tensors, strict=True)
    partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.pt2")
    torch._inductor.aoti_compile_and_package(program, package_path=str(partial))
    partial.replace(path)


class Library:
    """A built tick, loaded: called like the uncompiled tick, on trees of the same shapes."""

    def __init__(self, path: pathlib.Path, static: api.Rules, arguments: tuple) -> None:
        self._run = torch._inductor.aoti_load_package(str(path))
        self._positions = _tensor_positions(_pytree.tree_leaves(arguments))
        with torch.no_grad():  # the outputs' tree: the uncompiled tick's, once
            out = _tick_of(static)(*clone(arguments))
        self._out_leaves, self._out_spec = _pytree.tree_flatten(out)
        self._out_positions = _tensor_positions(self._out_leaves)

    def __call__(self, *arguments):
        leaves = _pytree.tree_leaves(arguments)
        tensors = self._run(*[leaves[position] for position in self._positions])
        out = list(self._out_leaves)
        for position, tensor in zip(self._out_positions, tensors, strict=True):
            out[position] = tensor
        return _pytree.tree_unflatten(out, self._out_spec)


def library(rules: api.Rules, static: api.Rules, state, types, live) -> Library:
    """The process's built tick for these static rules: loaded if this process has it,
    read from disk if it was built before, else built in a child process first (the
    calling thread waits without holding the interpreter lock)."""
    with _LOCK:
        if static in _BUILT:
            return _BUILT[static]
    path = library_path(static)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        child = multiprocessing.get_context("spawn").Process(
            target=_build,
            args=(path, rules, static, state, types, live),
            name="tick-build",
            daemon=True,
        )
        child.start()
        child.join()
        if child.exitcode != 0 or not path.exists():
            raise RuntimeError(f"building the compiled tick failed (exit code {child.exitcode})")
    loaded = Library(path, static, (state, types, live, *_throwaway_inputs(rules)))
    with _LOCK:
        return _BUILT.setdefault(static, loaded)


@dataclasses.dataclass(frozen=True)
class Keyframe:
    """Everything a match continues from: its state, every team's latest view and its
    generator's state."""

    state: api.State
    obs: Obs
    generator: torch.Tensor


class Realtime:
    """One match (B = 1) on the CPU for interactive play: its state, every team's latest
    view `obs`, the last tick's `events`, and the tick.

    compiled: use the compiled tick (`library`; without `background`, built or loaded
    before the constructor returns). background: get it on a thread while the match runs
    the uncompiled tick, then swap it in (`uses_compiled` says which ran last).
    generator: the match's draws (a fresh one seeded 0 when None). lineup: the ship type of
    each slot, the same for both teams (the game's alternating default when None).
    settings: tuned rules by `api.tuning` key (none: the game's); types: the ship card (None:
    the default card of the config's ship set); bullet_slots: per ship (None:
    `api.tuning.most_bullet_slots`, room for any fire rate and bullet life live tuning
    allows: a match never runs out of slots while it is tuned, and spare slots cost the tick
    next to nothing).
    """

    def __init__(
        self,
        config: GameConfig,
        compiled: bool = True,
        fogged: bool = True,
        generator: torch.Generator | None = None,
        lineup: tuple[int, ...] | None = None,
        background: bool = False,
        settings: dict[str, float] | None = None,
        types: api.ShipTypes | None = None,
        bullet_slots: int | None = None,
    ) -> None:
        types = api.ship_types(ship_set=config.ship_set) if types is None else types
        rules = api.rules_of(config)
        if lineup is not None:
            rules = dataclasses.replace(rules, team_types=tuple(lineup))
        if settings:
            rules = api.tuning.tuned(rules, settings)
        if bullet_slots is None:  # room for any fire rate and bullet life live tuning allows
            bullet_slots = api.tuning.most_bullet_slots(rules.tick_length)
        rules = dataclasses.replace(rules, bullet_slots=bullet_slots)
        self.config, self.rules = config, rules
        self.static, self.live = api.tuning.static_of(rules), api.tuning.live_of(rules)
        self.types = types
        self.generator = torch.Generator().manual_seed(0) if generator is None else generator
        fog_flags = torch.tensor([fogged])
        uniforms = api.uniforms(rules, 1, generator=self.generator)
        self.state = api.reset(rules, self.types, uniforms, fog_flags)
        self.state, self.obs, _ = api.perceive(rules, self.state)
        self.events: Events | None = None
        self._eager = _tick_of(self.static)
        self.compiled = compiled
        self.uses_compiled = False
        self._compile_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        if compiled and background and self.static not in _BUILT:
            self._thread = threading.Thread(target=self._compile, daemon=True)
            self._thread.start()
        elif compiled:
            self._compile()
            _ = self.compiling  # raises a failed build's error

    # The compiled tick --------------------------------------------------------------------

    def _draws(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            api.spread(self.rules, 1, generator=self.generator),
            api.uniforms(self.rules, 1, generator=self.generator),
        )

    def _compile(self) -> None:
        """Load the built tick for these rules (`library`: built in a child process first
        when no earlier run built it)."""
        try:
            library(self.rules, self.static, clone(self.state), self.types, self.live)
        except BaseException as error:  # reported by `compiling`; the match stays uncompiled
            self._compile_error = error

    @property
    def compiling(self) -> bool:
        """The compiled tick is still being built or loaded in the background."""
        if self._compile_error is not None:
            raise RuntimeError("building the compiled tick failed") from self._compile_error
        return self._thread is not None and self._thread.is_alive()

    def wait_compiled(self) -> None:
        """Block until a background build finishes."""
        if self._thread is not None:
            self._thread.join()
        _ = self.compiling  # raises a failed build's error

    def _tick(self, eager: bool) -> TickFunction:
        built = _BUILT.get(self.static) if self.compiled and not eager else None
        self.uses_compiled = built is not None
        return self._eager if built is None else built

    # Stepping, keyframes ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, actions: torch.Tensor, eager: bool = False) -> Obs:
        """Advance one tick with (N, 3) int32 actions (sim order); returns every team's view,
        shaped (1, 2, ...), and keeps the tick's events. Resets automatically at the end of
        an episode. eager: run the uncompiled tick this time (a replay repeats a tick as it
        was run)."""
        tick = self._tick(eager)
        spread, uniforms = self._draws()
        self.state, self.obs, self.events = tick(
            self.state, self.types, self.live, actions, spread, uniforms
        )
        return self.obs

    def retune(self, rules: api.Rules | None = None, types: api.ShipTypes | None = None) -> None:
        """Play on with new rules and/or a new ship card from the next tick, keeping the
        match: rules that differ only in their live values (`api.tuning.static_of`), and a
        card of the same shapes, or ValueError. Nothing recompiles. The state carries over
        (`api.retune`: the zone ring refit, shields and stores capped)."""
        rules = self.rules if rules is None else rules
        types = self.types if types is None else types
        if api.tuning.static_of(rules) != self.static:
            raise ValueError("live tuning keeps the sizes and the mode; a new match changes them")
        for name, value in vars(types).items():
            current = getattr(self.types, name)
            if value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError(f"the card's {name} must stay {tuple(current.shape)}")
        self.rules, self.live, self.types = rules, api.tuning.live_of(rules), types
        self.state = api.retune(rules, types, self.state)

    def keyframe(self) -> Keyframe:
        """A copy of everything the match continues from."""
        return Keyframe(
            state=clone(self.state),
            obs=clone(self.obs),
            generator=self.generator.get_state(),
        )

    def restore(self, keyframe: Keyframe) -> None:
        """Continue from a keyframe (of a match with the same rules); the last tick's events
        are gone."""
        self.state, self.obs = clone(keyframe.state), clone(keyframe.obs)
        self.events = None
        self.generator.set_state(keyframe.generator)
