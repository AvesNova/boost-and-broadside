"""One match on the CPU for interactive play: the same game functions at B = 1.

The tick is one compiled library (`bnb.runtime.built`: AOTInductor's C++), shared by every
match of the same rules in the process. It is built in a child process and kept on disk
(`library_path`, in `bnb.cache`: keyed by the game's code, torch's version and the rules),
then loaded in milliseconds. With `background=True` the match runs the uncompiled tick until
the library is ready, so a new match starts at once. The library agrees with the uncompiled
tick to float32 fusion rounding; a replay repeats each tick as it ran (`uses_compiled`), so
it reproduces the match either way. The match draws its randomness from its own
`torch.Generator`, outside the tick, so a seed, a keyframe and the actions decide every
state. Process-wide settings (thread count) belong to the entry point.

The tick takes the rules' tunable values as tensors (`api.tuning.live_of`): one library
serves every value of them, keyed on the rules with those values blanked
(`api.tuning.static_of`), and the ship card is an input too. So a live change
(`Realtime.retune`: new tunable values or a new card, of the same sizes) is new data on the
next tick, never a rebuild.
"""

import dataclasses
import pathlib
import threading
from collections.abc import Callable

import torch

from bnb import cache
from bnb.contracts import Action, Events, GameConfig, Obs
from bnb.game import api
from bnb.runtime import built
from bnb.runtime.built import clone

TickFunction = Callable[..., tuple[api.State, Obs, Events]]

_TICK_CODE = ("game", "contracts", "runtime/realtime.py", "runtime/built.py")  # what it holds


@dataclasses.dataclass(frozen=True)
class Tick:
    """The tick for these static rules: move, resolve and perceive, with the rules' live
    values, the card and the draws as inputs."""

    static: api.Rules

    def __call__(
        self,
        state: api.State,
        types: api.ShipTypes,
        live: dict[str, torch.Tensor],
        actions: torch.Tensor,
        spread: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> tuple[api.State, Obs, Events]:
        rules = api.tuning.with_live(self.static, live)
        moved, sweep = api.move(rules, types, state, Action.unpack(actions[None]), spread)
        stepped, events = api.resolve(rules, types, moved, sweep, uniforms)
        stepped, obs, _ = api.perceive(rules, stepped)
        return stepped, obs, events


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


def library(rules: api.Rules, static: api.Rules, state, types, live) -> built.Library:
    """The process's built tick for these static rules (`bnb.runtime.built.library`: built
    in a child process when no earlier run built it)."""
    tick, arguments = Tick(static), (state, types, live, *_throwaway_inputs(rules))
    with torch.no_grad():  # the outputs' structure: the uncompiled tick's (it draws nothing)
        outputs = tick(*clone(arguments))
    return built.library(library_path(static), tick, arguments, outputs, name="tick-build")


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
        if settings:  # as live tuning holds them: a time scale carries some past a slider's end
            rules = api.tuning.tuned(rules, settings, bounded=False)
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
        self._eager = Tick(self.static)
        self._path = library_path(self.static)
        self.compiled = compiled
        self.uses_compiled = False
        self._compile_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        if compiled and background and built.loaded(self._path) is None:
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
        loaded = built.loaded(self._path) if self.compiled and not eager else None
        self.uses_compiled = loaded is not None
        return self._eager if loaded is None else loaded

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
