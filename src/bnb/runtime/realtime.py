"""One match on the CPU for interactive play: the same game functions at B = 1.

The tick is one compiled function (Inductor's C++ backend), shared by every match of the
same rules in the process. It can compile in the background while the match runs the
uncompiled tick (`background=True`), so a new match starts at once. The match draws its
randomness from its own `torch.Generator`, outside the compiled function, so a seed, a
keyframe and the actions decide every state. Process-wide settings (thread count) belong
to the entry point.

The tick takes the rules' tunable values as tensors (`api.tuning.live_of`): one compiled tick
serves every value of them, keyed on the rules with those values blanked
(`api.tuning.static_of`), and the ship card is an input too. So a live change
(`Realtime.retune`: new tunable values or a new card, of the same sizes) is new data on the
next tick, never a recompile.
"""

import dataclasses
import threading
from collections.abc import Callable

import torch
from torch.utils import _pytree

from bnb.contracts import Action, Events, GameConfig, Obs
from bnb.game import api

TickFunction = Callable[..., tuple[api.State, Obs, Events]]

_COMPILED: dict[api.Rules, TickFunction] = {}  # by static rules
_READY: set[api.Rules] = set()  # static rules whose compiled tick has run once (compiled)
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


def compiled_tick(rules: api.Rules) -> TickFunction:
    """The process's compiled tick for these static rules (compiled on its first call)."""
    with _LOCK:
        if rules not in _COMPILED:
            _COMPILED[rules] = torch.compile(_tick_of(rules), fullgraph=True, dynamic=False)
        return _COMPILED[rules]


def clone(tree):
    """A deep copy of a tensor tree (a state, an observation)."""
    return _pytree.tree_map(torch.clone, tree)


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

    compiled: use the compiled tick. background: compile it on a thread while the match
    runs the uncompiled tick, then swap it in (`uses_compiled` says which ran last).
    generator: the match's draws (a fresh one seeded 0 when None). lineup: the ship type of
    each slot, the same for both teams (the game's alternating default when None).
    settings: tuned rules by `api.tuning` key (none: the game's); types: the ship card (None:
    the default card).
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
    ) -> None:
        rules = api.rules_of(config)
        if lineup is not None:
            rules = dataclasses.replace(rules, team_types=tuple(lineup))
        if settings:
            rules = api.tuning.tuned(rules, settings)
        self.config, self.rules = config, rules
        self.static, self.live = api.tuning.static_of(rules), api.tuning.live_of(rules)
        self.types = api.ship_types() if types is None else types
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
        if compiled and background and self.static not in _READY:
            self._thread = threading.Thread(target=self._compile, daemon=True)
            self._thread.start()

    # The compiled tick --------------------------------------------------------------------

    def _draws(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            api.spread(self.rules, 1, generator=self.generator),
            api.uniforms(self.rules, 1, generator=self.generator),
        )

    def _compile(self) -> None:
        """Run the compiled tick once on throwaway inputs, so it is ready to swap in."""
        try:
            generator = torch.Generator().manual_seed(0)
            inputs = (
                torch.zeros(self.rules.num_ships, 3, dtype=torch.int32),
                api.spread(self.rules, 1, generator=generator),
                api.uniforms(self.rules, 1, generator=generator),
            )
            with torch.no_grad():
                compiled_tick(self.static)(clone(self.state), self.types, self.live, *inputs)
            with _LOCK:
                _READY.add(self.static)
        except BaseException as error:  # reported by `compiling`; the match stays eager
            self._compile_error = error

    @property
    def compiling(self) -> bool:
        """The compiled tick is still being built in the background."""
        if self._compile_error is not None:
            raise RuntimeError("the background compile failed") from self._compile_error
        return self._thread is not None and self._thread.is_alive()

    def wait_compiled(self) -> None:
        """Block until a background compile finishes."""
        if self._thread is not None:
            self._thread.join()
        _ = self.compiling  # raises a failed compile's error

    def _tick(self, eager: bool) -> TickFunction:
        background = self._thread is not None and self.static not in _READY
        if self.compiled and not background and not eager:
            self.uses_compiled = True
            return compiled_tick(self.static)
        self.uses_compiled = False
        return self._eager

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
