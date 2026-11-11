"""Live tuning: change the game's numbers while a match plays, and see the effect on the
next tick.

Two kinds of number, both applied through `MatchLoop.retune`:

- **Rule settings** (`api.tuning`: the arena and clock, bullet life, the shared flight
  constants, spawns, each mode's rules). They apply on the next tick: the compiled tick
  takes their values as tensors (`Realtime.retune`), so nothing recompiles. A few apply
  from the next episode (`api.tuning.EPISODE`: the map and the spawns).
- **Handles**, per ship type (`api.Handles`: cruise speed, sustained turn, gun range, ...).
  A type's handles show at once: where its card and the rules its handles depend on
  (`SET_BY`) are its ship file's, they are the file's (each type solved exact); otherwise
  they are measured, one type at a time, when the panel shows the type. The card meeting a
  set of handles takes an exact solve that flies the game's code (seconds to minutes on
  the CPU), so a change goes in two steps:
  1. **at once**, `api.Estimate`: closed forms anchored on the last exact card, within a few
     percent for moderate changes, on the fast worker (tens of ms);
  2. **once the handles settle** (SETTLE seconds), the exact solve of the changed types on
     the slow worker, started from the estimate. Each of its iterations swaps a better card
     in and reports the handles that card actually flies (`flying`); the last one
     re-anchors the estimate. The scripted agent's characteristics follow each exact card
     (a worker of their own).
  A combat handle (damage, fire rate, shield, ...) of an exact card whose flight is
  measured skips both: the flight does not depend on it, so the exact card is that card
  with its combat fields in closed form, applied at once.

What each card flies (`flying`, behind the panel's off-target marker) is measured on the
measuring worker: the type shown first, then the match's types. A rule that changes what
the card flies (a flight constant) keeps the card and re-measures its handles; a type's
handles cannot be set until it is measured (`ready`). A rule set while a type is measured
restarts its measurement, and one set while the exact solve runs cancels it and solves
again under the new rules. Every change reaches the game on the main thread, in `update`,
which the shell calls each frame: the workers only compute. The workers are processes, not
threads: the solve and the estimate are thousands of small tensor operations, and on a
thread each would wait for the interpreter lock while the match runs (several times
slower). Each worker flies the compiled flight tick, compiled on its first flight (from
the on-disk cache after the first time on a machine, in seconds); the slow worker compiles
a solve's flights when the tuner starts, ahead of the first solve.

The **time scale** (`Tuner.time_scale`) makes the game k times faster at fixed map lengths.
The tuner keeps working on the design, at x1: its rule settings (`values`, `rules`), its
card (`card`), the handles' targets and what they fly (measured at x1) are the design's,
and the match plays the design k times faster (`Tuner.played`: speeds x k, rates x k,
energies x k^2, powers x k^3, the values held to whole ticks or Hz snapped). So the panel
shows and edits x1 values whatever k is, and a save writes the same files as at x1.

`Tuning` is what a match was tuned to (the rules' settings and the card, where they differ
from the game's, and the time scale's factor), as `MatchSetup.tuning`, replays and presets
hold it; `Presets` saves and loads named tunings.
"""

import dataclasses
import json
import math
import multiprocessing
import pathlib
import queue
import re
import threading
import time
from collections.abc import Callable

import platformdirs
import torch

from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.runtime.flights import compiled_flight_tick

SETTLE = 0.6  # s the handles must be still before the exact solve starts
STOP_WAIT = 0.5  # s a stopped worker has to finish before it is ended
PRESETS = pathlib.Path(platformdirs.user_config_dir("boost-and-broadside")) / "tunings"
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
TICK_HANDLES = ("shield_recharge_delay",)  # handles held to whole ticks
# Rule settings that change what a card flies: the card is kept and its handles re-measured.
MEASURED_BY = ("flight.",)
# The rule settings a card's handles depend on: a ship file's handles hold under its set's.
SET_BY = (*MEASURED_BY, "rules.bullet_life")


@dataclasses.dataclass(frozen=True)
class Tuning:
    """A match's tuned numbers, where they differ from the game's: rule settings by
    `api.tuning` key, and ship-card fields by name, one value per ship type (the form
    `TrainConfig.ship_card` and `bnb ship-card --card` read), relative to the default card of
    the match's ship set; both the design's, at x1. `time_factor`: the match plays the
    design that many times faster (`played`)."""

    rules: dict[str, float] = dataclasses.field(default_factory=dict)
    card: dict[str, tuple[float, ...]] = dataclasses.field(default_factory=dict)
    time_factor: float = 1.0

    def __bool__(self) -> bool:
        return bool(self.rules or self.card or self.time_factor != 1.0)

    def types(self, ship_set: str = api.STANDARD_SHIP_SET) -> api.ShipTypes:
        """The tuned ship card (the set's default card's other fields)."""
        values = {name: values for name, values in self.card.items()}
        return api.ship_types(ship_set=ship_set, **values)

    def played(
        self, base: api.Rules, ship_set: str = api.STANDARD_SHIP_SET
    ) -> tuple[dict[str, float], api.ShipTypes | None]:
        """The rule settings and the card a match of this tuning plays, as `Realtime` takes
        them (None: the set's default card): the design itself at x1; otherwise every
        setting and the card `time_factor` times faster (`time_scaled_settings`,
        `time_scaled_card`)."""
        types = self.types(ship_set) if self.card else None
        if self.time_factor == 1.0:
            return self.rules, types
        rules = api.tuning.tuned(base, self.rules, bounded=False)
        card = api.ship_types(ship_set=ship_set) if types is None else types
        return (
            time_scaled_settings(settings_of(rules), self.time_factor, rules.tick_length),
            time_scaled_card(card, rules, self.time_factor),
        )

    @classmethod
    def of(
        cls,
        rules: api.Rules,
        types: api.ShipTypes,
        base: api.Rules,
        ship_set: str = api.STANDARD_SHIP_SET,
        time_factor: float = 1.0,
    ) -> Tuning:
        """The tuning that makes `base` into `rules` and the set's default card into
        `types` (the design), played `time_factor` times faster."""
        return cls(
            rules=api.tuning.changed(rules, base),
            card=changed_card(types, ship_set),
            time_factor=time_factor,
        )


def settings_of(rules: api.Rules) -> dict[str, float]:
    """Every rule setting's value in these rules, by key."""
    return {s.key: api.tuning.value(rules, s.key) for s in api.tuning.settings_for(rules)}


def changed_card(
    types: api.ShipTypes, ship_set: str = api.STANDARD_SHIP_SET
) -> dict[str, tuple[float, ...]]:
    """The card's fields that differ from the set's default card's, compared in the card's
    own dtype (every field, when the type counts differ)."""
    default = vars(api.ship_types(dtype=types.max_energy.dtype, ship_set=ship_set))
    return {
        name: tuple(float(v) for v in value.tolist())
        for name, value in vars(types).items()
        if value.shape != default[name].shape or not torch.equal(value, default[name])
    }


def base_rules(config, lineup: tuple[int, ...] | None) -> api.Rules:
    """The game's rules for a match's config and lineup, untuned (what a `Tuning` is
    relative to)."""
    rules = api.rules_of(config)
    return rules if lineup is None else dataclasses.replace(rules, team_types=tuple(lineup))


def fire_rates(tick_length: float) -> list[float]:
    """The fire rates a card can hold: whole shots per second, a whole number of ticks
    apart (at 30 Hz: 1, 2, 3, 5, 6, 10, 15 and 30)."""
    tick_rate = round(1.0 / tick_length)
    return [float(rate) for rate in range(1, tick_rate + 1) if tick_rate % rate == 0]


def snap_handle(name: str, value: float, tick_length: float) -> float:
    """A handle's value as the card can hold it: the nearest of `fire_rates` for the fire
    rate, a shield recharge delay in whole ticks."""
    if name == "fire_rate":
        return min(fire_rates(tick_length), key=lambda rate: abs(math.log(rate / max(value, 1e-9))))
    if name in TICK_HANDLES:
        return max(round(value / tick_length), 0) * tick_length
    return value


def time_scaled_settings(
    values: dict[str, float], factor: float, tick_length: float
) -> dict[str, float]:
    """Rule settings in a game `factor` times faster at fixed map lengths (`similarity` with
    length 1 and time 1/factor: speeds x factor, accelerations x factor^2, rates x factor,
    times / factor), the times snapped to whole ticks."""
    scaled = api.similarity.scale_rules(values, 1.0, 1.0 / factor)
    return {
        key: api.tuning.snap(api.tuning.BY_KEY[key], number, tick_length, bounded=False)
        for key, number in scaled.items()
    }


def time_scaled_card(types: api.ShipTypes, rules: api.Rules, factor: float) -> api.ShipTypes:
    """A card in a game `factor` times faster (`similarity.scale_card`, length 1, time
    1/factor; energies x factor^2 with gravity, so a store's height, energy / g, stays). The
    shared values the card scales per type and that no setting holds (the exchanges, the
    bullet spread, the shield recharge) are scaled through the per-type scales; the fire
    rate snaps to `fire_rates` and the shield recharge delay to whole ticks."""
    scaled = api.similarity.scale_card(types, 1.0, 1.0 / factor)
    acceleration, rate = factor**2, factor
    tick = rules.tick_length
    delay = rules.shield_recharge_delay * types.shield_recharge_delay_scale / factor
    delay = torch.clamp(torch.round(delay / tick), min=0.0) * tick
    fire_rate = [
        snap_handle("fire_rate", float(value), tick) for value in scaled.fire_rate.flatten()
    ]
    return dataclasses.replace(
        scaled,
        coast_exchange_scale=types.coast_exchange_scale * acceleration,
        dive_exchange_scale=types.dive_exchange_scale * acceleration,
        climb_exchange_scale=types.climb_exchange_scale * acceleration,
        bullet_spread_scale=types.bullet_spread_scale * rate,
        shield_recharge_rate_scale=types.shield_recharge_rate_scale * rate,
        shield_recharge_delay_scale=delay / rules.shield_recharge_delay,
        fire_rate=torch.tensor(fire_rate, dtype=types.fire_rate.dtype).reshape(
            types.fire_rate.shape
        ),
    )


HANDLES, RULES, FIRST = 0, 1, 2  # the shared counters' slots; FIRST: the type to measure
CHECK_EVERY = 256  # ticks between a measurement's looks at whether it is still wanted
_STEP = None  # this worker process's compiled flight tick
_ESTIMATE: tuple | None = None  # this worker process's estimate: (its anchor's key, Estimate)


class _Compiled(Exception):
    """The warm-up's flight stops after its first tick: that call compiled it."""


class _Unwanted(Exception):
    """A measurement stops: the rules changed, or another type is wanted first."""


def columns(value, kinds):
    """The ship types `kinds` of a card or handles ((num_types,) fields), in that order."""
    index = torch.tensor(list(kinds), dtype=torch.long)
    return type(value)(**{name: field[index] for name, field in vars(value).items()})


def with_columns(value, kinds, part):
    """A card or handles with the ship types `kinds` taken from `part` (their columns, in
    that order), in `value`'s dtype."""
    index = torch.tensor(list(kinds), dtype=torch.long)
    fields = {}
    for name, field in vars(value).items():
        field = field.clone()
        field[index] = getattr(part, name).to(field.dtype)
        fields[name] = field
    return type(value)(**fields)


def joined(parts: list):
    """Cards or handles of one or more types each, side by side."""
    names = vars(parts[0])
    return type(parts[0])(**{name: torch.cat([getattr(p, name) for p in parts]) for name in names})


def _double(types: api.ShipTypes) -> api.ShipTypes:
    return api.ShipTypes(**{name: value.double() for name, value in vars(types).items()})


# ---------------------------------------------------------------- the workers' jobs
# Each runs in a worker process: (versions, report, *arguments) -> a result for `_apply`.
# `versions` are the tuner's shared counters (a job reads them to see it is stale);
# `report` sends a result before the job ends (a solve's iterations).


def _step(versions):
    """The compiled flight tick this worker flies with (compiled on its first call for each
    batch size, from the on-disk cache after the first time on a machine: seconds instead
    of about a minute)."""
    global _STEP
    if _STEP is None:
        _STEP = compiled_flight_tick()
    return _STEP


def _warm_job(versions, report, rules, card, handles):
    """Compile the solve's flights (one type's card and its perturbations) ahead of the
    first solve. `card` and `handles`: one type's. The flight stops after its first tick,
    the call that compiles."""
    step = _step(versions)

    def once(*arguments):
        step(*arguments)
        raise _Compiled

    try:
        api.solve_handles(
            handles, rules, iterations=1, strict=False, start=_double(card), tick_function=once
        )
    except _Compiled:
        pass
    return ("warmed",)


def _estimate_job(versions, report, version, kinds, key, rules, anchor, anchor_handles, targets):
    """The estimated card of the ship types `kinds` (`anchor`, `anchor_handles` and
    `targets`: their columns); `key` names the anchor, whose `Estimate` is kept while the
    key holds."""
    global _ESTIMATE
    try:
        if _ESTIMATE is None or _ESTIMATE[0] != key:
            _ESTIMATE = (key, api.Estimate(rules, anchor, anchor_handles))
        return ("estimated", version, kinds, _ESTIMATE[1].card(targets), targets)
    except api.Infeasible as error:
        return ("refused", version, str(error))


def _measure_job(versions, report, rules_version, kind, rules, card):
    """One ship type's handles (`card`: its column) under the rules; None if it stopped
    because the rules changed or another type is wanted first (`versions[FIRST]`, -1 for
    none), which it looks at every CHECK_EVERY ticks."""
    step, ticks = _step(versions), [0]

    def unwanted() -> bool:
        first = versions[FIRST]
        return versions[RULES] != rules_version or (first >= 0 and first != kind)

    def tick(*arguments):
        ticks[0] += 1
        if ticks[0] % CHECK_EVERY == 0 and unwanted():
            raise _Unwanted
        return step(*arguments)

    try:
        if unwanted():
            raise _Unwanted
        handles = api.measure_handles(_double(card), rules, tick)
    except _Unwanted:
        handles = None
    return ("measured", rules_version, kind, card, handles)


def _solve_job(versions, report, version, rules_version, kinds, targets, rules, start):
    """The exact cards of the ship types `kinds` (`targets` and `start`: their columns), one
    type at a time (the flights' batch size stays the compiled one)."""

    def stale() -> bool:
        return versions[HANDLES] != version or versions[RULES] != rules_version

    step = _step(versions)
    cards, flown_handles, converged, failures = [], [], [], []
    for index, kind in enumerate(kinds):
        target, last = columns(targets, [index]), []

        def progress(state, kind=kind) -> None:
            last[:] = [state]
            report(("progress", version, kind, state.types, state.flown))

        solution = api.solve_handles(
            target,
            rules,
            start=columns(start, [index]),
            progress=progress,
            cancel=stale,
            strict=False,
            tick_function=step,
        )
        if stale() or not last:
            return ("solved", version, rules_version, kinds, None, None, [], [])
        # The anchor's handles: the targets, with the flown ones as the solve's last flight
        # measured them (the others follow from the card in closed form).
        flown = last[0].flown.double()
        flown_handles.append(
            target.replace(**{name: flown[..., i] for i, name in enumerate(api.FLOWN_HANDLES)})
        )
        cards.append(solution.types)
        converged.append(bool(solution.converged.all()))
        failures += [failure.split(": ", 1)[-1] for failure in solution.failures]
    types, handles = joined(cards), joined(flown_handles)
    return ("solved", version, rules_version, kinds, types, handles, converged, failures)


def _characterise_job(versions, report, card, rules, types):
    return ("characterised", card, characteristics(types, rules))


def _serve(inbox, outbox, versions) -> None:
    """A worker process: run the latest job in the inbox (a job waiting behind a newer one
    is dropped), and put each result, tagged with the job's number and whether it is the
    job's last, in the outbox. None stops it."""
    while True:
        job = inbox.get()
        try:
            while job is not None:
                job = inbox.get_nowait()
        except queue.Empty:
            pass
        if job is None:
            return
        number, function, arguments = job

        def report(result, number=number) -> None:
            outbox.put((number, False, result))

        try:
            with torch.no_grad():
                result = function(versions, report, *arguments)
        except BaseException as error:  # reported to the panel; the match plays on
            result = ("error", f"{type(error).__name__}: {error}")
        outbox.put((number, True, result))


class _Worker:
    """A worker process (`_serve`) and its queues: `submit` a job, `results` collects what
    came back; `idle` once the latest job submitted has finished."""

    def __init__(self, context, name: str, versions) -> None:
        self._inbox, self._outbox = context.Queue(), context.Queue()
        self._process = context.Process(
            target=_serve, args=(self._inbox, self._outbox, versions), name=name, daemon=True
        )
        self._process.start()
        self._submitted = self._finished = 0

    @property
    def idle(self) -> bool:
        return self._finished == self._submitted

    def submit(self, function: Callable, *arguments) -> None:
        self._submitted += 1
        self._inbox.put((self._submitted, function, arguments))

    def results(self) -> list:
        found = []
        try:
            while True:
                number, last, result = self._outbox.get_nowait()
                if last:
                    self._finished = max(self._finished, number)
                found.append(result)
        except queue.Empty:
            return found

    def stop(self) -> None:
        """End the process (what it is computing is dropped) without waiting for it: a
        thread asks it to stop, then ends it if it is still busy after STOP_WAIT seconds.
        Jobs it never read are dropped too: the queues' feeder threads are not joined, so a
        full pipe to a process that is gone can never hold up this process (at a new match
        or at exit)."""
        self._inbox.cancel_join_thread()
        self._outbox.cancel_join_thread()
        self._inbox.put(None)
        process = self._process

        def end() -> None:
            process.join(timeout=STOP_WAIT)
            if process.is_alive():
                process.terminate()
                process.join()

        threading.Thread(target=end, name="tuning-stop", daemon=True).start()


def file_handles(files: list, count: int) -> tuple[api.Handles, list[bool]]:
    """The handles of `count` ship types as their ship files hold them, and which types have
    a file with every handle (a type past the files takes the first file's, as a stand-in)."""
    rows = [files[k].handles if k < len(files) else {} for k in range(count)]
    complete = [all(name in row for name in api.HANDLE_NAMES) for row in rows]
    first = files[0].handles
    handles = api.Handles(
        **{
            name: torch.tensor(
                [float(row.get(name, first.get(name, math.nan))) for row in rows],
                dtype=torch.float64,
            )
            for name in api.HANDLE_NAMES
        }
    )
    return handles, complete


class Tuner:
    """Live tuning of a `MatchLoop` (see the module), on the design at x1: `rules` and
    `card` are the design's, and the match plays them `time_factor` times faster
    (`played`). The panel reads `values` (each rule setting's design value), `targets`
    (each handle's set value, (num_types,) per name; a type's are meaningful once
    `ready(type)`), `flying` (each handle's value each type's applied card actually flies
    at x1, by name then type: None until a measurement or a solve iteration has reported
    it), `exact` (every applied card is the exact solve of its targets), `status` and
    `message` (the last refusal or error); it calls `show` (the type it shows, measured
    first), `set_setting`, `set_handle`, `measure_all` and `update`."""

    def __init__(self, loop, base: api.Rules) -> None:
        self.loop = loop
        self.ship_set = loop.setup.config.ship_set
        self.base = dataclasses.replace(base, bullet_slots=loop.rules.bullet_slots)
        tuning = loop.setup.tuning
        self.time_factor = tuning.time_factor
        if self.time_factor == 1.0:
            self.rules, self.card = loop.rules, loop.types
        else:
            self.rules = api.tuning.tuned(self.base, tuning.rules, bounded=False)
            self.card = tuning.types(self.ship_set)
        self.values = settings_of(self.rules)
        self.status = ""
        self.message = ""
        self.shown: dict[str, list[float]] = {}
        self.focus = 0  # the ship type the panel shows
        self._context = context = multiprocessing.get_context("spawn")
        self._versions = context.Array("q", 3, lock=False)
        self._fast = _Worker(context, "tuning-estimate", self._versions)
        self._slow = _Worker(context, "tuning-solve", self._versions)
        self._measurer = _Worker(context, "tuning-measure", self._versions)
        self._traits: _Worker | None = None  # the scripted agent's, started when one plays
        self._card = 0  # counts the cards the match has played, for the characteristics
        self._handles_changed: float | None = None
        self._solving: int | None = None  # the handles version a solve is running for
        self._measuring: int | None = None  # the ship type being measured
        self._wanted: set[int] = set()  # types to measure besides the shown one and the match's
        self._anchor_version = 0  # names the anchor (and the rules) an estimate is made with
        self._reset(characterise=False)
        self._slow.submit(  # dropped if a solve comes first
            _warm_job, self.rules, columns(self.card, [0]), columns(self.targets, [0])
        )
        self._show()

    # The version counters, shared with the workers: a job for an older one is stale.

    @property
    def _handles_version(self) -> int:
        return self._versions[HANDLES]

    @_handles_version.setter
    def _handles_version(self, value: int) -> None:
        self._versions[HANDLES] = value

    @property
    def _rules_version(self) -> int:
        return self._versions[RULES]

    @_rules_version.setter
    def _rules_version(self, value: int) -> None:
        self._versions[RULES] = value

    def _retune(self, rules=None, types=None) -> None:
        """Take new design rules and/or a new design card; the match plays them from the
        next tick (`played`; under a time scale both, as the card's scaling reads the
        rules)."""
        if rules is not None:
            self.rules = rules
        if types is not None:
            self.card = types
        if self.time_factor == 1.0:
            self.loop.retune(rules=rules, types=types)
        else:
            self.loop.retune(*self.played())
        if types is not None:
            self._card += 1
        self._show()

    def played(self) -> tuple[api.Rules, api.ShipTypes]:
        """The rules and the card the match plays: the design `time_factor` times faster
        (`time_scaled_settings`, `time_scaled_card`)."""
        if self.time_factor == 1.0:
            return self.rules, self.card
        values = time_scaled_settings(self.values, self.time_factor, self.base.tick_length)
        rules = api.tuning.tuned(self.base, values, bounded=False)
        return rules, time_scaled_card(self.card, self.rules, self.time_factor)

    def _show(self) -> None:
        """The values shown beside the handles for the design's card: its bleed rate
        (closed form, within 1% of the flown one)."""
        card = _double(self.card)
        self.shown = {"bleed_rate": api.bleed_rate(self.rules, card)[0].tolist()}

    @property
    def _count(self) -> int:
        return int(self.card.max_energy.shape[-1])

    def _reset(self, characterise: bool = True) -> None:
        """Start over from the design's card under its current rules: the types whose
        card and handle-setting rules are the ship files' take their handles from the files
        (each solved exact, so nothing is measured to show them); the others are measured
        when shown. Nothing flown is known yet."""
        files = api.ships.load(self.ship_set)
        count = self._count
        self.targets, complete = file_handles(files, count)
        rules = self.rules
        same_rules = all(
            api.tuning.value(rules, s.key) == api.tuning.value(self.base, s.key)
            for s in api.tuning.settings_for(rules)
            if s.key.startswith(SET_BY)
        )
        default = api.ship_types(ship_set=self.ship_set)
        same_card = [
            k < default.max_energy.shape[-1]
            and all(
                torch.equal(value[k], getattr(default, name)[k].to(value.dtype))
                for name, value in vars(self.card).items()
            )
            for k in range(count)
        ]
        self._known = [same_rules and complete[k] and same_card[k] for k in range(count)]
        self._applied = self.targets  # the targets each applied card was made for
        self._anchor = (self.card, self.targets)  # exact cards and their handles
        self._anchor_version += 1
        self._flown: list[dict[str, float] | None] = [None] * count  # each card's flight
        self.flying = {name: [None] * count for name in api.HANDLE_NAMES}
        self._estimated: set[int] = set()  # the types whose card is an estimate
        self._wanted = set()
        self._unmeasurable: set[int] = set()  # types whose measurement failed
        self.exact = True
        self._solving = None
        self._handles_changed = None
        self._handles_version += 1
        if characterise:
            self._characterise()

    # ---------------------------------------------------------------- reading

    def ready(self, ship_type: int | None = None) -> bool:
        """Whether a type's handles (every type's, by default) are known under the current
        rules, so they show and can be set."""
        if ship_type is None:
            return all(self._known)
        return 0 <= ship_type < len(self._known) and self._known[ship_type]

    @property
    def tuning(self) -> Tuning:
        """What the match is tuned to now: the design and the time scale's factor."""
        return Tuning.of(self.rules, self.card, self.base, self.ship_set, self.time_factor)

    def default(self, key: str) -> float:
        """A rule setting's value in the game's rules for this match."""
        return api.tuning.value(self.base, key)

    def name(self, ship_type: int) -> str:
        """A ship type's name (its file's)."""
        names = api.type_names(self.ship_set)
        return names[ship_type] if ship_type < len(names) else f"type {ship_type}"

    # ---------------------------------------------------------------- what is measured

    def show(self, ship_type: int) -> None:
        """The type the panel shows: it is measured first."""
        self.focus = ship_type

    def measure_all(self) -> None:
        """Measure every type whose handles are not known (before a save)."""
        self._wanted |= {k for k in range(self._count) if not self._known[k]}

    def _next_measurement(self) -> int | None:
        """The type to measure next: the shown one, the ones asked for, then the match's;
        each once its flight is unknown and its card is not being solved."""
        team_types = sorted(set(self.loop.rules.team_types))
        for kind in [self.focus, *sorted(self._wanted), *team_types]:
            if (
                0 <= kind < self._count
                and self._flown[kind] is None
                and kind not in self._estimated
                and kind not in self._unmeasurable
            ):
                return kind
        return None

    # ---------------------------------------------------------------- rule settings

    def set_setting(self, key: str, value: float) -> None:
        """Set a rule setting's design value (x1); it applies on the next tick (or the next
        episode), time scaled."""
        setting = api.tuning.BY_KEY[key]
        value = api.tuning.snap(setting, value, self.base.tick_length)
        if value == self.values.get(key):
            return
        values = dict(self.values, **{key: value})
        rules = api.tuning.tuned(self.base, values, bounded=False)  # only `key` is new
        self.values = values
        self.message = ""
        self._retune(rules=rules)
        self._rules_version += 1  # what the workers were computing under the old rules is stale
        self._anchor_version += 1
        if key.startswith(MEASURED_BY):
            self._reset()  # each type re-measured when shown
        elif key == "rules.bullet_life":
            self._rerange()
        if self._solving is not None or not self.exact:
            self._handles_changed = time.monotonic()  # the cancelled solve runs again

    def _rerange(self) -> None:
        """A new bullet life: the card is kept, so its gun range (a closed form of the
        bullet's drag, speed and life) changes; the estimate is re-anchored on it."""
        rules = self.rules
        gun_range = api.gun_range(rules, self.card, self.targets.cruise_speed)
        self.targets = self.targets.replace(gun_range=gun_range)
        self._applied = self._applied.replace(gun_range=gun_range)
        anchor, handles = self._anchor
        anchored = api.gun_range(rules, anchor, handles.cruise_speed)
        self._anchor = (anchor, handles.replace(gun_range=anchored))
        self.flying["gun_range"] = [
            None if value is None else float(gun_range[kind])
            for kind, value in enumerate(self.flying["gun_range"])
        ]
        self._handles_version += 1

    def load(self, tuning: Tuning) -> None:
        """Play on with a whole tuning (a preset, or the game's numbers): its rules, card and
        time scale from the next tick, the handles re-measured where they are not the
        files'."""
        rules = api.tuning.tuned(self.base, tuning.rules, bounded=False)
        self.values = settings_of(rules)
        self.time_factor = tuning.time_factor
        self._retune(rules=rules, types=tuning.types(self.ship_set))
        self.message = ""
        self._rules_version += 1
        self._reset()

    def time_scale(self, factor: float) -> str:
        """Make the whole game `factor` times faster than now with the map's lengths fixed:
        the match plays the design `time_factor` (now that many times larger) times faster
        (`played`: every rule setting and the card by their units, energies with gravity;
        an exact similarity up to the tick and the values held to whole ticks or Hz, which
        snap). The design, its handles and their measurements stay. Returns what it did,
        naming the values that snapped."""
        if factor <= 0.0 or factor == 1.0:
            return ""
        total = self.time_factor * factor
        self.time_factor = 1.0 if math.isclose(total, 1.0, rel_tol=1e-9) else total
        rules, types = self.played()
        self.loop.retune(rules=rules, types=types)
        self._card += 1
        self._characterise()
        self.message = ""
        exact = api.similarity.scale_rules(self.values, 1.0, 1.0 / self.time_factor)
        snapped = [
            f"{api.tuning.BY_KEY[key].label} {exact[key]:.4g} -> {self._played_value(key):.4g} s"
            for key in exact
            if abs(self._played_value(key) - exact[key]) > 1e-9 * max(1.0, abs(exact[key]))
        ]
        wanted = self.card.fire_rate * self.time_factor
        snapped += [
            f"fire rate {float(a):.4g} -> {float(b):.4g} Hz"
            for a, b in zip(wanted.flatten(), types.fire_rate.flatten())
            if abs(float(a) - float(b)) > 1e-6 * float(a)
        ]
        done = f"the game runs x{self.time_factor:g} (values stay at x1)"
        return done + ("; snapped: " + ", ".join(snapped) if snapped else "")

    def _played_value(self, key: str) -> float:
        """A rule setting's value as the match plays it (time scaled)."""
        return api.tuning.value(self.loop.rules, key)

    def close(self) -> None:
        """Stop the workers (a match that ends): what they are computing is dropped."""
        self._rules_version += 1
        self._handles_version += 1
        for worker in (self._fast, self._slow, self._measurer, self._traits):
            if worker is not None:
                worker.stop()

    # ---------------------------------------------------------------- handles

    def set_handle(self, name: str, ship_type: int, value: float) -> None:
        """Set one ship type's handle: a combat handle of an exact card whose flight is
        measured applies exactly at once (the flight does not change); otherwise the
        estimate applies within a frame or two, the exact card once the handles settle."""
        if not self.ready(ship_type):
            return
        value = snap_handle(name, value, self.base.tick_length)
        current = getattr(self.targets, name)
        if float(current[ship_type]) == value:
            return
        changed = current.clone()
        changed[ship_type] = value
        self.targets = self.targets.replace(**{name: changed})
        self.message = ""
        self._handles_version += 1
        if self._combat_only(ship_type):
            self._apply_combat(ship_type)
            changed = sorted(self._estimated)  # the others' estimates, made stale, run again
        else:
            changed = sorted(self._estimated | {ship_type})
        if changed:
            self.exact = False
            self._handles_changed = time.monotonic()
            self._estimate(changed)

    def _estimate(self, kinds: list[int]) -> None:
        anchor, handles = self._anchor
        self._fast.submit(
            _estimate_job,
            self._handles_version,
            kinds,
            (self._anchor_version, tuple(kinds)),
            self.rules,
            columns(anchor, kinds),
            columns(handles, kinds),
            columns(self.targets, kinds),
        )

    def _combat_only(self, kind: int) -> bool:
        """Whether a type's new targets differ from its exact card's only in combat handles
        and that card's measured flight meets them (within the solver's acceptance): its
        exact card is then that card with the combat fields in closed form."""
        flown = self._flown[kind]
        if flown is None or kind in self._estimated:
            return False
        if any(
            float(getattr(self.targets, name)[kind]) != float(getattr(self._applied, name)[kind])
            for name in api.FLIGHT_ONLY_HANDLES
        ):
            return False
        target = columns(self.targets, [kind])
        wanted = torch.tensor([float(getattr(target, n)[0]) for n in api.FLOWN_HANDLES])
        scale = torch.maximum(wanted.abs(), api.solve_error_floor(self.rules, target)[0].float())
        got = torch.tensor([flown[n] for n in api.FLOWN_HANDLES])
        return bool(((got - wanted).abs() / scale < api.SOLVE_ACCEPTANCE).all())

    def _apply_combat(self, kind: int) -> None:
        """Apply a type's exact card for a combat-only change (`_combat_only`), or set the
        targets back when no card can meet them."""
        target = columns(self.targets, [kind])
        try:
            combat = api.combat_card(self.rules, target, target.cruise_speed)
        except api.Infeasible as error:
            self.message = str(error)
            self.targets = self._applied
            return
        card = columns(self.card, [kind])
        card = dataclasses.replace(
            card, **{k: v.to(card.max_energy.dtype) for k, v in combat.items()}
        )
        types = with_columns(self.card, [kind], card)
        self._retune(types=types)
        self._applied = with_columns(self._applied, [kind], target)
        anchor, handles = self._anchor
        self._anchor = (
            with_columns(anchor, [kind], card),
            with_columns(
                handles,
                [kind],
                target.replace(
                    **{
                        n: torch.tensor([self._flown[kind][n]], dtype=torch.float64)
                        for n in api.FLOWN_HANDLES
                    }
                ),
            ),
        )
        self._anchor_version += 1
        self._set_flying(kind, target, self._flown[kind])
        self._characterise()

    def _set_flying(self, kind: int, handles: api.Handles | None, flown: dict | None) -> None:
        """A type's flown handles: `flown`'s (the measured ones), the others (closed forms)
        as `handles` (one type's) hold them; None for unknown."""
        for name in api.HANDLE_NAMES:
            if handles is None:
                value = None
            elif flown is not None and name in flown:
                value = flown[name]
            else:
                value = float(getattr(handles, name)[0])
            self.flying[name][kind] = value

    # ---------------------------------------------------------------- the frame

    def update(self) -> None:
        """Apply what the workers finished and start what is due (each frame, on the main
        thread)."""
        for result in self._measurer.results():
            if result[0] != "error":
                self._apply(result)
            elif self._measuring is not None:  # not again, until the rules or the card change
                self.message = f"measuring {self.name(self._measuring)}: {result[1]}"
                self._unmeasurable.add(self._measuring)
        found = self._fast.results() + self._slow.results()
        if self._traits is not None:
            found += self._traits.results()
        for result in found:
            self._apply(result)
        now = time.monotonic()
        if (
            self._handles_changed is not None
            and now - self._handles_changed > SETTLE
            and self._fast.idle
            and self._estimated
            and self._solving != self._handles_version
        ):
            self._handles_changed = None
            self._solve()
        focus = self.focus
        first = (
            0 <= focus < self._count and self._flown[focus] is None and focus not in self._estimated
        )
        self._versions[FIRST] = focus if first and focus not in self._unmeasurable else -1
        if self._measurer.idle:
            self._measuring = self._next_measurement()
            if self._measuring is not None:
                kind = self._measuring
                card = columns(self.card, [kind])
                self._measurer.submit(_measure_job, self._rules_version, kind, self.rules, card)
        self.status = self._status()

    def _status(self) -> str:
        parts = []
        if not self.ready(self.focus):
            parts.append(f"measuring {self.name(self.focus)}'s handles")
        if self._solving is not None:
            parts.append("solving the card exactly")
        elif not self.exact:
            parts.append("estimated card")
        else:
            parts.append("exact")
        if (
            self._measuring is not None
            and self.ready(self.focus)
            or (self._measuring is not None and self._measuring != self.focus)
        ):
            parts.append(f"measuring {self.name(self._measuring)}'s flight")
        return "; ".join(parts)

    def _apply(self, result) -> None:
        kind = result[0]
        if kind == "error":
            self.message = result[1]
            self._solving = None
            return
        if kind == "measured":
            _, rules_version, ship_type, card, handles = result
            if (
                handles is None
                or rules_version != self._rules_version
                or ship_type >= self._count
                or ship_type in self._estimated
                or not all(
                    torch.equal(value[ship_type : ship_type + 1], getattr(card, name))
                    for name, value in vars(self.card).items()
                )
            ):
                return  # measured under other rules, or of a card since changed
            flown = {name: float(getattr(handles, name)[0]) for name in api.FLOWN_HANDLES}
            if not self._known[ship_type]:  # its handles are what it flies
                self.targets = with_columns(self.targets, [ship_type], handles)
                self._applied = with_columns(self._applied, [ship_type], handles)
                self._known[ship_type] = True
            anchor, anchor_handles = self._anchor
            self._anchor = (
                with_columns(anchor, [ship_type], card),
                with_columns(anchor_handles, [ship_type], handles),
            )
            self._anchor_version += 1
            self._flown[ship_type] = flown
            self._set_flying(ship_type, handles, None)
            self._wanted.discard(ship_type)
            return
        if kind == "warmed":
            return
        if kind == "characterised":
            _, card, traits = result
            if card == self._card:  # still the card the match plays
                for side in self.loop.sides:
                    if hasattr(side, "retune"):
                        side.retune(traits)
            return
        if kind == "estimated":
            _, version, kinds, types, targets = result
            if version == self._handles_version:
                self._retune(types=with_columns(self.card, kinds, types))
                self._applied = with_columns(self._applied, kinds, targets)
                self._estimated |= set(kinds)
                for ship_type in kinds:
                    self._flown[ship_type] = None
                    self._set_flying(ship_type, None, None)
            return
        if kind == "refused":
            _, version, reason = result
            if version == self._handles_version:
                self.message = reason
                self.targets = self._applied  # back to what the applied card was made for
                self._handles_version += 1
                self.exact = not self._estimated
            return
        if kind == "progress":
            _, version, ship_type, types, flown = result
            if version == self._handles_version:
                self._retune(types=with_columns(self.card, [ship_type], types))
                values = {n: float(flown[..., i][0]) for i, n in enumerate(api.FLOWN_HANDLES)}
                self._set_flying(ship_type, columns(self.targets, [ship_type]), values)
            return
        if kind == "solved":
            _, version, rules_version, kinds, types, handles, converged, failures = result
            if self._solving == version:
                self._solving = None
            if (
                types is None
                or version != self._handles_version
                or rules_version != self._rules_version
            ):
                return
            self._retune(types=with_columns(self.card, kinds, types))
            anchor, anchor_handles = self._anchor
            self._anchor = (
                with_columns(anchor, kinds, types),
                with_columns(anchor_handles, kinds, handles),
            )
            self._anchor_version += 1
            for index, ship_type in enumerate(kinds):
                one = columns(handles, [index])
                flown = {n: float(getattr(one, n)[0]) for n in api.FLOWN_HANDLES}
                self._flown[ship_type] = flown
                self._set_flying(ship_type, columns(self.targets, [ship_type]), flown)
                if converged[index]:
                    self._estimated.discard(ship_type)
            self.exact = not self._estimated
            self.message = "; ".join(failures)
            self._characterise()

    def _characterise(self) -> None:
        """Re-measure the scripted agent's characteristics for the card the match plays
        (time scaled), when a side uses them (on a worker of their own, so the next solve
        does not wait for them)."""
        if not any(hasattr(side, "retune") for side in self.loop.sides):
            return
        if self._traits is None:
            self._traits = _Worker(self._context, "tuning-traits", self._versions)
        self._traits.submit(_characterise_job, self._card, self.loop.rules, self.loop.types)

    # ---------------------------------------------------------------- the slow worker

    def _solve(self) -> None:
        """The exact cards of the types whose card is an estimate, from the applied cards
        (slow worker)."""
        kinds = sorted(self._estimated)
        self._solving = self._handles_version
        self._slow.submit(
            _solve_job,
            self._handles_version,
            self._rules_version,
            kinds,
            columns(self.targets, kinds),
            self.rules,
            columns(self.card, kinds),
        )


class Presets:
    """Named tunings saved as JSON under `directory` (the user's config by default): the
    rules' settings and the full ship card, by field."""

    def __init__(self, directory: str | pathlib.Path = PRESETS) -> None:
        self.directory = pathlib.Path(directory)

    def names(self) -> list[str]:
        """The saved presets' names, sorted."""
        if not self.directory.is_dir():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json"))

    def path(self, name: str) -> pathlib.Path:
        """Where the preset of this name lives; fails for a name that is not lowercase
        words joined by hyphens."""
        if NAME.fullmatch(name) is None:
            raise ValueError(f"preset name {name!r}: use lowercase words joined by hyphens")
        return self.directory / f"{name}.json"

    def save(
        self,
        name: str,
        tuning: Tuning,
        types: api.ShipTypes,
        ship_set: str = api.STANDARD_SHIP_SET,
    ) -> pathlib.Path:
        """Write a preset: its ship set, the tuning's rule settings and every field of the
        design's card `types` (so the file is a whole `ship_card` for training), at x1, and
        the time scale's factor when it is not 1."""
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        card = {name: [float(v) for v in value.tolist()] for name, value in vars(types).items()}
        data = {"ship_set": ship_set, "rules": tuning.rules, "ship_card": card}
        if tuning.time_factor != 1.0:
            data["time_factor"] = tuning.time_factor
        path.write_text(json.dumps(data, indent=2))
        return path

    def load(self, name: str, ship_set: str = api.STANDARD_SHIP_SET) -> Tuning:
        """The tuning a preset holds (its card reduced to the fields that differ from the
        set's default); fails naming any entry it does not know, and for a preset of another
        ship set (a preset without one is the standard set's)."""
        data = json.loads(self.path(name).read_text())
        stored_set = data.get("ship_set", api.STANDARD_SHIP_SET)
        if stored_set != ship_set:
            raise ValueError(f"preset {name} tunes the {stored_set} ships, not the {ship_set} ones")
        stored = data.get("ship_card", {})
        unknown = sorted(set(stored) - set(api.default_card(ship_set))) + sorted(
            set(data.get("rules", {})) - set(api.tuning.BY_KEY)
        )
        if unknown:
            raise ValueError(f"preset {name}: unknown entries {unknown}")
        # The card as the match holds it (float32), with only the fields that differ kept.
        values = {field: tuple(values) for field, values in stored.items()}
        types = api.ship_types(ship_set=ship_set, **values)
        card = changed_card(types, ship_set)
        return Tuning(
            rules={k: float(v) for k, v in data.get("rules", {}).items()},
            card=card,
            time_factor=float(data.get("time_factor", 1.0)),
        )
