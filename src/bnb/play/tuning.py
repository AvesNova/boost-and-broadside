"""Live tuning: change the game's numbers while a match plays, and see the effect on the
next tick.

Two kinds of number, both applied through `MatchLoop.retune`:

- **Rule settings** (`api.tuning`: the arena and clock, bullet life, the shared flight
  constants, spawns, each mode's rules). They apply on the next tick: the compiled tick
  takes their values as tensors (`Realtime.retune`), so nothing recompiles. A few apply
  from the next episode (`api.tuning.EPISODE`: the map and the spawns).
- **Handles**, per ship type (`api.Handles`: cruise speed, sustained turn, gun range, ...).
  The card meeting a set of handles takes an exact solve that flies the game's code (tens of
  seconds on the CPU), so a change goes in two steps:
  1. **at once**, `api.Estimate`: closed forms anchored on the last exact card, within a few
     percent for moderate changes, on the fast worker (tens of ms);
  2. **once the handles settle** (SETTLE seconds), the exact solve on the slow worker,
     started from the estimate. Each of its iterations swaps a better card in and reports
     the handles that card actually flies (`flying`); the last one re-anchors the estimate.
     The scripted agent's characteristics follow each exact card (a third worker).

A rule that changes what the card flies (the flight constants, the bullet life) keeps the
card and re-measures its handles on the slow worker; until that finishes, handles cannot be
set (`measuring`). Any rule set while the handles are measured restarts the measurement, and
one set while the exact solve runs cancels it and solves again under the new rules. Every
change reaches the game on the main thread, in `update`, which the shell calls each frame:
the workers only compute. The workers are processes, not threads:
the solve and the estimate are thousands of small tensor operations, and on a thread each
would wait for the interpreter lock while the match runs (several times slower).

`Tuning` is what a match was tuned to (the rules' settings and the card, where they differ
from the game's), as `MatchSetup.tuning`, replays and presets hold it; `Presets` saves and
loads named tunings.
"""

import dataclasses
import json
import math
import multiprocessing
import pathlib
import queue
import re
import time
from collections.abc import Callable

import platformdirs
import torch

from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.runtime.flights import compiled_flight_tick

SETTLE = 0.6  # s the handles must be still before the exact solve starts
PRESETS = pathlib.Path(platformdirs.user_config_dir("boost-and-broadside")) / "tunings"
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
TICK_HANDLES = ("shield_recharge_delay",)  # handles held to whole ticks
# Rule settings that change what a card flies: the card is kept and its handles re-measured.
MEASURED_BY = ("flight.",)


@dataclasses.dataclass(frozen=True)
class Tuning:
    """A match's tuned numbers, where they differ from the game's: rule settings by
    `api.tuning` key, and ship-card fields by name, one value per ship type (the form
    `TrainConfig.ship_card` and `bnb ship-card --card` read), relative to the default card of
    the match's ship set."""

    rules: dict[str, float] = dataclasses.field(default_factory=dict)
    card: dict[str, tuple[float, ...]] = dataclasses.field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.rules or self.card)

    def types(self, ship_set: str = api.STANDARD_SHIP_SET) -> api.ShipTypes:
        """The tuned ship card (the set's default card's other fields)."""
        values = {name: values for name, values in self.card.items()}
        return api.ship_types(ship_set=ship_set, **values)

    @classmethod
    def of(
        cls,
        rules: api.Rules,
        types: api.ShipTypes,
        base: api.Rules,
        ship_set: str = api.STANDARD_SHIP_SET,
    ) -> Tuning:
        """The tuning that makes `base` into `rules` and the set's default card into
        `types`."""
        return cls(rules=api.tuning.changed(rules, base), card=changed_card(types, ship_set))


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


HANDLES, RULES, WARM = 0, 1, 2  # the shared counters' slots; WARM: the flights compiled
_STEP = None  # this worker process's compiled flight tick, once WARM is set


@dataclasses.dataclass
class _Anchor:
    """An exact card under the current rules, its handles, and the estimate anchored on
    them."""

    types: api.ShipTypes
    handles: api.Handles
    estimate: api.Estimate


# ---------------------------------------------------------------- the workers' jobs
# Each runs in a worker process: (versions, report, *arguments) -> a result for `_apply`.
# `versions` are the tuner's shared counters (a job reads them to see it is stale);
# `report` sends a result before the job ends (a solve's iterations).


def _step(versions):
    """The flight tick a worker flies with: compiled once the warm-up has filled the
    compile cache (so compiling here is quick), the game's own until then."""
    global _STEP
    if not versions[WARM]:
        return None
    if _STEP is None:
        _STEP = compiled_flight_tick()
    return _STEP


def _warm_job(versions, report, rules, types):
    """Compile the flights for both batch sizes the tuner flies (a measurement, and a solve
    iteration's card and perturbations) into the on-disk cache, then set WARM."""
    step = compiled_flight_tick()
    card = api.ShipTypes(**{n: v.double() for n, v in vars(types).items()})
    handles = api.measure_handles(card, rules, step)
    api.solve_handles(handles, rules, iterations=1, strict=False, start=types, tick_function=step)
    versions[WARM] = 1
    return ("warmed",)


def _estimate(versions, report, version, estimate, targets):
    try:
        return ("estimated", version, estimate.card(targets), targets)
    except api.Infeasible as error:
        return ("refused", version, str(error))


def _measure_job(versions, report, version, rules, types):
    card = api.ShipTypes(**{n: v.double() for n, v in vars(types).items()})
    handles = api.measure_handles(card, rules, _step(versions))
    anchor = _Anchor(types, handles, api.Estimate(rules, types, handles))
    flying = {name: getattr(handles, name).tolist() for name in api.HANDLE_NAMES}
    return ("measured", version, anchor, flying)


def _solve_job(versions, report, version, rules_version, targets, rules, start):
    def stale() -> bool:
        return versions[HANDLES] != version or versions[RULES] != rules_version

    last = []

    def progress(state) -> None:
        last[:] = [state]
        report(("progress", version, state.types, state.flown))

    solution = api.solve_handles(
        targets,
        rules,
        start=start,
        progress=progress,
        cancel=stale,
        strict=False,
        tick_function=_step(versions),
    )
    if stale() or not last:
        return ("solved", version, rules_version, None, False, "")
    # The anchor's handles: the targets, with the flown ones as the solve's last flight
    # measured them (the others follow from the card in closed form).
    flown = last[0].flown.double()
    handles = targets.replace(**{name: flown[..., i] for i, name in enumerate(api.FLOWN_HANDLES)})
    anchor = _Anchor(solution.types, handles, api.Estimate(rules, solution.types, handles))
    converged = bool(solution.converged.all())
    return ("solved", version, rules_version, anchor, converged, "; ".join(solution.failures))


def _anchor_job(versions, report, version, rules, types, handles):
    return ("anchored", version, _Anchor(types, handles, api.Estimate(rules, types, handles)))


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
        """End the process (what it is computing is dropped)."""
        self._inbox.put(None)
        self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.terminate()


class Tuner:
    """Live tuning of a `MatchLoop` (see the module). The panel reads `values` (each rule
    setting's current value), `targets` (each handle's set value, (num_types,) per name;
    None while measuring), `flying` (each handle's value the applied card actually flies,
    or None until a measurement or a solve iteration has reported it), `exact` (the applied
    card is the exact solve of the targets), `status` and `message` (the last refusal or
    error), and calls `set_setting`, `set_handle` and `update`."""

    def __init__(self, loop, base: api.Rules) -> None:
        self.loop = loop
        self.ship_set = loop.setup.config.ship_set
        self.base = dataclasses.replace(base, bullet_slots=loop.rules.bullet_slots)
        rules = loop.rules
        self.values = {
            s.key: api.tuning.value(rules, s.key) for s in api.tuning.settings_for(rules)
        }
        self.targets: api.Handles | None = None
        self.flying: dict[str, list[float]] | None = None
        self.exact = False
        self.status = ""
        self.message = ""
        self.shown: dict[str, list[float]] = {}
        self._anchor: _Anchor | None = None
        self._context = context = multiprocessing.get_context("spawn")
        self._versions = context.Array("q", 3, lock=False)
        self._fast = _Worker(context, "tuning-estimate", self._versions)
        self._slow = _Worker(context, "tuning-solve", self._versions)
        self._traits: _Worker | None = None  # the scripted agent's, started when one plays
        self._warm: _Worker | None = _Worker(context, "tuning-warm", self._versions)
        self._warm.submit(_warm_job, loop.rules, loop.types)
        self._card = 0  # counts the cards the match has played, for the characteristics
        self._handles_changed: float | None = None
        self._solving: int | None = None  # the handles version a solve is running for
        self._applied: api.Handles | None = None  # the targets the applied card was made for
        self._show()
        self._measure()

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
        self.loop.retune(rules=rules, types=types)
        if types is not None:
            self._card += 1
        self._show()

    def _show(self) -> None:
        """The values shown beside the handles for the card the match plays: its bleed rate
        (closed form, within 1% of the flown one)."""
        card = api.ShipTypes(**{n: v.double() for n, v in vars(self.loop.types).items()})
        self.shown = {"bleed_rate": api.bleed_rate(self.loop.rules, card)[0].tolist()}

    # ---------------------------------------------------------------- reading

    @property
    def measuring(self) -> bool:
        """The handles are being measured under new rules (they cannot be set meanwhile)."""
        return self.targets is None

    @property
    def tuning(self) -> Tuning:
        """What the match is tuned to now."""
        return Tuning.of(self.loop.rules, self.loop.types, self.base, self.ship_set)

    def default(self, key: str) -> float:
        """A rule setting's value in the game's rules for this match."""
        return api.tuning.value(self.base, key)

    # ---------------------------------------------------------------- rule settings

    def set_setting(self, key: str, value: float) -> None:
        """Set a rule setting; it applies on the next tick (or the next episode)."""
        setting = api.tuning.BY_KEY[key]
        value = api.tuning.snap(setting, value, self.base.tick_length)
        if value == self.values.get(key):
            return
        values = dict(self.values, **{key: value})
        rules = api.tuning.tuned(self.base, values)
        self.values = values
        self.message = ""
        self._retune(rules=rules)
        self._rules_version += 1  # what the workers were computing under the old rules is stale
        if key.startswith(MEASURED_BY) or self.targets is None:
            self._measure()  # a measurement under way restarts under the new rules
        elif key == "rules.bullet_life":
            self._rerange()
        if self.targets is not None and (self._solving is not None or not self.exact):
            self._handles_changed = time.monotonic()  # the cancelled solve runs again

    def _rerange(self) -> None:
        """A new bullet life: the card is kept, so its gun range (a closed form of the
        bullet's drag, speed and life) changes; the estimate is re-anchored on it."""
        gun_range = api.gun_range(self.loop.rules, self.loop.types, self.targets.cruise_speed)
        self.targets = self._applied = self.targets.replace(gun_range=gun_range)
        if self.flying is not None:
            self.flying = dict(self.flying, gun_range=gun_range.tolist())
        self._handles_version += 1
        self._fast.submit(
            _anchor_job, self._rules_version, self.loop.rules, self.loop.types, self.targets
        )

    def load(self, tuning: Tuning) -> None:
        """Play on with a whole tuning (a preset, or the game's numbers): its rules and card
        from the next tick, the handles re-measured."""
        rules = api.tuning.tuned(self.base, tuning.rules)
        self.values = {
            s.key: api.tuning.value(rules, s.key) for s in api.tuning.settings_for(rules)
        }
        self._retune(rules=rules, types=tuning.types(self.ship_set))
        self.message = ""
        self._rules_version += 1
        self._measure()

    def close(self) -> None:
        """Stop the workers (a match that ends): what they are computing is dropped."""
        self._rules_version += 1
        self._handles_version += 1
        for worker in (self._fast, self._slow, self._traits, self._warm):
            if worker is not None:
                worker.stop()

    # ---------------------------------------------------------------- handles

    def set_handle(self, name: str, ship_type: int, value: float) -> None:
        """Set one ship type's handle; the estimate applies within a frame or two, the exact
        card once the handles settle."""
        if self.targets is None:
            return
        value = snap_handle(name, value, self.base.tick_length)
        current = getattr(self.targets, name)
        if float(current[ship_type]) == value:
            return
        changed = current.clone()
        changed[ship_type] = value
        self.targets = self.targets.replace(**{name: changed})
        self.message = ""
        self.exact = False
        self._handles_version += 1
        self._handles_changed = time.monotonic()
        self._fast.submit(_estimate, self._handles_version, self._anchor.estimate, self.targets)

    # ---------------------------------------------------------------- the frame

    def update(self) -> None:
        """Apply what the workers finished and start what is due (each frame, on the main
        thread)."""
        found = self._fast.results() + self._slow.results()
        for worker in (self._traits, self._warm):
            if worker is not None:
                found += worker.results()
        for result in found:
            self._apply(result)
        now = time.monotonic()
        if (
            self._handles_changed is not None
            and now - self._handles_changed > SETTLE
            and self._fast.idle
            and self.targets is not None
            and self._solving != self._handles_version
        ):
            self._handles_changed = None
            self._solve()
        self.status = self._status()

    def _status(self) -> str:
        parts = []
        if self.targets is None:
            parts.append("measuring the handles")
        elif self._solving is not None:
            parts.append("solving the card exactly")
        elif not self.exact:
            parts.append("estimated card")
        return "; ".join(parts) or "exact"

    def _apply(self, result) -> None:
        kind = result[0]
        if kind == "error":
            self.message = result[1]
            self._solving = None
            return
        if kind == "measured":
            _, version, anchor, flying = result
            if version != self._rules_version:
                return
            self._anchor = anchor
            self.targets = self._applied = anchor.handles
            self.flying, self.exact = flying, True
            self._characterise(anchor.types)
            return
        if kind == "warmed":
            self._warm.stop()
            self._warm = None
            return
        if kind == "anchored":
            _, version, anchor = result
            if version == self._rules_version:
                self._anchor = anchor
            return
        if kind == "characterised":
            _, card, traits = result
            if card == self._card:  # still the card the match plays
                for side in self.loop.sides:
                    if hasattr(side, "retune"):
                        side.retune(traits)
            return
        if kind == "estimated":
            _, version, types, targets = result
            if version == self._handles_version:
                self._retune(types=types)
                self._applied = targets
                self.flying = None
            return
        if kind == "refused":
            _, version, reason = result
            if version == self._handles_version:
                self.message = reason
                self.targets = self._applied  # back to what the applied card was made for
                self._handles_version += 1
            return
        if kind == "progress":
            _, version, types, flown = result
            if version == self._handles_version:
                self._retune(types=types)
                self.flying = self._flying(flown)
            return
        if kind == "solved":
            _, version, rules_version, anchor, converged, reason = result
            if self._solving == version:
                self._solving = None
            if version != self._handles_version or rules_version != self._rules_version:
                return
            self._retune(types=anchor.types)
            self._anchor = anchor
            self.exact = converged
            self.message = "" if converged else reason
            self._characterise(anchor.types)

    def _flying(self, flown: torch.Tensor) -> dict[str, list[float]]:
        """Every handle the applied card flies: the flown ones measured by the solve, the
        others (closed forms) as set."""
        values = {name: getattr(self.targets, name).tolist() for name in api.HANDLE_NAMES}
        for index, name in enumerate(api.FLOWN_HANDLES):
            values[name] = flown[..., index].tolist()
        return values

    def _characterise(self, types: api.ShipTypes) -> None:
        """Re-measure the scripted agent's characteristics for this card, when a side uses
        them (on a worker of their own, so the next solve does not wait for them)."""
        if not any(hasattr(side, "retune") for side in self.loop.sides):
            return
        if self._traits is None:
            self._traits = _Worker(self._context, "tuning-traits", self._versions)
        self._traits.submit(_characterise_job, self._card, self.loop.rules, types)

    # ---------------------------------------------------------------- the slow worker

    def _measure(self) -> None:
        """Measure the applied card's handles under the current rules (slow worker), then
        anchor the estimate on them."""
        self.targets = None
        self.flying = None
        self._solving = None
        self._handles_version += 1
        self._slow.submit(_measure_job, self._rules_version, self.loop.rules, self.loop.types)

    def _solve(self) -> None:
        """The exact card for the current targets, from the applied card (slow worker)."""
        self._solving = self._handles_version
        self._slow.submit(
            _solve_job,
            self._handles_version,
            self._rules_version,
            self.targets,
            self.loop.rules,
            self.loop.types,
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
        """Write a preset: its ship set, the tuning's rule settings and every card field (so
        the file is a whole `ship_card` for training)."""
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        card = {name: [float(v) for v in value.tolist()] for name, value in vars(types).items()}
        data = {"ship_set": ship_set, "rules": tuning.rules, "ship_card": card}
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
        return Tuning(rules={k: float(v) for k, v in data.get("rules", {}).items()}, card=card)
