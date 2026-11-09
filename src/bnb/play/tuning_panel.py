"""The tuning panel: an ImGui window over a live match that changes its numbers as it plays
(`bnb.play.tuning`).

- **Ships**: choose a ship type; each of its handles is a slider with a number box beside it
  (click the box, type, Enter), and beside that the value the ship actually flies (red when
  it is more than 0.5% off; "..." until a solve iteration has measured it). The fire rate is
  a choice of the rates a card can hold; the bleed rate is shown, not set. **Save ships**
  writes every type to its ship file (`src/bnb/game/ships/`, tracked in git); **Add** copies
  the chosen type under a new name and **Remove** deletes it, each starting a new match.
- **Rules**, by group: each setting a slider and a number box within its bounds;
  "(next episode)" marks those drawn at an episode's start.
- **Presets**: save the tuning under a name (the rules' settings and the whole card), load
  one, or go back to the game's numbers (the ship files' and the rules' own).
"""

import math
import re

from imgui_bundle import imgui

from bnb.game import api

from .tuning import Presets, Tuning, fire_rates

PANEL_WIDTH = 640.0
TOP = 60.0
ERROR = imgui.ImVec4(0.85, 0.25, 0.25, 1.0)
DIM = imgui.ImVec4(0.55, 0.55, 0.55, 1.0)
FORMAT = "%.5g"
BOX_WIDTH = 90.0  # the number box beside each slider
SPAN = 4.0  # a handle's slider spans its saved value divided and multiplied by this
ENTER = imgui.InputTextFlags_.enter_returns_true


def _label(name: str) -> str:
    return name.replace("_", " ")


def begin_rows(name: str, extra: bool) -> bool:
    """A table of rows: the label, the slider, the number box and (extra) a note."""
    columns = 4 if extra else 3
    if not imgui.begin_table(name, columns, imgui.TableFlags_.sizing_stretch_prop):
        return False
    imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch, 1.6)
    imgui.table_setup_column("slider", imgui.TableColumnFlags_.width_stretch, 1.4)
    imgui.table_setup_column("box", imgui.TableColumnFlags_.width_fixed, BOX_WIDTH)
    if extra:
        imgui.table_setup_column("note", imgui.TableColumnFlags_.width_stretch, 0.9)
    return True


def number(
    label: str,
    key: str,
    value: float,
    low: float,
    high: float,
    integer: bool = False,
    log: bool = False,
):
    """One row: the label, a slider (logarithmic: its value sits mid-slider whatever its
    scale) and a number box that takes any typed value, applied on Enter or when the box
    loses focus. Returns (changed, value)."""
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.text(label)
    imgui.push_id(key)
    imgui.table_next_column()
    imgui.set_next_item_width(-1.0)
    if integer:
        slid, slider = imgui.slider_int("##slider", int(value), int(low), int(high), " ")
    elif log and low * high > 0.0:
        # A linear slider over the log of the magnitude: ImGui's own logarithmic slider takes
        # values below the format's precision for zero, and its format here shows nothing.
        start, end = log_axis(low, high)
        position = min(max(to_log_axis(value, low), start), end) if value * low > 0 else start
        slid, position = imgui.slider_float("##slider", position, start, end, " ")
        slider = from_log_axis(position, low)
    else:
        slid, slider = imgui.slider_float("##slider", value, low, high, " ")
    imgui.table_next_column()
    imgui.set_next_item_width(-1.0)
    if integer:
        typed, box = imgui.input_int("##box", int(value), 0, 0, ENTER)
    else:
        typed, box = imgui.input_float("##box", value, 0.0, 0.0, FORMAT, ENTER)
    typed = typed or imgui.is_item_deactivated_after_edit()
    imgui.pop_id()
    if typed:
        return True, float(box)
    return slid, float(slider)


def to_log_axis(value: float, low: float) -> float:
    """A value's place on a logarithmic slider whose ends share `low`'s sign: the log of its
    magnitude, negated for negative values so the slider still runs from low to high."""
    sign = math.copysign(1.0, low)
    return sign * math.log(abs(value))


def from_log_axis(position: float, low: float) -> float:
    """The value at a place on that slider (`to_log_axis`'s inverse)."""
    sign = math.copysign(1.0, low)
    return sign * math.exp(sign * position)


def log_axis(low: float, high: float) -> tuple[float, float]:
    """The slider's ends on that axis, in order (low and high of one sign)."""
    return tuple(sorted((to_log_axis(low, low), to_log_axis(high, low))))


def _range(saved: float) -> tuple[float, float]:
    """A handle slider's span around its saved value (either sign)."""
    if saved == 0.0:
        return -1.0, 1.0
    ends = (saved / SPAN, saved * SPAN)
    return min(ends), max(ends)


class TuningPanel:
    """open: whether the window shows; `ship`: the ship type shown. The tuner it drives is
    the session's (created when the panel first opens on a live match)."""

    def __init__(self, presets: Presets | None = None) -> None:
        self.open = False
        self.presets = Presets() if presets is None else presets
        self.ship = 0
        self.name = ""
        self.new_ship = ""
        self.chosen = 0
        self.message = ""
        self._files: list | None = None  # the ship files as last read

    def _saved(self) -> list:
        if self._files is None:
            self._files = api.ships.load()
        return self._files

    def draw(self, session) -> None:
        """The window, in the current ImGui frame."""
        if not self.open:
            return
        width, _ = imgui.get_io().display_size
        imgui.set_next_window_pos((width - PANEL_WIDTH - 10.0, TOP), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((PANEL_WIDTH, 680.0), imgui.Cond_.first_use_ever)
        expanded, self.open = imgui.begin("Tuning###tuning", self.open)
        if expanded:
            tuner = session.tuner
            if session.loop.replaying or tuner is None:
                imgui.text_wrapped("Tuning changes a live match; a replay plays as recorded.")
            else:
                self._status(tuner)
                self._ships(session, tuner)
                self._rules(tuner)
                self._presets(session, tuner)
        imgui.end()

    def _status(self, tuner) -> None:
        imgui.text_wrapped(tuner.status)
        message = tuner.message or self.message
        if message:
            imgui.push_style_color(imgui.Col_.text, ERROR if tuner.message else DIM)
            imgui.text_wrapped(message)
            imgui.pop_style_color()
        imgui.separator()

    # Ships -------------------------------------------------------------------------------

    def _ships(self, session, tuner) -> None:
        if not imgui.collapsing_header("Ships", imgui.TreeNodeFlags_.default_open):
            return
        names = api.type_names()
        self.ship = min(self.ship, len(names) - 1)
        imgui.set_next_item_width(180.0)
        _, self.ship = imgui.combo("##ship", self.ship, list(names))
        imgui.same_line()
        if imgui.button("Save ships"):
            self._save(session, tuner)
        imgui.same_line()
        if imgui.button("Remove") and len(names) > 1:
            self._remove(session, tuner)
        imgui.set_next_item_width(180.0)
        _, self.new_ship = imgui.input_text_with_hint(
            "##new ship", "new ship's name", self.new_ship
        )
        imgui.same_line()
        if imgui.button("Add a copy as this name"):
            self._add(session, tuner)
        targets = tuner.targets
        if targets is None:
            imgui.text_wrapped("Measuring the card's handles under the current rules...")
            return
        kind = self.ship
        if kind >= targets.ceiling.shape[-1]:
            imgui.text_wrapped("This type joins at the next match.")
            return
        saved = self._saved()[kind].handles if kind < len(self._saved()) else {}
        flying = tuner.flying
        if not begin_rows("handles", extra=True):
            return
        for name in api.HANDLE_NAMES:
            unit, factor = api.HANDLE_UNITS[name]
            value = float(getattr(targets, name)[kind]) * factor
            label = _label(name) + (f" ({unit})" if unit else "")
            if name == "fire_rate":
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.align_text_to_frame_padding()
                imgui.text(label)
                imgui.table_next_column()
                rates = fire_rates(tuner.base.tick_length)
                index = min(range(len(rates)), key=lambda i: abs(rates[i] - value))
                imgui.set_next_item_width(-1.0)
                changed, index = imgui.combo("##fire_rate", index, [f"{r:g}" for r in rates])
                if changed:
                    tuner.set_handle(name, kind, rates[index])
                imgui.table_next_column()
            else:
                low, high = _range(saved.get(name, value / factor) * factor)
                if name == "half_lift_drag":
                    low, high = 0.5**5, 0.5**0.5
                changed, new = number(label, name, value, low, high, log=True)
                if changed and math.isfinite(new):
                    tuner.set_handle(name, kind, new / factor)
            imgui.table_next_column()
            if flying is None:
                imgui.text_colored(DIM, "flying ...")
            else:
                flown = float(flying[name][kind]) * factor
                off = abs(flown - value) > 5e-3 * max(abs(value), 1e-9)
                imgui.text_colored(ERROR if off else DIM, f"flying {FORMAT % flown}")
        for name, values in tuner.shown.items():
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(f"{_label(name)} ({api.HANDLE_UNITS[name][0]})")
            imgui.table_next_column()
            imgui.text_colored(DIM, f"{FORMAT % values[kind]} (shown, not set)")
        imgui.end_table()

    def _current(self, tuner) -> list:
        """The ship types as the match plays them: the card's columns, and the handles they
        fly (the measured ones where known)."""
        types, files = tuner.loop.types, self._saved()
        out = []
        for kind, ship in enumerate(files):
            handles = {
                name: float(
                    (tuner.flying or {}).get(name, getattr(tuner.targets, name).tolist())[kind]
                )
                for name in api.HANDLE_NAMES
            }
            card = {field: float(value[kind]) for field, value in vars(types).items()}
            out.append(api.ships.Ship(ship.key, ship.name, handles, card))
        return out

    def _save(self, session, tuner) -> None:
        if tuner.targets is None or not tuner.exact:
            self.message = "wait for the exact card (the status line) before saving"
            return
        session.save_ships(self._current(tuner))
        self._files = None
        self.message = f"saved {len(api.type_names())} ships to src/bnb/game/ships/"

    def _add(self, session, tuner) -> None:
        name = self.new_ship.strip()
        key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not key or tuner.targets is None:
            self.message = "type the new ship's name first"
            return
        current = self._current(tuner)
        if key in [ship.key for ship in current]:
            self.message = f"a ship {key!r} exists already"
            return
        copy = current[self.ship]
        current.append(api.ships.Ship(key, name, dict(copy.handles), dict(copy.card)))
        moved = {kind: kind for kind in range(len(current) - 1)}
        session.save_ships(current, moved)
        self._files = None
        self.ship, self.new_ship = len(current) - 1, ""
        self.message = f"added {name}: a copy of {copy.name}, in a new match"

    def _remove(self, session, tuner) -> None:
        current = self._current(tuner) if tuner.targets is not None else list(self._saved())
        removed = current.pop(self.ship)
        moved = {
            kind: (None if kind == self.ship else kind - (kind > self.ship))
            for kind in range(len(current) + 1)
        }
        session.save_ships(current, moved)
        self._files = None
        self.ship = max(self.ship - 1, 0)
        self.message = f"removed {removed.name}; a new match started"

    # Rules -------------------------------------------------------------------------------

    def _rules(self, tuner) -> None:
        groups: dict[str, list] = {}
        for setting in api.tuning.settings_for(tuner.base):
            groups.setdefault(setting.group, []).append(setting)
        for group, settings in groups.items():
            if not imgui.collapsing_header(group) or not begin_rows(group, extra=False):
                continue
            for setting in settings:
                label = setting.label + (f" ({setting.unit})" if setting.unit else "")
                if setting.when == api.tuning.EPISODE:
                    label += " (next episode)"
                changed, new = number(
                    label,
                    setting.key,
                    tuner.values[setting.key],
                    setting.low,
                    setting.high,
                    setting.integer,
                )
                if changed and math.isfinite(new):
                    tuner.set_setting(setting.key, new)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"game's value: {tuner.default(setting.key):.5g}")
            imgui.end_table()

    # Presets -----------------------------------------------------------------------------

    def _presets(self, session, tuner) -> None:
        if not imgui.collapsing_header("Presets"):
            return
        _, self.name = imgui.input_text("name##preset", self.name)
        imgui.same_line()
        if imgui.button("Save##preset"):
            try:
                path = self.presets.save(self.name.strip(), tuner.tuning, session.loop.types)
                self.message = f"saved {path}"
            except (ValueError, OSError) as error:
                self.message = str(error)
        names = self.presets.names()
        if names:
            self.chosen = min(self.chosen, len(names) - 1)
            _, self.chosen = imgui.combo("##presets", self.chosen, names)
            imgui.same_line()
            if imgui.button("Load"):
                try:
                    session.retune_to(self.presets.load(names[self.chosen]))
                    self.message = f"loaded {names[self.chosen]}"
                except (ValueError, OSError, KeyError) as error:
                    self.message = str(error)
        if imgui.button("Game's numbers"):
            session.retune_to(Tuning())
            self.message = "back to the ship files' and the rules' own numbers"
