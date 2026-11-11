"""The tuning panel: an ImGui window over a live match that changes its numbers as it plays
(`bnb.play.tuning`).

The window opens at the right, below the top bar, 400 px wide per interface scale and as
tall as the room under the top bar (never taller; its contents scroll). Each row is a label
(cut short to its column; the full label and unit in its tooltip), a slider and a number box
(4 significant digits).

- **Ships**: choose a ship type; each of its handles is a slider with a number box beside it
  (click the box, type, Enter). A red dot after the box marks a handle the ships fly more
  than 0.5% off its set value; its tooltip gives the flown value, or that the type is not
  measured yet (the sliders show at once; the chosen type is measured first). The fire
  rate is a choice of the rates a card can hold; the bleed rate is shown, not set. **Save
  ships** writes every type to its ship file (`src/bnb/game/ships/`, tracked in git);
  **Add** copies the chosen type under a new name and **Remove** deletes it (after a popup
  asks; the last type stays), each starting a new match. Save and Add need every type's
  handles, and first measure the ones not known (after a flight constant changed).
- **Rules**, by group: each setting a slider and a number box within its bounds; "*" marks
  those drawn at an episode's start, and the tooltip gives the game's value.
- **Presets**: save the tuning under a name (the rules' settings and the whole card), load
  one, or go back to the game's numbers (the ship files' and the rules' own).
"""

import math
import re

from imgui_bundle import imgui

from bnb.game import api

from .interface.layout import MARGIN, TOP_BAR, Rect
from .tuning import Presets, Tuning, fire_rates

PANEL_WIDTH = 400.0  # pixels at interface scale 1, like every size here
LABEL_WIDTH = 150.0  # the label column
BOX_WIDTH = 72.0  # the number box beside each slider
MARKER_WIDTH = 10.0  # the column of a handle's off-target marker
ERROR = imgui.ImVec4(0.85, 0.25, 0.25, 1.0)
DIM = imgui.ImVec4(0.55, 0.55, 0.55, 1.0)
OFF = 5e-3  # a flown handle this far off its set value, relative, is marked
SPAN = 4.0  # a handle's slider spans its saved value divided and multiplied by this
ENTER = imgui.InputTextFlags_.enter_returns_true
REMOVE = "Remove a ship type###remove ship"
EPISODE_MARK = " *"  # after the label of a setting drawn at an episode's start


def _label(name: str) -> str:
    return name.replace("_", " ")


def number_format(value: float) -> str:
    """The printf format of a number in the panel: 4 significant digits, written out in
    full from 10000 up rather than with an exponent."""
    return "%.0f" if abs(value) >= 1e4 else "%.4g"


def short(value: float) -> str:
    """The number as the panel writes it (`number_format`)."""
    return number_format(value) % value


def fit(text: str, width: float) -> str:
    """The text, cut short with "..." if it is wider than `width` pixels."""
    if imgui.calc_text_size(text).x <= width:
        return text
    while text and imgui.calc_text_size(text + "...").x > width:
        text = text[:-1]
    return text.rstrip() + "..."


def tooltip(help: str) -> None:
    """The row's tooltip for the last item, when it is hovered."""
    if help and imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
        imgui.set_tooltip(help)


def begin_rows(name: str, scale: float) -> bool:
    """A table of rows: the label, the slider, the number box and a handle's off-target
    marker (blank for a rule, so every table's boxes line up)."""
    if not imgui.begin_table(name, 4, imgui.TableFlags_.sizing_fixed_fit):
        return False
    fixed, stretch = imgui.TableColumnFlags_.width_fixed, imgui.TableColumnFlags_.width_stretch
    imgui.table_setup_column("label", fixed, LABEL_WIDTH * scale)
    imgui.table_setup_column("slider", stretch)
    imgui.table_setup_column("box", fixed, BOX_WIDTH * scale)
    imgui.table_setup_column("marker", fixed, MARKER_WIDTH * scale)
    return True


def row_label(label: str, help: str, mark: str = "") -> None:
    """A row's label (cut to its column, `mark` after it, dim), its tooltip the full one."""
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    room = imgui.get_content_region_avail().x - imgui.calc_text_size(mark).x
    imgui.text(fit(label, room))
    tooltip(help)
    if mark:
        imgui.same_line(0.0, 0.0)
        imgui.text_colored(DIM, mark)
        tooltip(help)


def number(
    label: str,
    key: str,
    value: float,
    low: float,
    high: float,
    integer: bool = False,
    log: bool = False,
    help: str = "",
    mark: str = "",
):
    """One row: the label, a slider (logarithmic: its value sits mid-slider whatever its
    scale) and a number box that takes any typed value, applied on Enter or when the box
    loses focus; `help` is the row's tooltip. Returns (changed, value)."""
    row_label(label, help, mark)
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
    tooltip(help)
    imgui.table_next_column()
    imgui.set_next_item_width(-1.0)
    if integer:
        typed, box = imgui.input_int("##box", int(value), 0, 0, ENTER)
    else:
        typed, box = imgui.input_float("##box", value, 0.0, 0.0, number_format(value), ENTER)
    typed = typed or imgui.is_item_deactivated_after_edit()
    tooltip(help)
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


def placement(width: float, height: float, scale: float) -> tuple[float, float, float, float]:
    """The panel's first place on a display this size (x, y, width, height): at the right,
    below the top bar, as tall as the room under it."""
    margin, top = MARGIN * scale, (TOP_BAR + 2.0 * MARGIN) * scale
    panel_width = min(PANEL_WIDTH * scale, width - 2.0 * margin)
    return width - panel_width - margin, top, panel_width, max(height - top - margin, 0.0)


def is_off(flown: float, value: float) -> bool:
    """Whether a flown handle is more than `OFF` (relative) from its set value."""
    return abs(flown - value) > OFF * max(abs(value), 1e-9)


def flown_marker(value: float, flown: float | None, unit: str, scale: float) -> None:
    """A handle row's last cell: a red dot when what the ships fly is more than 0.5% off the
    set value, the flown value in its tooltip; blank otherwise (and until measured)."""
    side = imgui.get_frame_height()
    low = imgui.get_cursor_screen_pos()
    imgui.dummy((MARKER_WIDTH * scale, side))
    if flown is None:
        tooltip("not measured yet")
    else:
        flies = f"flies {short(flown)}" + (f" {unit}" if unit else "")
        if is_off(flown, value):
            centre = (low.x + 0.5 * MARKER_WIDTH * scale, low.y + 0.5 * side)
            colour = imgui.get_color_u32(ERROR)
            imgui.get_window_draw_list().add_circle_filled(centre, 3.5 * scale, colour)
            tooltip(f"{flies}: more than 0.5% off the set {short(value)}")
        else:
            tooltip(flies)


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
        self.removing = 0  # the ship type the remove popup asks about
        self.factor = 2.0  # the time scale's factor
        self.window: Rect | None = None  # where the window was last drawn
        self.message = ""
        self._files: list | None = None  # the ship files as last read
        self._files_set = ""  # the ship set they are of

    def _saved(self, ship_set: str) -> list:
        """The ship set's files as last read."""
        if self._files is None or self._files_set != ship_set:
            self._files, self._files_set = api.ships.load(ship_set), ship_set
        return self._files

    def draw(self, session) -> None:
        """The window, in the current ImGui frame."""
        if not self.open:
            return
        x, y, width, height = placement(*imgui.get_io().display_size, session.scale)
        imgui.set_next_window_pos((x, y), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((width, height), imgui.Cond_.first_use_ever)
        # Never taller than the room below the top bar: the contents scroll instead.
        imgui.set_next_window_size_constraints((0.0, 0.0), (math.inf, height))
        expanded, self.open = imgui.begin("Tuning###tuning", self.open)
        position, size = imgui.get_window_pos(), imgui.get_window_size()
        self.window = Rect(position.x, position.y, size.x, size.y)
        if expanded:
            tuner = session.tuner
            if session.loop.replaying or tuner is None:
                imgui.text_wrapped("Tuning changes a live match; a replay plays as recorded.")
            else:
                self._status(tuner)
                self._ships(session, tuner)
                self._rules(tuner, session.scale)
                self._time_scale(session, tuner)
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
        names = api.type_names(tuner.ship_set)
        self.ship = min(self.ship, len(names) - 1)
        scale = session.scale
        imgui.set_next_item_width(150.0 * scale)
        _, self.ship = imgui.combo("##ship", self.ship, list(names), len(names))
        imgui.same_line()
        if imgui.button("Save ships"):
            self._save(session, tuner)
        imgui.same_line()
        controls = session.interface.controls
        imgui.begin_disabled(len(names) < 2)  # the last type stays
        if imgui.button("Remove"):
            self.removing = self.ship
            imgui.open_popup(REMOVE)
        controls.record("tuning/remove")
        imgui.end_disabled()
        self._confirm_remove(session, tuner, names)
        imgui.set_next_item_width(150.0 * scale)
        _, self.new_ship = imgui.input_text_with_hint(
            "##new ship", "new ship's name", self.new_ship
        )
        controls.record("tuning/new ship")
        imgui.same_line()
        if imgui.button("Add a copy as this name"):
            self._add(session, tuner)
        kind = self.ship
        targets = tuner.targets
        if kind >= targets.ceiling.shape[-1]:
            imgui.text_wrapped("This type joins at the next match.")
            return
        tuner.show(kind)  # measured first
        if not tuner.ready(kind):
            imgui.text_wrapped(f"Measuring {names[kind]}'s handles under the current rules...")
            return
        files = self._saved(tuner.ship_set)
        saved = files[kind].handles if kind < len(files) else {}
        flying = tuner.flying
        if not begin_rows("handles", scale):
            return
        for name in api.HANDLE_NAMES:
            unit, factor = api.HANDLE_UNITS[name]
            value = float(getattr(targets, name)[kind]) * factor
            label = _label(name)
            help = label + (f", {unit}" if unit else "")
            if name == "fire_rate":
                row_label(label, help)
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
                changed, new = number(label, name, value, low, high, log=True, help=help)
                if changed and math.isfinite(new):
                    tuner.set_handle(name, kind, new / factor)
            imgui.table_next_column()
            flown = flying[name][kind]
            flown = None if flown is None else float(flown) * factor
            flown_marker(value, flown, unit, scale)
        for name, values in tuner.shown.items():
            unit = api.HANDLE_UNITS[name][0]
            row_label(_label(name), f"{_label(name)}, {unit}: shown, not set")
            imgui.table_next_column()
            imgui.align_text_to_frame_padding()
            imgui.text_colored(DIM, "shown, not set")
            imgui.table_next_column()
            imgui.text_colored(DIM, short(values[kind]))
        imgui.end_table()

    def _current(self, tuner) -> list:
        """The ship types as the match plays them: the card's columns, and the handles they
        fly (the measured ones where known)."""
        types, files = tuner.loop.types, self._saved(tuner.ship_set)
        out = []
        for kind, ship in enumerate(files):
            handles = {}
            for name in api.HANDLE_NAMES:
                flown = tuner.flying[name][kind]
                handles[name] = float(
                    getattr(tuner.targets, name)[kind] if flown is None else flown
                )
            card = {field: float(value[kind]) for field, value in vars(types).items()}
            out.append(api.ships.Ship(ship.key, ship.name, handles, card, dict(ship.notes)))
        return out

    def _measured(self, tuner) -> bool:
        """Whether every type's handles are known, so the files can be written; if not,
        they are measured and the message says so."""
        if tuner.ready():
            return True
        tuner.measure_all()
        self.message = (
            "measuring every type's handles under the current rules first; try again then"
        )
        return False

    def _save(self, session, tuner) -> None:
        if not self._measured(tuner):
            return
        if not tuner.exact:
            self.message = "wait for the exact card (the status line) before saving"
            return
        session.save_ships(self._current(tuner))
        self._files = None
        directory = api.ships.directory_of(tuner.ship_set)
        self.message = f"saved {len(api.type_names(tuner.ship_set))} ships to {directory}"

    def _add(self, session, tuner) -> None:
        name = self.new_ship.strip()
        key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not key:
            self.message = "type the new ship's name first"
            return
        if not self._measured(tuner):
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

    def _confirm_remove(self, session, tuner, names: tuple) -> None:
        """The modal that asks before **Remove** deletes a ship type's file."""
        if not imgui.begin_popup_modal(REMOVE, None, imgui.WindowFlags_.always_auto_resize)[0]:
            return
        controls = session.interface.controls
        imgui.text(f"Remove {names[self.removing]}?")
        imgui.push_style_color(imgui.Col_.text, DIM)
        imgui.text("This deletes its ship file and starts a new match.")
        imgui.pop_style_color()
        if imgui.button("Remove##confirm"):
            self.ship = self.removing
            self._remove(session, tuner)
            imgui.close_current_popup()
        controls.record("tuning/remove/confirm")
        imgui.same_line()
        if imgui.button("Cancel"):
            imgui.close_current_popup()
        controls.record("tuning/remove/cancel")
        imgui.end_popup()

    def _remove(self, session, tuner) -> None:
        current = self._current(tuner) if tuner.ready() else list(self._saved(tuner.ship_set))
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

    def _rules(self, tuner, scale: float) -> None:
        groups: dict[str, list] = {}
        for setting in api.tuning.settings_for(tuner.base):
            groups.setdefault(setting.group, []).append(setting)
        for group, settings in groups.items():
            if not imgui.collapsing_header(group) or not begin_rows(group, scale):
                continue
            for setting in settings:
                episode = setting.when == api.tuning.EPISODE
                help = "\n".join(
                    line
                    for line in (
                        setting.label + (f", {setting.unit}" if setting.unit else ""),
                        f"game's value: {short(tuner.default(setting.key))}",
                        "applies from the next episode" if episode else "",
                    )
                    if line
                )
                changed, new = number(
                    setting.label.split(" (")[0],
                    setting.key,
                    tuner.values[setting.key],
                    setting.low,
                    setting.high,
                    setting.integer,
                    help=help,
                    mark=EPISODE_MARK if episode else "",
                )
                if changed and math.isfinite(new):
                    tuner.set_setting(setting.key, new)
            imgui.end_table()
            if any(setting.when == api.tuning.EPISODE for setting in settings):
                imgui.text_colored(DIM, "* applies from the next episode")

    # Time scale --------------------------------------------------------------------------

    def _time_scale(self, session, tuner) -> None:
        """A factor and **Apply** (the game that many times faster, map lengths fixed), and
        **Make this speed 1×** (the playback speed becomes the game's own)."""
        if not imgui.collapsing_header("Time scale"):
            return
        controls = session.interface.controls
        help = (
            "Make the whole game this many times faster with the map's lengths fixed: speeds "
            "x k, accelerations and gravity x k^2, rates and fire rates x k, times / k, "
            "energies x k^2 (a store keeps its height, energy / g) and powers x k^3. Fire "
            "rates snap to whole Hz dividing the tick rate and bullet life, clocks and shield "
            "delays to whole ticks; the handles are re-measured."
        )
        width = 80.0 * session.scale
        imgui.align_text_to_frame_padding()
        imgui.text("faster by")
        imgui.same_line()
        imgui.set_next_item_width(width)
        _, self.factor = imgui.input_float("##time factor", self.factor, 0.0, 0.0, "%.4g")
        controls.record("tuning/time factor")
        tooltip(help)
        imgui.same_line()
        if imgui.button("Apply##time scale") and self.factor > 0.0:
            self.message = session.time_scale(self.factor)
        controls.record("tuning/time scale")
        tooltip(help)
        playback = session.loop.time_scale
        imgui.begin_disabled(playback == 1.0)
        if imgui.button(f"Make this speed 1× (playback ×{playback:g})"):
            self.message = session.make_speed_normal()
        imgui.end_disabled()
        controls.record("tuning/make speed normal")
        tooltip(
            "Take the playback speed as the factor: the game becomes that many times faster "
            "and playback goes back to ×1, so the match looks the same.\n" + help
        )

    # Presets -----------------------------------------------------------------------------

    def _presets(self, session, tuner) -> None:
        if not imgui.collapsing_header("Presets"):
            return
        width = 150.0 * session.scale
        imgui.set_next_item_width(width)
        _, self.name = imgui.input_text_with_hint("##preset name", "preset's name", self.name)
        imgui.same_line()
        if imgui.button("Save##preset"):
            try:
                path = self.presets.save(
                    self.name.strip(), tuner.tuning, session.loop.types, tuner.ship_set
                )
                self.message = f"saved {path}"
            except (ValueError, OSError) as error:
                self.message = str(error)
        names = self.presets.names()
        if names:
            self.chosen = min(self.chosen, len(names) - 1)
            imgui.set_next_item_width(width)
            _, self.chosen = imgui.combo("##presets", self.chosen, names)
            imgui.same_line()
            if imgui.button("Load"):
                try:
                    session.retune_to(self.presets.load(names[self.chosen], tuner.ship_set))
                    self.message = f"loaded {names[self.chosen]}"
                except (ValueError, OSError, KeyError) as error:
                    self.message = str(error)
        if imgui.button("Game's numbers"):
            session.retune_to(Tuning())
            self.message = "back to the ship files' and the rules' own numbers"
