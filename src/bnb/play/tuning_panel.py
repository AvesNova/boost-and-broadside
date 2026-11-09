"""The tuning panel: an ImGui window over a live match that changes its numbers as it plays
(`bnb.play.tuning`).

- **Ships**: each handle per ship type (fighter, heavy), dragged (Ctrl+click types a value).
  Beside each, the value the applied card actually flies once measured (red when it is more
  than 0.5% off): until a solve iteration has reported, the card is an estimate and the
  flown value shows as "...".
- **Rules**, by group: each setting on a slider within its bounds; "(next episode)" marks
  those drawn at an episode's start.
- **Presets**: save the tuning under a name (the rules' settings and the whole card, the
  form training's `ship_card` reads), load one, or go back to the game's numbers.
"""

from imgui_bundle import imgui

from bnb.game import api

from .tuning import Presets, Tuning

PANEL_WIDTH = 640.0
TOP = 60.0
TYPE_NAMES = ("fighter", "heavy")
ERROR = imgui.ImVec4(0.85, 0.25, 0.25, 1.0)
DIM = imgui.ImVec4(0.55, 0.55, 0.55, 1.0)
DRAG_FRACTION = 0.004  # a handle drag moves its value this fraction per pixel
FORMAT = "%.5g"


def _label(name: str) -> str:
    return name.replace("_", " ")


def _shown(name: str, value: float) -> tuple[float, str]:
    """A handle in its display unit, and the unit."""
    unit, factor = api.HANDLE_UNITS[name]
    return value * factor, unit


class TuningPanel:
    """open: whether the window shows. The tuner it drives is the session's (created when
    the panel first opens on a live match)."""

    def __init__(self, presets: Presets | None = None) -> None:
        self.open = False
        self.presets = Presets() if presets is None else presets
        self.name = ""
        self.chosen = 0
        self.message = ""

    def draw(self, session) -> None:
        """The window, in the current ImGui frame."""
        if not self.open:
            return
        width, _ = imgui.get_io().display_size
        imgui.set_next_window_pos((width - PANEL_WIDTH - 10.0, TOP), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((PANEL_WIDTH, 640.0), imgui.Cond_.first_use_ever)
        expanded, self.open = imgui.begin("Tuning###tuning", self.open)
        if expanded:
            tuner = session.tuner
            if session.loop.replaying or tuner is None:
                imgui.text_wrapped("Tuning changes a live match; a replay plays as recorded.")
            else:
                self._status(tuner)
                self._ships(tuner)
                self._rules(tuner)
                self._presets(session, tuner)
        imgui.end()

    def _status(self, tuner) -> None:
        imgui.text_wrapped(tuner.status)
        if tuner.message:
            imgui.push_style_color(imgui.Col_.text, ERROR)
            imgui.text_wrapped(tuner.message)
            imgui.pop_style_color()
        imgui.separator()

    def _ships(self, tuner) -> None:
        if not imgui.collapsing_header("Ships", imgui.TreeNodeFlags_.default_open):
            return
        targets = tuner.targets
        if targets is None:
            imgui.text_wrapped("Measuring the card's handles under the current rules...")
            return
        types = targets.ceiling.shape[-1]
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.row_bg
        if not imgui.begin_table("handles", 1 + 2 * types, flags):
            return
        imgui.table_setup_column("handle", imgui.TableColumnFlags_.width_stretch, 1.9)
        for kind in range(types):
            imgui.table_setup_column(
                TYPE_NAMES[kind] if kind < len(TYPE_NAMES) else f"type {kind}",
                imgui.TableColumnFlags_.width_stretch,
                1.0,
            )
            imgui.table_setup_column("flying", imgui.TableColumnFlags_.width_stretch, 0.7)
        imgui.table_headers_row()
        flying = tuner.flying
        for name in api.HANDLE_NAMES:
            imgui.table_next_row()
            imgui.table_next_column()
            unit = api.HANDLE_UNITS[name][0]
            imgui.align_text_to_frame_padding()
            imgui.text(_label(name) + (f" ({unit})" if unit else ""))
            for kind in range(types):
                imgui.table_next_column()
                set_value, _ = _shown(name, float(getattr(targets, name)[kind]))
                imgui.push_id(f"{name}/{kind}")
                imgui.set_next_item_width(-1.0)
                speed = max(abs(set_value) * DRAG_FRACTION, 1e-6)
                changed, value = imgui.drag_float("##set", set_value, speed, 0.0, 0.0, FORMAT)
                if changed:
                    tuner.set_handle(name, kind, value / api.HANDLE_UNITS[name][1])
                imgui.pop_id()
                imgui.table_next_column()
                imgui.align_text_to_frame_padding()
                if flying is None:
                    imgui.text_colored(DIM, "...")
                else:
                    flown, _ = _shown(name, float(flying[name][kind]))
                    off = abs(flown - set_value) > 5e-3 * max(abs(set_value), 1e-9)
                    imgui.text_colored(ERROR if off else DIM, FORMAT % flown)
        imgui.end_table()

    def _rules(self, tuner) -> None:
        groups: dict[str, list] = {}
        for setting in api.tuning.settings_for(tuner.base):
            groups.setdefault(setting.group, []).append(setting)
        for group, settings in groups.items():
            if not imgui.collapsing_header(group):
                continue
            imgui.push_item_width(-230.0)
            for setting in settings:
                value = tuner.values[setting.key]
                label = setting.label + (f" ({setting.unit})" if setting.unit else "")
                if setting.when == api.tuning.EPISODE:
                    label += " (next episode)"
                label += f"##{setting.key}"
                if setting.integer:
                    changed, new = imgui.slider_int(
                        label, int(value), int(setting.low), int(setting.high)
                    )
                else:
                    changed, new = imgui.slider_float(
                        label, value, setting.low, setting.high, FORMAT
                    )
                if changed:
                    tuner.set_setting(setting.key, float(new))
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"game's value: {tuner.default(setting.key):.5g}")
            imgui.pop_item_width()

    def _presets(self, session, tuner) -> None:
        if not imgui.collapsing_header("Presets", imgui.TreeNodeFlags_.default_open):
            return
        _, self.name = imgui.input_text("name##preset", self.name)
        imgui.same_line()
        if imgui.button("Save"):
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
            self.message = "back to the game's numbers"
        if self.message:
            imgui.text_wrapped(self.message)
