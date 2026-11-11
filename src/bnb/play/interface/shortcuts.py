"""The keyboard: optional shortcuts for on-screen controls, and the keys that fly a ship.

Every shortcut names a `Session` method and appears in its control's tooltip; `KEYS` is
the reference the key list (F1) and the guide show.
"""

from imgui_bundle import imgui

# Key, its label, the Session method it calls, what it does.
SHORTCUTS = (
    (imgui.Key.escape, "Esc", "toggle_menu", "open or close the menu (pauses a live match)"),
    (imgui.Key.p, "P", "toggle_pause", "pause or play"),
    (imgui.Key.n, "N", "single_step", "one tick, while paused"),
    (imgui.Key.equal, "+", "faster", "faster"),
    (imgui.Key.keypad_add, "+", "faster", "faster"),
    (imgui.Key.minus, "-", "slower", "slower"),
    (imgui.Key.keypad_subtract, "-", "slower", "slower"),
    (imgui.Key.v, "V", "cycle_view", "next view: spectator, team 0, team 1"),
    (imgui.Key.f, "F", "toggle_follow", "follow the selected ship"),
    (imgui.Key.c, "C", "toggle_control", "take control of the selected ship, or release it"),
    (imgui.Key.home, "Home", "fit", "fit the arena in the view"),
    (imgui.Key.z, "Z", "toggle_true_size", "ships at their true size, or never too small to see"),
    (imgui.Key.g, "G", "cycle_grid", "dot grid: off, square, triangle, hex"),
    (imgui.Key.t, "T", "cycle_theme", "next theme"),
    (imgui.Key.e, "E", "toggle_editor", "theme editor"),
    (imgui.Key.f2, "F2", "toggle_tuning", "tuning panel: the ships' handles and the rules"),
    (imgui.Key.left_arrow, "Left", "step_back", "replay: one tick back, while paused"),
    (imgui.Key.right_arrow, "Right", "single_step", "replay: one tick on, while paused"),
    (imgui.Key.f1, "F1", "toggle_keys", "this list"),
)
LABELS = {method: label for _, label, method, _ in SHORTCUTS}

# The flight keys: each INPUTS entry (bnb.controllers.human) and the keys that hold it.
FLIGHT_KEYS = {
    "boost": (imgui.Key.w,),
    "reverse": (imgui.Key.s,),
    "left": (imgui.Key.a,),
    "right": (imgui.Key.d,),
    "full": (imgui.Key.left_shift, imgui.Key.right_shift),
    "lightest": (imgui.Key.left_ctrl, imgui.Key.right_ctrl),
    "shoot": (imgui.Key.space,),
}

KEYS = (
    ("W / S", "boost / reverse (neither: coast), while flying"),
    ("A / D", "turn left / right"),
    ("Shift / Ctrl", "full / lightest turn"),
    ("Space", "shoot"),
    ("click", "select a ship, field or zone"),
    ("drag / wheel", "pan / zoom"),
) + tuple(
    (label, meaning)
    for index, (_, label, method, meaning) in enumerate(SHORTCUTS)
    if method not in {entry[2] for entry in SHORTCUTS[:index]}
)


def shortcut(method: str) -> str | None:
    """The key label of the shortcut that calls `method`, if any."""
    return LABELS.get(method)
