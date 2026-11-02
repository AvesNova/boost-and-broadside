"""The window's key tables (`bnb.play.window`): ImGui hears every key, and every shell
key calls a real action. Needs only the GLFW and ImGui modules, not a display."""

import glfw

from bnb.play.window import FLIGHT_KEYS, IMGUI_KEYS, SHELL_KEYS, Window

UNMAPPED = {"KEY_UNKNOWN", "KEY_LAST", "KEY_WORLD_1", "KEY_WORLD_2", "KEY_F25"}  # not in ImGui


def test_imgui_hears_every_key_glfw_names():
    keys = {
        value
        for name, value in vars(glfw).items()
        if name.startswith("KEY_") and name not in UNMAPPED
    }
    assert keys <= IMGUI_KEYS.keys()
    assert len(set(IMGUI_KEYS.values())) == len(IMGUI_KEYS)  # no two keys share one


def test_every_shell_and_flight_key_reaches_imgui():
    flight = {key for keys in FLIGHT_KEYS.values() for key in keys}
    assert (SHELL_KEYS.keys() | flight) <= IMGUI_KEYS.keys()


def test_every_shell_key_calls_a_window_action():
    assert all(callable(getattr(Window, action, None)) for action in SHELL_KEYS.values())
