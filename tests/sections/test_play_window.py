"""The window's key table (`bnb.play.window`) and the shortcuts
(`bnb.play.interface.shortcuts`): ImGui hears every key, and every shortcut calls a
session method. Needs only the GLFW and ImGui modules, not a display."""

import glfw

from bnb.controllers.human import INPUTS
from bnb.play.app import Session
from bnb.play.interface.shortcuts import FLIGHT_KEYS, SHORTCUTS
from bnb.play.window import IMGUI_KEYS

UNMAPPED = {"KEY_UNKNOWN", "KEY_LAST", "KEY_WORLD_1", "KEY_WORLD_2", "KEY_F25"}  # not in ImGui


def test_imgui_hears_every_key_glfw_names():
    keys = {
        value
        for name, value in vars(glfw).items()
        if name.startswith("KEY_") and name not in UNMAPPED
    }
    assert keys <= IMGUI_KEYS.keys()
    assert len(set(IMGUI_KEYS.values())) == len(IMGUI_KEYS)  # no two keys share one


def test_every_shortcut_and_flight_key_is_a_key_the_window_sends():
    sent = set(IMGUI_KEYS.values())
    assert {key for key, *_ in SHORTCUTS} <= sent
    assert {key for keys in FLIGHT_KEYS.values() for key in keys} <= sent
    assert set(FLIGHT_KEYS) == set(INPUTS)


def test_every_shortcut_calls_a_session_method():
    assert all(callable(getattr(Session, method, None)) for _, _, method, _ in SHORTCUTS)


def test_no_shortcut_is_a_flight_key():
    flight = {key for keys in FLIGHT_KEYS.values() for key in keys}
    assert not flight & {key for key, *_ in SHORTCUTS}
