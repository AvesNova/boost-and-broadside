"""The desktop window: GLFW for the window and its input, the session for everything else.

Every key, character, mouse move, button and wheel step goes to ImGui's input state; the
session reads it there (its interface, the battlefield's clicks, drags and wheel, the
flight keys and the shortcuts), so the window adds no behaviour of its own.

The window opens without the desktop's input method (`without_input_method`): on X11 an
input method such as IBus with a Hangul engine takes the letter keys, and GLFW, which has no
composition support, then hands ImGui garbled text (digits pass straight through). Typed
text is the key's own character.
"""

import contextlib
import os
import string
import time

import glfw
import moderngl
from imgui_bundle import imgui

from .app import Options, Session

# The named keys GLFW and ImGui share (the names below are both libraries' own).
NAMED_KEYS = {
    "TAB": "tab",
    "LEFT": "left_arrow",
    "RIGHT": "right_arrow",
    "UP": "up_arrow",
    "DOWN": "down_arrow",
    "PAGE_UP": "page_up",
    "PAGE_DOWN": "page_down",
    "HOME": "home",
    "END": "end",
    "INSERT": "insert",
    "DELETE": "delete",
    "BACKSPACE": "backspace",
    "SPACE": "space",
    "ENTER": "enter",
    "ESCAPE": "escape",
    "APOSTROPHE": "apostrophe",
    "COMMA": "comma",
    "MINUS": "minus",
    "PERIOD": "period",
    "SLASH": "slash",
    "SEMICOLON": "semicolon",
    "EQUAL": "equal",
    "LEFT_BRACKET": "left_bracket",
    "BACKSLASH": "backslash",
    "RIGHT_BRACKET": "right_bracket",
    "GRAVE_ACCENT": "grave_accent",
    "CAPS_LOCK": "caps_lock",
    "SCROLL_LOCK": "scroll_lock",
    "NUM_LOCK": "num_lock",
    "PRINT_SCREEN": "print_screen",
    "PAUSE": "pause",
    "MENU": "menu",
    "KP_DECIMAL": "keypad_decimal",
    "KP_DIVIDE": "keypad_divide",
    "KP_MULTIPLY": "keypad_multiply",
    "KP_SUBTRACT": "keypad_subtract",
    "KP_ADD": "keypad_add",
    "KP_ENTER": "keypad_enter",
    "KP_EQUAL": "keypad_equal",
    "LEFT_SHIFT": "left_shift",
    "LEFT_CONTROL": "left_ctrl",
    "LEFT_ALT": "left_alt",
    "LEFT_SUPER": "left_super",
    "RIGHT_SHIFT": "right_shift",
    "RIGHT_CONTROL": "right_ctrl",
    "RIGHT_ALT": "right_alt",
    "RIGHT_SUPER": "right_super",
}
FUNCTION_KEYS = 24  # F1 to F24: ImGui has no F25


def _imgui_keys() -> dict[int, imgui.Key]:
    """Every GLFW key that ImGui has, to its ImGui key: letters, digits, the keypad, the
    function keys and the named keys (as Dear ImGui's own GLFW backend maps them)."""
    keys = {
        getattr(glfw, f"KEY_{letter}"): getattr(imgui.Key, letter.lower())
        for letter in string.ascii_uppercase
    }
    for digit in range(10):
        keys[getattr(glfw, f"KEY_{digit}")] = getattr(imgui.Key, f"_{digit}")
        keys[getattr(glfw, f"KEY_KP_{digit}")] = getattr(imgui.Key, f"keypad{digit}")
    for number in range(1, FUNCTION_KEYS + 1):
        keys[getattr(glfw, f"KEY_F{number}")] = getattr(imgui.Key, f"f{number}")
    for glfw_name, imgui_name in NAMED_KEYS.items():
        keys[getattr(glfw, f"KEY_{glfw_name}")] = getattr(imgui.Key, imgui_name)
    return keys


IMGUI_KEYS = _imgui_keys()
MODIFIERS = (
    (glfw.MOD_CONTROL, imgui.Key.mod_ctrl),
    (glfw.MOD_SHIFT, imgui.Key.mod_shift),
    (glfw.MOD_ALT, imgui.Key.mod_alt),
    (glfw.MOD_SUPER, imgui.Key.mod_super),
)


@contextlib.contextmanager
def without_input_method():
    """GLFW's start without an X input method: XMODIFIERS, which X reads when GLFW
    starts, names none (and is restored afterwards, so other programs started from here
    keep the desktop's)."""
    saved = os.environ.get("XMODIFIERS")
    os.environ["XMODIFIERS"] = "@im=none"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("XMODIFIERS", None)
        else:
            os.environ["XMODIFIERS"] = saved


class Window:
    """The desktop window: GLFW's window and input, feeding one session."""

    def __init__(self, options: Options) -> None:
        with without_input_method():
            started = glfw.init()
        if not started:
            raise RuntimeError("GLFW could not start (is there a display?)")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        self.window = glfw.create_window(
            options.width, options.height, "Boost and Broadside", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW could not open an OpenGL 3.3 window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if options.vsync else 0)
        self.context = moderngl.create_context(require=330)
        width, height = glfw.get_framebuffer_size(self.window)
        self.session = Session(options, self.context, width, height)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_char_callback(self.window, self.on_char)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor)
        glfw.set_mouse_button_callback(self.window, self.on_button)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_framebuffer_size_callback(self.window, self.on_resize)

    def _scale(self) -> tuple[float, float]:
        """Framebuffer pixels per window unit (above 1 on high-density displays)."""
        width, height = glfw.get_window_size(self.window)
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        return framebuffer_width / max(width, 1), framebuffer_height / max(height, 1)

    def on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """GLFW's key callback: the modifiers and the key to ImGui."""
        io = imgui.get_io()
        for flag, imgui_key in MODIFIERS:
            io.add_key_event(imgui_key, bool(mods & flag))
        if key in IMGUI_KEYS and action in (glfw.PRESS, glfw.RELEASE):
            io.add_key_event(IMGUI_KEYS[key], action == glfw.PRESS)

    def on_char(self, window, codepoint: int) -> None:
        """GLFW's character callback: typed text to ImGui."""
        imgui.get_io().add_input_character(codepoint)

    def on_cursor(self, window, x: float, y: float) -> None:
        """GLFW's cursor callback: the position, in framebuffer pixels, to ImGui."""
        scale_x, scale_y = self._scale()
        imgui.get_io().add_mouse_pos_event(x * scale_x, y * scale_y)

    def on_button(self, window, button: int, action: int, mods: int) -> None:
        """GLFW's mouse button callback: to ImGui."""
        if button < 3:
            imgui.get_io().add_mouse_button_event(button, action == glfw.PRESS)

    def on_scroll(self, window, x_offset: float, y_offset: float) -> None:
        """GLFW's scroll callback: to ImGui."""
        imgui.get_io().add_mouse_wheel_event(x_offset, y_offset)

    def on_resize(self, window, width: int, height: int) -> None:
        """GLFW's framebuffer size callback: resize the session."""
        if width > 0 and height > 0:
            self.session.resize(width, height)

    def run(self) -> None:
        """Run frames until the window closes or the menu quits, then close the session
        (saving the live match's replay)."""
        last = time.perf_counter()
        while not glfw.window_should_close(self.window) and not self.session.quit_requested:
            glfw.poll_events()
            now = time.perf_counter()
            elapsed, last = now - last, now
            self.session.update(elapsed)
            self.session.draw(self.context.screen)
            glfw.swap_buffers(self.window)
        self.session.close()
        glfw.terminate()


def run_window(options: Options) -> None:
    """Open a window on the options' match and run it."""
    Window(options).run()
