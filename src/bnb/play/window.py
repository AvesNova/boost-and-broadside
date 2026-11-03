"""The desktop window: GLFW for the window and input, the session for everything else.

Keys that fly the ship are read each frame into the pilot's input array; the others act
when pressed (`bnb.play.render.hud.KEYS` lists them all). The mouse pans and zooms the
camera, and drives ImGui's panels when it is over one (ImGui then takes the keyboard
too, so typing a hex code does not fly the ship).
"""

import string
import time

import glfw
import moderngl
import torch
from imgui_bundle import imgui

from bnb.controllers.human import INPUT, INPUTS

from .app import Options, Session

WHEEL_ZOOM = 1.15  # per wheel step

FLIGHT_KEYS = {
    "boost": (glfw.KEY_W,),
    "reverse": (glfw.KEY_S,),
    "left": (glfw.KEY_A,),
    "right": (glfw.KEY_D,),
    "full": (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT),
    "lightest": (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL),
    "shoot": (glfw.KEY_SPACE,),
}

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

# The keys that act once when pressed, to the `Window` method they call.
SHELL_KEYS = {
    glfw.KEY_ESCAPE: "quit",
    glfw.KEY_V: "cycle_view",
    glfw.KEY_T: "cycle_theme",
    glfw.KEY_E: "toggle_editor",
    glfw.KEY_F: "toggle_follow",
    glfw.KEY_F1: "toggle_help",
    glfw.KEY_P: "toggle_pause",
    glfw.KEY_N: "single_step",
    glfw.KEY_TAB: "next_pilot",
    glfw.KEY_EQUAL: "faster",
    glfw.KEY_KP_ADD: "faster",
    glfw.KEY_MINUS: "slower",
    glfw.KEY_KP_SUBTRACT: "slower",
    glfw.KEY_HOME: "fit",
}


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


class Window:
    """The desktop window: GLFW's window and input, feeding one session."""

    def __init__(self, options: Options) -> None:
        if not glfw.init():
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
        self.dragging = False
        self.cursor = (0.0, 0.0)
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

    # Callbacks ---------------------------------------------------------------------------

    def on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """GLFW's key callback: every key to ImGui, and a shell key's action when ImGui does
        not want the keyboard."""
        io = imgui.get_io()
        for flag, imgui_key in MODIFIERS:
            io.add_key_event(imgui_key, bool(mods & flag))
        if key in IMGUI_KEYS and action in (glfw.PRESS, glfw.RELEASE):
            io.add_key_event(IMGUI_KEYS[key], action == glfw.PRESS)
        if action != glfw.PRESS or io.want_capture_keyboard:
            return
        if key in SHELL_KEYS:
            getattr(self, SHELL_KEYS[key])()

    # The shell's keys (SHELL_KEYS) ---------------------------------------------------------

    def quit(self) -> None:
        """Close the window."""
        glfw.set_window_should_close(self.window, True)

    def cycle_view(self) -> None:
        """Switch to the next view."""
        self.session.cycle_view()

    def cycle_theme(self) -> None:
        """Switch to the next theme."""
        self.session.cycle_theme()

    def toggle_editor(self) -> None:
        """Show or hide the theme editor."""
        self.session.editor.open = not self.session.editor.open

    def toggle_follow(self) -> None:
        """Start or stop following the flown ship."""
        self.session.follow = not self.session.follow

    def toggle_help(self) -> None:
        """Show or hide the key list."""
        self.session.help = not self.session.help

    def toggle_pause(self) -> None:
        """Pause or resume the match."""
        self.session.loop.paused = not self.session.loop.paused

    def single_step(self) -> None:
        """One tick while paused."""
        self.session.loop.single_step()

    def next_pilot(self) -> None:
        """Fly the next ship."""
        self.session.loop.next_pilot()

    def faster(self) -> None:
        """The next larger time scale."""
        self.session.loop.faster()

    def slower(self) -> None:
        """The next smaller time scale."""
        self.session.loop.slower()

    def fit(self) -> None:
        """Fit the arena in the view."""
        self.session.camera.fit(self.session.frame.arena_radius)

    def on_char(self, window, codepoint: int) -> None:
        """GLFW's character callback: typed text to ImGui."""
        imgui.get_io().add_input_character(codepoint)

    def on_cursor(self, window, x: float, y: float) -> None:
        """GLFW's cursor callback: the position to ImGui, and a drag pans the camera."""
        scale_x, scale_y = self._scale()
        x, y = x * scale_x, y * scale_y
        imgui.get_io().add_mouse_pos_event(x, y)
        if self.dragging:
            self.session.camera.pan(x - self.cursor[0], y - self.cursor[1])
            self.session.follow = False
        self.cursor = (x, y)

    def on_button(self, window, button: int, action: int, mods: int) -> None:
        """GLFW's mouse button callback: to ImGui, and a press outside the panels starts a drag."""
        io = imgui.get_io()
        if button < 3:
            io.add_mouse_button_event(button, action == glfw.PRESS)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS and not io.want_capture_mouse

    def on_scroll(self, window, x_offset: float, y_offset: float) -> None:
        """GLFW's scroll callback: to ImGui, or a zoom about the cursor outside the panels."""
        io = imgui.get_io()
        io.add_mouse_wheel_event(x_offset, y_offset)
        if not io.want_capture_mouse:
            self.session.camera.zoom(WHEEL_ZOOM**y_offset, *self.cursor)

    def on_resize(self, window, width: int, height: int) -> None:
        """GLFW's framebuffer size callback: resize the session."""
        if width > 0 and height > 0:
            self.session.resize(width, height)

    # The loop ----------------------------------------------------------------------------

    def _inputs(self) -> torch.Tensor:
        inputs = torch.zeros(len(INPUTS), dtype=torch.bool)
        if imgui.get_io().want_capture_keyboard:
            return inputs
        for name, keys in FLIGHT_KEYS.items():
            inputs[INPUT[name]] = any(glfw.get_key(self.window, key) == glfw.PRESS for key in keys)
        return inputs

    def run(self) -> None:
        """Run frames until the window closes, then release the session."""
        last = time.perf_counter()
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            now = time.perf_counter()
            elapsed, last = now - last, now
            self.session.loop.inputs = self._inputs()
            self.session.update(elapsed)
            self.session.draw(self.context.screen)
            glfw.swap_buffers(self.window)
        self.session.close()
        glfw.terminate()


def run_window(options: Options) -> None:
    """Open a window on the options' match and run it."""
    Window(options).run()
