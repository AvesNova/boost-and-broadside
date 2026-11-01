"""The desktop window: GLFW for the window and input, the session for everything else.

Keys that fly the ship are read each frame into the pilot's input array; the others act
when pressed (`bnb.play.render.hud.KEYS` lists them all). The mouse pans and zooms the
camera, and drives ImGui's panels when it is over one (ImGui then takes the keyboard
too, so typing a hex code does not fly the ship).
"""

import time

import glfw
import moderngl
import torch
from imgui_bundle import imgui

from bnb.controllers.human import INPUT, INPUTS

from .app import Options, Session

WHEEL_ZOOM = 1.15  # per wheel step

FLIGHT_KEYS = {
    "boost": (glfw.KEY_W,), "reverse": (glfw.KEY_S,), "left": (glfw.KEY_A,),
    "right": (glfw.KEY_D,), "full": (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT),
    "lightest": (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL), "shoot": (glfw.KEY_SPACE,),
}

# The keys ImGui's text fields and navigation need.
IMGUI_KEYS = {
    glfw.KEY_TAB: imgui.Key.tab, glfw.KEY_LEFT: imgui.Key.left_arrow,
    glfw.KEY_RIGHT: imgui.Key.right_arrow, glfw.KEY_UP: imgui.Key.up_arrow,
    glfw.KEY_DOWN: imgui.Key.down_arrow, glfw.KEY_HOME: imgui.Key.home,
    glfw.KEY_END: imgui.Key.end, glfw.KEY_DELETE: imgui.Key.delete,
    glfw.KEY_BACKSPACE: imgui.Key.backspace, glfw.KEY_ENTER: imgui.Key.enter,
    glfw.KEY_KP_ENTER: imgui.Key.keypad_enter, glfw.KEY_ESCAPE: imgui.Key.escape,
    glfw.KEY_A: imgui.Key.a, glfw.KEY_C: imgui.Key.c, glfw.KEY_V: imgui.Key.v,
    glfw.KEY_X: imgui.Key.x, glfw.KEY_Z: imgui.Key.z,
    glfw.KEY_LEFT_CONTROL: imgui.Key.left_ctrl, glfw.KEY_RIGHT_CONTROL: imgui.Key.right_ctrl,
    glfw.KEY_LEFT_SHIFT: imgui.Key.left_shift, glfw.KEY_RIGHT_SHIFT: imgui.Key.right_shift,
    glfw.KEY_LEFT_ALT: imgui.Key.left_alt, glfw.KEY_RIGHT_ALT: imgui.Key.right_alt,
}
MODIFIERS = ((glfw.MOD_CONTROL, imgui.Key.mod_ctrl), (glfw.MOD_SHIFT, imgui.Key.mod_shift),
             (glfw.MOD_ALT, imgui.Key.mod_alt))


class Window:
    def __init__(self, options: Options) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW could not start (is there a display?)")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        self.window = glfw.create_window(options.width, options.height, "Boost and Broadside",
                                         None, None)
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
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        return fb_width / max(width, 1), fb_height / max(height, 1)

    # Callbacks ---------------------------------------------------------------------------

    def on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        io = imgui.get_io()
        for flag, imgui_key in MODIFIERS:
            io.add_key_event(imgui_key, bool(mods & flag))
        if key in IMGUI_KEYS and action in (glfw.PRESS, glfw.RELEASE):
            io.add_key_event(IMGUI_KEYS[key], action == glfw.PRESS)
        if action != glfw.PRESS or io.want_capture_keyboard:
            return
        session, loop = self.session, self.session.loop
        actions = {
            glfw.KEY_ESCAPE: lambda: glfw.set_window_should_close(window, True),
            glfw.KEY_V: session.cycle_view,
            glfw.KEY_T: session.cycle_theme,
            glfw.KEY_E: lambda: setattr(session.editor, "open", not session.editor.open),
            glfw.KEY_F: lambda: setattr(session, "follow", not session.follow),
            glfw.KEY_F1: lambda: setattr(session, "help", not session.help),
            glfw.KEY_P: lambda: setattr(loop, "paused", not loop.paused),
            glfw.KEY_N: loop.single_step,
            glfw.KEY_TAB: loop.next_pilot,
            glfw.KEY_EQUAL: loop.faster, glfw.KEY_KP_ADD: loop.faster,
            glfw.KEY_MINUS: loop.slower, glfw.KEY_KP_SUBTRACT: loop.slower,
            glfw.KEY_HOME: lambda: session.camera.fit(session.frame.arena_radius),
        }
        if key in actions:
            actions[key]()

    def on_char(self, window, codepoint: int) -> None:
        imgui.get_io().add_input_character(codepoint)

    def on_cursor(self, window, x: float, y: float) -> None:
        scale_x, scale_y = self._scale()
        x, y = x * scale_x, y * scale_y
        imgui.get_io().add_mouse_pos_event(x, y)
        if self.dragging:
            self.session.camera.pan(x - self.cursor[0], y - self.cursor[1])
            self.session.follow = False
        self.cursor = (x, y)

    def on_button(self, window, button: int, action: int, mods: int) -> None:
        io = imgui.get_io()
        if button < 3:
            io.add_mouse_button_event(button, action == glfw.PRESS)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS and not io.want_capture_mouse

    def on_scroll(self, window, x_offset: float, y_offset: float) -> None:
        io = imgui.get_io()
        io.add_mouse_wheel_event(x_offset, y_offset)
        if not io.want_capture_mouse:
            self.session.camera.zoom(WHEEL_ZOOM ** y_offset, *self.cursor)

    def on_resize(self, window, width: int, height: int) -> None:
        if width > 0 and height > 0:
            self.session.resize(width, height)

    # The loop ----------------------------------------------------------------------------

    def _inputs(self) -> torch.Tensor:
        inputs = torch.zeros(len(INPUTS), dtype=torch.bool)
        if imgui.get_io().want_capture_keyboard:
            return inputs
        for name, keys in FLIGHT_KEYS.items():
            inputs[INPUT[name]] = any(glfw.get_key(self.window, key) == glfw.PRESS
                                      for key in keys)
        return inputs

    def run(self) -> None:
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
    Window(options).run()
