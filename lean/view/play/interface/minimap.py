"""The minimap card (lower right): the whole arena, drawn by the renderer into its own
texture (`bnb.play.render.minimap`), with the camera's view as a rectangle. Click or drag
on it to move the camera there. In a team view it shows that team's fog and only the
ships it sees, as the battlefield does.

`to_world` and `to_minimap` map between the minimap's pixels and the world; they are
each other's inverse.
"""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ..camera import Camera
from . import cards
from .controls import collapse_icon, icon_button, u32
from .layout import Rect

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Layout


def to_world(map_rect: Rect, minimap: Camera, px: float, py: float) -> tuple[float, float]:
    """The world point under the screen pixel (px, py) inside the minimap's square."""
    return minimap.world(px - map_rect.x, py - map_rect.y)


def to_minimap(map_rect: Rect, minimap: Camera, x: float, y: float) -> tuple[float, float]:
    """The screen pixel inside the minimap's square where the world point (x, y) is."""
    px, py = minimap.pixel(x, y)
    return px + map_rect.x, py + map_rect.y


def view_rect(map_rect: Rect, minimap: Camera, camera: Camera) -> Rect:
    """The battlefield camera's view, as a rectangle on the minimap (screen pixels)."""
    left, top = to_minimap(map_rect, minimap, *camera.world(0.0, 0.0))
    right, bottom = to_minimap(map_rect, minimap, *camera.world(camera.width, camera.height))
    return Rect(left, top, right - left, bottom - top)


def draw(interface: Interface, layout: Layout) -> None:
    """The minimap card, or its tab."""
    session, controls = interface.session, interface.controls
    if "minimap" not in interface.open:
        cards.tab(interface, "minimap", layout.minimap)
        return
    padding = layout.minimap_map.x - layout.minimap.x
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(padding, padding))
    cards.window("###minimap", layout.minimap, cards.FLAGS | imgui.WindowFlags_.no_scrollbar)
    imgui.pop_style_var()
    square = layout.minimap_map
    side = int(round(square.width))
    texture = session.draw_minimap(side)
    imgui.set_cursor_screen_pos(imgui.ImVec2(square.x, square.y))
    imgui.image(
        imgui.ImTextureRef(texture),
        imgui.ImVec2(side, side),
        imgui.ImVec2(0.0, 1.0),
        imgui.ImVec2(1.0, 0.0),
    )
    imgui.set_cursor_screen_pos(imgui.ImVec2(square.x, square.y))
    imgui.set_next_item_allow_overlap()
    imgui.invisible_button("minimap", imgui.ImVec2(side, side))
    controls.record("minimap/map")
    if imgui.is_item_active():
        mouse = imgui.get_io().mouse_pos
        x, y = to_world(square, session.minimap_camera, mouse.x, mouse.y)
        session.following = None
        session.camera.follow(x, y)
    if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip) and not imgui.is_item_active():
        imgui.set_tooltip("Click or drag to move the camera")
    draw_list = imgui.get_window_draw_list()
    draw_list.push_clip_rect(
        imgui.ImVec2(square.x, square.y), imgui.ImVec2(square.right, square.bottom)
    )
    view = view_rect(square, session.minimap_camera, session.camera)
    draw_list.add_rect(
        imgui.ImVec2(view.x, view.y),
        imgui.ImVec2(view.right, view.bottom),
        u32(session.theme.text, 0.85),
        0.0,
        1.5 * session.scale,
    )
    draw_list.pop_clip_rect()
    button = imgui.get_frame_height() * 0.8
    imgui.set_cursor_screen_pos(imgui.ImVec2(square.right - button - 2.0, square.y + 2.0))
    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0.0, 0.0))
    if icon_button(controls, "minimap/collapse", collapse_icon(False), "Collapse the map card"):
        interface.toggle("minimap")
    imgui.pop_style_var()
    imgui.end()
