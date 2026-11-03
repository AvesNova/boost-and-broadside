"""Dear ImGui's draw data through moderngl, so the panels and the HUD share the renderer's
context (imgui-bundle's own OpenGL backends need PyOpenGL). Follows the reference OpenGL 3
backend: textures created and updated on request (the font atlas grows as text needs
glyphs), one draw per command with its clip rectangle."""

import ctypes

import moderngl
from imgui_bundle import imgui

from .programs import program, set_uniforms


class ImguiRenderer:
    def __init__(self, context: moderngl.Context) -> None:
        self.context = context
        self.program = program(context, "imgui")
        self.vertices = context.buffer(reserve=1 << 16, dynamic=True)
        self.indices = context.buffer(reserve=1 << 16, dynamic=True)
        self.vertex_array = context.vertex_array(
            self.program,
            [(self.vertices, "2f 2f 4f1", "position", "uv", "tint")],
            index_buffer=self.indices,
            index_element_size=imgui.INDEX_SIZE,
        )
        self.textures: dict[int, moderngl.Texture] = {}
        io = imgui.get_io()
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        largest = context.info["GL_MAX_TEXTURE_SIZE"]
        imgui.get_platform_io().renderer_texture_max_width = largest
        imgui.get_platform_io().renderer_texture_max_height = largest

    def _update_textures(self) -> None:
        for texture in imgui.get_platform_io().textures:
            status = texture.status
            foreign = texture.tex_id != 0 and texture.tex_id not in self.textures
            if foreign and status != imgui.ImTextureStatus.destroyed:
                # Another renderer made it (an ImGui context can outlive one): make our own.
                status = imgui.ImTextureStatus.want_create
            if status == imgui.ImTextureStatus.want_create:
                created = self.context.texture(
                    (texture.width, texture.height), 4, texture.get_pixels_array().tobytes()
                )
                created.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.textures[created.glo] = created
                texture.set_tex_id(created.glo)
                texture.status = imgui.ImTextureStatus.ok
            elif status == imgui.ImTextureStatus.want_updates:
                self.textures[texture.tex_id].write(texture.get_pixels_array().tobytes())
                texture.status = imgui.ImTextureStatus.ok
            elif status == imgui.ImTextureStatus.want_destroy and texture.unused_frames > 0:
                self.textures.pop(texture.tex_id).release()
                texture.set_tex_id(0)
                texture.status = imgui.ImTextureStatus.destroyed

    def render(self, draw_data) -> None:
        """Draw this frame's ImGui output over the bound framebuffer."""
        self._update_textures()
        width, height = draw_data.display_size
        if width <= 0 or height <= 0:
            return
        context = self.context
        context.enable(moderngl.BLEND)
        context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        set_uniforms(self.program, display_size=(width, height), atlas=0)
        for commands in draw_data.cmd_lists:
            for buffer, vector, size in (
                (self.vertices, commands.vtx_buffer, imgui.VERTEX_SIZE),
                (self.indices, commands.idx_buffer, imgui.INDEX_SIZE),
            ):
                nbytes = vector.size() * size
                if buffer.size < nbytes:
                    buffer.orphan(2 * nbytes)
                buffer.write(ctypes.string_at(vector.data_address(), nbytes))
            for command in commands.cmd_buffer:
                self.textures[command.tex_ref.get_tex_id()].use(0)
                left, top, right, bottom = command.clip_rect
                if right <= left or bottom <= top:
                    continue
                context.scissor = (
                    int(left),
                    int(height - bottom),
                    int(right - left),
                    int(bottom - top),
                )
                self.vertex_array.render(
                    moderngl.TRIANGLES, vertices=command.elem_count, first=command.idx_offset
                )
        context.scissor = None
