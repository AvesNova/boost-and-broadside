#version 330
// Dear ImGui's vertices: pixel positions (y down), atlas coordinates, 8-bit colours.
uniform vec2 display_size;
in vec2 position;
in vec2 uv;
in vec4 tint;
out vec2 v_uv;
out vec4 v_colour;

void main() {
    v_uv = uv;
    v_colour = tint;
    gl_Position = vec4(position.x / display_size.x * 2.0 - 1.0,
                       1.0 - position.y / display_size.y * 2.0, 0.0, 1.0);
}
