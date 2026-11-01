#version 330
#include "common.glsl"
// One instance per circle or arc: a quad around it.
in vec2 corner;  // the quad's corner, in [-1, 1]
in vec2 centre;
in float radius;
in vec2 width;  // the line's width: the larger of metres (x) and pixels (y)
in vec4 fill;
in vec4 line;
in vec2 arc;  // the arc's middle angle and half its span (rad); a half span of pi is a circle

out vec2 local;
flat out float v_radius;
flat out float v_width;
flat out vec4 v_fill;
flat out vec4 v_line;
flat out vec2 v_arc;

void main() {
    v_width = max(width.x, width.y * pixel);
    float reach = radius + 0.5 * v_width + 2.0 * pixel;
    local = corner * reach;
    v_radius = radius;
    v_fill = fill;
    v_line = line;
    v_arc = arc;
    gl_Position = clip_of(centre + local);
}
