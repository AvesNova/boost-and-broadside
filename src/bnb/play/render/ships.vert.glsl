#version 330
#include "common.glsl"
// Instanced silhouettes. The mesh holds the outline (the outline flag set: the polygon
// grown along its miters) and then the fill, so each ship's outline is drawn under its
// own fill and over the ships before it.
in vec2 position;  // nose along +x, in units of the ring radius
in vec2 miter;  // how far a vertex moves per unit of outline width
in float outline;
in vec2 centre;
in float heading;
in float size;  // the ring radius (m)
in vec4 fill;

uniform vec4 outline_colour;
uniform float outline_pixels;
uniform float outline_fraction;  // of the ring radius, when wider than outline_pixels

flat out vec4 v_colour;

void main() {
    float grow = outline * max(outline_pixels * pixel, outline_fraction * size);
    vec2 local = position * size + miter * grow;
    vec2 nose = vec2(cos(heading), sin(heading));
    vec2 turned = vec2(local.x * nose.x - local.y * nose.y, local.x * nose.y + local.y * nose.x);
    v_colour = outline > 0.5 ? outline_colour : fill;
    gl_Position = clip_of(centre + turned);
}
