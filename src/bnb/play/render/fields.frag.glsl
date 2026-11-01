#version 330
#include "common.glsl"
// The topographic map of ln n. The terraces fill by sign (thin air below 0, thick above)
// more strongly with |level|; contour lines sit at every third of a level. A point inside
// several fields' outer discs is drawn by the first of them only.
uniform sampler2D index_map;  // ln n, texel (i, j) at origin + (i + 0.5, j + 0.5) texel
uniform vec2 index_grid;  // origin, texel (m)
uniform sampler2D outer_discs;  // (M, 1): centre x, y and outer radius per field
uniform vec4 sky;
uniform vec4 thin_air;
uniform vec4 thick_air;
uniform vec4 thin_contour;
uniform vec4 thick_contour;
uniform float fill_strength;
uniform float contour_step;  // ln n between contours: a third of a level

const float LINE_WIDTH = 1.0;  // pixels
const float ON_CORE = 0.02;  // contours sit this far inside each whole step (off the plateau)
const float LEVELS = 6.0;  // terrace steps at which the fill is strongest (two levels)

in vec2 world;
flat in int field_index;
out vec4 colour;

vec4 catmull_rom(float t) {
    float t2 = t * t, t3 = t2 * t;
    return vec4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

// Bicubic (Catmull-Rom) through the texel centres, so the contours stay smooth between
// texels.
float log_index(vec2 point) {
    vec2 texel = (point - index_grid.x) / index_grid.y - 0.5;
    vec2 base = floor(texel);
    vec2 fraction = texel - base;
    vec4 weight_x = catmull_rom(fraction.x), weight_y = catmull_rom(fraction.y);
    ivec2 size = textureSize(index_map, 0);
    float sum = 0.0;
    for (int j = 0; j < 4; ++j) {
        float row = 0.0;
        for (int i = 0; i < 4; ++i) {
            ivec2 at = clamp(ivec2(base) + ivec2(i - 1, j - 1), ivec2(0), size - 1);
            row += weight_x[i] * texelFetch(index_map, at, 0).r;
        }
        sum += weight_y[j] * row;
    }
    return sum;
}

void main() {
    vec3 own = texelFetch(outer_discs, ivec2(field_index, 0), 0).xyz;
    if (distance(world, own.xy) > own.z) discard;
    for (int i = 0; i < field_index; ++i) {
        vec3 other = texelFetch(outer_discs, ivec2(i, 0), 0).xyz;
        if (distance(world, other.xy) <= other.z) discard;
    }
    float steps = log_index(world) / contour_step;
    float magnitude = abs(steps) + ON_CORE;
    float terrace = floor(magnitude);
    bool thick = steps > 0.0;
    vec3 air = thick ? thick_air.rgb : thin_air.rgb;
    vec3 fill = mix(sky.rgb, air, min(terrace / LEVELS, 1.0) * fill_strength);
    float width = max(fwidth(steps), 1e-6);  // steps per pixel
    float nearest = floor(magnitude + 0.5);
    float line = nearest >= 1.0
        ? line_coverage(abs(magnitude - nearest) / width, LINE_WIDTH) : 0.0;
    line *= clamp(1.5 - 3.0 * width, 0.0, 1.0);  // fade lines closer together than ~3 px
    vec4 ink = thick ? thick_contour : thin_contour;
    colour = vec4(mix(fill, ink.rgb, ink.a * line), 1.0);
}
