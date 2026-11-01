#version 330
#include "common.glsl"
// Trails drawn straight from the trail store, without vertex buffers: vertex v belongs to
// segment v / 6 (ship, then sample k to k + 1) and is one corner of its two triangles.
// A segment with an end too old, unsampled or unseen by the view collapses off screen.
uniform sampler2D samples;  // (P, N): x, y, age (s), heading
uniform sampler2D turns;  // (P, N): angle of attack over the stall angle, seen by the view
uniform sampler2D ships;  // (1, N): team, ring radius (m), wingtip (ring radii)
uniform int sample_count;
uniform float seconds;  // a trail's length in time
uniform bool vortex;  // draw the wingtip vortices instead of the trails
uniform vec4 trail_colours[2];
uniform vec4 vortex_colour;
uniform float vortex_threshold;
uniform float vortex_ramp;
uniform vec2 trail_pixels;  // width at the oldest and the newest end
uniform float vortex_pixels;

out vec4 v_colour;

const int CORNER_SAMPLE[6] = int[6](0, 0, 1, 1, 0, 1);
const float CORNER_SIDE[6] = float[6](-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);

vec2 point(int ship, int k, vec4 ship_row) {
    vec4 s = texelFetch(samples, ivec2(k, ship), 0);
    if (!vortex) return s.xy;
    float turn = texelFetch(turns, ivec2(k, ship), 0).r;
    vec2 left = vec2(-sin(s.w), cos(s.w));
    return s.xy + left * sign(turn) * ship_row.z * ship_row.y;
}

void main() {
    int segments = sample_count - 1;
    int segment = gl_VertexID / 6, corner = gl_VertexID % 6;
    int ship = segment / segments, k = segment % segments;
    int at = k + CORNER_SAMPLE[corner];
    vec4 ship_row = texelFetch(ships, ivec2(0, ship), 0);
    float age_start = texelFetch(samples, ivec2(k, ship), 0).z;
    float age_end = texelFetch(samples, ivec2(k + 1, ship), 0).z;
    float age = texelFetch(samples, ivec2(at, ship), 0).z;
    float turn = abs(texelFetch(turns, ivec2(at, ship), 0).r);
    bool seen = texelFetch(turns, ivec2(k, ship), 0).g > 0.5
                && texelFetch(turns, ivec2(k + 1, ship), 0).g > 0.5;
    bool valid = seen && max(age_start, age_end) <= seconds;
    float strength = 1.0;
    if (vortex) {
        float threshold = max(vortex_threshold, 1e-6);
        strength = clamp((turn - threshold) / threshold * vortex_ramp, 0.0, 1.0);
        float other = abs(texelFetch(turns, ivec2(at == k ? k + 1 : k, ship), 0).r);
        valid = valid && (turn > threshold || other > threshold);
    }
    if (!valid) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);  // outside the clip volume: culled
        v_colour = vec4(0.0);
        return;
    }
    vec2 here = point(ship, at, ship_row);
    vec2 before = point(ship, max(at - 1, 0), ship_row);
    vec2 after = point(ship, min(at + 1, sample_count - 1), ship_row);
    vec2 along = after - before;
    vec2 normal = length(along) > 0.0 ? normalize(vec2(-along.y, along.x)) : vec2(0.0);
    float fresh = clamp(1.0 - age / seconds, 0.0, 1.0);  // 1 at the ship, 0 at the end
    float width = vortex ? vortex_pixels : mix(trail_pixels.x, trail_pixels.y, fresh);
    vec4 colour = vortex ? vortex_colour : trail_colours[int(ship_row.x)];
    v_colour = vec4(colour.rgb, colour.a * fresh * strength);
    gl_Position = clip_of(here + normal * CORNER_SIDE[corner] * 0.5 * width * pixel);
}
