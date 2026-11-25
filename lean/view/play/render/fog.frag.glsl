#version 330
#include "common.glsl"
// Multiplies the scene (blend: destination times source) by the fog colour where the
// view's team cannot see, by 1 where it can.
uniform sampler2D fog_mask;
uniform vec4 fog;

out vec4 colour;

void main() {
    // Four bilinear taps half a mask texel apart: a tent filter that softens the mask's
    // steps.
    vec2 texel = 1.0 / vec2(textureSize(fog_mask, 0));
    vec2 at = gl_FragCoord.xy / viewport;
    float seen = 0.25 * (texture(fog_mask, at + vec2(-0.5, -0.5) * texel).r
                         + texture(fog_mask, at + vec2(0.5, -0.5) * texel).r
                         + texture(fog_mask, at + vec2(-0.5, 0.5) * texel).r
                         + texture(fog_mask, at + vec2(0.5, 0.5) * texel).r);
    colour = vec4(mix(fog.rgb, vec3(1.0), seen), 1.0);
}
