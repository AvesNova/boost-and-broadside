#version 330
#include "common.glsl"
// One instance per field: a quad over its outer disc (radius plus half the band).
in vec2 corner;
in vec3 field;  // centre x, y and outer radius (m)

out vec2 world;
flat out int field_index;

void main() {
    float reach = field.z + 2.0 * pixel;
    world = field.xy + corner * reach;
    field_index = gl_InstanceID;
    gl_Position = clip_of(world);
}
