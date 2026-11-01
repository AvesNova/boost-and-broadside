#version 330
uniform sampler2D atlas;
in vec2 v_uv;
in vec4 v_colour;
out vec4 colour;

void main() {
    colour = v_colour * texture(atlas, v_uv);
}
