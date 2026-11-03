"""Shader programs: GLSL 3.30 files next to the passes, `name.vert.glsl` and
`name.frag.glsl`. A line `#include "common.glsl"` pulls in the shared camera helpers."""

import pathlib

import moderngl
import numpy as np

HERE = pathlib.Path(__file__).parent
INCLUDE = '#include "common.glsl"'


def source(name: str) -> str:
    text = (HERE / name).read_text()
    return text.replace(INCLUDE, (HERE / "common.glsl").read_text())


def program(context: moderngl.Context, name: str, vertex: str | None = None) -> moderngl.Program:
    """The program built from `name.vert.glsl` (or `vertex.vert.glsl`) and
    `name.frag.glsl`."""
    return context.program(
        vertex_shader=source(f"{vertex or name}.vert.glsl"),
        fragment_shader=source(f"{name}.frag.glsl"),
    )


def set_uniforms(program: moderngl.Program, **values) -> None:
    """Set the uniforms the program has (the compiler drops the ones it does not use)."""
    for name, value in values.items():
        if name in program:
            uniform = program[name]
            if isinstance(value, np.ndarray):
                uniform.write(value.astype(np.float32).tobytes())
            else:
                uniform.value = value
