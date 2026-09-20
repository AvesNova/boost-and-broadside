"""Independent Pygame/ModernGL Frontline prototype driven by render snapshots."""

# ruff: noqa: E501

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass

import pygame
import torch

from boost_and_broadside.ui.gpu_snapshot import (
    PackedRenderSnapshot,
    RenderSnapshot,
    SnapshotCore,
    VisionPerspective,
)

MAX_OBSERVERS = 50
MAX_FOG_CORES = 128  # Frontline 50v50 has room for 77+ opaque primitives.


def toroidal_delta(
    position: tuple[float, float], center: tuple[float, float], world_size: tuple[float, float]
) -> tuple[float, float]:
    return tuple(
        coordinate - origin - size * math.floor((coordinate - origin) / size + 0.5)
        for coordinate, origin, size in zip(position, center, world_size, strict=True)
    )  # type: ignore[return-value]


def interpolate_toroidal_position(
    previous: tuple[float, float],
    current: tuple[float, float],
    alpha: float,
    world_size: tuple[float, float],
) -> tuple[float, float]:
    motion = toroidal_delta(current, previous, world_size)
    return tuple(
        (start + delta * max(0.0, min(1.0, alpha))) % size
        for start, delta, size in zip(previous, motion, world_size, strict=True)
    )  # type: ignore[return-value]


def segment_clear_of_core(
    observer: tuple[float, float],
    target: tuple[float, float],
    core: SnapshotCore,
    world_size: tuple[float, float],
) -> bool:
    """Exact minimum-image segment/core test used by the fog shader.

    Boundary contact is clear (strict comparisons); endpoints inside the same
    core are exempt because their disk segment never crosses its boundary.
    """

    segment = toroidal_delta(target, observer, world_size)
    center = toroidal_delta(core.position, observer, world_size)
    length_squared = segment[0] ** 2 + segment[1] ** 2
    projection = (
        0.0
        if length_squared == 0.0
        else (center[0] * segment[0] + center[1] * segment[1]) / length_squared
    )
    projection = max(0.0, min(1.0, projection))
    closest = (center[0] - projection * segment[0], center[1] - projection * segment[1])
    radius_squared = core.radius**2
    observer_inside = center[0] ** 2 + center[1] ** 2 < radius_squared
    target_inside = (center[0] - segment[0]) ** 2 + (center[1] - segment[1]) ** 2 < radius_squared
    return not (
        closest[0] ** 2 + closest[1] ** 2 < radius_squared
        and not (observer_inside and target_inside)
    )


def is_in_vision_range(
    observer: tuple[float, float],
    target: tuple[float, float],
    vision_range: float,
    world_size: tuple[float, float],
) -> bool:
    """Authoritative perception uses an inclusive range boundary."""

    delta = toroidal_delta(target, observer, world_size)
    return delta[0] ** 2 + delta[1] ** 2 <= vision_range**2


def fog_uniform_payload(
    snapshot: RenderSnapshot, perspective: VisionPerspective
) -> dict[str, object] | None:
    """Bounded team fog inputs; omniscient perspective intentionally has no fog."""

    if perspective == "full":
        return None
    team = 0 if perspective == "team0" else 1
    observers = snapshot.fog.observer_positions[team][:MAX_OBSERVERS]
    cores = snapshot.fog.opaque_cores[:MAX_FOG_CORES]
    return {
        "observer_positions": observers,
        "vision_range": snapshot.fog.vision_range,
        "opaque_cores": cores,
        # ModernGL validates fixed GLSL uniform array sizes on ``write``.
        "observer_data": tuple(value for point in observers for value in point)
        + (0.0,) * (2 * (MAX_OBSERVERS - len(observers))),
        "core_data": tuple(value for core in cores for value in (*core.position, core.radius))
        + (0.0,) * (3 * (MAX_FOG_CORES - len(cores))),
    }


@dataclass(frozen=True, slots=True)
class PackedFogUniformPayload:
    """Fixed-size float32 buffers required by the GLSL fog uniform arrays."""

    observer_count: int
    core_count: int
    vision_range: float
    observer_data: torch.Tensor
    core_data: torch.Tensor


def packed_fog_uniform_payload(
    snapshot: PackedRenderSnapshot, perspective: VisionPerspective
) -> PackedFogUniformPayload | None:
    """Pad packed fog rows once, without rebuilding snapshot dataclasses."""

    fog_rows = snapshot.fog_rows(perspective)
    if fog_rows is None or snapshot.vision_range is None:
        return None
    observers, cores = fog_rows
    observer_data = torch.zeros((MAX_OBSERVERS, 2), dtype=torch.float32)
    core_data = torch.zeros((MAX_FOG_CORES, 3), dtype=torch.float32)
    observer_data[: len(observers)] = observers
    core_data[: len(cores)] = cores
    return PackedFogUniformPayload(
        len(observers),
        len(cores),
        float(snapshot.vision_range),
        observer_data.flatten(),
        core_data.flatten(),
    )


def map_instance_payload(snapshot: RenderSnapshot) -> tuple[tuple[float, ...], ...]:
    """One batched instance row per field core, zone, and playable boundary.

    Row layout: x, y, radius, primitive kind (0 field/1 zone/2 boundary),
    team/role information. Field cores are the field-only prefix of fog cores.
    """

    rows = [
        (core.position[0], core.position[1], core.radius, 0.0, 0.0)
        for core in snapshot.fog.field_cores
    ]
    rows.extend(
        (zone.position[0], zone.position[1], zone.radius, 1.0, float(zone.role))
        for zone in snapshot.zones
    )
    rows.append((*snapshot.map_center, snapshot.playable_boundary_radius, 2.0, 0.0))
    return tuple(rows)


@dataclass(slots=True)
class GPUCamera:
    world_size: tuple[float, float]
    viewport_size: tuple[int, int]
    center: tuple[float, float] | None = None
    zoom: float = 1.0

    def __post_init__(self) -> None:
        if self.center is None:
            self.center = (self.world_size[0] / 2.0, self.world_size[1] / 2.0)

    @property
    def scale(self) -> float:
        return (
            min(
                self.viewport_size[0] / self.world_size[0],
                self.viewport_size[1] / self.world_size[1],
            )
            * self.zoom
        )

    def pan_pixels(self, dx: float, dy: float) -> None:
        assert self.center is not None
        self.center = (
            (self.center[0] - dx / self.scale) % self.world_size[0],
            (self.center[1] - dy / self.scale) % self.world_size[1],
        )

    def zoom_at(self, factor: float) -> None:
        self.zoom = max(1.0, min(64.0, self.zoom * factor))


_POINT_VERTEX = """#version 330
in vec2 in_previous; in vec2 in_current; in float in_team; in float in_visible;
uniform vec2 world_size, camera_center, viewport; uniform float scale, alpha;
out float team; out float visible;
vec2 shortest(vec2 p) { return p-world_size*floor(p/world_size+vec2(.5)); }
void main() { vec2 p=shortest(in_previous+shortest(in_current-in_previous)*alpha-camera_center); vec2 q=p*scale;
 gl_Position=vec4(q.x/(viewport.x*.5),-q.y/(viewport.y*.5),0,1); gl_PointSize=10; team=in_team; visible=in_visible; }"""
_POINT_FRAGMENT = """#version 330
in float team; in float visible; uniform int perspective; uniform vec3 projectile_color; out vec4 color;
void main() { if(perspective>=0 && (int(visible+.5)&(1<<perspective))==0) discard; if(length(gl_PointCoord-vec2(.5))>.5) discard;
 vec3 teams[2]=vec3[2](vec3(.2,.7,1),vec3(1,.3,.25)); color=vec4(projectile_color==vec3(0)?teams[int(team+.5)]:projectile_color,1); }"""
_MAP_VERTEX = """#version 330
in vec2 in_position; in float in_radius, in_kind, in_role; uniform vec2 world_size,camera_center,viewport; uniform float scale;
out float kind; out float role; vec2 shortest(vec2 p){return p-world_size*floor(p/world_size+vec2(.5));}
void main(){vec2 q=shortest(in_position-camera_center)*scale; gl_Position=vec4(q.x/(viewport.x*.5),-q.y/(viewport.y*.5),0,1); gl_PointSize=max(2.,in_radius*scale*2.);kind=in_kind;role=in_role;}"""
_MAP_FRAGMENT = """#version 330
in float kind; in float role; out vec4 color;
void main(){float d=length(gl_PointCoord-vec2(.5)); if(d>.5)discard; if(kind>1.5 && d<.47)discard;
 vec3 zone=role<1.?vec3(.2,.55,1):vec3(1,.3,.25); color=vec4(kind<.5?vec3(.18,.18,.24):(kind<1.5?zone:vec3(.85,.85,.85)),kind>1.5?.8:.28);}"""
_QUAD_VERTEX = """#version 330
in vec2 in_vert; out vec2 uv; void main(){uv=(in_vert+1.)*.5;gl_Position=vec4(in_vert,0,1);}"""
_FOG_FRAGMENT = f"""#version 330
in vec2 uv; uniform vec2 world_size,camera_center,viewport; uniform float scale,vision_range; uniform int observer_count,core_count;
uniform vec2 observers[{MAX_OBSERVERS}]; uniform vec3 cores[{MAX_FOG_CORES}]; out vec4 color;
vec2 shortest(vec2 p){{return p-world_size*floor(p/world_size+vec2(.5));}}
bool clear(vec2 a,vec2 b,vec3 core){{vec2 s=shortest(b-a),c=shortest(core.xy-a);float den=max(dot(s,s),1e-12);float t=clamp(dot(c,s)/den,0.,1.);vec2 close=c-t*s;
 bool ai=dot(c,c)<core.z*core.z,bi=dot(c-s,c-s)<core.z*core.z;return !(dot(close,close)<core.z*core.z && !(ai&&bi));}}
void main(){{vec2 target=camera_center+(uv-.5)*vec2(viewport.x,-viewport.y)/scale;float lit=0.;for(int i=0;i<observer_count;i++){{vec2 d=shortest(target-observers[i]);bool ok=length(d)<=vision_range;for(int c=0;c<core_count;c++)ok=ok&&clear(observers[i],target,cores[c]);if(ok)lit=1.;}}color=vec4(0,0,0,1.-lit);}}"""
_COMPOSITE_FRAGMENT = """#version 330
in vec2 uv; uniform sampler2D fog_texture; out vec4 color; void main(){color=texture(fog_texture,uv);}"""


class FrontlineGPURenderer:
    """Batched GPU scene plus quarter-resolution, exact-LOS team fog pass."""

    def __init__(
        self,
        snapshot: RenderSnapshot | PackedRenderSnapshot,
        viewport_size: tuple[int, int] = (900, 900),
    ) -> None:
        try:
            import moderngl
        except ImportError as error:
            raise RuntimeError(
                "ModernGL is required for FrontlineGPURenderer; install 'moderngl'."
            ) from error
        self._moderngl = moderngl
        pygame.init()
        pygame.display.set_mode(viewport_size, pygame.OPENGL | pygame.DOUBLEBUF)
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._program = self._ctx.program(
            vertex_shader=_POINT_VERTEX, fragment_shader=_POINT_FRAGMENT
        )
        self._map_program = self._ctx.program(
            vertex_shader=_MAP_VERTEX, fragment_shader=_MAP_FRAGMENT
        )
        self._fog_program = self._ctx.program(
            vertex_shader=_QUAD_VERTEX, fragment_shader=_FOG_FRAGMENT
        )
        self._composite_program = self._ctx.program(
            vertex_shader=_QUAD_VERTEX, fragment_shader=_COMPOSITE_FRAGMENT
        )
        self._buffer = self._ctx.buffer(reserve=1)
        self._map_buffer = self._ctx.buffer(reserve=1)
        self._quad = self._ctx.buffer(array("f", (-1, -1, 1, -1, -1, 1, 1, 1)))
        self._vao = self._ctx.vertex_array(
            self._program,
            [(self._buffer, "2f 2f 1f 1f", "in_previous", "in_current", "in_team", "in_visible")],
        )
        self._map_vao = self._ctx.vertex_array(
            self._map_program,
            [(self._map_buffer, "2f 1f 1f 1f", "in_position", "in_radius", "in_kind", "in_role")],
        )
        self._fog_vao = self._ctx.vertex_array(self._fog_program, [(self._quad, "2f", "in_vert")])
        self._composite_vao = self._ctx.vertex_array(
            self._composite_program, [(self._quad, "2f", "in_vert")]
        )
        fog_size = (max(1, viewport_size[0] // 4), max(1, viewport_size[1] // 4))
        self._fog_texture = self._ctx.texture(fog_size, 4)
        self._fog_fbo = self._ctx.framebuffer([self._fog_texture])
        self.camera = GPUCamera(snapshot.world_size, viewport_size)
        self.perspective: VisionPerspective = "full"
        self.running = True

    def close(self) -> None:
        for resource in (
            self._vao,
            self._map_vao,
            self._fog_vao,
            self._composite_vao,
            self._buffer,
            self._map_buffer,
            self._quad,
            self._fog_texture,
            self._fog_fbo,
            self._program,
            self._map_program,
            self._fog_program,
            self._composite_program,
            self._ctx,
        ):
            resource.release()
        pygame.quit()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEWHEEL:
                self.camera.zoom_at(1.2**event.y)
            elif event.type == pygame.MOUSEMOTION and event.buttons[2]:
                self.camera.pan_pixels(*event.rel)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                self.perspective = {"full": "team0", "team0": "team1", "team1": "full"}[
                    self.perspective
                ]  # type: ignore[assignment]

    def render(self, snapshot: RenderSnapshot, interpolation: float = 1.0) -> None:
        if snapshot.world_size != self.camera.world_size:
            raise ValueError("snapshot world size changed")
        self._ctx.screen.use()
        self._ctx.clear(0.015, 0.02, 0.04)
        self._set_scene_uniforms(self._map_program, snapshot, 1.0)
        self._draw_map(snapshot)
        self._set_scene_uniforms(self._program, snapshot, interpolation)
        self._program["perspective"].value = {"full": -1, "team0": 0, "team1": 1}[self.perspective]
        self._draw_entities(snapshot.ships, False)
        self._draw_entities(snapshot.projectiles, True)
        if (payload := fog_uniform_payload(snapshot, self.perspective)) is not None and payload[
            "vision_range"
        ] is not None:
            self._draw_fog(snapshot, payload)
        pygame.display.flip()

    def render_packed(
        self,
        snapshot: PackedRenderSnapshot,
        previous: PackedRenderSnapshot | None = None,
        interpolation: float = 1.0,
    ) -> None:
        """Render the opt-in tensor packet path without entity reconstruction.

        Callers retain the prior immutable packet and pass it as ``previous``;
        the first packet deliberately snaps all transforms to current state.
        """

        if snapshot.world_size != self.camera.world_size:
            raise ValueError("snapshot world size changed")
        if previous is not None and (
            previous.num_ships != snapshot.num_ships or previous.max_bullets != snapshot.max_bullets
        ):
            raise ValueError("packed snapshot layout changed")
        self._ctx.screen.use()
        self._ctx.clear(0.015, 0.02, 0.04)
        self._set_scene_uniforms(self._map_program, snapshot, 1.0)
        self._draw_packed_map(snapshot.map_instances())
        self._set_scene_uniforms(self._program, snapshot, interpolation)
        self._program["perspective"].value = {"full": -1, "team0": 0, "team1": 1}[self.perspective]
        self._draw_packed_entities(snapshot.ship_instances(previous), projectile=False)
        self._draw_packed_entities(snapshot.projectile_instances(previous), projectile=True)
        if (payload := packed_fog_uniform_payload(snapshot, self.perspective)) is not None:
            self._draw_packed_fog(snapshot, payload)
        pygame.display.flip()

    def _set_scene_uniforms(
        self, program: object, snapshot: RenderSnapshot | PackedRenderSnapshot, alpha: float
    ) -> None:
        for name, value in (
            ("world_size", snapshot.world_size),
            ("camera_center", self.camera.center),
            ("viewport", self.camera.viewport_size),
        ):
            program[name].value = value
        program["scale"].value = self.camera.scale
        if "alpha" in program:
            program["alpha"].value = max(0.0, min(1.0, alpha))

    def _draw_map(self, snapshot: RenderSnapshot) -> None:
        data = array("f", (value for row in map_instance_payload(snapshot) for value in row))
        self._map_buffer.orphan(len(data) * 4)
        self._map_buffer.write(data)
        self._map_vao.render(self._moderngl.POINTS, vertices=len(data) // 5)

    def _draw_packed_map(self, rows: torch.Tensor) -> None:
        """Upload a contiguous float32 map packet via the ndarray buffer protocol."""

        self._map_buffer.orphan(rows.numel() * rows.element_size())
        self._map_buffer.write(rows.numpy())
        self._map_vao.render(self._moderngl.POINTS, vertices=rows.shape[0])

    def _draw_entities(self, entities: object, projectile: bool) -> None:
        data = array("f")
        for entity in entities:
            if not projectile and not entity.alive:
                continue
            if projectile and not entity.active:
                continue
            data.extend(
                (
                    *entity.previous_position,
                    *entity.current_position,
                    float(entity.owner_team if projectile else entity.team),
                    float(entity.visibility_bits),
                )
            )
        if data:
            self._buffer.orphan(len(data) * 4)
            self._buffer.write(data)
            self._program["projectile_color"].value = (
                (1.0, 0.8, 0.2) if projectile else (0.0, 0.0, 0.0)
            )
            self._vao.render(self._moderngl.POINTS, vertices=len(data) // 6)

    def _draw_packed_entities(self, rows: torch.Tensor, projectile: bool) -> None:
        """Upload packed entity rows directly; no per-ship Python loop."""

        if not len(rows):
            return
        self._buffer.orphan(rows.numel() * rows.element_size())
        self._buffer.write(rows.numpy())
        self._program["projectile_color"].value = (1.0, 0.8, 0.2) if projectile else (0.0, 0.0, 0.0)
        self._vao.render(self._moderngl.POINTS, vertices=rows.shape[0])

    def _draw_fog(self, snapshot: RenderSnapshot, payload: dict[str, object]) -> None:
        self._draw_fog_buffers(
            snapshot,
            len(payload["observer_positions"]),
            len(payload["opaque_cores"]),
            float(payload["vision_range"]),
            array("f", payload["observer_data"]),
            array("f", payload["core_data"]),
        )

    def _draw_packed_fog(
        self, snapshot: PackedRenderSnapshot, payload: PackedFogUniformPayload
    ) -> None:
        self._draw_fog_buffers(
            snapshot,
            payload.observer_count,
            payload.core_count,
            payload.vision_range,
            payload.observer_data.numpy(),
            payload.core_data.numpy(),
        )

    def _draw_fog_buffers(
        self,
        snapshot: RenderSnapshot | PackedRenderSnapshot,
        observer_count: int,
        core_count: int,
        vision_range: float,
        observer_data: object,
        core_data: object,
    ) -> None:
        self._fog_fbo.use()
        # The fog shader writes its final alpha; blending into the transparent
        # FBO would square it under source-alpha blending.
        self._ctx.disable(self._moderngl.BLEND)
        self._ctx.clear(0.0, 0.0, 0.0, 0.0)
        self._set_scene_uniforms(self._fog_program, snapshot, 1.0)
        self._fog_program["observer_count"].value = observer_count
        self._fog_program["core_count"].value = core_count
        self._fog_program["vision_range"].value = vision_range
        self._fog_program["observers"].write(observer_data)
        self._fog_program["cores"].write(core_data)
        self._fog_vao.render(self._moderngl.TRIANGLE_STRIP)
        self._ctx.screen.use()
        self._ctx.enable(self._moderngl.BLEND)
        self._fog_texture.use(0)
        self._composite_program["fog_texture"].value = 0
        self._composite_vao.render(self._moderngl.TRIANGLE_STRIP)
