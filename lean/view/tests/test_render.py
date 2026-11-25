"""One offscreen frame in v2's look, when an OpenGL 3.3 context can be made (skipped
otherwise)."""

import numpy as np
import pytest

from lean.view.match import Match, MatchSettings


def test_renders_a_frame(tmp_path):
    moderngl = pytest.importorskip("moderngl")
    try:
        moderngl.create_standalone_context(backend="egl").release()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no EGL context: {error}")
    from lean.view import video
    from lean.view.play.index_map import IndexGrid

    match = Match(MatchSettings(ships_per_team=(1, 1), seed=0))
    frames = [frame for frame, _ in zip(match.snapshots(), range(20), strict=False)]
    index = match.index
    grid = IndexGrid(origin=index.origin, texel=index.texel, size=index.size)
    renderer = video.VideoRenderer(frames, grid, index.values, -1, "mint-pink", "test", True)
    pixels = np.frombuffer(renderer.frame(len(frames) - 1), np.uint8)
    assert pixels.size == video.WIDTH * video.HEIGHT * 3
    assert len(np.unique(pixels.reshape(-1, 3), axis=0)) > 50  # drawn, not one colour
