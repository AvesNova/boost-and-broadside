"""Optional Warp LOS prototype for performance_audit.py; no runtime dependency.

One native thread tests one observer/target pair, exiting at its first blocker.
This changes execution and arithmetic, not the intended visibility rule.
"""

import torch
import warp as wp

wp.config.kernel_cache_dir = "/tmp/bnb-audit-warp-cache"
wp.init()
wp.set_module_options({"enable_backward": False, "fast_math": False, "enable_fp_fusion": False})


@wp.func
def minimum_image(value: float, extent: float):
    shifted = value + extent * 0.5
    return shifted - wp.floor(shifted / extent) * extent - extent * 0.5


@wp.kernel
def los_kernel(
    ox: wp.array2d(dtype=float),
    oy: wp.array2d(dtype=float),
    tx: wp.array2d(dtype=float),
    ty: wp.array2d(dtype=float),
    cx: wp.array2d(dtype=float),
    cy: wp.array2d(dtype=float),
    radius: wp.array2d(dtype=float),
    width: float,
    height: float,
    output: wp.array3d(dtype=wp.bool),
):
    b, o, t = wp.tid()
    sx = minimum_image(tx[b, t] - ox[b, o], width)
    sy = minimum_image(ty[b, t] - oy[b, o], height)
    denom = wp.max(sx * sx + sy * sy, 1.0e-6)
    clear = bool(True)  # noqa: UP018 - Warp requires a dynamic loop variable.
    for c in range(radius.shape[1]):
        dx = minimum_image(cx[b, c] - ox[b, o], width)
        dy = minimum_image(cy[b, c] - oy[b, o], height)
        proj = wp.clamp((dx * sx + dy * sy) / denom, 0.0, 1.0)
        near_x = dx - proj * sx
        near_y = dy - proj * sy
        r2 = radius[b, c] * radius[b, c]
        observer_inside = dx * dx + dy * dy < r2
        ex = dx - sx
        ey = dy - sy
        target_inside = ex * ex + ey * ey < r2
        if near_x * near_x + near_y * near_y < r2:
            if not (observer_inside and target_inside):
                clear = False
                break
    output[b, o, t] = clear


def warp_los(observer, target, centers, radius, world):
    # Conversion and allocation are deliberately included in prototype timings.
    tensors = (
        observer.real,
        observer.imag,
        target.real,
        target.imag,
        centers.real,
        centers.imag,
        radius,
    )
    arrays = [wp.from_torch(tensor, dtype=wp.float32) for tensor in tensors]
    output = torch.empty(
        (observer.shape[0], observer.shape[1], target.shape[1]),
        dtype=torch.bool,
        device=observer.device,
    )
    stream = (
        wp.stream_from_torch(torch.cuda.current_stream(observer.device))
        if observer.is_cuda
        else None
    )
    wp.launch(
        los_kernel,
        dim=output.shape,
        inputs=[*arrays, *world],
        outputs=[wp.from_torch(output, dtype=wp.bool)],
        device=str(observer.device),
        stream=stream,
    )
    return output
