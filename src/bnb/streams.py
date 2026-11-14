"""Keyed random streams: every random draw of training, evaluation, BC and the rating
matches is a pure function of the run's seed and of what the draw is for, never a position
in a shared sequence.

Two forms, one key scheme (`stream_key`: the seed and a stream's name, crc32):
- counter-based draws (`keyed_hash`, `keyed_uniform`, `keyed_normal`): a 32-bit integer
  hash of a key and integer coordinates (an environment, an episode, a tick, a ship, a
  slot), on any device, inside a captured graph or a compiled stage, with no state and no
  host read. The same arguments give the same draws on every device.
- a seeded generator per stream (`stream_generator`), for draws made once on the host
  (a parameter's initial values, the learner's minibatch order).

Draws keyed this way couple runs that differ in something unrelated: an environment's k-th
episode gets the same map, its tick t the same game draws and its decisions the same
uniforms in every run of one seed, whatever else the runs draw (common random numbers).
`inverse_cdf` turns a uniform into a category, so two policies that are close mostly pick
the same action from the same uniform.
"""

import zlib

import torch

MASK = (1 << 32) - 1
SCALE = 1.0 / (1 << 24)  # a uniform's 24 bits (float32's mantissa) in [0, 1)


def stream_key(seed: int, name: str) -> int:
    """A stream's 32-bit key, from the run's seed and the stream's name."""
    return zlib.crc32(f"{seed}/{name}".encode()) & MASK


def purpose_key(name: str) -> int:
    """A 32-bit key for one use of an already keyed value (a decision's key, a tick's):
    the use's name alone, as the value carries the seed."""
    return zlib.crc32(name.encode()) & MASK


def stream_generator(seed: int, name: str, device: torch.device | None = None) -> torch.Generator:
    """The stream `name`'s own generator, seeded by `stream_key`, so a new stream shifts no
    other's draws. A 32-bit seed: the CPU generator (Mersenne twister) reads only a seed's
    low 32 bits."""
    return torch.Generator(device).manual_seed(stream_key(seed, name))


def _multiply(value: torch.Tensor, constant: int) -> torch.Tensor:
    """(value * constant) mod 2^32 for int64 values below 2^32, without int64 overflow."""
    high, low = constant >> 16, constant & 0xFFFF
    return (value * low + (((value * high) & 0xFFFF) << 16)) & MASK


def _mix(value: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer hash (lowbias32), on int64 values below 2^32."""
    value = value ^ (value >> 16)
    value = _multiply(value, 0x7FEB352D)
    value = value ^ (value >> 15)
    value = _multiply(value, 0x846CA68B)
    return value ^ (value >> 16)


def keyed_hash(key: int, *coordinates: torch.Tensor) -> torch.Tensor:
    """(...) int64 hashes in [0, 2^32) of the key and the coordinates (int64 tensors in
    [0, 2^32), broadcast against each other), one hash round per coordinate, in order."""
    if not coordinates:
        raise ValueError("a keyed hash needs at least one coordinate")
    value = key
    for coordinate in coordinates:
        value = _mix((coordinate & MASK) ^ value)
    return value


def unit(hashed: torch.Tensor) -> torch.Tensor:
    """Uniforms in [0, 1) float32 from 32-bit hashes (their top 24 bits)."""
    return (hashed >> 8).to(torch.float32) * SCALE


def keyed_uniform(key: int, *coordinates: torch.Tensor) -> torch.Tensor:
    """Uniforms in [0, 1) float32, a pure function of the key and the coordinates (as
    `keyed_hash`)."""
    return unit(keyed_hash(key, *coordinates))


def keyed_normal(key: int, *coordinates: torch.Tensor) -> torch.Tensor:
    """Standard normals float32 by the inverse CDF of a keyed uniform in (0, 1) (as
    `keyed_hash`); within 5.4 standard deviations, from 24 bits."""
    uniform = ((keyed_hash(key, *coordinates) >> 8).to(torch.float32) + 0.5) * SCALE
    return torch.erfinv(2.0 * uniform - 1.0) * 2.0**0.5


def element_index(shape: tuple[int, ...], device: torch.device | None = None) -> torch.Tensor:
    """(*shape) int64 each element's flat index: the coordinate that keys an array of draws
    per row (`row_uniforms`)."""
    count = 1
    for size in shape:
        count *= size
    return torch.arange(count, device=device).reshape(shape)


def row_uniforms(key: int, rows: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """(R, *shape) uniforms in [0, 1): for each row's key (R,) int64 (a decision's), an
    array of draws keyed by `key` (the use's), the row's key and the element's index."""
    index = element_index(shape, rows.device)
    return keyed_uniform(key, rows.reshape(rows.shape + (1,) * len(shape)), index)


def inverse_cdf(weights: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """One category per row of non-negative `weights` (..., K), by the inverse CDF of
    `uniform` (...): int64. A category of weight 0 is never drawn; the weights need not be
    normalised."""
    cumulative = weights.cumsum(-1)
    target = uniform[..., None] * cumulative[..., -1:]
    return (target >= cumulative[..., :-1]).sum(-1)
