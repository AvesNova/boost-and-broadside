"""Write an RGB image as a PNG file (8 bits per channel, no filtering), for offscreen
renders, without an image library."""

import pathlib
import struct
import zlib

import numpy as np


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def write_png(path: pathlib.Path, pixels: np.ndarray) -> None:
    """pixels (height, width, 3) uint8, top row first."""
    height, width, _ = pixels.shape
    rows = np.concatenate([np.zeros((height, 1), np.uint8), pixels.reshape(height, -1)], 1)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
                     + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + _chunk(b"IEND", b""))
