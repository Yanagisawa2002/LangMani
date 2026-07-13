"""Raw RGB sequence identity and deterministic lossy-video quality helpers."""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

from langmani.datasets.identity import canonical_json


class FrameAlignmentError(ValueError):
    """Raised when a raw or decoded frame violates the M3B camera contract."""


class RawRenderDigest:
    """Incremental unambiguous digest of ordered uint8 HWC RGB frames."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"langmani-m3b-raw-rgb-sequence-v1\0")
        self._next_index = 0

    @property
    def frame_count(self) -> int:
        return self._next_index

    def add(self, frame_index: int, frame: np.ndarray) -> None:
        if frame_index != self._next_index:
            raise FrameAlignmentError(
                f"raw RGB frame index must be {self._next_index}, got {frame_index}"
            )
        if not isinstance(frame, np.ndarray):
            raise TypeError("raw RGB frame must be a NumPy array")
        if frame.dtype != np.dtype(np.uint8):
            raise FrameAlignmentError(f"raw RGB frame must use uint8, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise FrameAlignmentError(
                f"raw RGB frame must be HWC with 3 channels, got {frame.shape}"
            )
        contiguous = np.ascontiguousarray(frame)
        header = canonical_json(
            {
                "dtype": contiguous.dtype.str,
                "frame_index": frame_index,
                "shape": list(contiguous.shape),
            }
        ).encode("utf-8")
        payload = memoryview(contiguous).cast("B")
        self._digest.update(struct.pack(">Q", len(header)))
        self._digest.update(header)
        self._digest.update(struct.pack(">Q", payload.nbytes))
        self._digest.update(payload)
        self._next_index += 1

    def hexdigest(self) -> str:
        if self._next_index == 0:
            raise FrameAlignmentError("cannot finalize an empty raw RGB sequence")
        return self._digest.hexdigest()


def deterministic_video_sample_indices(
    frame_count: int, *, sample_count: int = 5
) -> tuple[int, ...]:
    """Return all short episodes or evenly spaced first/interior/final frames."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 3:
        raise ValueError("sample_count must be an integer >= 3")
    if frame_count <= sample_count:
        return tuple(range(frame_count))
    indices = {
        round(position * (frame_count - 1) / (sample_count - 1)) for position in range(sample_count)
    }
    # Rounding is unique when frame_count > sample_count, but keep this guard explicit.
    if len(indices) != sample_count:
        raise FrameAlignmentError("could not construct a unique deterministic video sample")
    return tuple(sorted(indices))


def rgb_quality_metrics(reference: np.ndarray, decoded: np.ndarray) -> tuple[float, float]:
    """Return mean absolute pixel error and PSNR for two uint8 HWC frames."""
    for label, value in (("reference", reference), ("decoded", decoded)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{label} frame must be a NumPy array")
        if value.dtype != np.dtype(np.uint8):
            raise FrameAlignmentError(f"{label} frame must use uint8")
        if value.ndim != 3 or value.shape[-1] != 3:
            raise FrameAlignmentError(f"{label} frame must be HWC RGB")
    if reference.shape != decoded.shape:
        raise FrameAlignmentError(
            f"video/reference shapes differ: {decoded.shape} != {reference.shape}"
        )
    reference_float = reference.astype(np.float64)
    decoded_float = decoded.astype(np.float64)
    difference = decoded_float - reference_float
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    psnr = math.inf if mse == 0 else float(10.0 * math.log10((255.0**2) / mse))
    return mae, psnr


__all__ = [
    "FrameAlignmentError",
    "RawRenderDigest",
    "deterministic_video_sample_indices",
    "rgb_quality_metrics",
]
