from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from langmani.datasets.frame_alignment import (
    FrameAlignmentError,
    RawRenderDigest,
    deterministic_video_sample_indices,
    rgb_quality_metrics,
)
from langmani.datasets.lerobot_source import RawEpisodeData


def _frame(value: int = 0, *, shape: tuple[int, int, int] = (4, 5, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _digest(frames: tuple[np.ndarray, ...]) -> str:
    digest = RawRenderDigest()
    for index, frame in enumerate(frames):
        digest.add(index, frame)
    return digest.hexdigest()


def test_t_actions_yield_exactly_t_pre_action_pairs_and_exclude_terminal_state() -> None:
    transition_count = 4
    actions = np.arange(transition_count * 8, dtype=np.float32).reshape(transition_count, 8)
    states = tuple(
        {"marker": np.asarray([index], dtype=np.float32)} for index in range(transition_count + 1)
    )
    episode = RawEpisodeData(
        record=SimpleNamespace(raw_trajectory_id="raw-fixture"),  # type: ignore[arg-type]
        actions=actions,
        _states=states,
    )

    pairs = tuple(episode.iter_training_pairs())

    assert episode.transition_count == transition_count
    assert episode.state_count == transition_count + 1
    assert episode.terminal_state_index == transition_count
    assert len(pairs) == transition_count
    for index, (state, action) in enumerate(pairs):
        np.testing.assert_array_equal(state["marker"], [index])
        np.testing.assert_array_equal(action, actions[index])
    np.testing.assert_array_equal(episode.state_at(transition_count)["marker"], [transition_count])
    with pytest.raises(IndexError, match=r"\[0, T-1\]"):
        episode.training_state_at(transition_count)

    pairs[0][0]["marker"][0] = -1
    pairs[0][1][0] = -1
    np.testing.assert_array_equal(episode.state_at(0)["marker"], [0])
    assert episode.actions[0, 0] == 0.0


def test_raw_render_digest_is_deterministic_for_identical_semantic_frames() -> None:
    first = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    noncontiguous = first[:, ::-1, :]
    contiguous = np.ascontiguousarray(noncontiguous)
    second = _frame(91, shape=(4, 6, 3))

    digest_a = _digest((noncontiguous, second))
    digest_b = _digest((contiguous, second.copy()))

    assert digest_a == digest_b
    assert len(digest_a) == 64
    assert set(digest_a) <= set("0123456789abcdef")


def test_raw_render_digest_changes_with_pixels_order_or_shape() -> None:
    first = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    changed = first.copy()
    changed[0, 0, 0] += 1
    reshaped = first.reshape(3, 2, 3)

    baseline = _digest((first, _frame(7, shape=(2, 3, 3))))

    assert _digest((changed, _frame(7, shape=(2, 3, 3)))) != baseline
    assert _digest((_frame(7, shape=(2, 3, 3)), first)) != baseline
    assert _digest((reshaped,)) != _digest((first,))


def test_raw_render_digest_requires_exact_sequential_indices() -> None:
    digest = RawRenderDigest()
    with pytest.raises(FrameAlignmentError, match="must be 0"):
        digest.add(1, _frame())
    digest.add(0, _frame())
    assert digest.frame_count == 1
    with pytest.raises(FrameAlignmentError, match="must be 1"):
        digest.add(0, _frame())
    assert digest.frame_count == 1


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (np.zeros((4, 5, 3), dtype=np.float32), FrameAlignmentError, "uint8"),
        (np.zeros((3, 4, 5), dtype=np.uint8), FrameAlignmentError, "HWC"),
        (np.zeros((4, 5, 4), dtype=np.uint8), FrameAlignmentError, "3 channels"),
        (np.zeros((4, 5), dtype=np.uint8), FrameAlignmentError, "HWC"),
        ([[[0, 0, 0]]], TypeError, "NumPy"),
    ],
)
def test_raw_render_digest_rejects_invalid_shape_or_dtype(
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        RawRenderDigest().add(0, value)  # type: ignore[arg-type]


def test_raw_render_digest_rejects_empty_sequence() -> None:
    with pytest.raises(FrameAlignmentError, match="empty"):
        RawRenderDigest().hexdigest()


def test_deterministic_samples_include_first_interior_and_final_frames() -> None:
    assert deterministic_video_sample_indices(101) == (0, 25, 50, 75, 100)
    assert deterministic_video_sample_indices(101, sample_count=3) == (0, 50, 100)
    assert deterministic_video_sample_indices(4) == (0, 1, 2, 3)
    assert deterministic_video_sample_indices(101) == deterministic_video_sample_indices(101)


@pytest.mark.parametrize(
    ("frame_count", "sample_count", "message"),
    [
        (0, 5, "positive integer"),
        (True, 5, "positive integer"),
        (5, 2, ">= 3"),
        (5, True, ">= 3"),
    ],
)
def test_deterministic_samples_reject_invalid_bounds(
    frame_count: object,
    sample_count: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        deterministic_video_sample_indices(  # type: ignore[arg-type]
            frame_count,
            sample_count=sample_count,  # type: ignore[arg-type]
        )


def test_rgb_quality_metrics_identical_frames_have_zero_mae_and_infinite_psnr() -> None:
    reference = _frame(37)

    mae, psnr = rgb_quality_metrics(reference, reference.copy())

    assert mae == 0.0
    assert math.isinf(psnr) and psnr > 0


def test_rgb_quality_metrics_match_known_constant_error() -> None:
    reference = _frame(0)
    decoded = _frame(10)

    mae, psnr = rgb_quality_metrics(reference, decoded)

    assert mae == pytest.approx(10.0)
    assert psnr == pytest.approx(10.0 * math.log10((255.0**2) / 100.0))


@pytest.mark.parametrize(
    ("reference", "decoded", "error", "message"),
    [
        (_frame().astype(np.float32), _frame(), FrameAlignmentError, "uint8"),
        (_frame(), np.zeros((4, 5), dtype=np.uint8), FrameAlignmentError, "HWC"),
        (_frame(), _frame(shape=(5, 4, 3)), FrameAlignmentError, "shapes differ"),
        ([[[0, 0, 0]]], _frame(), TypeError, "NumPy"),
    ],
)
def test_rgb_quality_metrics_reject_invalid_frames(
    reference: object,
    decoded: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        rgb_quality_metrics(reference, decoded)  # type: ignore[arg-type]
