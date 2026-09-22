"""Tests for the RepairMind Edge perception pipeline."""

import cv2
import numpy as np
import pytest

from repairmind_edge.models import EvidenceType, LocalizationMethod
from repairmind_edge.perception import PCBPerception, PerceptionError


def _synthetic_pcb() -> np.ndarray:
    """Create a controlled synthetic perspective view of a PCB."""

    image = np.full((720, 960, 3), 25, dtype=np.uint8)

    board = np.array(
        [
            [180, 140],
            [790, 105],
            [835, 590],
            [130, 625],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(image, board, (45, 125, 70))
    cv2.polylines(
        image,
        [board.reshape((-1, 1, 2))],
        True,
        (235, 235, 235),
        8,
    )

    # Synthetic electronic components.
    cv2.rectangle(image, (300, 260), (470, 390), (30, 30, 30), -1)
    cv2.rectangle(image, (570, 210), (690, 300), (180, 180, 180), -1)

    for x in range(250, 750, 80):
        cv2.circle(image, (x, 500), 15, (200, 180, 60), -1)

    return image


def _synthetic_chamfered_pcb() -> np.ndarray:
    """Create a synthetic PCB with chamfered corners."""

    image = np.full((720, 960, 3), 25, dtype=np.uint8)

    # Eight-point PCB outline: deliberately not a quadrilateral.
    board = np.array(
        [
            [190, 120],
            [770, 120],
            [825, 175],
            [825, 535],
            [760, 610],
            [200, 610],
            [135, 535],
            [135, 180],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(image, board, (45, 125, 70))

    cv2.polylines(
        image,
        [board.reshape((-1, 1, 2))],
        True,
        (235, 235, 235),
        8,
    )

    # Synthetic components create realistic internal edges.
    cv2.rectangle(image, (280, 245), (470, 390), (30, 30, 30), -1)
    cv2.rectangle(image, (570, 205), (700, 305), (180, 180, 180), -1)

    for x in range(240, 760, 80):
        cv2.circle(image, (x, 500), 15, (200, 180, 60), -1)

    return image


def test_rejects_empty_image() -> None:
    perception = PCBPerception()
    empty = np.array([], dtype=np.uint8)

    with pytest.raises(PerceptionError):
        perception.inspect(empty)


def test_rejects_grayscale_input() -> None:
    perception = PCBPerception()
    grayscale = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(PerceptionError):
        perception.inspect(grayscale)


def test_measures_image_quality() -> None:
    perception = PCBPerception()
    image = _synthetic_pcb()

    output = perception.inspect(image)

    assert output.result.quality.width == 960
    assert output.result.quality.height == 720
    assert output.result.quality.channels == 3
    assert output.result.quality.brightness >= 0.0
    assert output.result.quality.sharpness >= 0.0


def test_localizes_synthetic_pcb() -> None:
    perception = PCBPerception()
    output = perception.inspect(_synthetic_pcb())

    assert output.result.localization.detected is True
    assert output.result.localization.corners is not None
    assert output.result.localization.area_ratio > 0.10
    assert output.result.localization.confidence >= 0.55

    evidence_types = {evidence.evidence_type for evidence in output.result.evidence}

    assert EvidenceType.IMAGE_QUALITY in evidence_types
    assert EvidenceType.BOARD_LOCALIZED in evidence_types
    assert EvidenceType.PERSPECTIVE_RECTIFIED in evidence_types


def test_rectifies_synthetic_pcb() -> None:
    perception = PCBPerception()
    output = perception.inspect(_synthetic_pcb())

    assert output.normalized_image.ndim == 3
    assert output.normalized_image.shape[2] == 3
    assert output.normalized_image.shape[0] > 300
    assert output.normalized_image.shape[1] > 500

    assert output.result.normalized_width == output.normalized_image.shape[1]
    assert output.result.normalized_height == output.normalized_image.shape[0]


def test_handles_missing_board_without_inventing_detection() -> None:
    perception = PCBPerception()
    blank = np.full((480, 640, 3), 30, dtype=np.uint8)

    output = perception.inspect(blank)

    assert output.result.localization.detected is False
    assert output.result.localization.corners is None

    evidence_types = {evidence.evidence_type for evidence in output.result.evidence}

    assert EvidenceType.BOARD_LOCALIZATION_UNCERTAIN in evidence_types

    assert np.array_equal(output.normalized_image, blank)


def test_localizes_chamfered_pcb() -> None:
    """A non-quadrilateral PCB should still be localizable."""

    perception = PCBPerception()
    output = perception.inspect(_synthetic_chamfered_pcb())

    assert output.result.localization.detected is True
    assert output.result.localization.method == LocalizationMethod.DOMINANT_CONTOUR
    assert output.result.localization.corners is not None
    assert output.result.localization.area_ratio > 0.10
