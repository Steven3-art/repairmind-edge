"""Tests for the RepairMind Edge perception pipeline."""

import cv2
import numpy as np
import pytest

from repairmind_edge.models import (
    BoundingBox,
    EvidenceType,
    LocalizationMethod,
    Point,
    RegionOfInterest,
)
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


def test_cleans_source_contour_without_losing_board_geometry() -> None:
    """Contour cleanup should remove minor noise without distorting the PCB."""

    perception = PCBPerception()

    contour = np.array(
        [
            [100, 100],
            [300, 100],
            [500, 100],
            [505, 102],  # Minor boundary perturbation.
            [700, 100],
            [760, 160],
            [760, 500],
            [700, 560],
            [500, 560],
            [300, 560],
            [100, 560],
            [40, 500],
            [40, 160],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    raw_area = float(cv2.contourArea(contour))

    cleaned = perception._clean_source_contour(
        contour,
        image_width=800,
        image_height=700,
    )

    cleaned_area = float(cv2.contourArea(cleaned))
    area_drift_ratio = abs(cleaned_area - raw_area) / raw_area

    assert len(cleaned) < len(contour)
    assert len(cleaned) >= perception._MIN_SOURCE_CONTOUR_VERTICES
    assert area_drift_ratio <= perception._MAX_SOURCE_CONTOUR_AREA_DRIFT_RATIO


def test_source_contour_cleanup_removes_small_border_spike() -> None:
    """Contour cleanup should remove a small acute border spike."""

    perception = PCBPerception()

    contour = np.array(
        [
            [100, 100],
            [700, 100],
            [760, 160],
            [760, 500],
            [700, 560],
            [180, 560],
            [0, 540],  # Small acute parasite excursion.
            [100, 560],
            [40, 500],
            [40, 160],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    raw_area = float(cv2.contourArea(contour))

    cleaned = perception._clean_source_contour(
        contour,
        image_width=800,
        image_height=700,
    )

    cleaned_points = {tuple(map(int, point[0])) for point in cleaned}

    cleaned_area = float(cv2.contourArea(cleaned))
    area_drift_ratio = abs(cleaned_area - raw_area) / raw_area

    assert (0, 540) not in cleaned_points
    assert area_drift_ratio <= perception._MAX_SOURCE_CONTOUR_AREA_DRIFT_RATIO


def test_source_contour_cleanup_preserves_large_border_excursion() -> None:
    """Cleanup should not remove a border vertex when geometry changes too much."""

    perception = PCBPerception()

    contour = np.array(
        [
            [100, 100],
            [700, 100],
            [760, 160],
            [760, 500],
            [700, 560],
            [400, 560],
            [0, 430],  # Large geometry-changing excursion.
            [100, 560],
            [40, 500],
            [40, 160],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cleaned = perception._clean_source_contour(
        contour,
        image_width=800,
        image_height=700,
    )

    cleaned_points = {tuple(map(int, point[0])) for point in cleaned}

    assert (0, 430) in cleaned_points


def test_dense_contour_denoising_removes_local_excursion() -> None:
    """Dense contour denoising should remove a short local excursion."""

    perception = PCBPerception()

    contour = np.array(
        [
            [100, 100],
            [150, 100],
            [200, 100],
            [250, 100],
            [300, 100],
            [350, 100],
            [400, 100],
            [450, 100],
            [500, 100],
            [550, 100],
            [600, 100],
            [650, 100],
            [700, 100],
            [700, 150],
            [700, 200],
            [700, 250],
            [700, 300],
            [700, 350],
            [700, 400],
            [700, 450],
            [700, 500],
            [650, 500],
            [600, 500],
            [550, 500],
            [500, 500],
            [450, 500],
            [400, 500],
            [350, 500],
            [300, 500],
            [250, 500],
            [200, 500],
            [150, 500],
            [100, 500],
            [100, 450],
            [100, 400],
            [70, 375],  # Short parasite excursion.
            [100, 350],
            [100, 300],
            [100, 250],
            [100, 200],
            [100, 150],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cleaned = perception._denoise_dense_contour(contour)

    cleaned_points = {tuple(map(int, point[0])) for point in cleaned}

    assert (70, 375) not in cleaned_points
    assert (100, 400) in cleaned_points
    assert (100, 350) in cleaned_points

    assert len(cleaned) == len(contour) - 1


def test_dense_contour_denoising_preserves_structural_chamfer() -> None:
    """Dense contour denoising should preserve a genuine PCB chamfer."""

    perception = PCBPerception()

    contour = np.array(
        [
            [100, 100],
            [150, 100],
            [200, 100],
            [250, 100],
            [300, 100],
            [350, 100],
            [400, 100],
            [450, 100],
            [500, 100],
            [550, 100],
            [600, 100],
            [650, 100],
            [700, 100],
            [700, 150],
            [700, 200],
            [700, 250],
            [700, 300],
            [700, 350],
            [700, 400],
            [700, 450],
            [700, 500],
            [650, 550],  # Genuine lower-right chamfer.
            [600, 600],
            [550, 600],
            [500, 600],
            [450, 600],
            [400, 600],
            [350, 600],
            [300, 600],
            [250, 600],
            [200, 600],
            [150, 600],
            [100, 550],  # Genuine lower-left chamfer.
            [100, 500],
            [100, 450],
            [100, 400],
            [100, 350],
            [100, 300],
            [100, 250],
            [100, 200],
            [100, 150],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cleaned = perception._denoise_dense_contour(contour)

    cleaned_points = {tuple(map(int, point[0])) for point in cleaned}

    assert (650, 550) in cleaned_points
    assert (600, 600) in cleaned_points
    assert (150, 600) in cleaned_points
    assert (100, 550) in cleaned_points

    assert len(cleaned) == len(contour)


def test_dense_contour_denoising_rejects_local_parasite_path() -> None:
    """Dense contour denoising should reject a multi-point parasite path."""

    perception = PCBPerception()

    contour = np.array(
        [
            [100, 100],
            [200, 100],
            [300, 100],
            [400, 100],
            [500, 100],
            [600, 100],
            [700, 100],
            [700, 200],
            [700, 300],
            [700, 400],
            [700, 500],
            [600, 500],
            [500, 500],
            [400, 500],
            [300, 500],
            [200, 500],
            [100, 500],
            # Expected PCB boundary would continue near x=100.
            # Instead, the observed contour leaves the board,
            # follows a parasite path, then returns.
            [100, 450],
            [70, 440],
            [45, 420],
            [20, 390],
            [5, 350],
            [25, 315],
            [50, 290],
            [75, 275],
            [100, 250],
            [100, 200],
            [100, 150],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cleaned = perception._reject_local_path_excursions(contour)

    cleaned_points = {tuple(map(int, point[0])) for point in cleaned}

    parasite_points = {
        (70, 440),
        (45, 420),
        (20, 390),
        (5, 350),
        (25, 315),
        (50, 290),
        (75, 275),
    }

    assert parasite_points.isdisjoint(cleaned_points)

    # The two physical anchor points must survive.
    assert (100, 450) in cleaned_points
    assert (100, 250) in cleaned_points

    # Unrelated PCB geometry must remain untouched.
    assert (700, 100) in cleaned_points
    assert (700, 500) in cleaned_points
    assert (100, 500) in cleaned_points


def test_local_path_excursion_rejection_preserves_structural_chamfer() -> None:
    """Local path rejection must preserve a legitimate PCB chamfer."""

    perception = PCBPerception()

    contour = np.array(
        [
            [100, 100],
            [200, 100],
            [300, 100],
            [400, 100],
            [500, 100],
            [600, 100],
            [700, 100],
            [700, 200],
            [700, 300],
            [700, 400],
            [700, 500],
            # Legitimate lower-right PCB chamfer.
            [680, 530],
            [650, 560],
            [620, 585],
            [600, 600],
            [500, 600],
            [400, 600],
            [300, 600],
            [200, 600],
            # Legitimate lower-left PCB chamfer.
            [180, 585],
            [150, 560],
            [120, 530],
            [100, 500],
            [100, 400],
            [100, 300],
            [100, 200],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cleaned = perception._reject_local_path_excursions(contour)

    np.testing.assert_array_equal(cleaned, contour)


def test_board_region_segmentation_isolates_coherent_foreground() -> None:
    """Board segmentation should isolate the coherent PCB foreground region."""

    perception = PCBPerception()

    image = np.full(
        (600, 800, 3),
        110,
        dtype=np.uint8,
    )

    # Synthetic PCB body.
    board_polygon = np.array(
        [
            [180, 100],
            [620, 100],
            [680, 450],
            [600, 520],
            [200, 520],
            [120, 450],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        image,
        board_polygon,
        (45, 145, 70),
    )

    # Internal PCB details. These must remain part of the board region
    # despite having colors different from the PCB substrate.
    cv2.rectangle(
        image,
        (300, 210),
        (500, 350),
        (35, 35, 35),
        thickness=-1,
    )
    cv2.circle(
        image,
        (240, 180),
        22,
        (190, 190, 190),
        thickness=-1,
    )

    # External visual clutter deliberately disconnected from the PCB.
    cv2.line(
        image,
        (20, 350),
        (100, 480),
        (40, 150, 75),
        thickness=14,
    )
    cv2.rectangle(
        image,
        (20, 520),
        (130, 580),
        (50, 140, 70),
        thickness=-1,
    )

    coarse_contour = board_polygon.reshape((-1, 1, 2))

    board_mask = perception._segment_board_region(
        image,
        coarse_contour,
    )

    assert board_mask.shape == image.shape[:2]
    assert board_mask.dtype == np.uint8

    # PCB substrate.
    assert board_mask[150, 400] == 255

    # Dark component inside the PCB must still belong to the board region.
    assert board_mask[280, 400] == 255

    # Background must remain excluded.
    assert board_mask[50, 50] == 0

    # External clutter must not be absorbed into the PCB.
    assert board_mask[550, 70] == 0


def test_segmented_board_contour_selects_main_region() -> None:
    """Segmented contour extraction should select the main PCB region."""

    perception = PCBPerception()

    mask = np.zeros(
        (600, 800),
        dtype=np.uint8,
    )

    board_polygon = np.array(
        [
            [180, 100],
            [620, 100],
            [680, 450],
            [600, 520],
            [200, 520],
            [120, 450],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        mask,
        board_polygon,
        255,
    )

    # Independent foreground contamination that must not become
    # part of the extracted PCB boundary.
    cv2.rectangle(
        mask,
        (20, 500),
        (100, 580),
        255,
        thickness=-1,
    )

    contour = perception._extract_segmented_board_contour(mask)

    assert contour is not None

    contour_area = float(cv2.contourArea(contour))
    expected_area = float(cv2.contourArea(board_polygon))

    assert expected_area > 0.0
    assert abs(contour_area - expected_area) / expected_area < 0.02

    x, y, width, height = cv2.boundingRect(contour)

    # The selected component must correspond to the PCB, not the
    # smaller foreground contaminant.
    assert x >= 100
    assert y >= 90
    assert width >= 500
    assert height >= 400


def test_board_mask_refinement_removes_narrow_attached_region() -> None:
    """Board mask refinement should reject a narrow attached region."""

    perception = PCBPerception()

    mask = np.zeros(
        (700, 800),
        dtype=np.uint8,
    )

    # Main PCB body with legitimate lower chamfers.
    board_polygon = np.array(
        [
            [180, 80],
            [620, 80],
            [680, 430],
            [600, 540],
            [200, 540],
            [120, 430],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        mask,
        board_polygon,
        255,
    )

    # Narrow attachment below the PCB.
    cv2.rectangle(
        mask,
        (385, 540),
        (415, 590),
        255,
        thickness=-1,
    )

    # Larger parasite connected only through the narrow neck.
    cv2.rectangle(
        mask,
        (330, 590),
        (470, 670),
        255,
        thickness=-1,
    )

    refined = perception._refine_board_mask(mask)

    assert refined.shape == mask.shape
    assert refined.dtype == np.uint8
    assert set(np.unique(refined)).issubset({0, 255})

    # Main PCB body must survive.
    assert refined[300, 400] == 255

    # Legitimate lower PCB geometry must survive.
    assert refined[520, 250] == 255
    assert refined[520, 550] == 255

    # Narrow attachment and its parasite must be rejected.
    assert refined[565, 400] == 0
    assert refined[630, 400] == 0

    # Background must remain background.
    assert refined[650, 100] == 0

    # Refinement must not mutate its input.
    assert mask[565, 400] == 255
    assert mask[630, 400] == 255


def test_board_mask_refinement_removes_irregular_narrow_attachment() -> None:
    """Board mask refinement should reject an irregular narrow attachment."""

    perception = PCBPerception()

    mask = np.zeros(
        (700, 800),
        dtype=np.uint8,
    )

    # Main PCB body with legitimate lower chamfers.
    board_polygon = np.array(
        [
            [180, 80],
            [620, 80],
            [680, 430],
            [600, 540],
            [200, 540],
            [120, 430],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        mask,
        board_polygon,
        255,
    )

    # Irregular narrow bridge attached to the lower PCB boundary.
    bridge_polygon = np.array(
        [
            [370, 538],
            [402, 540],
            [414, 555],
            [407, 570],
            [425, 585],
            [416, 602],
            [382, 600],
            [374, 582],
            [383, 566],
            [368, 552],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        [bridge_polygon],
        255,
    )

    # Larger irregular foreground region connected through the bridge.
    parasite_polygon = np.array(
        [
            [350, 595],
            [430, 590],
            [475, 620],
            [460, 670],
            [335, 675],
            [310, 640],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        [parasite_polygon],
        255,
    )

    refined = perception._refine_board_mask(mask)

    # Main PCB body must remain.
    assert refined[300, 400] == 255

    # Legitimate lower-board geometry must remain.
    assert refined[520, 250] == 255
    assert refined[520, 550] == 255

    # Irregular bridge must be removed.
    assert refined[570, 395] == 0

    # Attached parasite must be removed.
    assert refined[640, 390] == 0

    # Refinement must remain conservative on PCB area.
    original_board_area = float(cv2.contourArea(board_polygon))
    refined_contour = perception._extract_segmented_board_contour(refined)

    assert refined_contour is not None

    refined_area = float(cv2.contourArea(refined_contour))
    area_drift = abs(refined_area - original_board_area) / original_board_area

    assert area_drift < 0.03


def test_terminal_foreground_extension_is_rejected_from_board_body() -> None:
    """Board envelope should reject a broad terminal foreground extension."""

    perception = PCBPerception()

    mask = np.zeros(
        (800, 800),
        dtype=np.uint8,
    )

    # Main PCB body.
    board_polygon = np.array(
        [
            [150, 80],
            [650, 80],
            [680, 420],
            [600, 650],
            [200, 650],
            [120, 420],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        mask,
        board_polygon,
        255,
    )

    # Broad connection deliberately designed to survive ordinary
    # morphological opening.
    cv2.rectangle(
        mask,
        (330, 645),
        (470, 705),
        255,
        thickness=-1,
    )

    # Terminal foreground extension below the physical PCB body.
    extension_polygon = np.array(
        [
            [300, 700],
            [500, 700],
            [530, 770],
            [270, 770],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        mask,
        extension_polygon,
        255,
    )

    cleaned = perception._reject_terminal_foreground_extension(mask)

    assert cleaned.shape == mask.shape
    assert cleaned.dtype == np.uint8
    assert set(np.unique(cleaned)).issubset({0, 255})

    # Main board body must remain.
    assert cleaned[300, 400] == 255
    assert cleaned[600, 400] == 255

    # Legitimate lower PCB geometry must remain.
    assert cleaned[640, 250] == 255
    assert cleaned[640, 550] == 255

    # Broad connection below the physical board termination must go.
    assert cleaned[680, 400] == 0

    # Terminal extension must go.
    assert cleaned[740, 400] == 0

    # Input must remain untouched.
    assert mask[680, 400] == 255
    assert mask[740, 400] == 255


def test_guided_boundary_fusion_recovers_detail_and_rejects_excursion() -> None:
    """Boundary fusion should recover nearby detail and reject distant noise."""

    perception = PCBPerception()

    reliable_mask = np.zeros(
        (600, 800),
        dtype=np.uint8,
    )

    # Conservative segmented board body.
    reliable_board = np.array(
        [
            [180, 100],
            [620, 100],
            [660, 430],
            [590, 520],
            [210, 520],
            [140, 430],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        reliable_mask,
        reliable_board,
        255,
    )

    # Observed boundary contains:
    #
    # 1. a small legitimate local feature near the upper-left corner,
    #    intentionally missing from the conservative mask;
    #
    # 2. a large distant excursion on the lower-left side that must
    #    not survive the fusion.
    observed_contour = np.array(
        [
            # Upper-left reliable anchor.
            [180, 100],
            # Legitimate local detail:
            # leaves the conservative boundary and returns nearby.
            [165, 92],
            [150, 100],
            [160, 115],
            [180, 120],
            # Reliable upper/right/lower boundary.
            [620, 100],
            [660, 430],
            [590, 520],
            [210, 520],
            # First reliable anchor before the parasite excursion.
            [165, 470],
            [145, 445],
            # Large parasite excursion.
            [100, 500],
            [45, 470],
            [25, 420],
            [80, 390],
            # Return to the reliable board boundary.
            [140, 430],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    fused = perception._fuse_observed_and_segmented_boundary(
        observed_contour,
        reliable_mask,
    )

    assert fused is not None

    detail_signed_distance = float(
        cv2.pointPolygonTest(
            fused,
            (150.0, 100.0),
            True,
        )
    )

    assert detail_signed_distance >= 0.0

    # The distant excursion must remain outside the fused board region.
    parasite_points = [
        (100.0, 500.0),
        (45.0, 470.0),
        (25.0, 420.0),
    ]

    for parasite_point in parasite_points:
        parasite_signed_distance = float(
            cv2.pointPolygonTest(
                fused,
                parasite_point,
                True,
            )
        )

        assert parasite_signed_distance < 0.0

    # The fused boundary must still describe a substantial board.
    fused_area = float(cv2.contourArea(fused))
    reliable_area = float(cv2.contourArea(reliable_board))

    assert fused_area > 0.0
    assert reliable_area > 0.0

    area_drift = abs(fused_area - reliable_area) / reliable_area

    assert area_drift < 0.10


def test_guided_boundary_fusion_creates_connected_recovery_region() -> None:
    """Boundary fusion should create a connected local recovery region."""

    perception = PCBPerception()

    reliable_mask = np.zeros(
        (600, 800),
        dtype=np.uint8,
    )

    reliable_board = np.array(
        [
            [180, 100],
            [620, 100],
            [660, 430],
            [590, 520],
            [210, 520],
            [140, 430],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        reliable_mask,
        reliable_board,
        255,
    )

    # The observed boundary contains one legitimate outward feature
    # near the upper-left corner and one distant parasite excursion.
    observed_contour = np.array(
        [
            [180, 100],
            # Legitimate local PCB geometry missing from the
            # conservative segmented mask.
            [165, 92],
            [150, 100],
            [160, 115],
            [180, 120],
            [620, 100],
            [660, 430],
            [590, 520],
            [210, 520],
            [165, 470],
            [145, 445],
            # Distant background excursion that must not be fused.
            [100, 500],
            [45, 470],
            [25, 420],
            [80, 390],
            [140, 430],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    reliable_contour = perception._extract_segmented_board_contour(reliable_mask)

    assert reliable_contour is not None

    fused_contour = perception._fuse_observed_and_segmented_boundary(
        observed_contour,
        reliable_mask,
    )

    assert fused_contour is not None

    reliable_area = float(cv2.contourArea(reliable_contour))
    fused_area = float(cv2.contourArea(fused_contour))

    # The recovered detail lies outside the conservative mask.
    assert reliable_mask[100, 150] == 0

    # The recovered boundary must remain connected to the main board region.
    # the local observed feature must become part of the fused
    # external geometry rather than merely an isolated polyline.
    recovered_detail_distance = float(
        cv2.pointPolygonTest(
            fused_contour,
            (150.0, 100.0),
            True,
        )
    )

    assert recovered_detail_distance >= 0.0

    # Recovering a genuine outward feature must increase the
    # external board area.
    assert fused_area > reliable_area

    # The distant parasite must remain outside the fused board.
    parasite_points = (
        (100.0, 500.0),
        (45.0, 470.0),
        (25.0, 420.0),
    )

    for point in parasite_points:
        assert (
            cv2.pointPolygonTest(
                fused_contour,
                point,
                False,
            )
            < 0.0
        )

    # Rasterize the final contour as a filled region and verify
    # that the resulting board hypothesis is a single component.
    fused_region = np.zeros_like(reliable_mask)

    cv2.drawContours(
        fused_region,
        [fused_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    component_count, _labels = cv2.connectedComponents(fused_region)

    # connectedComponents includes background label 0.
    assert component_count - 1 == 1


def test_guided_boundary_fusion_connects_nearby_detached_anchors() -> None:
    """Fusion should connect accepted detached anchors to the reliable region."""

    perception = PCBPerception()

    reliable_mask = np.zeros(
        (500, 700),
        dtype=np.uint8,
    )

    cv2.rectangle(
        reliable_mask,
        (200, 120),
        (600, 400),
        255,
        thickness=-1,
    )

    # The local feature is deliberately detached from the reliable mask.
    #
    # The two endpoints at x=194 are only 6 px from the reliable
    # boundary at x=200, so they qualify as geometric anchors.
    #
    # The excursion reaches x=180, i.e. 20 px from the reliable mask:
    # outside the anchor band, but still below the maximum accepted
    # excursion for this geometry.
    observed_contour = np.array(
        [
            [200, 120],
            # Detached but geometrically acceptable recovery run.
            [194, 140],
            [186, 135],
            [180, 150],
            [186, 165],
            [194, 160],
            [200, 180],
            [600, 180],
            [600, 400],
            [200, 400],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    reliable_contour = perception._extract_segmented_board_contour(reliable_mask)

    assert reliable_contour is not None

    reliable_area = float(cv2.contourArea(reliable_contour))

    # Explicitly verify the topology represented by this fixture.
    assert reliable_mask[140, 194] == 0
    assert reliable_mask[160, 194] == 0
    assert reliable_mask[150, 180] == 0

    fused_contour = perception._fuse_observed_and_segmented_boundary(
        observed_contour,
        reliable_mask,
    )

    assert fused_contour is not None

    fused_area = float(cv2.contourArea(fused_contour))

    feature_distance = float(
        cv2.pointPolygonTest(
            fused_contour,
            (180.0, 150.0),
            True,
        )
    )

    # The accepted detached feature must be connected back to the
    # reliable PCB and become part of its external geometry.
    assert feature_distance >= 0.0

    assert fused_area > reliable_area

    # The final hypothesis must remain one coherent PCB region.
    fused_region = np.zeros_like(reliable_mask)

    cv2.drawContours(
        fused_region,
        [fused_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    component_count, _labels = cv2.connectedComponents(fused_region)

    assert component_count - 1 == 1


def test_guided_boundary_fusion_rejects_degenerate_same_projection() -> None:
    """Fusion should reject a recovery whose anchors share one projection."""

    perception = PCBPerception()

    reliable_mask = np.zeros(
        (500, 700),
        dtype=np.uint8,
    )

    cv2.rectangle(
        reliable_mask,
        (200, 120),
        (600, 400),
        255,
        thickness=-1,
    )

    reliable_contour = perception._extract_segmented_board_contour(reliable_mask)

    assert reliable_contour is not None

    reliable_area = float(cv2.contourArea(reliable_contour))

    # Both nearby anchors lie around the same reliable corner.
    # Their nearest reliable contour projection is expected to
    # collapse onto the same local boundary point.
    observed_contour = np.array(
        [
            [200, 120],
            [194, 114],
            [188, 110],
            [186, 116],
            [194, 118],
            [200, 130],
            [600, 130],
            [600, 400],
            [200, 400],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    fused_contour = perception._fuse_observed_and_segmented_boundary(
        observed_contour,
        reliable_mask,
    )

    assert fused_contour is not None

    fused_area = float(cv2.contourArea(fused_contour))

    # A degenerate recovery must not alter the reliable board.
    assert fused_area == pytest.approx(reliable_area)

    # The artificial excursion must remain outside.
    assert (
        cv2.pointPolygonTest(
            fused_contour,
            (188.0, 110.0),
            False,
        )
        < 0.0
    )


def test_roi_extraction_produces_deterministic_visual_regions() -> None:
    """Meaningful board structures should receive stable ROI proposals."""

    board = np.full(
        (480, 640, 3),
        (45, 125, 70),
        dtype=np.uint8,
    )

    # Three deliberately distinct visual structures.
    structures = (
        (80, 70, 210, 180),
        (380, 80, 530, 170),
        (220, 300, 420, 390),
    )

    for x1, y1, x2, y2 in structures:
        cv2.rectangle(
            board,
            (x1, y1),
            (x2, y2),
            (25, 25, 25),
            -1,
        )

    # Give the second structure a different appearance while preserving
    # the same geometric test contract.
    cv2.rectangle(
        board,
        (380, 80),
        (530, 170),
        (190, 190, 190),
        -1,
    )

    # Tiny isolated structures must not dominate useful proposals.
    cv2.circle(
        board,
        (40, 430),
        2,
        (255, 255, 255),
        -1,
    )
    cv2.circle(
        board,
        (600, 430),
        2,
        (255, 255, 255),
        -1,
    )

    first_regions = PCBPerception._extract_regions(board)
    second_regions = PCBPerception._extract_regions(board)

    assert first_regions == second_regions

    assert 3 <= len(first_regions) <= PCBPerception._ROI_MAX_PROPOSALS

    assert [region.roi_id for region in first_regions] == [
        f"roi_{index:03d}" for index in range(1, len(first_regions) + 1)
    ]

    for region in first_regions:
        assert isinstance(region, RegionOfInterest)

        box = region.bounding_box

        assert box.width > 0
        assert box.height > 0
        assert box.x + box.width <= board.shape[1]
        assert box.y + box.height <= board.shape[0]

        assert 0.0 < region.area_ratio < 1.0
        assert 0.0 < region.edge_density <= 1.0
        assert region.local_contrast > 0.0
        assert 0.0 <= region.confidence <= 1.0

    def structure_is_covered(
        structure: tuple[int, int, int, int],
    ) -> bool:
        x1, y1, x2, y2 = structure

        structure_area = float((x2 - x1) * (y2 - y1))

        for region in first_regions:
            box = region.bounding_box

            intersection_left = max(
                x1,
                box.x,
            )
            intersection_top = max(
                y1,
                box.y,
            )
            intersection_right = min(
                x2,
                box.x + box.width,
            )
            intersection_bottom = min(
                y2,
                box.y + box.height,
            )

            intersection_width = max(
                0,
                intersection_right - intersection_left,
            )
            intersection_height = max(
                0,
                intersection_bottom - intersection_top,
            )

            intersection_area = float(intersection_width * intersection_height)

            coverage_ratio = intersection_area / structure_area

            if coverage_ratio >= 0.05:
                return True

        return False

    assert all(structure_is_covered(structure) for structure in structures)


def test_region_evidence_preserves_roi_observations() -> None:
    """ROI evidence should preserve measured visual observations."""

    regions = (
        RegionOfInterest(
            roi_id="roi_001",
            bounding_box=BoundingBox(
                x=40,
                y=60,
                width=120,
                height=80,
            ),
            center=Point(
                x=100.0,
                y=100.0,
            ),
            area_ratio=0.05,
            edge_density=0.24,
            local_contrast=48.5,
            confidence=0.82,
        ),
        RegionOfInterest(
            roi_id="roi_002",
            bounding_box=BoundingBox(
                x=220,
                y=180,
                width=90,
                height=70,
            ),
            center=Point(
                x=265.0,
                y=215.0,
            ),
            area_ratio=0.032,
            edge_density=0.31,
            local_contrast=61.0,
            confidence=0.91,
        ),
    )

    first_evidence = PCBPerception._build_region_evidence(regions)
    second_evidence = PCBPerception._build_region_evidence(regions)

    assert first_evidence == second_evidence
    assert len(first_evidence) == len(regions)

    assert [item.evidence_id for item in first_evidence] == [
        "evidence_roi_001",
        "evidence_roi_002",
    ]

    for region, item in zip(
        regions,
        first_evidence,
        strict=True,
    ):
        box = region.bounding_box

        assert item.evidence_type == EvidenceType.VISUAL_REGION
        assert item.source_roi_id == region.roi_id
        assert item.confidence == region.confidence

        assert item.description == (
            "Visually active inspection region detected on the normalized PCB."
        )

        assert item.measurements == {
            "x": box.x,
            "y": box.y,
            "width": box.width,
            "height": box.height,
            "center_x": region.center.x,
            "center_y": region.center.y,
            "area_ratio": region.area_ratio,
            "edge_density": region.edge_density,
            "local_contrast": region.local_contrast,
            "proposal_score": region.confidence,
        }


def test_inspection_exposes_visual_region_evidence() -> None:
    """Successful inspection should expose ROI evidence."""

    perception = PCBPerception()

    output = perception.inspect(_synthetic_pcb())

    assert output.result.localization.detected

    region_evidence = tuple(
        item for item in output.result.evidence if item.evidence_type == EvidenceType.VISUAL_REGION
    )

    assert region_evidence

    assert len(region_evidence) <= PCBPerception._ROI_MAX_PROPOSALS

    assert [item.source_roi_id for item in region_evidence] == [
        f"roi_{index:03d}"
        for index in range(
            1,
            len(region_evidence) + 1,
        )
    ]

    for item in region_evidence:
        assert item.source_roi_id is not None
        assert item.evidence_id == (f"evidence_{item.source_roi_id}")

        assert "x" in item.measurements
        assert "y" in item.measurements
        assert "width" in item.measurements
        assert "height" in item.measurements
        assert "edge_density" in item.measurements
        assert "local_contrast" in item.measurements
        assert "proposal_score" in item.measurements
