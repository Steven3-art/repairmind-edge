"""OpenCV-based perception pipeline for RepairMind Edge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from repairmind_edge.models import (
    BoardLocalization,
    EvidenceType,
    ImageQuality,
    InspectionResult,
    LocalizationMethod,
    Point,
    VisualEvidence,
)

ImageArray = NDArray[np.uint8]


class PerceptionError(ValueError):
    """Raised when an input cannot be processed as a valid image."""


@dataclass(frozen=True)
class PerceptionOutput:
    """Complete perception output including structured and visual artifacts."""

    result: InspectionResult
    normalized_image: ImageArray
    annotated_image: ImageArray
    debug_image: ImageArray


@dataclass(frozen=True)
class _BoardCandidate:
    """Internal representation of a possible PCB boundary."""

    source_contour: cv2.typing.MatLike
    corners: NDArray[np.float32]
    area_ratio: float
    confidence: float
    method: LocalizationMethod

@dataclass(frozen=True)
class _ContourDiagnostic:
    """Diagnostic information for a contour considered during PCB localization."""

    contour: cv2.typing.MatLike
    area_ratio: float
    vertex_count: int

class PCBPerception:
    """Detect, measure, and normalize PCB observations with OpenCV."""

    def __init__(
        self,
        *,
        minimum_board_area_ratio: float = 0.10,
        minimum_confidence: float = 0.55,
    ) -> None:
        if not 0.0 < minimum_board_area_ratio < 1.0:
            raise ValueError("minimum_board_area_ratio must be between 0 and 1")

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")

        self._minimum_board_area_ratio = minimum_board_area_ratio
        self._minimum_confidence = minimum_confidence

    def inspect(self, image: ImageArray) -> PerceptionOutput:
        """Inspect an image and return structured visual evidence."""

        validated = self._validate_image(image)
        quality = self._measure_quality(validated)

        evidence: list[VisualEvidence] = [
            VisualEvidence(
                evidence_id="VE-0001",
                evidence_type=EvidenceType.IMAGE_QUALITY,
                confidence=1.0,
                description="Objective input image quality measurements.",
                measurements={
                    "width": quality.width,
                    "height": quality.height,
                    "channels": quality.channels,
                    "brightness": quality.brightness,
                    "sharpness": quality.sharpness,
                },
            )
        ]

        candidate, diagnostics = self._find_board(validated)
        debug_image = self._render_contour_diagnostics(
            validated,
            diagnostics,
        )

        if candidate is None or candidate.confidence < self._minimum_confidence:
            localization = BoardLocalization(
                detected=False,
                method=None,
                confidence=0.0 if candidate is None else candidate.confidence,
                area_ratio=0.0 if candidate is None else candidate.area_ratio,
                corners=None,
            )

            evidence.append(
                VisualEvidence(
                    evidence_id="VE-0002",
                    evidence_type=EvidenceType.BOARD_LOCALIZATION_UNCERTAIN,
                    confidence=localization.confidence,
                    description=(
                        "A reliable PCB boundary could not be established. "
                        "A clearer complete-board observation may be required."
                    ),
                    measurements={
                        "area_ratio": localization.area_ratio,
                    },
                )
            )

            normalized = validated.copy()
            annotated = validated.copy()

        else:
            ordered_corners = self._order_corners(candidate.corners)
            normalized = self._rectify(validated, ordered_corners)

            points: tuple[Point, Point, Point, Point] = (
                Point(x=float(ordered_corners[0][0]), y=float(ordered_corners[0][1])),
                Point(x=float(ordered_corners[1][0]), y=float(ordered_corners[1][1])),
                Point(x=float(ordered_corners[2][0]), y=float(ordered_corners[2][1])),
                Point(x=float(ordered_corners[3][0]), y=float(ordered_corners[3][1])),
            )

            localization = BoardLocalization(
                detected=True,
                method=candidate.method,
                confidence=candidate.confidence,
                area_ratio=candidate.area_ratio,
                corners=points,
            )

            if candidate.method == LocalizationMethod.QUADRILATERAL:
                localization_description = (
                    "PCB boundary localized from an observed quadrilateral contour."
                )
            else:
                localization_description = (
                    "PCB boundary localized from a dominant non-quadrilateral contour."
                )

            evidence.append(
                VisualEvidence(
                    evidence_id="VE-0002",
                    evidence_type=EvidenceType.BOARD_LOCALIZED,
                    confidence=candidate.confidence,
                    description=localization_description,
                    measurements={
                        "area_ratio": candidate.area_ratio,
                        "corners": [
                            {"x": point.x, "y": point.y}
                            for point in points
                        ],
                    },
                )
            )

            evidence.append(
                VisualEvidence(
                    evidence_id="VE-0003",
                    evidence_type=EvidenceType.PERSPECTIVE_RECTIFIED,
                    confidence=candidate.confidence,
                    description="PCB perspective normalized from detected board corners.",
                    measurements={
                        "normalized_width": int(normalized.shape[1]),
                        "normalized_height": int(normalized.shape[0]),
                    },
                )
            )

            annotated = self._annotate_board(
                validated,
                candidate.source_contour,
            )

        result = InspectionResult(
            quality=quality,
            localization=localization,
            evidence=tuple(evidence),
            normalized_width=int(normalized.shape[1]),
            normalized_height=int(normalized.shape[0]),
        )

        return PerceptionOutput(
            result=result,
            normalized_image=normalized,
            annotated_image=annotated,
            debug_image=debug_image,
        )

    @staticmethod
    def _validate_image(image: ImageArray) -> ImageArray:
        if not isinstance(image, np.ndarray):
            raise PerceptionError("Input must be a NumPy image array.")

        if image.size == 0:
            raise PerceptionError("Input image is empty.")

        if image.dtype != np.uint8:
            raise PerceptionError("Input image must use uint8 pixels.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise PerceptionError(
                "Input image must be a three-channel BGR image."
            )

        return image

    @staticmethod
    def _measure_quality(image: ImageArray) -> ImageQuality:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return ImageQuality(
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            channels=int(image.shape[2]),
            brightness=brightness,
            sharpness=max(0.0, sharpness),
        )

    def _find_board(
        self,
        image: ImageArray,
    ) -> tuple[_BoardCandidate | None, tuple[_ContourDiagnostic, ...]]:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = float(image.shape[0] * image.shape[1])

        diagnostics = self._build_contour_diagnostics(
            contours=contours,
            image_area=image_area,
        )

        best: _BoardCandidate | None = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area

            if area_ratio < self._minimum_board_area_ratio:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0:
                continue

            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(polygon) != 4:
                continue

            if not cv2.isContourConvex(polygon):
                continue

            corners = polygon.reshape(4, 2).astype(np.float32)

            rectangularity = self._rectangularity(corners, area)
            coverage_score = min(
                1.0,
                area_ratio / max(self._minimum_board_area_ratio * 3.0, 0.30),
            )

            confidence = float(
                np.clip(
                    (0.65 * rectangularity) + (0.35 * coverage_score),
                    0.0,
                    1.0,
                )
            )

            candidate = _BoardCandidate(
                corners=corners,
                source_contour=contour,
                area_ratio=area_ratio,
                confidence=confidence,
                method=LocalizationMethod.QUADRILATERAL,

            )

            if best is None or candidate.confidence > best.confidence:
                best = candidate

        if best is not None:
            return best, diagnostics

        dominant_candidate = self._find_dominant_contour(
            contours=contours,
            image_area=image_area,
        )

        return dominant_candidate, diagnostics


    def _find_dominant_contour(
        self,
        contours: Sequence[cv2.typing.MatLike],
        image_area: float,
    ) -> _BoardCandidate | None:
        """Derive board geometry from a dominant non-quadrilateral contour."""

        best: _BoardCandidate | None = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area

            if area_ratio < self._minimum_board_area_ratio:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0.0:
                continue

            polygon = cv2.approxPolyDP(
                contour,
                0.015 * perimeter,
                True,
            )

            # Quadrilaterals belong to the primary strategy.
            if len(polygon) <= 4:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))

            if hull_area <= 0.0:
                continue

            solidity = float(
                np.clip(
                    area / hull_area,
                    0.0,
                    1.0,
                )
            )

            rectangle = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rectangle).astype(np.float32)

            rectangle_width, rectangle_height = rectangle[1]
            rectangle_area = float(rectangle_width * rectangle_height)

            if rectangle_area <= 0.0:
                continue

            rectangularity = float(
                np.clip(
                    area / rectangle_area,
                    0.0,
                    1.0,
                )
            )

            coverage_score = min(
                1.0,
                area_ratio / max(
                    self._minimum_board_area_ratio * 3.0,
                    0.30,
                ),
            )

            confidence = float(
                np.clip(
                    (0.45 * solidity)
                    + (0.35 * rectangularity)
                    + (0.20 * coverage_score),
                    0.0,
                    1.0,
                )
            )

            candidate = _BoardCandidate(
                corners=box,
                source_contour=contour,
                area_ratio=area_ratio,
                confidence=confidence,
                method=LocalizationMethod.DOMINANT_CONTOUR,
            )

            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best




    def _build_contour_diagnostics(
        self,
        contours: Sequence[cv2.typing.MatLike],
        image_area: float,
    ) -> tuple[_ContourDiagnostic, ...]:
        """Build diagnostics for contours large enough to be localization candidates."""

        diagnostics: list[_ContourDiagnostic] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area

            if area_ratio < self._minimum_board_area_ratio:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0.0:
                continue

            polygon = cv2.approxPolyDP(
                contour,
                0.015 * perimeter,
                True,
            )

            diagnostics.append(
                _ContourDiagnostic(
                    contour=contour,
                    area_ratio=area_ratio,
                    vertex_count=len(polygon),
                )
            )

        return tuple(
            sorted(
                diagnostics,
                key=lambda diagnostic: diagnostic.area_ratio,
                reverse=True,
            )
        )


    def _render_contour_diagnostics(
        self,
        image: ImageArray,
        diagnostics: tuple[_ContourDiagnostic, ...],
    ) -> ImageArray:
        """Render contour diagnostics without affecting localization decisions."""

        debug_image = image.copy()

        for index, diagnostic in enumerate(diagnostics[:8], start=1):
            contour = diagnostic.contour

            cv2.drawContours(
                debug_image,
                [contour],
                -1,
                (0, 255, 255),
                2,
            )

            moments = cv2.moments(contour)

            if moments["m00"] != 0.0:
                label_x = int(moments["m10"] / moments["m00"])
                label_y = int(moments["m01"] / moments["m00"])
            else:
                x, y, width, height = cv2.boundingRect(contour)
                label_x = x + (width // 2)
                label_y = y + (height // 2)

            label = (
                f"C{index} "
                f"A={diagnostic.area_ratio:.3f} "
                f"V={diagnostic.vertex_count}"
            )

            cv2.putText(
                debug_image,
                label,
                (max(5, label_x - 80), max(20, label_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return debug_image

    @staticmethod
    def _rectangularity(corners: NDArray[np.float32], contour_area: float) -> float:
        rectangle = cv2.minAreaRect(corners)
        width, height = rectangle[1]
        rectangle_area = float(width * height)

        if rectangle_area <= 0.0:
            return 0.0

        return float(np.clip(contour_area / rectangle_area, 0.0, 1.0))

    @staticmethod
    def _order_corners(corners: NDArray[np.float32]) -> NDArray[np.float32]:
        ordered = np.zeros((4, 2), dtype=np.float32)

        coordinate_sum = corners.sum(axis=1)
        coordinate_diff = np.diff(corners, axis=1).reshape(-1)

        ordered[0] = corners[np.argmin(coordinate_sum)]
        ordered[2] = corners[np.argmax(coordinate_sum)]
        ordered[1] = corners[np.argmin(coordinate_diff)]
        ordered[3] = corners[np.argmax(coordinate_diff)]

        return ordered

    @staticmethod
    def _rectify(
        image: ImageArray,
        corners: NDArray[np.float32],
    ) -> ImageArray:
        top_left, top_right, bottom_right, bottom_left = corners

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        height_right = np.linalg.norm(bottom_right - top_right)
        height_left = np.linalg.norm(bottom_left - top_left)

        width = max(1, int(round(max(width_top, width_bottom))))
        height = max(1, int(round(max(height_right, height_left))))

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        transform = cv2.getPerspectiveTransform(corners, destination)

        rectified = cv2.warpPerspective(
            image,
            transform,
            (width, height),
        )

        return cast(ImageArray, rectified)

    @staticmethod
    def _annotate_board(
        image: ImageArray,
        source_contour: cv2.typing.MatLike,
    ) -> ImageArray:
        """Annotate the PCB boundary actually observed by the perception pipeline."""

        annotated = image.copy()
        contour = np.round(source_contour).astype(np.int32).reshape((-1, 1, 2))

        cv2.drawContours(
            annotated,
            [contour],
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=3,
        )

        return annotated
