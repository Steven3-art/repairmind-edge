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

    _SOURCE_CONTOUR_APPROXIMATION_RATIO = 0.01
    _MAX_SOURCE_CONTOUR_AREA_DRIFT_RATIO = 0.05
    _MIN_SOURCE_CONTOUR_VERTICES = 5
    _MAX_SOURCE_CONTOUR_SPIKE_ANGLE_DEGREES = 60.0
    _DENSE_CONTOUR_MIN_DETOUR_RATIO = 1.10
    _DENSE_CONTOUR_MIN_DEVIATION_PIXELS = 10.0
    _DENSE_CONTOUR_MAX_DIRECTION_CHANGE_DEGREES = 20.0
    _PATH_EXCURSION_MIN_INTERMEDIATE_POINTS = 3
    _PATH_EXCURSION_MAX_INTERMEDIATE_POINTS = 12
    _PATH_EXCURSION_MIN_DETOUR_RATIO = 1.20
    _PATH_EXCURSION_MIN_DEVIATION_PIXELS = 15.0
    _PATH_EXCURSION_MAX_ANCHOR_DIRECTION_CHANGE_DEGREES = 20.0
    _BOARD_TERMINAL_SEARCH_START_RATIO = 0.60
    _BOARD_TERMINAL_WIDTH_RATIO = 0.55
    _BOARD_TERMINAL_PERSISTENCE_RATIO = 0.03
    _BOUNDARY_FUSION_DISTANCE_RATIO = 0.03
    _BOUNDARY_FUSION_MAX_EXCURSION_RATIO = 0.08
    _BOUNDARY_FUSION_MAX_PATH_LENGTH_RATIO = 0.25

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
                        "corners": [{"x": point.x, "y": point.y} for point in points],
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
            raise PerceptionError("Input image must be a three-channel BGR image.")

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
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
        )

        return dominant_candidate, diagnostics

    @classmethod
    def _segment_board_region(
        cls,
        image: ImageArray,
        coarse_contour: cv2.typing.MatLike,
    ) -> ImageArray:
        """Segment a coherent PCB region from a coarse board hypothesis."""

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            raise PerceptionError("Cannot segment an empty image.")

        contour = np.asarray(coarse_contour)

        if contour.size == 0:
            raise PerceptionError("Cannot segment a board without a coarse contour.")

        contour_area = float(cv2.contourArea(contour))

        if contour_area <= 0.0:
            raise PerceptionError("Cannot segment a board from a degenerate contour.")

        # GrabCut label mask:
        #   GC_BGD     = definite background
        #   GC_FGD     = definite foreground
        #   GC_PR_BGD  = probable background
        #   GC_PR_FGD  = probable foreground
        grabcut_mask = np.full(
            (height, width),
            cv2.GC_BGD,
            dtype=np.uint8,
        )

        # The coarse contour defines the region in which the board can exist.
        # It is initially marked as probable foreground rather than definite
        # foreground so GrabCut can still reject contamination near its edges.
        cv2.drawContours(
            grabcut_mask,
            [contour],
            contourIdx=-1,
            color=cv2.GC_PR_FGD,
            thickness=cv2.FILLED,
        )

        # Erode the coarse region to obtain a conservative foreground seed.
        coarse_binary = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        cv2.drawContours(
            coarse_binary,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

        x, y, bounding_width, bounding_height = cv2.boundingRect(contour)

        minimum_dimension = min(
            bounding_width,
            bounding_height,
        )

        erosion_radius = max(
            3,
            int(round(minimum_dimension * 0.03)),
        )

        kernel_size = (erosion_radius * 2) + 1

        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        definite_foreground = cv2.erode(
            coarse_binary,
            erosion_kernel,
            iterations=1,
        )

        grabcut_mask[definite_foreground == 255] = cv2.GC_FGD

        background_model = np.zeros(
            (1, 65),
            dtype=np.float64,
        )
        foreground_model = np.zeros(
            (1, 65),
            dtype=np.float64,
        )

        cv2.grabCut(
            image,
            grabcut_mask,
            (0, 0, width, height),
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )

        foreground_labels = np.logical_or(
            grabcut_mask == cv2.GC_FGD,
            grabcut_mask == cv2.GC_PR_FGD,
        )

        segmented_mask = np.where(
            foreground_labels,
            255,
            0,
        ).astype(np.uint8)

        # Components mounted on the PCB may produce internal segmentation
        # holes. Fill only enclosed holes; do not expand the outer boundary.
        flood_fill = segmented_mask.copy()

        flood_mask = np.zeros(
            (height + 2, width + 2),
            dtype=np.uint8,
        )

        cv2.floodFill(
            flood_fill,
            flood_mask,
            seedPoint=(0, 0),
            newVal=255,
        )

        enclosed_holes = cv2.bitwise_not(flood_fill)

        segmented_mask = np.asarray(
            cv2.bitwise_or(
                segmented_mask,
                enclosed_holes,
            ),
            dtype=np.uint8,
        )

        return segmented_mask

    @classmethod
    def _refine_board_mask(
        cls,
        mask: ImageArray,
    ) -> ImageArray:
        """Remove narrow attached regions from a segmented board mask."""

        if mask.ndim != 2:
            raise PerceptionError("Board segmentation mask must be a single-channel image.")

        if mask.dtype != np.uint8:
            raise PerceptionError("Board segmentation mask must use uint8 pixels.")

        if mask.size == 0:
            raise PerceptionError("Cannot refine an empty board segmentation mask.")

        binary_mask = np.where(
            mask > 0,
            255,
            0,
        ).astype(np.uint8)

        contours, _hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        valid_contours = [contour for contour in contours if float(cv2.contourArea(contour)) > 0.0]

        if not valid_contours:
            return np.zeros_like(binary_mask)

        dominant_contour = max(
            valid_contours,
            key=cv2.contourArea,
        )

        _x, _y, width, height = cv2.boundingRect(dominant_contour)

        minimum_dimension = min(width, height)

        if minimum_dimension <= 0:
            return binary_mask.copy()

        # Derive the opening scale from the detected object itself.
        # The kernel remains small relative to the main board body while
        # being large enough to disconnect narrow foreground bridges.
        opening_radius = max(
            2,
            int(round(minimum_dimension * 0.03)),
        )

        kernel_size = (opening_radius * 2) + 1

        opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        opened_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            opening_kernel,
            iterations=1,
        )

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            opened_mask,
            connectivity=8,
        )

        if component_count <= 1:
            return binary_mask.copy()

        component_areas = stats[
            1:,
            cv2.CC_STAT_AREA,
        ]

        largest_component_label = int(np.argmax(component_areas)) + 1

        main_component = np.where(
            labels == largest_component_label,
            255,
            0,
        ).astype(np.uint8)

        # Recover the main PCB region from the original mask without
        # reintroducing disconnected foreground appendages.
        recovery_seed = cv2.bitwise_and(
            binary_mask,
            main_component,
        )

        recovery_contours, _hierarchy = cv2.findContours(
            recovery_seed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not recovery_contours:
            return main_component

        recovered_contour = max(
            recovery_contours,
            key=cv2.contourArea,
        )

        refined_mask = np.zeros_like(binary_mask)

        cv2.drawContours(
            refined_mask,
            [recovered_contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

        return refined_mask

    @classmethod
    def _reject_terminal_foreground_extension(
        cls,
        mask: ImageArray,
    ) -> ImageArray:
        """Reject foreground extending beyond the main board body's termination."""

        if mask.ndim != 2:
            raise PerceptionError("Board mask must be a single-channel image.")

        if mask.dtype != np.uint8:
            raise PerceptionError("Board mask must use uint8 pixels.")

        if mask.size == 0:
            raise PerceptionError("Cannot analyze an empty board mask.")

        binary_mask = np.where(
            mask > 0,
            255,
            0,
        ).astype(np.uint8)

        foreground_rows = np.flatnonzero(np.any(binary_mask > 0, axis=1))

        if foreground_rows.size == 0:
            return binary_mask.copy()

        first_row = int(foreground_rows[0])
        last_row = int(foreground_rows[-1])

        object_height = last_row - first_row + 1

        if object_height <= 1:
            return binary_mask.copy()

        row_spans = np.zeros(
            binary_mask.shape[0],
            dtype=np.int32,
        )

        for row_index in range(first_row, last_row + 1):
            foreground_columns = np.flatnonzero(binary_mask[row_index] > 0)

            if foreground_columns.size == 0:
                continue

            row_spans[row_index] = int(foreground_columns[-1] - foreground_columns[0] + 1)

        positive_spans = row_spans[row_spans > 0]

        if positive_spans.size == 0:
            return binary_mask.copy()

        # A high percentile is more robust than a single maximum row,
        # which could itself contain foreground contamination.
        body_width = float(
            np.percentile(
                positive_spans,
                90,
            )
        )

        if body_width <= 0.0:
            return binary_mask.copy()

        terminal_width_limit = body_width * cls._BOARD_TERMINAL_WIDTH_RATIO

        search_start = first_row + int(
            round(object_height * cls._BOARD_TERMINAL_SEARCH_START_RATIO)
        )

        persistence_rows = max(
            3,
            int(round(object_height * cls._BOARD_TERMINAL_PERSISTENCE_RATIO)),
        )

        termination_row: int | None = None

        for row_index in range(
            search_start,
            last_row - persistence_rows + 2,
        ):
            window = row_spans[row_index : row_index + persistence_rows]

            if window.size < persistence_rows:
                break

            if np.all((window > 0) & (window < terminal_width_limit)):
                termination_row = row_index
                break

        if termination_row is None:
            return binary_mask.copy()

        refined_mask = binary_mask.copy()

        refined_mask[termination_row:, :] = 0

        return refined_mask

    @staticmethod
    def _project_point_to_closed_contour(
        point: NDArray[np.float64],
        contour: cv2.typing.MatLike,
    ) -> NDArray[np.float64]:
        """Project a point onto the nearest segment of a closed contour."""

        contour_points = np.asarray(
            contour,
            dtype=np.float64,
        ).reshape((-1, 2))

        if len(contour_points) < 2:
            raise PerceptionError("A closed contour requires at least two points.")

        query_point = np.asarray(
            point,
            dtype=np.float64,
        ).reshape(2)

        best_projection: NDArray[np.float64] | None = None
        best_squared_distance = float("inf")

        for index in range(len(contour_points)):
            segment_start = contour_points[index]
            segment_end = contour_points[(index + 1) % len(contour_points)]

            segment = segment_end - segment_start
            segment_squared_length = float(
                np.dot(
                    segment,
                    segment,
                )
            )

            if segment_squared_length <= 0.0:
                projection = segment_start.copy()
            else:
                relative_point = query_point - segment_start

                segment_position = float(
                    np.dot(
                        relative_point,
                        segment,
                    )
                    / segment_squared_length
                )

                segment_position = float(
                    np.clip(
                        segment_position,
                        0.0,
                        1.0,
                    )
                )

                projection = segment_start + segment_position * segment

            offset = query_point - projection

            squared_distance = float(
                np.dot(
                    offset,
                    offset,
                )
            )

            if squared_distance < best_squared_distance:
                best_squared_distance = squared_distance
                best_projection = projection

        if best_projection is None:
            raise PerceptionError("Unable to project point onto contour.")

        return best_projection

    @classmethod
    def _fuse_observed_and_segmented_boundary(
        cls,
        observed_contour: cv2.typing.MatLike,
        reliable_mask: ImageArray,
    ) -> cv2.typing.MatLike | None:
        """Fuse locally plausible observed subpaths with a reliable board region."""

        if reliable_mask.ndim != 2:
            raise PerceptionError("Reliable board mask must be a single-channel image.")

        if reliable_mask.dtype != np.uint8:
            raise PerceptionError("Reliable board mask must use uint8 pixels.")

        if reliable_mask.size == 0:
            raise PerceptionError("Cannot fuse a boundary with an empty board mask.")

        binary_mask = np.where(
            reliable_mask > 0,
            255,
            0,
        ).astype(np.uint8)

        reliable_contour = cls._extract_segmented_board_contour(binary_mask)

        if reliable_contour is None:
            return None

        observed_points = np.asarray(
            observed_contour,
            dtype=np.int32,
        ).reshape((-1, 2))

        if len(observed_points) < 3:
            return reliable_contour

        _x, _y, width, height = cv2.boundingRect(reliable_contour)

        minimum_dimension = min(
            width,
            height,
        )

        if minimum_dimension <= 0:
            return reliable_contour

        anchor_distance = max(
            2.0,
            minimum_dimension * cls._BOUNDARY_FUSION_DISTANCE_RATIO,
        )

        maximum_excursion = max(
            anchor_distance,
            minimum_dimension * cls._BOUNDARY_FUSION_MAX_EXCURSION_RATIO,
        )

        outside_mask = cv2.bitwise_not(binary_mask)

        distance_map = cv2.distanceTransform(
            outside_mask,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )

        image_height, image_width = binary_mask.shape

        point_distances = np.full(
            len(observed_points),
            np.inf,
            dtype=np.float64,
        )

        for index, point in enumerate(observed_points):
            x = int(point[0])
            y = int(point[1])

            if 0 <= x < image_width and 0 <= y < image_height:
                point_distances[index] = float(distance_map[y, x])

        anchor_flags = point_distances <= anchor_distance

        fusion_mask = binary_mask.copy()

        point_count = len(observed_points)

        # A closed contour has no natural first point. Start immediately
        # after an anchor so excursion runs cannot be split at index zero.
        anchor_indices = np.flatnonzero(anchor_flags)

        if anchor_indices.size == 0:
            return reliable_contour

        start_index = int(anchor_indices[0])

        ordered_indices = [(start_index + offset) % point_count for offset in range(point_count)]

        position = 0

        while position < point_count:
            current_index = ordered_indices[position]

            if anchor_flags[current_index]:
                position += 1
                continue

            run_start_position = position

            while position < point_count and not anchor_flags[ordered_indices[position]]:
                position += 1

            run_end_position = position - 1

            previous_position = run_start_position - 1

            if previous_position < 0:
                continue

            previous_anchor_index = ordered_indices[previous_position]

            if position >= point_count:
                next_anchor_index = start_index
            else:
                next_anchor_index = ordered_indices[position]

            if not anchor_flags[previous_anchor_index] or not anchor_flags[next_anchor_index]:
                continue

            excursion_indices = [
                ordered_indices[index]
                for index in range(
                    run_start_position,
                    run_end_position + 1,
                )
            ]

            if not excursion_indices:
                continue

            excursion_distances = point_distances[excursion_indices]

            maximum_run_excursion = float(np.max(excursion_distances))

            if not np.isfinite(maximum_run_excursion) or maximum_run_excursion > maximum_excursion:
                continue

            path_indices = [
                previous_anchor_index,
                *excursion_indices,
                next_anchor_index,
            ]

            path_points = observed_points[path_indices].astype(np.float64)

            path_segments = np.diff(
                path_points,
                axis=0,
            )

            path_length = float(
                np.sum(
                    np.linalg.norm(
                        path_segments,
                        axis=1,
                    )
                )
            )

            maximum_path_length = minimum_dimension * cls._BOUNDARY_FUSION_MAX_PATH_LENGTH_RATIO

            if path_length > maximum_path_length:
                continue
            raster_path = np.rint(path_points).astype(np.int32)

            first_anchor = raster_path[0]
            last_anchor = raster_path[-1]

            first_anchor = raster_path[0]
            last_anchor = raster_path[-1]

            first_projection_float = cls._project_point_to_closed_contour(
                first_anchor.astype(np.float64),
                reliable_contour,
            )

            last_projection_float = cls._project_point_to_closed_contour(
                last_anchor.astype(np.float64),
                reliable_contour,
            )

            projection_span = float(np.linalg.norm(last_projection_float - first_projection_float))

            # A recovery whose anchors physically collapse onto the same
            # reliable boundary location does not define a meaningful
            # boundary segment.
            if projection_span <= np.finfo(np.float64).eps:
                continue

            first_projection = np.rint(first_projection_float).astype(np.int32)

            last_projection = np.rint(last_projection_float).astype(np.int32)

            # Integer rasterization may collapse two distinct continuous
            # projections onto the same pixel. Such a recovery cannot form
            # a meaningful raster region.
            if np.array_equal(
                first_projection,
                last_projection,
            ):
                continue
            recovery_polygon = np.vstack(
                (
                    first_projection.reshape(1, 2),
                    raster_path,
                    last_projection.reshape(1, 2),
                )
            ).astype(np.int32)

            # Convert the accepted observed excursion into a connected local
            # recovery region. The projection points bridge geometrically valid
            # anchors back to the reliable PCB component.
            cv2.fillPoly(
                fusion_mask,
                [recovery_polygon.reshape((-1, 1, 2))],
                color=255,
            )

        fused_contour = cls._extract_segmented_board_contour(fusion_mask)

        if fused_contour is None:
            return reliable_contour

        reliable_area = float(cv2.contourArea(reliable_contour))
        fused_area = float(cv2.contourArea(fused_contour))

        if reliable_area <= 0.0:
            return reliable_contour

        area_growth = (fused_area - reliable_area) / reliable_area

        # Fusion may recover conservative segmentation losses, but it must
        # never redefine the global board geometry.
        if area_growth > 0.10:
            return reliable_contour

        return fused_contour

    @classmethod
    def _extract_segmented_board_contour(
        cls,
        mask: ImageArray,
    ) -> cv2.typing.MatLike | None:
        """Extract the largest coherent external contour from a board mask."""

        if mask.ndim != 2:
            raise PerceptionError("Board segmentation mask must be a single-channel image.")

        if mask.dtype != np.uint8:
            raise PerceptionError("Board segmentation mask must use uint8 pixels.")

        if mask.size == 0:
            raise PerceptionError("Cannot extract a board contour from an empty mask.")

        # Normalize the input so contour extraction always operates on a
        # strict binary 0/255 representation.
        binary_mask = np.where(
            mask > 0,
            255,
            0,
        ).astype(np.uint8)

        contours, _hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        valid_contours = [contour for contour in contours if float(cv2.contourArea(contour)) > 0.0]

        if not valid_contours:
            return None

        return max(
            valid_contours,
            key=cv2.contourArea,
        )

    @classmethod
    def _denoise_dense_contour(
        cls,
        contour: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        """Remove isolated excursions while preserving structural contour turns."""

        points = np.asarray(contour).reshape((-1, 2)).copy()

        if len(points) < cls._MIN_SOURCE_CONTOUR_VERTICES:
            return contour

        raw_area = float(cv2.contourArea(contour))

        if raw_area <= 0.0:
            return contour

        point_count = len(points)
        candidate_excursions: list[tuple[int, float]] = []

        for index, point in enumerate(points):
            previous_index = (index - 1) % point_count
            previous_previous_index = (index - 2) % point_count
            next_index = (index + 1) % point_count
            next_next_index = (index + 2) % point_count

            previous_previous_point = points[previous_previous_index]
            previous_point = points[previous_index]
            next_point = points[next_index]
            next_next_point = points[next_next_index]

            previous_previous_float = previous_previous_point.astype(np.float64)
            previous_float = previous_point.astype(np.float64)
            point_float = point.astype(np.float64)
            next_float = next_point.astype(np.float64)
            next_next_float = next_next_point.astype(np.float64)

            # Structural direction before and after the candidate.
            incoming_structure = previous_float - previous_previous_float
            outgoing_structure = next_next_float - next_float

            incoming_structure_length = float(np.linalg.norm(incoming_structure))
            outgoing_structure_length = float(np.linalg.norm(outgoing_structure))

            if incoming_structure_length <= 0.0 or outgoing_structure_length <= 0.0:
                continue

            direction_cosine = float(
                np.dot(incoming_structure, outgoing_structure)
                / (incoming_structure_length * outgoing_structure_length)
            )
            direction_cosine = float(np.clip(direction_cosine, -1.0, 1.0))

            direction_change_degrees = float(np.degrees(np.arccos(direction_cosine)))

            # A genuine structural corner changes the local trajectory.
            # A short excursion normally leaves and returns to approximately
            # the same trajectory.
            if direction_change_degrees > cls._DENSE_CONTOUR_MAX_DIRECTION_CHANGE_DEGREES:
                continue

            incoming_length = float(np.linalg.norm(point_float - previous_float))
            outgoing_length = float(np.linalg.norm(next_float - point_float))
            direct_length = float(np.linalg.norm(next_float - previous_float))

            if direct_length <= 0.0:
                continue

            detour_length = incoming_length + outgoing_length
            detour_ratio = detour_length / direct_length

            local_segment = next_float - previous_float
            point_offset = point_float - previous_float

            cross_product = abs(
                float(local_segment[0] * point_offset[1] - local_segment[1] * point_offset[0])
            )
            deviation = cross_product / direct_length

            if (
                detour_ratio >= cls._DENSE_CONTOUR_MIN_DETOUR_RATIO
                and deviation >= cls._DENSE_CONTOUR_MIN_DEVIATION_PIXELS
            ):
                candidate_excursions.append((index, deviation))

        if not candidate_excursions:
            return contour

        # Process the strongest geometric excursions first.
        candidate_excursions.sort(
            key=lambda candidate: candidate[1],
            reverse=True,
        )

        removed_indices: set[int] = set()
        protected_indices: set[int] = set()

        for index, _deviation in candidate_excursions:
            previous_index = (index - 1) % point_count
            next_index = (index + 1) % point_count

            if index in protected_indices:
                continue

            trial_removed_indices = removed_indices | {index}

            remaining_points = np.asarray(
                [
                    point
                    for point_index, point in enumerate(points)
                    if point_index not in trial_removed_indices
                ],
                dtype=points.dtype,
            )

            if len(remaining_points) < cls._MIN_SOURCE_CONTOUR_VERTICES:
                continue

            trial_contour = remaining_points.reshape((-1, 1, 2))

            trial_area = float(cv2.contourArea(trial_contour))

            if trial_area <= 0.0:
                continue

            area_drift_ratio = abs(trial_area - raw_area) / raw_area

            if area_drift_ratio > cls._MAX_SOURCE_CONTOUR_AREA_DRIFT_RATIO:
                continue

            removed_indices.add(index)

            # Prevent one accepted correction from triggering local erosion.
            protected_indices.update(
                {
                    previous_index,
                    index,
                    next_index,
                }
            )

        if not removed_indices:
            return contour

        cleaned_points = np.asarray(
            [
                point
                for point_index, point in enumerate(points)
                if point_index not in removed_indices
            ],
            dtype=points.dtype,
        )

        return cleaned_points.reshape((-1, 1, 2))

    @classmethod
    def _reject_local_path_excursions(
        cls,
        contour: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        """Reject a local parasite path while preserving structural geometry."""

        points = np.asarray(contour).reshape((-1, 2)).copy()

        if len(points) < cls._MIN_SOURCE_CONTOUR_VERTICES:
            return contour

        raw_area = float(cv2.contourArea(contour))

        if raw_area <= 0.0:
            return contour

        point_count = len(points)
        best_candidate: tuple[int, int, float] | None = None

        max_intermediate_points = min(
            cls._PATH_EXCURSION_MAX_INTERMEDIATE_POINTS,
            point_count - 2,
        )

        for start_index in range(point_count):
            for intermediate_count in range(
                cls._PATH_EXCURSION_MIN_INTERMEDIATE_POINTS,
                max_intermediate_points + 1,
            ):
                end_index = start_index + intermediate_count + 1

                # Keep the first implementation conservative:
                # no window may wrap across contour closure.
                if end_index >= point_count:
                    continue

                previous_anchor_index = start_index - 1
                next_anchor_index = end_index + 1

                # Both anchors need an external structural neighbor.
                if previous_anchor_index < 0 or next_anchor_index >= point_count:
                    continue

                anchor_start = points[start_index].astype(np.float64)
                anchor_end = points[end_index].astype(np.float64)

                previous_anchor = points[previous_anchor_index].astype(np.float64)
                next_anchor = points[next_anchor_index].astype(np.float64)

                # Estimate the structural trajectory immediately before
                # and after the candidate path.
                incoming_direction = anchor_start - previous_anchor
                outgoing_direction = next_anchor - anchor_end

                incoming_length = float(np.linalg.norm(incoming_direction))
                outgoing_length = float(np.linalg.norm(outgoing_direction))

                if incoming_length <= 0.0 or outgoing_length <= 0.0:
                    continue

                direction_cosine = float(
                    np.dot(incoming_direction, outgoing_direction)
                    / (incoming_length * outgoing_length)
                )
                direction_cosine = float(np.clip(direction_cosine, -1.0, 1.0))

                direction_change_degrees = float(np.degrees(np.arccos(direction_cosine)))

                # A parasite excursion should leave and return to
                # approximately the same structural trajectory.
                if (
                    direction_change_degrees
                    > cls._PATH_EXCURSION_MAX_ANCHOR_DIRECTION_CHANGE_DEGREES
                ):
                    continue

                path_points = points[start_index : end_index + 1].astype(np.float64)

                direct_vector = anchor_end - anchor_start
                direct_length = float(np.linalg.norm(direct_vector))

                if direct_length <= 0.0:
                    continue

                segment_vectors = np.diff(path_points, axis=0)
                segment_lengths = np.linalg.norm(
                    segment_vectors,
                    axis=1,
                )
                observed_length = float(np.sum(segment_lengths))

                detour_ratio = observed_length / direct_length

                if detour_ratio < cls._PATH_EXCURSION_MIN_DETOUR_RATIO:
                    continue

                intermediate_points = path_points[1:-1]
                offsets = intermediate_points - anchor_start

                cross_products = np.abs(
                    direct_vector[0] * offsets[:, 1] - direct_vector[1] * offsets[:, 0]
                )
                deviations = cross_products / direct_length

                max_deviation = float(np.max(deviations))

                if max_deviation < cls._PATH_EXCURSION_MIN_DEVIATION_PIXELS:
                    continue

                # Build the candidate correction now, before ranking it.
                # Both anchors are preserved; only the intermediate path
                # is removed.
                trial_points = np.concatenate(
                    (
                        points[: start_index + 1],
                        points[end_index:],
                    ),
                    axis=0,
                )

                if len(trial_points) < cls._MIN_SOURCE_CONTOUR_VERTICES:
                    continue

                trial_contour = trial_points.reshape((-1, 1, 2)).astype(
                    contour.dtype,
                    copy=False,
                )

                trial_area = float(cv2.contourArea(trial_contour))

                if trial_area <= 0.0:
                    continue

                area_drift_ratio = abs(trial_area - raw_area) / raw_area

                if area_drift_ratio > cls._MAX_SOURCE_CONTOUR_AREA_DRIFT_RATIO:
                    continue

                candidate_score = detour_ratio * max_deviation

                if best_candidate is None or candidate_score > best_candidate[2]:
                    best_candidate = (
                        start_index,
                        end_index,
                        candidate_score,
                    )

        if best_candidate is None:
            return contour

        start_index, end_index, _score = best_candidate

        cleaned_points = np.concatenate(
            (
                points[: start_index + 1],
                points[end_index:],
            ),
            axis=0,
        )

        return cleaned_points.reshape((-1, 1, 2)).astype(
            contour.dtype,
            copy=False,
        )

    @classmethod
    def _clean_source_contour(
        cls,
        contour: cv2.typing.MatLike,
        *,
        image_width: int,
        image_height: int,
    ) -> cv2.typing.MatLike:
        """Simplify an observed contour and remove isolated border excursions."""

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            return contour

        cleaned = cv2.approxPolyDP(
            contour,
            cls._SOURCE_CONTOUR_APPROXIMATION_RATIO * perimeter,
            True,
        )

        if len(cleaned) < cls._MIN_SOURCE_CONTOUR_VERTICES:
            return contour

        raw_area = float(cv2.contourArea(contour))
        if raw_area <= 0.0:
            return contour

        cleaned = cls._remove_border_excursions(
            cleaned,
            image_width=image_width,
            image_height=image_height,
            reference_area=raw_area,
        )

        if len(cleaned) < cls._MIN_SOURCE_CONTOUR_VERTICES:
            return contour

        cleaned_area = float(cv2.contourArea(cleaned))
        if cleaned_area <= 0.0:
            return contour

        area_drift_ratio = abs(cleaned_area - raw_area) / raw_area

        print(
            "PATH DEBUG:",
            "raw_area=",
            raw_area,
            "cleaned_area=",
            cleaned_area,
            "area_drift=",
            area_drift_ratio,
        )

        if area_drift_ratio > cls._MAX_SOURCE_CONTOUR_AREA_DRIFT_RATIO:
            return contour

        return cleaned

    @classmethod
    def _remove_border_excursions(
        cls,
        contour: cv2.typing.MatLike,
        *,
        image_width: int,
        image_height: int,
        reference_area: float,
    ) -> cv2.typing.MatLike:
        """Remove small acute contour spikes that touch the image border."""

        points = np.asarray(contour).reshape((-1, 2)).copy()

        if len(points) <= cls._MIN_SOURCE_CONTOUR_VERTICES:
            return contour

        max_x = image_width - 1
        max_y = image_height - 1

        changed = True

        while changed and len(points) > cls._MIN_SOURCE_CONTOUR_VERTICES:
            changed = False

            for index, point in enumerate(points):
                x = int(point[0])
                y = int(point[1])

                touches_border = x <= 0 or y <= 0 or x >= max_x or y >= max_y

                if not touches_border:
                    continue

                previous_point = points[index - 1]
                next_point = points[(index + 1) % len(points)]

                vector_to_previous = previous_point.astype(np.float64) - point.astype(np.float64)
                vector_to_next = next_point.astype(np.float64) - point.astype(np.float64)

                previous_length = float(np.linalg.norm(vector_to_previous))
                next_length = float(np.linalg.norm(vector_to_next))

                if previous_length <= 0.0 or next_length <= 0.0:
                    continue

                cosine = float(
                    np.dot(vector_to_previous, vector_to_next) / (previous_length * next_length)
                )
                cosine = float(np.clip(cosine, -1.0, 1.0))

                angle_degrees = float(np.degrees(np.arccos(cosine)))

                if angle_degrees > cls._MAX_SOURCE_CONTOUR_SPIKE_ANGLE_DEGREES:
                    continue

                trial_points = np.delete(points, index, axis=0)

                if len(trial_points) < cls._MIN_SOURCE_CONTOUR_VERTICES:
                    continue

                trial_contour = trial_points.reshape((-1, 1, 2)).astype(
                    contour.dtype,
                    copy=False,
                )

                trial_area = float(cv2.contourArea(trial_contour))

                if trial_area <= 0.0:
                    continue

                area_drift_ratio = abs(trial_area - reference_area) / reference_area

                if area_drift_ratio > cls._MAX_SOURCE_CONTOUR_AREA_DRIFT_RATIO:
                    continue

                points = trial_points
                changed = True
                break

        return points.reshape((-1, 1, 2)).astype(
            contour.dtype,
            copy=False,
        )

    def _find_dominant_contour(
        self,
        contours: Sequence[cv2.typing.MatLike],
        image_area: float,
        image_width: int,
        image_height: int,
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
                area_ratio
                / max(
                    self._minimum_board_area_ratio * 3.0,
                    0.30,
                ),
            )

            confidence = float(
                np.clip(
                    (0.45 * solidity) + (0.35 * rectangularity) + (0.20 * coverage_score),
                    0.0,
                    1.0,
                )
            )

            cleaned_source_contour = self._denoise_dense_contour(contour)
            cleaned_source_contour = self._reject_local_path_excursions(cleaned_source_contour)
            candidate = _BoardCandidate(
                corners=box,
                source_contour=cleaned_source_contour,
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

            label = f"C{index} A={diagnostic.area_ratio:.3f} V={diagnostic.vertex_count}"

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
