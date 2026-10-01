"""Domain models for RepairMind Edge visual perception."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceType(StrEnum):
    """Types of visual evidence produced by the perception pipeline."""

    IMAGE_QUALITY = "IMAGE_QUALITY"
    BOARD_LOCALIZED = "BOARD_LOCALIZED"
    BOARD_LOCALIZATION_UNCERTAIN = "BOARD_LOCALIZATION_UNCERTAIN"
    PERSPECTIVE_RECTIFIED = "PERSPECTIVE_RECTIFIED"
    VISUAL_REGION = "VISUAL_REGION"
    ELECTRICAL_MEASUREMENT = "ELECTRICAL_MEASUREMENT"


class LocalizationMethod(StrEnum):
    """Methods available for localizing a PCB."""

    QUADRILATERAL = "quadrilateral"
    DOMINANT_CONTOUR = "dominant_contour"


class Point(BaseModel):
    """A two-dimensional image coordinate."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class BoundingBox(BaseModel):
    """Axis-aligned rectangular region in image coordinates."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RegionOfInterest(BaseModel):
    """Deterministic visual region proposed for further inspection."""

    model_config = ConfigDict(frozen=True)

    roi_id: str = Field(min_length=1)
    bounding_box: BoundingBox
    center: Point
    area_ratio: float = Field(gt=0.0, le=1.0)
    edge_density: float = Field(ge=0.0, le=1.0)
    local_contrast: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class ImageQuality(BaseModel):
    """Objective measurements describing the input image."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(gt=0)
    brightness: float = Field(ge=0.0, le=255.0)
    sharpness: float = Field(ge=0.0)


class BoardLocalization(BaseModel):
    """Result of attempting to localize the PCB boundary."""

    model_config = ConfigDict(frozen=True)

    detected: bool
    method: LocalizationMethod | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    area_ratio: float = Field(ge=0.0, le=1.0)
    corners: tuple[Point, Point, Point, Point] | None = None


class VisualEvidence(BaseModel):
    """Immutable evidence generated from a visual observation."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    evidence_type: EvidenceType
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    source_roi_id: str | None = None
    measurements: dict[str, Any] = Field(default_factory=dict)


class InspectionResult(BaseModel):
    """Structured result returned by the perception pipeline."""

    model_config = ConfigDict(frozen=True)

    quality: ImageQuality
    localization: BoardLocalization
    evidence: tuple[VisualEvidence, ...]
    normalized_width: int = Field(gt=0)
    normalized_height: int = Field(gt=0)
