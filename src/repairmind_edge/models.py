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


class LocalizationMethod(StrEnum):
    """Methods available for localizing a PCB."""

    QUADRILATERAL = "quadrilateral"
    DOMINANT_CONTOUR = "dominant_contour"


class Point(BaseModel):
    """A two-dimensional image coordinate."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float


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
    measurements: dict[str, Any] = Field(default_factory=dict)


class InspectionResult(BaseModel):
    """Structured result returned by the perception pipeline."""

    model_config = ConfigDict(frozen=True)

    quality: ImageQuality
    localization: BoardLocalization
    evidence: tuple[VisualEvidence, ...]
    normalized_width: int = Field(gt=0)
    normalized_height: int = Field(gt=0)
