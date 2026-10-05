"""Typed models for RepairMind Edge cloud inspection sessions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CloudArtifactType(StrEnum):
    """Artifacts that may be published as part of an inspection session."""

    INSPECTION_REPORT = "INSPECTION_REPORT"
    ANNOTATED_IMAGE = "ANNOTATED_IMAGE"
    NORMALIZED_IMAGE = "NORMALIZED_IMAGE"
    ROI_OVERLAY = "ROI_OVERLAY"


class CloudArtifact(BaseModel):
    """Metadata describing an explicitly approved cloud artifact."""

    model_config = ConfigDict(frozen=True)

    artifact_type: CloudArtifactType
    filename: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class InspectionSessionManifest(BaseModel):
    """Immutable manifest describing a RepairMind inspection session."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    session_id: str = Field(min_length=1)
    created_at: datetime
    report_version: str = Field(min_length=1)
    inspection_status: str = Field(min_length=1)
    artifacts: tuple[CloudArtifact, ...]


class CloudPublishResult(BaseModel):
    """Result returned after publishing an inspection session."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    manifest_uri: str = Field(min_length=1)
    artifact_count: int = Field(ge=0)