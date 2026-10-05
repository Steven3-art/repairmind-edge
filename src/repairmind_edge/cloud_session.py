"""Build validated cloud publication sessions from RepairMind artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from repairmind_edge.cloud import ArtifactPayload
from repairmind_edge.cloud_models import (
    CloudArtifact,
    CloudArtifactType,
    InspectionSessionManifest,
)
from repairmind_edge.reporting import InspectionReport

_ARTIFACT_ALLOWLIST: Mapping[str, tuple[CloudArtifactType, str]] = {
    "inspection_report.json": (
        CloudArtifactType.INSPECTION_REPORT,
        "application/json",
    ),
    "annotated.jpg": (
        CloudArtifactType.ANNOTATED_IMAGE,
        "image/jpeg",
    ),
    "normalized.jpg": (
        CloudArtifactType.NORMALIZED_IMAGE,
        "image/jpeg",
    ),
    "roi_overlay.jpg": (
        CloudArtifactType.ROI_OVERLAY,
        "image/jpeg",
    ),
}


def _read_required_report(directory: Path) -> InspectionReport:
    """Load and validate the canonical inspection report."""

    report_path = directory / "inspection_report.json"

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Inspection report not found: {report_path}"
        )

    try:
        return InspectionReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid inspection report: {report_path}"
        ) from exc


def _artifact_from_file(
    path: Path,
    *,
    artifact_type: CloudArtifactType,
    content_type: str,
) -> tuple[CloudArtifact, ArtifactPayload]:
    """Create verified artifact metadata and its binary payload."""

    content = path.read_bytes()

    artifact = CloudArtifact(
        artifact_type=artifact_type,
        filename=path.name,
        content_type=content_type,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )

    return artifact, ArtifactPayload(content=content)


def build_cloud_session(
    artifact_directory: Path,
    *,
    session_id: str | None = None,
    created_at: datetime | None = None,
) -> tuple[InspectionSessionManifest, dict[str, ArtifactPayload]]:
    """Build a cloud session from explicitly approved inspection artifacts."""

    directory = artifact_directory.resolve()

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Artifact directory not found: {artifact_directory}"
        )

    report = _read_required_report(directory)

    resolved_session_id = session_id or f"inspection-{uuid4()}"

    if not resolved_session_id.strip():
        raise ValueError("session_id must not be empty")

    resolved_created_at = created_at or datetime.now(UTC)

    if resolved_created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")

    artifacts: list[CloudArtifact] = []
    payloads: dict[str, ArtifactPayload] = {}

    for filename, (artifact_type, content_type) in _ARTIFACT_ALLOWLIST.items():
        path = directory / filename

        if not path.is_file():
            if artifact_type == CloudArtifactType.INSPECTION_REPORT:
                raise FileNotFoundError(
                    f"Required cloud artifact not found: {path}"
                )
            continue

        artifact, payload = _artifact_from_file(
            path,
            artifact_type=artifact_type,
            content_type=content_type,
        )

        artifacts.append(artifact)
        payloads[filename] = payload

    manifest = InspectionSessionManifest(
        session_id=resolved_session_id,
        created_at=resolved_created_at,
        report_version=report.report_version,
        inspection_status=report.status.value,
        artifacts=tuple(artifacts),
    )

    return manifest, payloads


def is_generated_session_id(session_id: str) -> bool:
    """Return whether a session ID uses the generated RepairMind format."""

    prefix = "inspection-"

    if not session_id.startswith(prefix):
        return False

    try:
        UUID(session_id.removeprefix(prefix))
    except ValueError:
        return False

    return True