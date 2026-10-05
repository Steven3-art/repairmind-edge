"""Tests for RepairMind Edge cloud publishing abstractions."""

from datetime import UTC, datetime

import pytest

from repairmind_edge.cloud import ArtifactPayload, InMemoryCloudPublisher
from repairmind_edge.cloud_models import (
    CloudArtifact,
    CloudArtifactType,
    InspectionSessionManifest,
)


def _manifest() -> InspectionSessionManifest:
    artifact = CloudArtifact(
        artifact_type=CloudArtifactType.INSPECTION_REPORT,
        filename="inspection_report.json",
        content_type="application/json",
        sha256="a" * 64,
        size_bytes=2,
    )

    return InspectionSessionManifest(
        session_id="session-001",
        created_at=datetime(2026, 10, 5, tzinfo=UTC),
        report_version="1.0",
        inspection_status="ADDITIONAL_EVIDENCE_REQUIRED",
        artifacts=(artifact,),
    )


def test_in_memory_publisher_publishes_complete_session() -> None:
    publisher = InMemoryCloudPublisher()
    manifest = _manifest()

    result = publisher.publish(
        manifest,
        {
            "inspection_report.json": ArtifactPayload(content=b"{}"),
        },
    )

    assert result.session_id == "session-001"
    assert result.provider == "memory"
    assert result.artifact_count == 1
    assert result.manifest_uri == (
        "memory://sessions/session-001/manifest.json"
    )
    assert publisher.sessions["session-001"] == manifest


def test_in_memory_publisher_rejects_missing_payload() -> None:
    publisher = InMemoryCloudPublisher()

    with pytest.raises(
        ValueError,
        match="Missing artifact payloads: inspection_report.json",
    ):
        publisher.publish(_manifest(), {})


def test_in_memory_publisher_rejects_unexpected_payload() -> None:
    publisher = InMemoryCloudPublisher()

    with pytest.raises(ValueError, match="Unexpected artifact payloads: extra.jpg"):
        publisher.publish(
            _manifest(),
            {
                "inspection_report.json": ArtifactPayload(content=b"{}"),
                "extra.jpg": ArtifactPayload(content=b"data"),
            },
        )


def test_in_memory_publisher_rejects_duplicate_session() -> None:
    publisher = InMemoryCloudPublisher()
    manifest = _manifest()
    payloads = {
        "inspection_report.json": ArtifactPayload(content=b"{}"),
    }

    publisher.publish(manifest, payloads)

    with pytest.raises(
        ValueError,
        match="Inspection session already published: session-001",
    ):
        publisher.publish(manifest, payloads)