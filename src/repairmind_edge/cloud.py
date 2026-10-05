"""Cloud publishing abstractions for RepairMind Edge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from repairmind_edge.cloud_models import (
    CloudPublishResult,
    InspectionSessionManifest,
)


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    """Binary payload associated with an artifact in a session manifest."""

    content: bytes


class CloudPublisher(ABC):
    """Interface implemented by RepairMind cloud publication providers."""

    @abstractmethod
    def publish(
        self,
        manifest: InspectionSessionManifest,
        payloads: Mapping[str, ArtifactPayload],
    ) -> CloudPublishResult:
        """Publish one validated inspection session."""


class InMemoryCloudPublisher(CloudPublisher):
    """Deterministic publisher used for local tests and development."""

    def __init__(self) -> None:
        self._sessions: dict[str, InspectionSessionManifest] = {}
        self._payloads: dict[str, dict[str, bytes]] = {}

    @property
    def sessions(self) -> Mapping[str, InspectionSessionManifest]:
        """Return published session manifests."""

        return self._sessions

    def publish(
        self,
        manifest: InspectionSessionManifest,
        payloads: Mapping[str, ArtifactPayload],
    ) -> CloudPublishResult:
        """Persist a session in memory after validating its payload set."""

        expected = {artifact.filename for artifact in manifest.artifacts}
        supplied = set(payloads)

        missing = expected - supplied
        unexpected = supplied - expected

        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing artifact payloads: {names}")

        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"Unexpected artifact payloads: {names}")

        if manifest.session_id in self._sessions:
            raise ValueError(
                f"Inspection session already published: {manifest.session_id}"
            )

        stored_payloads = {
            filename: payload.content
            for filename, payload in payloads.items()
        }

        self._sessions[manifest.session_id] = manifest
        self._payloads[manifest.session_id] = stored_payloads

        return CloudPublishResult(
            session_id=manifest.session_id,
            provider="memory",
            manifest_uri=f"memory://sessions/{manifest.session_id}/manifest.json",
            artifact_count=len(manifest.artifacts),
        )