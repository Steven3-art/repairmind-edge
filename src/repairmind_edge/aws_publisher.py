"""AWS publication provider for RepairMind Edge inspection sessions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import boto3  # type: ignore[import-untyped]
from botocore.auth import SigV4Auth  # type: ignore[import-untyped]
from botocore.awsrequest import AWSRequest  # type: ignore[import-untyped]
from botocore.credentials import ReadOnlyCredentials  # type: ignore[import-untyped]

from repairmind_edge.cloud import ArtifactPayload, CloudPublisher
from repairmind_edge.cloud_models import (
    CloudArtifact,
    CloudPublishResult,
    InspectionSessionManifest,
)


class AwsPublishError(RuntimeError):
    """Raised when an AWS cloud publication operation fails."""


@dataclass(frozen=True, slots=True)
class UploadInstruction:
    """Validated artifact upload instruction returned by the AWS backend."""

    artifact_type: str
    upload_url: str
    content_type: str
    sha256: str


def _manifest_payload(
    manifest: InspectionSessionManifest,
) -> dict[str, Any]:
    """Translate the local manifest into the AWS ingestion contract."""

    return {
        "schema_version": manifest.schema_version,
        "session_id": manifest.session_id,
        "created_at": manifest.created_at.isoformat(),
        "artifacts": [
            {
                "artifact_type": artifact.artifact_type.value,
                "sha256": artifact.sha256,
                "byte_count": artifact.size_bytes,
            }
            for artifact in manifest.artifacts
        ],
    }


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialize JSON deterministically for signing and transmission."""

    return json.dumps(
        document,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _read_http_error(exc: HTTPError) -> str:
    """Return a bounded HTTP error body without exposing request credentials."""

    try:
        body = exc.read(4096).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""

    return body


def _frozen_credentials(
    *,
    profile_name: str | None,
    region_name: str,
) -> ReadOnlyCredentials:
    """Resolve AWS credentials through the standard boto3 credential chain."""

    session = boto3.Session(
        profile_name=profile_name,
        region_name=region_name,
    )

    credentials = session.get_credentials()

    if credentials is None:
        raise AwsPublishError("AWS credentials could not be resolved from the configured profile.")

    return credentials.get_frozen_credentials()


def _signed_post(
    *,
    url: str,
    body: bytes,
    region_name: str,
    credentials: ReadOnlyCredentials,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Send one SigV4-authenticated POST request to API Gateway."""

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    aws_request = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers=headers,
    )

    SigV4Auth(
        credentials,
        "execute-api",
        region_name,
    ).add_auth(aws_request)

    prepared_headers = {str(name): str(value) for name, value in aws_request.headers.items()}

    request = Request(
        url=url,
        data=body,
        headers=prepared_headers,
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = _read_http_error(exc)

        message = f"AWS session API returned HTTP {exc.code}"

        if detail:
            message += f": {detail}"

        raise AwsPublishError(message) from exc
    except URLError as exc:
        raise AwsPublishError(f"AWS session API could not be reached: {exc.reason}") from exc
    except OSError as exc:
        raise AwsPublishError(f"AWS session API request failed: {exc}") from exc

    try:
        document: Any = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AwsPublishError("AWS session API returned invalid JSON.") from exc

    if not isinstance(document, dict):
        raise AwsPublishError("AWS session API response must be a JSON object.")

    return document


def _require_response_string(
    document: Mapping[str, Any],
    name: str,
) -> str:
    """Read one required non-empty string from a backend response."""

    value = document.get(name)

    if not isinstance(value, str) or not value:
        raise AwsPublishError(f"AWS session API response is missing a valid {name!r}.")

    return value


def _manifest_uri(
    document: Mapping[str, Any],
) -> str:
    """Build the canonical S3 manifest URI from the backend response."""

    raw_manifest = document.get("manifest")

    if not isinstance(raw_manifest, dict):
        raise AwsPublishError("AWS session API response is missing the manifest object.")

    bucket = raw_manifest.get("bucket")
    object_key = raw_manifest.get("object_key")

    if not isinstance(bucket, str) or not bucket:
        raise AwsPublishError("AWS session API response has no valid manifest bucket.")

    if not isinstance(object_key, str) or not object_key:
        raise AwsPublishError("AWS session API response has no valid manifest object_key.")

    return f"s3://{bucket}/{object_key}"


def _parse_uploads(
    document: Mapping[str, Any],
) -> tuple[UploadInstruction, ...]:
    """Validate presigned upload instructions returned by the AWS backend."""

    raw_uploads = document.get("uploads")

    if not isinstance(raw_uploads, list):
        raise AwsPublishError("AWS session API response is missing the uploads array.")

    uploads: list[UploadInstruction] = []

    for index, raw_upload in enumerate(raw_uploads):
        if not isinstance(raw_upload, dict):
            raise AwsPublishError(f"AWS upload instruction {index} must be a JSON object.")

        artifact_type = raw_upload.get("artifact_type")

        if not isinstance(artifact_type, str) or not artifact_type:
            raise AwsPublishError(f"AWS upload instruction {index} has no valid artifact_type.")

        upload_url = raw_upload.get("upload_url")

        if not isinstance(upload_url, str) or not upload_url.startswith("https://"):
            raise AwsPublishError(f"AWS upload instruction {index} has no valid HTTPS upload URL.")

        content_type = raw_upload.get("content_type")

        if not isinstance(content_type, str) or not content_type:
            raise AwsPublishError(f"AWS upload instruction {index} has no valid content_type.")

        sha256 = raw_upload.get("sha256")

        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise AwsPublishError(f"AWS upload instruction {index} has no valid SHA-256.")

        uploads.append(
            UploadInstruction(
                artifact_type=artifact_type,
                upload_url=upload_url,
                content_type=content_type,
                sha256=sha256,
            )
        )

    return tuple(uploads)


def _artifact_by_type(
    manifest: InspectionSessionManifest,
) -> dict[str, CloudArtifact]:
    """Index manifest artifacts by their unique artifact type."""

    indexed: dict[str, CloudArtifact] = {}

    for artifact in manifest.artifacts:
        artifact_type = artifact.artifact_type.value

        if artifact_type in indexed:
            raise AwsPublishError(f"Duplicate artifact type in manifest: {artifact_type}")

        indexed[artifact_type] = artifact

    return indexed


def _verify_payload(
    artifact: CloudArtifact,
    payload: ArtifactPayload,
) -> None:
    """Verify payload size and digest immediately before cloud upload."""

    if len(payload.content) != artifact.size_bytes:
        raise AwsPublishError(f"Artifact size changed before upload: {artifact.filename}")

    digest = hashlib.sha256(payload.content).hexdigest()

    if digest != artifact.sha256:
        raise AwsPublishError(f"Artifact SHA-256 changed before upload: {artifact.filename}")


def _upload_artifact(
    *,
    instruction: UploadInstruction,
    artifact: CloudArtifact,
    payload: ArtifactPayload,
    timeout_seconds: float,
) -> None:
    """Upload one artifact through its server-issued presigned S3 URL."""

    _verify_payload(
        artifact,
        payload,
    )

    if instruction.content_type != artifact.content_type:
        raise AwsPublishError(f"AWS upload content type does not match {artifact.filename}.")

    if instruction.sha256 != artifact.sha256:
        raise AwsPublishError(f"AWS upload SHA-256 does not match {artifact.filename}.")

    headers = {
        "Content-Type": instruction.content_type,
        "x-amz-meta-repairmind-sha256": instruction.sha256,
        "x-amz-server-side-encryption": "AES256",
    }

    request = Request(
        url=instruction.upload_url,
        data=payload.content,
        headers=headers,
        method="PUT",
    )

    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            response.read()
    except HTTPError as exc:
        raise AwsPublishError(
            f"S3 rejected upload for {artifact.filename} with HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise AwsPublishError(f"S3 upload failed for {artifact.filename}: {exc.reason}") from exc
    except OSError as exc:
        raise AwsPublishError(f"S3 upload failed for {artifact.filename}: {exc}") from exc


class AwsCloudPublisher(CloudPublisher):
    """Publish validated RepairMind sessions to the AWS backend."""

    def __init__(
        self,
        *,
        api_url: str,
        region_name: str,
        profile_name: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        normalized_api_url = api_url.rstrip("/")

        if not normalized_api_url.startswith("https://"):
            raise ValueError("api_url must use HTTPS")

        if not region_name.strip():
            raise ValueError("region_name must not be empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._api_url = normalized_api_url
        self._region_name = region_name
        self._profile_name = profile_name
        self._timeout_seconds = timeout_seconds

    def publish(
        self,
        manifest: InspectionSessionManifest,
        payloads: Mapping[str, ArtifactPayload],
    ) -> CloudPublishResult:
        """Create an AWS session and upload every authorized artifact."""

        expected_filenames = {artifact.filename for artifact in manifest.artifacts}
        supplied_filenames = set(payloads)

        missing = expected_filenames - supplied_filenames
        unexpected = supplied_filenames - expected_filenames

        if missing:
            raise AwsPublishError("Missing artifact payloads: " + ", ".join(sorted(missing)))

        if unexpected:
            raise AwsPublishError("Unexpected artifact payloads: " + ", ".join(sorted(unexpected)))

        credentials = _frozen_credentials(
            profile_name=self._profile_name,
            region_name=self._region_name,
        )

        response = _signed_post(
            url=f"{self._api_url}/sessions",
            body=_json_bytes(_manifest_payload(manifest)),
            region_name=self._region_name,
            credentials=credentials,
            timeout_seconds=self._timeout_seconds,
        )

        response_session_id = _require_response_string(
            response,
            "session_id",
        )

        if response_session_id != manifest.session_id:
            raise AwsPublishError("AWS session API returned a different session_id.")

        manifest_uri = _manifest_uri(response)

        uploads = _parse_uploads(response)

        artifacts_by_type = _artifact_by_type(manifest)

        if len(uploads) != len(manifest.artifacts):
            raise AwsPublishError("AWS session API returned an unexpected number of uploads.")

        seen_types: set[str] = set()

        for instruction in uploads:
            if instruction.artifact_type in seen_types:
                raise AwsPublishError(
                    "AWS session API returned a duplicate upload instruction "
                    f"for {instruction.artifact_type}."
                )

            seen_types.add(instruction.artifact_type)

            artifact = artifacts_by_type.get(instruction.artifact_type)

            if artifact is None:
                raise AwsPublishError(
                    "AWS session API authorized an unexpected artifact type: "
                    f"{instruction.artifact_type}"
                )

            payload = payloads[artifact.filename]

            _upload_artifact(
                instruction=instruction,
                artifact=artifact,
                payload=payload,
                timeout_seconds=self._timeout_seconds,
            )

        if seen_types != set(artifacts_by_type):
            raise AwsPublishError("AWS session API did not authorize every manifest artifact.")

        return CloudPublishResult(
            session_id=manifest.session_id,
            provider="aws",
            manifest_uri=manifest_uri,
            artifact_count=len(manifest.artifacts),
        )
