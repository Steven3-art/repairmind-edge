"""RepairMind Edge AWS session-ingestion API.

This Lambda validates an inspection-session manifest, persists the validated
manifest to S3, and returns short-lived presigned PUT URLs for the artifacts.

Artifact object keys are always derived server-side. Client-supplied S3 keys
are never accepted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

SCHEMA_VERSION = "1.0"

ALLOWED_ARTIFACT_TYPES = frozenset(
    {
        "INSPECTION_REPORT",
        "ANNOTATED_IMAGE",
        "NORMALIZED_IMAGE",
        "ROI_OVERLAY",
    }
)

REQUIRED_ARTIFACT_TYPES = frozenset({"INSPECTION_REPORT"})

ARTIFACT_FILENAMES = {
    "INSPECTION_REPORT": "inspection_report.json",
    "ANNOTATED_IMAGE": "annotated.jpg",
    "NORMALIZED_IMAGE": "normalized.jpg",
    "ROI_OVERLAY": "roi_overlay.jpg",
}

ARTIFACT_CONTENT_TYPES = {
    "INSPECTION_REPORT": "application/json",
    "ANNOTATED_IMAGE": "image/jpeg",
    "NORMALIZED_IMAGE": "image/jpeg",
    "ROI_OVERLAY": "image/jpeg",
}

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

MAX_ARTIFACT_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 256 * 1024
DEFAULT_PRESIGNED_URL_TTL_SECONDS = 900


class RequestValidationError(ValueError):
    """Raised when an API request violates the RepairMind cloud contract."""


def _response(
    status_code: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "content-type": "application/json",
            "cache-control": "no-store",
        },
        "body": json.dumps(payload, separators=(",", ":"), sort_keys=True),
    }


def _error_response(
    status_code: int,
    code: str,
    message: str,
) -> dict[str, Any]:
    return _response(
        status_code,
        {
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


def _require_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(
            f"{field_name} must be a non-empty string."
        )
    return value.strip()


def _validate_timestamp(value: Any) -> str:
    timestamp = _require_string(value, "created_at")

    normalized = timestamp
    if timestamp.endswith("Z"):
        normalized = f"{timestamp[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RequestValidationError(
            "created_at must be a valid ISO 8601 timestamp."
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RequestValidationError(
            "created_at must include timezone information."
        )

    return timestamp


def _decode_event_body(event: dict[str, Any]) -> bytes:
    body = event.get("body")

    if not isinstance(body, str) or not body:
        raise RequestValidationError(
            "Request body must contain a JSON object."
        )

    try:
        if event.get("isBase64Encoded") is True:
            raw_body = base64.b64decode(body, validate=True)
        else:
            raw_body = body.encode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise RequestValidationError(
            "Request body encoding is invalid."
        ) from exc

    if len(raw_body) > MAX_REQUEST_BYTES:
        raise RequestValidationError(
            "Request body exceeds the maximum allowed size."
        )

    return raw_body


def parse_request(event: dict[str, Any]) -> dict[str, Any]:
    """Parse an API Gateway event into a JSON object."""

    raw_body = _decode_event_body(event)

    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RequestValidationError(
            "Request body must be valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise RequestValidationError(
            "Request body must be a JSON object."
        )

    return payload


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a RepairMind inspection-session manifest."""

    schema_version = _require_string(
        payload.get("schema_version"),
        "schema_version",
    )
    if schema_version != SCHEMA_VERSION:
        raise RequestValidationError(
            f"Unsupported schema_version: {schema_version}."
        )

    raw_session_id = payload.get("session_id")
    session_id = _require_string(raw_session_id, "session_id")

    if raw_session_id != session_id:
        raise RequestValidationError(
            "session_id must not contain leading or trailing whitespace."
        )

    if SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise RequestValidationError(
            "session_id contains unsupported characters or has invalid length."
        )

    created_at = _validate_timestamp(payload.get("created_at"))

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RequestValidationError(
            "artifacts must be a non-empty list."
        )

    normalized_artifacts: list[dict[str, Any]] = []
    seen_types: set[str] = set()

    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise RequestValidationError(
                f"artifacts[{index}] must be an object."
            )

        artifact_type = _require_string(
            artifact.get("artifact_type"),
            f"artifacts[{index}].artifact_type",
        )

        if artifact_type not in ALLOWED_ARTIFACT_TYPES:
            raise RequestValidationError(
                f"Unsupported artifact type: {artifact_type}."
            )

        if artifact_type in seen_types:
            raise RequestValidationError(
                f"Duplicate artifact type: {artifact_type}."
            )
        seen_types.add(artifact_type)

        sha256 = _require_string(
            artifact.get("sha256"),
            f"artifacts[{index}].sha256",
        ).lower()
        if SHA256_PATTERN.fullmatch(sha256) is None:
            raise RequestValidationError(
                f"Invalid SHA-256 for artifact {artifact_type}."
            )

        byte_count = artifact.get("byte_count")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or byte_count > MAX_ARTIFACT_BYTES
        ):
            raise RequestValidationError(
                f"Invalid byte_count for artifact {artifact_type}."
            )

        normalized_artifacts.append(
            {
                "artifact_type": artifact_type,
                "filename": ARTIFACT_FILENAMES[artifact_type],
                "content_type": ARTIFACT_CONTENT_TYPES[artifact_type],
                "byte_count": byte_count,
                "sha256": sha256,
            }
        )

    missing_types = REQUIRED_ARTIFACT_TYPES - seen_types
    if missing_types:
        missing = ", ".join(sorted(missing_types))
        raise RequestValidationError(
            f"Missing required artifact type(s): {missing}."
        )

    normalized_artifacts.sort(key=lambda item: item["artifact_type"])

    return {
        "schema_version": schema_version,
        "session_id": session_id,
        "created_at": created_at,
        "artifacts": normalized_artifacts,
    }


def build_artifact_key(
    session_id: str,
    artifact_type: str,
) -> str:
    """Derive an artifact object key from trusted server-side mappings."""

    filename = ARTIFACT_FILENAMES[artifact_type]
    return f"sessions/{session_id}/artifacts/{filename}"


def build_manifest_key(session_id: str) -> str:
    """Return the canonical S3 key for a validated session manifest."""

    return f"sessions/{session_id}/manifest.json"


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(
        manifest,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _manifest_sha256(manifest_bytes: bytes) -> str:
    return hashlib.sha256(manifest_bytes).hexdigest()


def create_session(
    payload: dict[str, Any],
    *,
    s3_client: BaseClient,
    bucket_name: str,
    presigned_url_ttl_seconds: int,
) -> dict[str, Any]:
    """Persist a validated manifest and create artifact upload URLs."""

    manifest = validate_manifest(payload)
    session_id = manifest["session_id"]
    manifest_key = build_manifest_key(session_id)
    serialized_manifest = _manifest_bytes(manifest)

    s3_client.put_object(
        Bucket=bucket_name,
        Key=manifest_key,
        Body=serialized_manifest,
        ContentType="application/json",
        ServerSideEncryption="AES256",
        Metadata={
            "repairmind-session-id": session_id,
            "repairmind-sha256": _manifest_sha256(serialized_manifest),
        },
    )

    uploads: list[dict[str, Any]] = []

    for artifact in manifest["artifacts"]:
        artifact_type = artifact["artifact_type"]
        object_key = build_artifact_key(session_id, artifact_type)
        content_type = artifact["content_type"]

        upload_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket_name,
                "Key": object_key,
                "ContentType": content_type,
                "Metadata": {
                    "repairmind-sha256": artifact["sha256"],
                },
                "ServerSideEncryption": "AES256",
            },
            ExpiresIn=presigned_url_ttl_seconds,
            HttpMethod="PUT",
        )

        uploads.append(
            {
                "artifact_type": artifact_type,
                "filename": artifact["filename"],
                "content_type": content_type,
                "byte_count": artifact["byte_count"],
                "sha256": artifact["sha256"],
                "object_key": object_key,
                "upload_url": upload_url,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "status": "UPLOAD_PENDING",
        "manifest": {
            "bucket": bucket_name,
            "object_key": manifest_key,
        },
        "uploads": uploads,
        "upload_url_expires_in_seconds": presigned_url_ttl_seconds,
    }


def _get_configuration() -> tuple[str, int]:
    bucket_name = os.environ.get("SESSION_BUCKET_NAME", "").strip()
    if not bucket_name:
        raise RuntimeError("SESSION_BUCKET_NAME is not configured.")

    raw_ttl = os.environ.get(
        "PRESIGNED_URL_TTL_SECONDS",
        str(DEFAULT_PRESIGNED_URL_TTL_SECONDS),
    )

    try:
        ttl = int(raw_ttl)
    except ValueError as exc:
        raise RuntimeError(
            "PRESIGNED_URL_TTL_SECONDS must be an integer."
        ) from exc

    if not 60 <= ttl <= 3600:
        raise RuntimeError(
            "PRESIGNED_URL_TTL_SECONDS must be between 60 and 3600."
        )

    return bucket_name, ttl


def lambda_handler(
    event: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """AWS Lambda entry point."""

    del context

    try:
        payload = parse_request(event)
        bucket_name, ttl = _get_configuration()
        s3_client = boto3.client("s3")

        result = create_session(
            payload,
            s3_client=s3_client,
            bucket_name=bucket_name,
            presigned_url_ttl_seconds=ttl,
        )

        LOGGER.info(
            "RepairMind inspection session accepted: session_id=%s artifacts=%d",
            result["session_id"],
            len(result["uploads"]),
        )

        return _response(201, result)

    except RequestValidationError as exc:
        LOGGER.warning("Rejected RepairMind session request: %s", exc)
        return _error_response(
            400,
            "INVALID_REQUEST",
            str(exc),
        )

    except (BotoCoreError, ClientError):
        LOGGER.exception("AWS S3 operation failed.")
        return _error_response(
            502,
            "STORAGE_ERROR",
            "The inspection session could not be persisted.",
        )

    except RuntimeError:
        LOGGER.exception("RepairMind Lambda configuration is invalid.")
        return _error_response(
            500,
            "CONFIGURATION_ERROR",
            "The session API is not configured correctly.",
        )

    except Exception:
        LOGGER.exception("Unexpected RepairMind session API failure.")
        return _error_response(
            500,
            "INTERNAL_ERROR",
            "An unexpected server error occurred.",
        )