"""Tests for the RepairMind Edge AWS session API."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "session_api"
    / "app.py"
)

SPEC = importlib.util.spec_from_file_location(
    "repairmind_aws_session_api",
    APP_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class FakeS3Client:
    """Minimal S3 test double for session creation tests."""

    def __init__(self) -> None:
        self.put_calls: list[dict[str, Any]] = []
        self.presign_calls: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        return {"ETag": '"test-etag"'}

    def generate_presigned_url(
        self,
        client_method: str,
        *,
        Params: dict[str, Any],
        ExpiresIn: int,
        HttpMethod: str,
    ) -> str:
        self.presign_calls.append(
            {
                "client_method": client_method,
                "Params": Params,
                "ExpiresIn": ExpiresIn,
                "HttpMethod": HttpMethod,
            }
        )
        key = Params["Key"]
        return f"https://example.invalid/upload/{key}"


def _artifact(
    artifact_type: str,
    *,
    sha256: str | None = None,
    byte_count: int = 128,
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "filename": "client-controlled-name.bin",
        "byte_count": byte_count,
        "sha256": sha256 or ("a" * 64),
    }


def _valid_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "session_id": "scanner-demo-001",
        "created_at": "2026-10-07T00:00:00+01:00",
        "artifacts": [
            _artifact("INSPECTION_REPORT", byte_count=1490),
            _artifact(
                "ANNOTATED_IMAGE",
                sha256="b" * 64,
                byte_count=215809,
            ),
            _artifact(
                "NORMALIZED_IMAGE",
                sha256="c" * 64,
                byte_count=219529,
            ),
            _artifact(
                "ROI_OVERLAY",
                sha256="d" * 64,
                byte_count=238274,
            ),
        ],
    }


def test_validate_manifest_accepts_supported_artifacts() -> None:
    manifest = app.validate_manifest(_valid_manifest())

    assert manifest["schema_version"] == "1.0"
    assert manifest["session_id"] == "scanner-demo-001"

    artifact_types = {
        artifact["artifact_type"]
        for artifact in manifest["artifacts"]
    }
    assert artifact_types == {
        "INSPECTION_REPORT",
        "ANNOTATED_IMAGE",
        "NORMALIZED_IMAGE",
        "ROI_OVERLAY",
    }


def test_validate_manifest_ignores_client_filename() -> None:
    manifest = app.validate_manifest(_valid_manifest())

    filenames = {
        artifact["artifact_type"]: artifact["filename"]
        for artifact in manifest["artifacts"]
    }

    assert filenames["INSPECTION_REPORT"] == "inspection_report.json"
    assert filenames["ANNOTATED_IMAGE"] == "annotated.jpg"
    assert filenames["NORMALIZED_IMAGE"] == "normalized.jpg"
    assert filenames["ROI_OVERLAY"] == "roi_overlay.jpg"

    assert "client-controlled-name.bin" not in filenames.values()


def test_validate_manifest_requires_inspection_report() -> None:
    payload = _valid_manifest()
    payload["artifacts"] = [
        artifact
        for artifact in payload["artifacts"]
        if artifact["artifact_type"] != "INSPECTION_REPORT"
    ]

    with pytest.raises(
        app.RequestValidationError,
        match="Missing required artifact",
    ):
        app.validate_manifest(payload)


def test_validate_manifest_rejects_unsupported_artifact() -> None:
    payload = _valid_manifest()
    payload["artifacts"].append(
        _artifact("DEBUG_CONTOURS")
    )

    with pytest.raises(
        app.RequestValidationError,
        match="Unsupported artifact type",
    ):
        app.validate_manifest(payload)


def test_validate_manifest_rejects_duplicate_artifact_type() -> None:
    payload = _valid_manifest()
    payload["artifacts"].append(
        _artifact(
            "ROI_OVERLAY",
            sha256="e" * 64,
        )
    )

    with pytest.raises(
        app.RequestValidationError,
        match="Duplicate artifact type",
    ):
        app.validate_manifest(payload)


@pytest.mark.parametrize(
    "session_id",
    [
        "",
        "../escape",
        "scanner/demo",
        " scanner-demo",
        "scanner demo",
        "x" * 129,
    ],
)
def test_validate_manifest_rejects_invalid_session_id(
    session_id: str,
) -> None:
    payload = _valid_manifest()
    payload["session_id"] = session_id

    with pytest.raises(app.RequestValidationError):
        app.validate_manifest(payload)


def test_validate_manifest_rejects_invalid_sha256() -> None:
    payload = _valid_manifest()
    payload["artifacts"][0]["sha256"] = "not-a-sha256"

    with pytest.raises(
        app.RequestValidationError,
        match="Invalid SHA-256",
    ):
        app.validate_manifest(payload)


@pytest.mark.parametrize(
    "byte_count",
    [
        0,
        -1,
        True,
        app.MAX_ARTIFACT_BYTES + 1,
    ],
)
def test_validate_manifest_rejects_invalid_byte_count(
    byte_count: Any,
) -> None:
    payload = _valid_manifest()
    payload["artifacts"][0]["byte_count"] = byte_count

    with pytest.raises(
        app.RequestValidationError,
        match="Invalid byte_count",
    ):
        app.validate_manifest(payload)


def test_validate_manifest_requires_timezone_aware_timestamp() -> None:
    payload = _valid_manifest()
    payload["created_at"] = "2026-10-07T00:00:00"

    with pytest.raises(
        app.RequestValidationError,
        match="timezone",
    ):
        app.validate_manifest(payload)


def test_build_artifact_key_is_server_controlled() -> None:
    key = app.build_artifact_key(
        "scanner-demo-001",
        "ROI_OVERLAY",
    )

    assert (
        key
        == "sessions/scanner-demo-001/artifacts/roi_overlay.jpg"
    )


def test_create_session_persists_manifest_and_generates_uploads() -> None:
    s3_client = FakeS3Client()

    result = app.create_session(
        _valid_manifest(),
        s3_client=s3_client,
        bucket_name="repairmind-test-bucket",
        presigned_url_ttl_seconds=900,
    )

    assert result["status"] == "UPLOAD_PENDING"
    assert result["session_id"] == "scanner-demo-001"
    assert result["upload_url_expires_in_seconds"] == 900

    assert result["manifest"] == {
        "bucket": "repairmind-test-bucket",
        "object_key": "sessions/scanner-demo-001/manifest.json",
    }

    assert len(s3_client.put_calls) == 1
    put_call = s3_client.put_calls[0]

    assert put_call["Bucket"] == "repairmind-test-bucket"
    assert (
        put_call["Key"]
        == "sessions/scanner-demo-001/manifest.json"
    )
    assert put_call["ContentType"] == "application/json"
    assert put_call["ServerSideEncryption"] == "AES256"

    stored_manifest = json.loads(put_call["Body"])
    assert stored_manifest["session_id"] == "scanner-demo-001"

    assert len(result["uploads"]) == 4
    assert len(s3_client.presign_calls) == 4

    for presign_call in s3_client.presign_calls:
        assert presign_call["client_method"] == "put_object"
        assert presign_call["ExpiresIn"] == 900
        assert presign_call["HttpMethod"] == "PUT"

        params = presign_call["Params"]
        assert params["Bucket"] == "repairmind-test-bucket"
        assert params["Key"].startswith(
            "sessions/scanner-demo-001/artifacts/"
        )
        assert params["ServerSideEncryption"] == "AES256"
        assert "repairmind-sha256" in params["Metadata"]


def test_parse_request_accepts_json_object() -> None:
    event = {
        "body": json.dumps(_valid_manifest()),
        "isBase64Encoded": False,
    }

    payload = app.parse_request(event)

    assert payload["session_id"] == "scanner-demo-001"


def test_parse_request_rejects_invalid_json() -> None:
    event = {
        "body": "{invalid-json",
        "isBase64Encoded": False,
    }

    with pytest.raises(
        app.RequestValidationError,
        match="valid JSON",
    ):
        app.parse_request(event)


def test_parse_request_rejects_non_object_json() -> None:
    event = {
        "body": '["not", "an", "object"]',
        "isBase64Encoded": False,
    }

    with pytest.raises(
        app.RequestValidationError,
        match="JSON object",
    ):
        app.parse_request(event)


def test_lambda_handler_returns_400_for_invalid_request() -> None:
    response = app.lambda_handler(
        {
            "body": "{invalid-json",
            "isBase64Encoded": False,
        },
        None,
    )

    assert response["statusCode"] == 400

    body = json.loads(response["body"])
    assert body["error"]["code"] == "INVALID_REQUEST"