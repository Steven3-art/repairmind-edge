"""Command-line interface for RepairMind Edge."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
from pydantic import ValidationError

from repairmind_edge.agent import (
    InspectionAgent,
    InspectionDecision,
    InspectionState,
    build_electrical_measurement_evidence,
)
from repairmind_edge.aws_publisher import (
    AwsCloudPublisher,
    AwsPublishError,
)
from repairmind_edge.cloud import CloudPublisher, InMemoryCloudPublisher
from repairmind_edge.cloud_session import build_cloud_session
from repairmind_edge.models import (
    ElectricalMeasurement,
    InspectionResult,
    VisualEvidence,
)
from repairmind_edge.perception import (
    ImageArray,
    PCBPerception,
    PerceptionError,
)
from repairmind_edge.reporting import build_inspection_report


def _build_parser() -> argparse.ArgumentParser:
    """Build the RepairMind Edge command-line parser."""

    parser = argparse.ArgumentParser(
        prog="repairmind",
        description=(
            "Evidence-driven visual inspection and troubleshooting for electronic and edge devices."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect an electronic board image.",
    )

    inspect_parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image.",
    )

    inspect_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts") / "inspection",
        help="Directory where inspection artifacts will be written.",
    )

    reason_parser = subparsers.add_parser(
        "reason",
        help="Select the next inspection action from structured evidence.",
    )

    reason_parser.add_argument(
        "result",
        type=Path,
        help="Path to an inspection result.json file.",
    )

    reason_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=("Path for decision.json. Defaults to the inspection result directory."),
    )

    reason_parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Inspection iteration number.",
    )

    measure_parser = subparsers.add_parser(
        "measure",
        help=("Add an electrical measurement and evaluate the next inspection action."),
    )

    measure_parser.add_argument(
        "result",
        type=Path,
        help="Path to an inspection result.json file.",
    )

    measure_parser.add_argument(
        "--quantity",
        required=True,
        help="Measured electrical quantity, for example voltage.",
    )

    measure_parser.add_argument(
        "--value",
        required=True,
        type=float,
        help="Observed numeric measurement value.",
    )

    measure_parser.add_argument(
        "--unit",
        required=True,
        help="Measurement unit, for example V.",
    )

    measure_parser.add_argument(
        "--point",
        required=True,
        dest="measurement_point",
        help="Logical measurement point, for example board_supply.",
    )

    measure_parser.add_argument(
        "--iteration",
        type=int,
        default=2,
        help="Inspection iteration number. Defaults to 2.",
    )

    measure_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Directory for measurement and decision artifacts. "
            "Defaults to the inspection result directory."
        ),
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the reproducible end-to-end RepairMind Edge demo.",
    )

    demo_parser.add_argument(
        "image",
        type=Path,
        help="Path to the input PCB image.",
    )

    demo_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts") / "demo",
        help="Directory where demo artifacts will be written.",
    )

    demo_parser.add_argument(
        "--measurement-value",
        type=float,
        default=None,
        help=(
            "Optional observed electrical measurement. "
            "When omitted, the demo stops after iteration 1."
        ),
    )

    demo_parser.add_argument(
        "--measurement-quantity",
        default="voltage",
        help="Electrical quantity associated with the measurement.",
    )

    demo_parser.add_argument(
        "--measurement-unit",
        default="V",
        help="Unit associated with the measurement.",
    )

    demo_parser.add_argument(
        "--measurement-point",
        default="board_supply",
        help="Logical point where the measurement was collected.",
    )

    cloud_parser = subparsers.add_parser(
        "cloud",
        help="Manage cloud publication of RepairMind inspection sessions.",
    )

    cloud_subparsers = cloud_parser.add_subparsers(
        dest="cloud_command",
        required=True,
    )

    cloud_publish_parser = cloud_subparsers.add_parser(
        "publish",
        help="Publish a validated inspection session.",
    )

    cloud_publish_parser.add_argument(
        "artifacts",
        type=Path,
        help="Directory containing the inspection report and generated artifacts.",
    )

    cloud_publish_parser.add_argument(
        "--provider",
        choices=("memory", "aws"),
        default="memory",
        help="Cloud publication provider.",
    )

    cloud_publish_parser.add_argument(
        "--session-id",
        default=None,
        help="Optional explicit inspection session identifier.",
    )

    cloud_publish_parser.add_argument(
        "--api-url",
        default=None,
        help="RepairMind AWS API base URL. Required for provider aws.",
    )

    cloud_publish_parser.add_argument(
        "--profile",
        default=None,
        help="Optional AWS CLI profile used for SigV4 authentication.",
    )

    cloud_publish_parser.add_argument(
        "--region",
        default="eu-north-1",
        help="AWS region used for SigV4 authentication.",
    )

    return parser


def _write_image(path: Path, image: ImageArray) -> None:
    """Write an OpenCV image and fail explicitly if writing fails."""

    success = cv2.imwrite(str(path), image)

    if not success:
        raise OSError(f"OpenCV could not write image: {path}")


def _inspect(image_path: Path, output_directory: Path) -> int:
    """Run a PCB inspection and persist its evidence artifacts."""

    if not image_path.exists():
        print(
            f"Error: input image does not exist: {image_path}",
            file=sys.stderr,
        )
        return 2

    if not image_path.is_file():
        print(
            f"Error: input path is not a file: {image_path}",
            file=sys.stderr,
        )
        return 2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        print(
            f"Error: OpenCV could not decode image: {image_path}",
            file=sys.stderr,
        )
        return 2

    image = cast(ImageArray, image.astype("uint8", copy=False))

    perception = PCBPerception()

    try:
        output = perception.inspect(image)
    except PerceptionError as exc:
        print(
            f"Error: perception failed: {exc}",
            file=sys.stderr,
        )
        return 3

    try:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        annotated_path = output_directory / "annotated.jpg"
        normalized_path = output_directory / "normalized.jpg"
        debug_path = output_directory / "debug_contours.jpg"
        roi_overlay_path = output_directory / "roi_overlay.jpg"
        result_path = output_directory / "result.json"

        _write_image(
            annotated_path,
            output.annotated_image,
        )

        _write_image(
            normalized_path,
            output.normalized_image,
        )

        _write_image(
            debug_path,
            output.debug_image,
        )

        _write_image(
            roi_overlay_path,
            output.roi_overlay_image,
        )

        result_path.write_text(
            json.dumps(
                output.result.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    except OSError as exc:
        print(
            f"Error: could not write inspection artifacts: {exc}",
            file=sys.stderr,
        )
        return 4

    localization = output.result.localization

    print("RepairMind Edge inspection complete")
    print(f"Input:       {image_path}")
    print(f"Output:      {output_directory}")
    print("Board:       " + ("detected" if localization.detected else "uncertain"))
    print(f"Confidence:  {localization.confidence:.3f}")
    print(f"Area ratio:  {localization.area_ratio:.3f}")
    print(f"Evidence:    {len(output.result.evidence)} item(s)")
    print()
    print(f"Result:      {result_path}")
    print(f"Annotated:   {annotated_path}")
    print(f"Normalized:  {normalized_path}")
    print(f"Debug:       {debug_path}")
    print(f"ROI Overlay: {roi_overlay_path}")

    return 0


def _load_result_evidence(
    result_path: Path,
) -> tuple[VisualEvidence, ...]:
    """Load and validate evidence from an inspection result."""

    try:
        raw_document: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in inspection result: {exc}") from exc

    if not isinstance(raw_document, dict):
        raise ValueError("inspection result must contain a JSON object")

    raw_evidence = raw_document.get("evidence")

    if not isinstance(raw_evidence, list):
        raise ValueError("inspection result must contain an evidence array")

    try:
        return tuple(VisualEvidence.model_validate(item) for item in raw_evidence)
    except ValidationError as exc:
        raise ValueError(f"inspection result contains invalid evidence: {exc}") from exc


def _reason(
    result_path: Path,
    output_path: Path | None,
    iteration: int,
) -> int:
    """Select and persist the next action from inspection evidence."""

    if not result_path.exists():
        print(
            f"Error: inspection result does not exist: {result_path}",
            file=sys.stderr,
        )
        return 2

    if not result_path.is_file():
        print(
            f"Error: inspection result path is not a file: {result_path}",
            file=sys.stderr,
        )
        return 2

    if iteration < 0:
        print(
            "Error: iteration must be greater than or equal to zero.",
            file=sys.stderr,
        )
        return 2

    try:
        evidence = _load_result_evidence(result_path)
    except (OSError, ValueError) as exc:
        print(
            f"Error: could not load inspection result: {exc}",
            file=sys.stderr,
        )
        return 3

    state = InspectionState(
        evidence=evidence,
        iteration=iteration,
    )

    decision = InspectionAgent().decide(state)

    decision_path = output_path if output_path is not None else result_path.parent / "decision.json"

    try:
        decision_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        decision_path.write_text(
            decision.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(
            f"Error: could not write agent decision: {exc}",
            file=sys.stderr,
        )
        return 4

    print("RepairMind Edge agent decision")
    print(f"Evidence:   {len(state.evidence)} item(s)")
    print(f"Iteration:  {decision.iteration}")
    print(f"Action:     {decision.action.action_type}")
    print(
        "Target ROI: "
        + (decision.action.target_roi_id if decision.action.target_roi_id is not None else "-")
    )
    print()
    print("Reason:")
    print(decision.action.reason)
    print()
    print("Next:")
    print(decision.action.instruction)
    print()
    print(f"Evidence IDs used: {len(decision.evidence_ids)}")
    print(f"Decision:   {decision_path}")

    return 0


def _measure(
    result_path: Path,
    *,
    quantity: str,
    value: float,
    unit: str,
    measurement_point: str,
    iteration: int,
    output_directory: Path | None,
) -> int:
    """Add an electrical measurement and evaluate the next action."""

    if not result_path.exists():
        print(
            f"Error: inspection result does not exist: {result_path}",
            file=sys.stderr,
        )
        return 2

    if not result_path.is_file():
        print(
            f"Error: inspection result path is not a file: {result_path}",
            file=sys.stderr,
        )
        return 2

    if iteration < 0:
        print(
            "Error: iteration must be greater than or equal to zero.",
            file=sys.stderr,
        )
        return 2

    try:
        existing_evidence = _load_result_evidence(result_path)

        measurement = ElectricalMeasurement(
            quantity=quantity,
            value=value,
            unit=unit,
            measurement_point=measurement_point,
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(
            f"Error: could not build measurement evidence: {exc}",
            file=sys.stderr,
        )
        return 3

    measurement_evidence = build_electrical_measurement_evidence(
        measurement,
        evidence_id="measurement_001",
    )

    state = InspectionState(
        evidence=existing_evidence + (measurement_evidence,),
        iteration=iteration,
    )

    decision = InspectionAgent().decide(state)

    artifact_directory = output_directory if output_directory is not None else result_path.parent

    measurement_path = artifact_directory / "measurement.json"
    decision_path = artifact_directory / f"decision_iteration_{iteration}.json"

    try:
        artifact_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        measurement_path.write_text(
            measurement_evidence.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        decision_path.write_text(
            decision.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(
            f"Error: could not write measurement artifacts: {exc}",
            file=sys.stderr,
        )
        return 4

    print("RepairMind Edge measurement accepted")
    print(f"Input:       {result_path}")
    print(f"Quantity:    {measurement.quantity}")
    print(f"Value:       {measurement.value:g} {measurement.unit}")
    print(f"Point:       {measurement.measurement_point}")
    print(f"Evidence:    {len(state.evidence)} item(s)")
    print(f"Iteration:   {decision.iteration}")
    print(f"Action:      {decision.action.action_type}")
    print()
    print("Reason:")
    print(decision.action.reason)
    print()
    print("Next:")
    print(decision.action.instruction)
    print()
    print(f"Measurement: {measurement_path}")
    print(f"Decision:    {decision_path}")

    return 0


def _load_json_model(
    path: Path,
    model_type: type[InspectionResult] | type[InspectionDecision] | type[VisualEvidence],
) -> InspectionResult | InspectionDecision | VisualEvidence:
    """Load and validate a persisted RepairMind Edge artifact."""

    try:
        raw_document: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in artifact {path}: {exc}") from exc

    try:
        return model_type.model_validate(raw_document)
    except ValidationError as exc:
        raise ValueError(f"invalid artifact {path}: {exc}") from exc


def _write_demo_report(
    *,
    result_path: Path,
    iteration_1_path: Path,
    output_directory: Path,
    measurement: ElectricalMeasurement | None = None,
    iteration_2_path: Path | None = None,
) -> Path:
    """Build and persist the consolidated demo inspection report."""

    inspection_result = _load_json_model(
        result_path,
        InspectionResult,
    )
    iteration_1 = _load_json_model(
        iteration_1_path,
        InspectionDecision,
    )

    if not isinstance(inspection_result, InspectionResult):
        raise TypeError("expected an InspectionResult artifact")

    if not isinstance(iteration_1, InspectionDecision):
        raise TypeError("expected an InspectionDecision artifact")

    iteration_2: InspectionDecision | None = None

    if iteration_2_path is not None:
        loaded_iteration_2 = _load_json_model(
            iteration_2_path,
            InspectionDecision,
        )

        if not isinstance(
            loaded_iteration_2,
            InspectionDecision,
        ):
            raise TypeError("expected an InspectionDecision artifact")

        iteration_2 = loaded_iteration_2

    report = build_inspection_report(
        inspection_result=inspection_result,
        iteration_1=iteration_1,
        measurement=measurement,
        iteration_2=iteration_2,
    )

    report_path = output_directory / "inspection_report.json"

    report_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    return report_path


def _demo(
    image_path: Path,
    output_directory: Path,
    *,
    measurement_value: float | None,
    measurement_quantity: str,
    measurement_unit: str,
    measurement_point: str,
) -> int:
    """Run the reproducible end-to-end inspection demo."""

    print("RepairMind Edge end-to-end demo")
    print("=" * 35)
    print()

    print("[1/3] OpenCV 5 perception")
    inspect_exit_code = _inspect(
        image_path=image_path,
        output_directory=output_directory,
    )

    if inspect_exit_code != 0:
        return inspect_exit_code

    result_path = output_directory / "result.json"
    decision_iteration_1_path = output_directory / "decision_iteration_1.json"

    print()
    print("[2/3] Agent reasoning — iteration 1")

    reason_exit_code = _reason(
        result_path=result_path,
        output_path=decision_iteration_1_path,
        iteration=1,
    )

    if reason_exit_code != 0:
        return reason_exit_code

    if measurement_value is None:
        try:
            report_path = _write_demo_report(
                result_path=result_path,
                iteration_1_path=decision_iteration_1_path,
                output_directory=output_directory,
            )
        except (OSError, ValueError, TypeError) as exc:
            print(
                f"Error: could not write inspection report: {exc}",
                file=sys.stderr,
            )
            return 4

        print()
        print("Demo paused")
        print("The agent requested additional evidence. No electrical measurement was supplied.")
        print()
        print("Re-run the demo with --measurement-value to continue the inspection loop.")
        print()
        print(f"Report:      {report_path}")

        return 0

    print()
    print("[3/3] Electrical evidence — iteration 2")

    measure_exit_code = _measure(
        result_path=result_path,
        quantity=measurement_quantity,
        value=measurement_value,
        unit=measurement_unit,
        measurement_point=measurement_point,
        iteration=2,
        output_directory=output_directory,
    )

    if measure_exit_code != 0:
        return measure_exit_code

    measurement = ElectricalMeasurement(
        quantity=measurement_quantity,
        value=measurement_value,
        unit=measurement_unit,
        measurement_point=measurement_point,
    )

    decision_iteration_2_path = output_directory / "decision_iteration_2.json"

    try:
        report_path = _write_demo_report(
            result_path=result_path,
            iteration_1_path=decision_iteration_1_path,
            output_directory=output_directory,
            measurement=measurement,
            iteration_2_path=decision_iteration_2_path,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(
            f"Error: could not write inspection report: {exc}",
            file=sys.stderr,
        )
        return 4

    print()
    print("=" * 35)
    print("RepairMind Edge demo complete")
    print(f"Artifacts: {output_directory}")
    print(f"Report:    {report_path}")

    return 0


def _cloud_publish(
    artifact_directory: Path,
    *,
    provider: str,
    session_id: str | None,
    api_url: str | None,
    profile: str | None,
    region: str,
) -> int:
    """Build and publish a validated RepairMind inspection session."""

    try:
        manifest, payloads = build_cloud_session(
            artifact_directory,
            session_id=session_id,
        )
    except (OSError, ValueError) as exc:
        print(
            f"Error: could not build cloud session: {exc}",
            file=sys.stderr,
        )
        return 3

    publisher: CloudPublisher

    if provider == "memory":
        publisher = InMemoryCloudPublisher()
    elif provider == "aws":
        if api_url is None or not api_url.strip():
            print(
                "Error: --api-url is required when --provider aws is used.",
                file=sys.stderr,
            )
            return 2

        try:
            publisher = AwsCloudPublisher(
                api_url=api_url,
                region_name=region,
                profile_name=profile,
            )
        except ValueError as exc:
            print(
                f"Error: invalid AWS cloud configuration: {exc}",
                file=sys.stderr,
            )
            return 2
    else:
        print(
            f"Error: unsupported cloud provider: {provider}",
            file=sys.stderr,
        )
        return 2

    try:
        result = publisher.publish(
            manifest,
            payloads,
        )
    except (ValueError, AwsPublishError) as exc:
        print(
            f"Error: cloud publication failed: {exc}",
            file=sys.stderr,
        )
        return 4

    print("RepairMind Edge cloud publication complete")
    print(f"Session:   {result.session_id}")
    print(f"Provider:  {result.provider}")
    print(f"Artifacts: {result.artifact_count}")
    print(f"Manifest:  {result.manifest_uri}")
    print()
    print("Integrity:")

    for artifact in manifest.artifacts:
        print(f"  {artifact.filename}: {artifact.size_bytes} bytes sha256={artifact.sha256}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RepairMind Edge CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        return _inspect(
            image_path=args.image,
            output_directory=args.output,
        )

    if args.command == "reason":
        return _reason(
            result_path=args.result,
            output_path=args.output,
            iteration=args.iteration,
        )

    if args.command == "measure":
        return _measure(
            result_path=args.result,
            quantity=args.quantity,
            value=args.value,
            unit=args.unit,
            measurement_point=args.measurement_point,
            iteration=args.iteration,
            output_directory=args.output,
        )

    if args.command == "demo":
        return _demo(
            image_path=args.image,
            output_directory=args.output,
            measurement_value=args.measurement_value,
            measurement_quantity=args.measurement_quantity,
            measurement_unit=args.measurement_unit,
            measurement_point=args.measurement_point,
        )

    if args.command == "cloud":
        if args.cloud_command == "publish":
            return _cloud_publish(
                artifact_directory=args.artifacts,
                provider=args.provider,
                session_id=args.session_id,
                api_url=args.api_url,
                profile=args.profile,
                region=args.region,
            )

        parser.error(f"Unsupported cloud command: {args.cloud_command}")
        return 2

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
