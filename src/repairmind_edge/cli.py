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
    InspectionState,
    build_electrical_measurement_evidence,
)
from repairmind_edge.models import ElectricalMeasurement, VisualEvidence
from repairmind_edge.perception import (
    ImageArray,
    PCBPerception,
    PerceptionError,
)


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
        print()
        print("Demo paused")
        print("The agent requested additional evidence. No electrical measurement was supplied.")
        print()
        print("Re-run the demo with --measurement-value to continue the inspection loop.")

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

    print()
    print("=" * 35)
    print("RepairMind Edge demo complete")
    print(f"Artifacts: {output_directory}")

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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
