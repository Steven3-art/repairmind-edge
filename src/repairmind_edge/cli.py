"""Command-line interface for RepairMind Edge."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2

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
    print(f"Input:      {image_path}")
    print(f"Output:     {output_directory}")
    print("Board:      " + ("detected" if localization.detected else "uncertain"))
    print(f"Confidence: {localization.confidence:.3f}")
    print(f"Area ratio: {localization.area_ratio:.3f}")
    print(f"Evidence:   {len(output.result.evidence)} item(s)")
    print()
    print(f"Result:     {result_path}")
    print(f"Annotated:  {annotated_path}")
    print(f"Normalized: {normalized_path}")
    print(f"Debug:      {debug_path}")
    print(f"ROI Overlay: {roi_overlay_path}")

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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
