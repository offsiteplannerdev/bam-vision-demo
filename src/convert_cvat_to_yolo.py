from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from src.config import CLASSES, PROCESSED_DIR, RAW_DIR


CLASS_TO_ID = {class_name: class_id for class_id, class_name in enumerate(CLASSES)}
IMAGE_SPLITS = {"train", "val", "test"}
OK_CLASS_NAME = "ok"


@dataclass(frozen=True)
class YoloBox:
    """Single YOLO bounding-box annotation."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        """Serializes the box to one YOLO label row.

        Returns:
            Space-separated label row with six-decimal normalized coordinates.
        """
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} "
            f"{self.y_center:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


@dataclass
class ImageLabels:
    """YOLO labels generated for one CVAT image entry."""

    image_name: str
    label_relative_path: Path
    boxes: list[YoloBox] = field(default_factory=list)


@dataclass
class ConversionSummary:
    """Aggregate counts for a CVAT-to-YOLO conversion run."""

    image_count: int = 0
    label_file_count: int = 0
    box_count: int = 0
    class_counts: dict[str, int] = field(default_factory=lambda: {class_name: 0 for class_name in CLASSES})

    def record_image(self, labels: ImageLabels) -> None:
        """Adds one converted image to the summary.

        Args:
            labels: Labels generated for an image entry.
        """
        self.image_count += 1
        self.label_file_count += 1
        if not labels.boxes:
            self.class_counts[OK_CLASS_NAME] += 1
            return

        self.box_count += len(labels.boxes)
        for box in labels.boxes:
            self.class_counts[CLASSES[box.class_id]] += 1


def convert_cvat_annotations(
    annotations_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> ConversionSummary:
    """Converts CVAT XML 1.1 image annotations to YOLO label files.

    The converter writes one `.txt` label file per CVAT `<image>` entry. Images with
    no boxes are treated as `ok` samples in the summary and receive an empty label
    file, which is the YOLO object-detection convention for negative images.

    Args:
        annotations_dir: Directory containing CVAT XML files.
        output_dir: Root directory for YOLO labels and `data.yaml`.
        dry_run: If true, only parse annotations and print the summary.
        overwrite: If true, replace existing label files and `data.yaml`.

    Returns:
        Summary of converted images and per-class counts.

    Raises:
        FileNotFoundError: If `annotations_dir` is missing or contains no XML files.
        ValueError: If annotations are malformed, unsupported, or unsafe to write.
        FileExistsError: If output files already exist and `overwrite` is false.
    """
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    xml_paths = _find_xml_files(annotations_dir)
    image_labels = _parse_xml_files(xml_paths)
    summary = _summarize_labels(image_labels)

    if not dry_run:
        _write_yolo_dataset(image_labels, output_dir, overwrite=overwrite)

    print_summary(summary, dry_run=dry_run, output_dir=output_dir)
    return summary


def print_summary(summary: ConversionSummary, *, dry_run: bool, output_dir: Path) -> None:
    """Prints a conversion summary by class.

    Args:
        summary: Conversion counts to display.
        dry_run: Whether this run skipped writing files.
        output_dir: Configured YOLO output directory.
    """
    mode = "dry-run" if dry_run else "write"
    print(f"CVAT to YOLO conversion summary ({mode})")
    print(f"Output directory: {Path(output_dir)}")
    print(f"Images: {summary.image_count}")
    print(f"Label files: {summary.label_file_count}")
    print(f"Boxes: {summary.box_count}")
    print("Classes:")
    for class_name in CLASSES:
        print(f"  {class_name}: {summary.class_counts[class_name]}")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the converter.

    Returns:
        Parsed command-line arguments.
    """
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Convert CVAT XML 1.1 annotations to YOLO labels.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=project_root / RAW_DIR / "annotations",
        help="Directory containing CVAT XML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / PROCESSED_DIR / "yolo",
        help="YOLO output root where labels/ and data.yaml are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse annotations and print the summary without writing output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing YOLO label files and data.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    convert_cvat_annotations(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


def _find_xml_files(annotations_dir: Path) -> list[Path]:
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"CVAT annotations directory does not exist: {annotations_dir}")

    xml_paths = sorted(annotations_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No CVAT XML files found in: {annotations_dir}")

    return xml_paths


def _parse_xml_files(xml_paths: list[Path]) -> list[ImageLabels]:
    image_labels: list[ImageLabels] = []
    seen_label_paths: set[Path] = set()

    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Failed to parse CVAT XML file: {xml_path}") from exc

        _validate_cvat_version(root, xml_path)
        for image_element in root.findall("image"):
            labels = _parse_image_element(image_element, xml_path)
            if labels.label_relative_path in seen_label_paths:
                raise ValueError(
                    "Duplicate image label path generated from CVAT annotations: "
                    f"{labels.label_relative_path}"
                )

            seen_label_paths.add(labels.label_relative_path)
            image_labels.append(labels)

    return image_labels


def _validate_cvat_version(root: ET.Element, xml_path: Path) -> None:
    version = root.findtext("version")
    if version != "1.1":
        raise ValueError(f"Expected CVAT XML version 1.1 in {xml_path}, got: {version!r}")


def _parse_image_element(image_element: ET.Element, xml_path: Path) -> ImageLabels:
    image_name = image_element.get("name")
    if not image_name:
        raise ValueError(f"CVAT image entry without a name in: {xml_path}")

    image_width = _required_positive_float(image_element, "width", xml_path, image_name)
    image_height = _required_positive_float(image_element, "height", xml_path, image_name)
    label_relative_path = _label_relative_path(image_name)
    boxes = [
        _parse_box_element(box_element, image_width, image_height, xml_path, image_name)
        for box_element in image_element.findall("box")
    ]
    return ImageLabels(image_name=image_name, label_relative_path=label_relative_path, boxes=boxes)


def _parse_box_element(
    box_element: ET.Element,
    image_width: float,
    image_height: float,
    xml_path: Path,
    image_name: str,
) -> YoloBox:
    label = box_element.get("label")
    if label not in CLASS_TO_ID:
        raise ValueError(
            f"Unknown CVAT label {label!r} in image {image_name!r} from {xml_path}. "
            f"Expected one of: {', '.join(CLASSES)}"
        )

    xtl = _required_float(box_element, "xtl", xml_path, image_name)
    ytl = _required_float(box_element, "ytl", xml_path, image_name)
    xbr = _required_float(box_element, "xbr", xml_path, image_name)
    ybr = _required_float(box_element, "ybr", xml_path, image_name)

    clipped_xtl = min(max(xtl, 0.0), image_width)
    clipped_ytl = min(max(ytl, 0.0), image_height)
    clipped_xbr = min(max(xbr, 0.0), image_width)
    clipped_ybr = min(max(ybr, 0.0), image_height)

    box_width = clipped_xbr - clipped_xtl
    box_height = clipped_ybr - clipped_ytl
    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError(f"Invalid or out-of-bounds box for image {image_name!r} in {xml_path}")

    return YoloBox(
        class_id=CLASS_TO_ID[label],
        x_center=((clipped_xtl + clipped_xbr) / 2.0) / image_width,
        y_center=((clipped_ytl + clipped_ybr) / 2.0) / image_height,
        width=box_width / image_width,
        height=box_height / image_height,
    )


def _required_positive_float(element: ET.Element, attribute: str, xml_path: Path, image_name: str) -> float:
    value = _required_float(element, attribute, xml_path, image_name)
    if value <= 0.0:
        raise ValueError(f"Expected positive {attribute!r} for image {image_name!r} in {xml_path}")
    return value


def _required_float(element: ET.Element, attribute: str, xml_path: Path, image_name: str) -> float:
    raw_value = element.get(attribute)
    if raw_value is None:
        raise ValueError(f"Missing {attribute!r} for image {image_name!r} in {xml_path}")

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid {attribute!r} value for image {image_name!r} in {xml_path}: {raw_value}") from exc

    if not math.isfinite(value):
        raise ValueError(f"Non-finite {attribute!r} value for image {image_name!r} in {xml_path}: {raw_value}")
    return value


def _label_relative_path(image_name: str) -> Path:
    normalized_name = image_name.replace("\\", "/")
    if normalized_name.startswith("/"):
        raise ValueError(f"CVAT image names must be relative paths, got: {image_name}")

    parts = [part for part in normalized_name.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe CVAT image path: {image_name}")

    if parts[0] == "images":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"CVAT image path does not include a file name: {image_name}")

    return Path(*parts).with_suffix(".txt")


def _summarize_labels(image_labels: list[ImageLabels]) -> ConversionSummary:
    summary = ConversionSummary()
    for labels in image_labels:
        summary.record_image(labels)
    return summary


def _write_yolo_dataset(image_labels: list[ImageLabels], output_dir: Path, *, overwrite: bool) -> None:
    labels_dir = output_dir / "labels"
    data_yaml_path = output_dir / "data.yaml"
    expected_label_paths = {labels_dir / labels.label_relative_path for labels in image_labels}
    output_paths = sorted(expected_label_paths) + [data_yaml_path]

    if labels_dir.exists():
        stale_label_paths = sorted(path for path in labels_dir.rglob("*.txt") if path not in expected_label_paths)
        if stale_label_paths:
            stale_labels = "\n".join(f"  - {path}" for path in stale_label_paths[:10])
            raise FileExistsError(
                "YOLO labels directory contains stale label files not present in the current CVAT XML. "
                "Use a clean output directory or remove stale labels manually:\n"
                f"{stale_labels}"
            )

    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        existing = "\n".join(f"  - {path}" for path in existing_paths[:10])
        raise FileExistsError(
            "YOLO output files already exist. Pass --overwrite to replace them:\n" f"{existing}"
        )

    for labels in image_labels:
        label_path = labels_dir / labels.label_relative_path
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_text = "\n".join(box.to_line() for box in labels.boxes)
        if label_text:
            label_text = f"{label_text}\n"
        label_path.write_text(label_text, encoding="utf-8")

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text(_build_data_yaml(image_labels, output_dir), encoding="utf-8")


def _build_data_yaml(image_labels: list[ImageLabels], output_dir: Path) -> str:
    splits = sorted(
        {
            labels.label_relative_path.parts[0]
            for labels in image_labels
            if labels.label_relative_path.parts and labels.label_relative_path.parts[0] in IMAGE_SPLITS
        }
    )
    train_split = "train" if "train" in splits else splits[0] if splits else ""
    val_split = "val" if "val" in splits else train_split

    data: dict[str, object] = {"path": str(output_dir), "names": {index: name for index, name in enumerate(CLASSES)}}
    if train_split:
        data["train"] = f"images/{train_split}"
        data["val"] = f"images/{val_split}"
    else:
        data["train"] = "images"
        data["val"] = "images"
    if "test" in splits:
        data["test"] = "images/test"

    return yaml.safe_dump(data, sort_keys=False)


if __name__ == "__main__":
    main()
