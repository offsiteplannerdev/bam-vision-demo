from pathlib import Path

import yaml

from src.config import CLASSES
from src.convert_cvat_to_yolo import convert_cvat_annotations


def test_convert_cvat_annotations_writes_yolo_labels_and_data_yaml(tmp_path: Path) -> None:
    """CVAT boxes should become normalized YOLO rows and empty images become ok."""
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / "task.xml").write_text(
        """
<annotations>
  <version>1.1</version>
  <image id="0" name="images/train/part_001.jpg" width="100" height="200">
    <box label="scratch" xtl="10" ytl="20" xbr="50" ybr="100" />
  </image>
  <image id="1" name="images/train/part_002.jpg" width="100" height="200" />
</annotations>
""".strip(),
        encoding="utf-8",
    )

    output_dir = tmp_path / "yolo"

    summary = convert_cvat_annotations(annotations_dir, output_dir)

    assert summary.image_count == 2
    assert summary.class_counts["ok"] == 1
    assert summary.class_counts["scratch"] == 1
    assert (output_dir / "labels/train/part_001.txt").read_text(encoding="utf-8") == "1 0.300000 0.300000 0.400000 0.400000\n"
    assert (output_dir / "labels/train/part_002.txt").read_text(encoding="utf-8") == ""

    data_yaml = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/train"
    assert data_yaml["names"] == {index: name for index, name in enumerate(CLASSES)}


def test_convert_cvat_annotations_dry_run_does_not_write_files(tmp_path: Path) -> None:
    """Dry-run should parse and summarize without creating YOLO output."""
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / "task.xml").write_text(
        """
<annotations>
  <version>1.1</version>
  <image id="0" name="part_001.jpg" width="100" height="100" />
</annotations>
""".strip(),
        encoding="utf-8",
    )

    output_dir = tmp_path / "yolo"

    summary = convert_cvat_annotations(annotations_dir, output_dir, dry_run=True)

    assert summary.image_count == 1
    assert summary.class_counts["ok"] == 1
    assert not output_dir.exists()
