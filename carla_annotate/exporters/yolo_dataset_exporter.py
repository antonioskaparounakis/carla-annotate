import shutil
from pathlib import Path
from typing import Tuple

import cv2
import yaml

from carla_annotate.domain import AnnotatedImage, Category
from carla_annotate.utils import bbox_to_yolo, rgb_to_opencv_image


class YoloDatasetExporter:
    VAL_RATIO: float = 0.2

    CATEGORY_TO_CLASS_INDEX = {Category.TRAFFIC_LIGHT: 0}

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.images_train_dir = self.dataset_dir / "images" / "train"
        self.images_val_dir = dataset_dir / "images" / "val"
        self.labels_train_dir = dataset_dir / "labels" / "train"
        self.labels_val_dir = dataset_dir / "labels" / "val"
        self.sample_idx = 0

    def __enter__(self):
        self._create_dirs()
        self._write_yaml()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize(val_ratio=self.VAL_RATIO)
        return False

    @property
    def summary(self) -> Tuple[int, Path]:
        return self.sample_idx, self.dataset_dir

    def export(self, annotated_image: AnnotatedImage):
        self._write_image(self.images_train_dir, annotated_image)
        self._write_label(self.labels_train_dir, annotated_image)
        self.sample_idx += 1

    def _finalize(self, val_ratio: float):
        images = sorted(self.images_train_dir.glob("*.jpg"))
        labels = sorted(self.labels_train_dir.glob("*.txt"))
        n_val = int(len(images) * val_ratio)
        if n_val == 0:
            return

        for img, lbl in zip(images[-n_val:], labels[-n_val:]):
            shutil.move(str(img), self.images_val_dir / img.name)
            shutil.move(str(lbl), self.labels_val_dir / lbl.name)

    def _create_dirs(self):
        self.images_train_dir.mkdir(parents=True, exist_ok=True)
        self.images_val_dir.mkdir(parents=True, exist_ok=True)
        self.labels_train_dir.mkdir(parents=True, exist_ok=True)
        self.labels_val_dir.mkdir(parents=True, exist_ok=True)

    def _write_yaml(self):
        data = {
            "path": str(self.dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {
                class_index: category.name.lower()
                for category, class_index in self.CATEGORY_TO_CLASS_INDEX.items()
            },
        }
        text = yaml.safe_dump(data, sort_keys=False)
        path = self.dataset_dir / f"{self.dataset_dir.stem}.yaml"
        path.write_text(text, encoding="utf-8")

    def _write_image(self, output_dir: Path, annotated_image: AnnotatedImage) -> Path:
        output_path = output_dir / f"{self.sample_idx:06d}.jpg"
        bgr = rgb_to_opencv_image(annotated_image.image)
        cv2.imwrite(str(output_path), bgr)
        return output_path

    def _write_label(self, output_dir: Path, annotated_image: AnnotatedImage):
        output_path = output_dir / f"{self.sample_idx:06d}.txt"
        rows = []
        for instance in annotated_image.instances:
            class_index = self.CATEGORY_TO_CLASS_INDEX[instance.category]
            x_center, y_center, width, height = bbox_to_yolo(
                instance.bbox, annotated_image.image_width, annotated_image.image_height
            )
            rows.append(
                f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
        output_path.write_text("\n".join(rows), encoding="utf-8")
