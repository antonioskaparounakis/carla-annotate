from typing import Tuple

import cv2
import numpy as np

from carla_annotate.domain import AnnotatedImage
from carla_annotate.utils import rgb_to_opencv_image


class OpencvVisualizer:
    _BBOX_COLOR: Tuple[int, int, int] = (0, 255, 0)
    _BBOX_THICKNESS: int = 1

    def __init__(self, window_name: str) -> None:
        self._window_name = window_name

    def __enter__(self):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyWindow(self._window_name)
        pass

    def visualize(self, annotated_image: AnnotatedImage) -> None:
        bgr = rgb_to_opencv_image(annotated_image.image)
        for instance in annotated_image.instances:
            self._draw_bounding_box(bgr, instance.bbox)
        cv2.imshow(self._window_name, bgr)
        cv2.waitKey(1)

    def _draw_bounding_box(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> None:
        x_min, y_min, x_max, y_max = bbox
        top_left = x_min, y_min
        bottom_right = x_max, y_max
        cv2.rectangle(
            image, top_left, bottom_right, self._BBOX_COLOR, self._BBOX_THICKNESS
        )
