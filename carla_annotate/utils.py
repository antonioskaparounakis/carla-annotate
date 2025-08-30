from typing import Tuple

import numpy as np

import carla


def euclidean_distance(loc1: carla.Location, loc2: carla.Location) -> float:
    return loc1.distance(loc2)


def carla_image_to_rgb(image: carla.Image) -> np.ndarray:
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4
    )
    rgb = bgra[..., :3][..., ::-1].copy()
    return rgb


def rgb_to_opencv_image(img: np.ndarray) -> np.ndarray:
    bgr = img[..., ::-1].copy()
    return bgr


def bbox_to_yolo(
    bbox: Tuple[int, int, int, int], image_width: int, image_height: int
) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height
