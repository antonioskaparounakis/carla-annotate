import carla
import numpy as np


def euclidean_distance(loc1: carla.Location, loc2: carla.Location) -> float:
    return loc1.distance(loc2)


def carla_image_to_rgb(image: carla.Image) -> np.ndarray:
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    rgb = bgra[..., :3][..., ::-1].copy()
    return rgb


def rgb_to_opencv_image(img: np.ndarray) -> np.ndarray:
    bgr = img[..., ::-1].copy()
    return bgr
