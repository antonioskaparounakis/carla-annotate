from typing import Optional, Tuple

import carla
import numpy as np

class CarlaCameraProjector:
    _MIN_DEPTH_M: float = 1e-2

    def __init__(self, camera: carla.Sensor) -> None:
        self._camera: carla.Sensor = camera

        self._image_width: int = int(self._camera.attributes["image_size_x"])
        self._image_height: int = int(self._camera.attributes["image_size_y"])
        self._fov: float = np.deg2rad(float(self._camera.attributes["fov"]))

        self._intrinsics: np.ndarray = self._compute_intrinsics()
        self._focal_length: Tuple[float, float] = float(self._intrinsics[0, 0]), float(self._intrinsics[1, 1])
        self._principal: Tuple[float, float] = float(self._intrinsics[0, 2]), float(self._intrinsics[1, 2])

    def project_bbox(self, bbox: carla.BoundingBox, transform: carla.Transform) -> Optional[Tuple[float, float, float, float]]:
        world_to_camera = self._world_to_camera()

        image_points = []
        vertices = bbox.get_world_vertices(transform)
        for vertex in vertices:
            image_point = self._project_image_point(vertex, world_to_camera)
            if image_point is not None:
                image_points.append(image_point)

        if not image_points:
            return None

        image_points = np.asarray(image_points, dtype=np.float64)

        xmin, ymin = image_points.min(axis=0)
        xmax, ymax = image_points.max(axis=0)

        xmin, ymin, xmax, ymax = self._clamp_bbox(xmin, ymin, xmax, ymax)

        if xmax <= xmin or ymax <= ymin:
            return None

        return xmin, ymin, xmax, ymax

    def _compute_intrinsics(self) -> np.ndarray:
        fx = (self._image_width / 2.0) / np.tan(self._fov / 2.0)
        fy = fx
        cx, cy = self._image_width / 2.0, self._image_height / 2.0
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    def _world_to_camera(self) -> np.ndarray:
        return np.array(self._camera.get_transform().get_inverse_matrix(), dtype=np.float64)

    def _project_image_point(self, p_w: carla.Location, T_w2c: np.ndarray) -> Optional[Tuple[float, float]]:
        fx, fy = self._focal_length
        cx, cy = self._principal

        X_w = np.array([p_w.x, p_w.y, p_w.z, 1.0], dtype=np.float64)
        x_c, y_c, z_c, _ = T_w2c @ X_w

        if x_c <= self._MIN_DEPTH_M or not np.isfinite(x_c):
            return None

        u = cx + fx * (y_c / x_c)
        v = cy - fy * (z_c / x_c)

        if not (np.isfinite(u) and np.isfinite(v)):
            return None

        return float(u), float(v)

    def _clamp_bbox(self, xmin: float, ymin: float, xmax: float, ymax: float) -> Tuple[float, float, float, float]:
        xmin = float(np.clip(xmin, 0, self._image_width - 1))
        xmax = float(np.clip(xmax, 0, self._image_width - 1))
        ymin = float(np.clip(ymin, 0, self._image_height - 1))
        ymax = float(np.clip(ymax, 0, self._image_height - 1))
        return xmin, ymin, xmax, ymax
