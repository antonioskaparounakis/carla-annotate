from typing import List, Tuple

import numpy as np

import carla


class CameraProjector:
    NEAR_PLANE_METERS: float = 1e-2

    def __init__(self, camera: carla.Sensor):
        self._camera = camera
        self._image_width = int(self._camera.attributes["image_size_x"])
        self._image_height = int(self._camera.attributes["image_size_y"])
        self._fov_rad = np.deg2rad(float(self._camera.attributes["fov"]))
        self._intrinsics: np.ndarray = self._compute_intrinsics()

    def project(
        self, bboxes: List[Tuple[carla.Actor, carla.BoundingBox]]
    ) -> List[Tuple[carla.Actor, Tuple[int, int, int, int]]]:
        T_w2c = np.array(
            self._camera.get_transform().get_inverse_matrix(), dtype=np.float32
        )
        results = []
        for actor, bbox3d in bboxes:
            actor_tf = (
                carla.Transform()
                if isinstance(actor, carla.TrafficLight)
                else actor.get_transform()
            )

            vertices = bbox3d.get_world_vertices(actor_tf)
            points = []
            for vertex in vertices:
                projected = self._project_vertex(vertex, T_w2c)
                if projected is not None:
                    points.append(projected)

            if not points:
                continue

            xs, ys = zip(*points)
            x_min, x_max = np.clip([min(xs), max(xs)], 0, self._image_width - 1)
            y_min, y_max = np.clip([min(ys), max(ys)], 0, self._image_height - 1)

            if x_max <= x_min or y_max <= y_min:
                continue

            results.append((actor, (int(x_min), int(y_min), int(x_max), int(y_max))))

        return results

    def _compute_intrinsics(self) -> np.ndarray:
        fx = (self._image_width / 2.0) / np.tan(self._fov_rad / 2.0)
        fy = fx
        cx, cy = self._image_width / 2.0, self._image_height / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        return K

    def _project_vertex(self, vertex: carla.Location, T_w2c: np.ndarray):
        fx, fy = self._intrinsics[0, 0], self._intrinsics[1, 1]
        cx, cy = self._intrinsics[0, 2], self._intrinsics[1, 2]

        X_w = np.array([vertex.x, vertex.y, vertex.z, 1.0], dtype=np.float32)
        x_c, y_c, z_c, _ = T_w2c @ X_w

        if x_c <= self.NEAR_PLANE_METERS or not np.isfinite(x_c):
            return None

        u = cx + fx * (y_c / x_c)
        v = cy - fy * (z_c / x_c)

        if not (np.isfinite(u) and np.isfinite(v)):
            return None

        return u, v
