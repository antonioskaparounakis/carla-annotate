import math
from typing import List, Tuple

import carla


class CameraVisibilityFilter:
    def __init__(self, camera: carla.Sensor, world: carla.World):
        self._camera = camera
        self._world = world
        self._camera_fov_half_rad = math.radians(
            float(self._camera.attributes["fov"]) / 2.0
        )

    def filter_visible(
        self, bboxes: List[Tuple[carla.Actor, carla.BoundingBox]]
    ) -> List[Tuple[carla.Actor, carla.BoundingBox]]:
        camera_tf = self._camera.get_transform()
        camera_loc = camera_tf.location
        camera_fwd = camera_tf.get_forward_vector()
        results = []
        for actor, bbox in bboxes:
            bbox_loc = bbox.location
            if self._is_in_fov(camera_loc, camera_fwd, bbox_loc) and self._is_in_sight(
                camera_loc, bbox_loc, actor
            ):
                results.append((actor, bbox))
        return results

    def _is_in_fov(
        self,
        camera_loc: carla.Location,
        camera_fwd: carla.Vector3D,
        bbox_loc: carla.Location,
    ) -> bool:
        camera_to_bbox = bbox_loc - camera_loc
        angle = camera_fwd.get_vector_angle(camera_to_bbox)
        return angle < self._camera_fov_half_rad

    # def _is_in_sight(
    #     self, camera_loc: carla.Location, bbox_loc: carla.Location, actor: carla.Actor
    # ) -> bool:
    #     hits = self._world.cast_ray(camera_loc, bbox_loc)
    #     idx, tolerance = (1, 0.1) if isinstance(actor, carla.TrafficLight) else (0, 0)
    #     hit = hits[idx]
    #     dist_to_hit = camera_loc.distance(hit.location)
    #     dist_to_bbox = camera_loc.distance(bbox_loc)
    #     if hit.label in actor.semantic_tags and math.isclose(
    #         dist_to_hit, dist_to_bbox, abs_tol=tolerance
    #     ):
    #         return True
    #     return False

    def _is_in_sight(self, camera_loc, bbox_loc, actor):
        hits = self._world.cast_ray(camera_loc, bbox_loc)
        return True if len(hits) == 0 else False # Temp fix for Town010HD
