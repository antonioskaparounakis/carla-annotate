from typing import List, Tuple

import carla

from carla_annotate.carla.camera_projector import CameraProjector
from carla_annotate.carla.camera_visibility_filter import CameraVisibilityFilter
from carla_annotate.types import AnnotatedImage, Category, Instance
from carla_annotate.utils import carla_image_to_rgb


class ImageAnnotator:
    TRAFFIC_LIGHT_BLUEPRINT_FILTER: str = "traffic.traffic_light"

    def __init__(self, camera: carla.Sensor, world: carla.World):
        self._visibility_filter = CameraVisibilityFilter(camera, world)
        self._projector = CameraProjector(camera)
        self._static_actors_bboxes = self._expand_static_actors(world)

    def annotate(self, image: carla.Image, actors: carla.ActorList) -> AnnotatedImage:
        bboxes = []

        # Filter visible
        visible = self._visibility_filter.filter_visible(self._static_actors_bboxes)

        # Project visible
        projected = self._projector.project(visible)

        instances = []
        for actor, bbox2d in projected:
            category = Category.TRAFFIC_LIGHT
            instances.append(Instance(category, bbox2d))

        rgb = carla_image_to_rgb(image)
        return AnnotatedImage(rgb, instances)

    def _expand_static_actors(self, world: carla.World) -> List[Tuple[carla.Actor, carla.BoundingBox]]:
        actors = world.get_actors()
        results = []
        # Light boxes
        for traffic_light in actors.filter(self.TRAFFIC_LIGHT_BLUEPRINT_FILTER):
            for light_box in traffic_light.get_light_boxes():
                results.append((traffic_light, light_box))
        return results
