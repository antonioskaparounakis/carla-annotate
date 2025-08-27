import queue
from pathlib import Path
from queue import Queue
from typing import Optional, Iterator, Tuple

import carla

from carla_annotate.carla.image_annotator import ImageAnnotator
from carla_annotate.types import WeatherPreset, ServerConfig, AnnotatedImage


class SimulationReplayer:
    SYNCHRONOUS_MODE_FIXED_DELTA_SECONDS: float = 0.05
    EGO_CAMERA_BLUEPRINT_ID: str = "sensor.camera.rgb"
    EGO_CAMERA_IMAGE_WIDTH: int = 640
    EGO_CAMERA_IMAGE_HEIGHT: int = 640
    EGO_CAMERA_FOV: float = 90
    EGO_CAMERA_SENSOR_TICK: float = 1.0

    def __init__(self, recording_file: Path, weather_preset: WeatherPreset, server_config: ServerConfig):
        self._recording_file = recording_file.resolve()
        self._weather_preset = weather_preset
        self._server_config = server_config
        self._client: Optional[carla.Client] = None
        self._recording_frames: Optional[int] = None
        self._world: Optional[carla.World] = None
        self._ego_vehicle: Optional[carla.Vehicle] = None
        self._ego_camera: Optional[carla.Sensor] = None
        self._ego_camera_image_queue: Optional[queue.Queue] = None
        self._image_annotator: Optional[ImageAnnotator] = None
        self._recorded_frames: int = 0

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    @property
    def summary(self) -> Tuple[int, Path]:
        return self._recorded_frames, self._recording_file

    def replay(self) -> Iterator[AnnotatedImage]:
        self._ego_camera.listen(lambda img: self._ego_camera_image_queue.put(img))

        frame = self._world.tick()
        self._ego_camera_image_queue.get()

        try:
            while self._recording_frames > 0:
                for _ in range(int(self.EGO_CAMERA_SENSOR_TICK / self.SYNCHRONOUS_MODE_FIXED_DELTA_SECONDS)):
                    frame = self._world.tick()
                    self._recording_frames -= 1
                    self._recorded_frames += 1

                image = self._ego_camera_image_queue.get()
                while image.frame < frame:
                    image = self._ego_camera_image_queue.get()

                yield self._image_annotator.annotate(image, self._world.get_actors())
        finally:
            self._ego_camera.stop()
            self._client.stop_recorder()

    def _setup(self):
        self._client = self._create_client()
        self._recording_frames = self._parse_recording_frames()
        self._client.replay_file(str(self._recording_file), 0.0, 0, False)
        self._world = self._client.get_world()
        self._world.wait_for_tick()
        self._set_synchronous_mode(synchronous=True)
        self._world.tick()
        self._set_weather_preset()
        self._ego_vehicle = self._find_ego_vehicle()
        self._ego_camera = self._spawn_ego_camera()
        self._ego_camera_image_queue = Queue()
        self._image_annotator = ImageAnnotator(self._ego_camera, self._world)

    def _cleanup(self):
        self._destroy_ego_camera()
        self._destroy_ego_vehicle()
        self._set_synchronous_mode(synchronous=False)
        self._world.wait_for_tick()

    def _create_client(self) -> carla.Client:
        client = carla.Client(self._server_config.host, self._server_config.port)
        client.set_timeout(self._server_config.timeout)
        return client

    def _set_synchronous_mode(self, synchronous: bool) -> None:
        settings = self._world.get_settings()
        settings.synchronous_mode = synchronous
        settings.fixed_delta_seconds = self.SYNCHRONOUS_MODE_FIXED_DELTA_SECONDS if synchronous else None
        self._world.apply_settings(settings)

    def _set_weather_preset(self) -> None:
        weather_preset = getattr(carla.WeatherParameters, self._weather_preset.value)
        self._world.set_weather(weather_preset)

    def _parse_recording_frames(self) -> int:
        info = self._client.show_recorder_file_info(str(self._recording_file), False)
        lines = info.splitlines()
        for line in reversed(lines):
            if line.startswith("Frames:"):
                return int(line.split()[1])
        raise ValueError("Cannot parse frames")

    def _find_ego_vehicle(self) -> carla.Vehicle:
        for actor in self._world.get_actors():
            if actor.attributes.get("role_name") == "ego":
                return actor
        raise ValueError("Cannot find ego vehicle")

    def _destroy_ego_vehicle(self) -> None:
        if self._ego_vehicle is not None and self._ego_vehicle.is_alive:
            self._ego_vehicle.destroy()

    def _destroy_ego_camera(self) -> None:
        if self._ego_camera is not None and self._ego_camera.is_alive:
            self._ego_camera.destroy()

    def _spawn_ego_camera(self) -> carla.Sensor:
        blueprint_library = self._world.get_blueprint_library()
        blueprint = blueprint_library.find(str(self.EGO_CAMERA_BLUEPRINT_ID))
        blueprint.set_attribute("image_size_x", str(self.EGO_CAMERA_IMAGE_WIDTH))
        blueprint.set_attribute("image_size_y", str(self.EGO_CAMERA_IMAGE_HEIGHT))
        blueprint.set_attribute("fov", str(self.EGO_CAMERA_FOV))
        blueprint.set_attribute("sensor_tick", str(self.EGO_CAMERA_SENSOR_TICK))
        ego_vehicle_bbox = self._ego_vehicle.bounding_box
        loc = carla.Location(z=ego_vehicle_bbox.location.z + ego_vehicle_bbox.extent.z + 1)
        return self._world.spawn_actor(blueprint, carla.Transform(loc, carla.Rotation()), self._ego_vehicle)
