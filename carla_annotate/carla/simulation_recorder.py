from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

import carla
from carla_annotate.carla.route_planner import RoutePlanner
from carla_annotate.domain import ServerConfig, Town


class SimulationRecorder:
    SYNCHRONOUS_MODE_FIXED_DELTA_SECONDS: float = 0.05
    EGO_VEHICLE_ROUTE_STRATEGY: str = "full_coverage"
    EGO_VEHICLE_BLUEPRINT_ID: str = "vehicle.tesla.model3"
    EGO_VEHICLE_TARGET_SPEED: float = 20.0

    def __init__(self, output_dir: Path, town: Town, server_config: ServerConfig):
        self._output_dir = output_dir
        self._town = town
        self._server_config = server_config
        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._map: Optional[carla.Map] = None
        self._route_planner: Optional[RoutePlanner] = None
        self._route: Optional[List[Tuple[carla.Waypoint, RoadOption]]] = None
        self._ego_vehicle: Optional[carla.Vehicle] = None
        self._agent: Optional[BasicAgent] = None
        self._recording_file: Optional[Path] = None
        self._recording_completed = False
        self._recorded_frames: int = 0

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False

    @property
    def summary(self) -> Tuple[int, Path]:
        return self._recorded_frames, self._recording_file

    def record(self) -> None:
        self._recording_file = self._make_recording_file()
        self._client.start_recorder(str(self._recording_file), True)
        try:
            while not self._agent.done():
                self._world.tick()
                self._recorded_frames += 1
                self._ego_vehicle.apply_control(self._agent.run_step())
            self._recording_completed = True
        finally:
            self._client.stop_recorder()

    def _setup(self) -> None:
        self._client = self._create_client()
        self._world = self._client.load_world(self._town.value)
        self._world.wait_for_tick()
        self._set_rendering_mode(rendering=False)
        self._set_synchronous_mode(synchronous=True)
        self._world.tick()
        self._map = self._world.get_map()
        self._route_planner = RoutePlanner(self._map)
        self._route = self._route_planner.plan(self.EGO_VEHICLE_ROUTE_STRATEGY)
        self._ego_vehicle = self._spawn_ego_vehicle()
        self._agent = self._create_ego_agent()

    def _cleanup(self) -> None:
        if not self._recording_completed:
            try:
                self._recording_file.unlink()
            except FileNotFoundError:
                pass
        self._destroy_ego_vehicle()
        self._set_synchronous_mode(synchronous=False)
        self._world.wait_for_tick()
        self._set_rendering_mode(rendering=True)

    def _create_client(self) -> carla.Client:
        client = carla.Client(self._server_config.host, self._server_config.port)
        client.set_timeout(self._server_config.timeout)
        return client

    def _set_rendering_mode(self, rendering: bool) -> None:
        settings = self._world.get_settings()
        settings.no_rendering_mode = not rendering
        self._world.apply_settings(settings)

    def _set_synchronous_mode(self, synchronous: bool) -> None:
        settings = self._world.get_settings()
        settings.synchronous_mode = synchronous
        settings.fixed_delta_seconds = (
            self.SYNCHRONOUS_MODE_FIXED_DELTA_SECONDS if synchronous else None
        )
        self._world.apply_settings(settings)

    def _spawn_ego_vehicle(self) -> carla.Vehicle:
        blueprint = self._world.get_blueprint_library().find(
            self.EGO_VEHICLE_BLUEPRINT_ID
        )
        blueprint.set_attribute("role_name", "ego")
        spawn_point = self._route[0][0].transform
        spawn_point.location.z += 0.5
        return self._world.spawn_actor(blueprint, spawn_point)

    def _destroy_ego_vehicle(self) -> None:
        if self._ego_vehicle is not None and self._ego_vehicle.is_alive:
            self._ego_vehicle.destroy()

    def _create_ego_agent(self) -> BasicAgent:
        agent = BasicAgent(
            self._ego_vehicle,
            self.EGO_VEHICLE_TARGET_SPEED,
            {},
            self._map,
            self._route_planner.grp,
        )
        agent.ignore_traffic_lights(True)
        agent.set_global_plan(self._route)
        return agent

    def _make_recording_file(self) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._town.value}_{timestamp}.log"
        return (self._output_dir / filename).resolve()
