from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    timeout: float


class Town(Enum):
    TOWN_01 = "Town01"
    TOWN_02 = "Town02"
    TOWN_10HD = "Town10HD"


class WeatherPreset(Enum):
    CLEAR_NOON = "ClearNoon"
    CLOUDY_NOON = "CloudyNoon"
    WET_NOON = "WetNoon"
    SOFT_RAIN_NOON = "SoftRainNoon"
    HARD_RAIN_NOON = "HardRainNoon"
    CLEAR_SUNSET = "ClearSunset"


class RoutePlanningStrategy(Enum):
    FULL_COVERAGE = "full_coverage"


class Category(Enum):
    TRAFFIC_LIGHT = "traffic_light"


@dataclass
class Instance:
    category: Category
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class AnnotatedImage:
    image: np.ndarray
    instances: List[Instance]

    @property
    def image_width(self) -> int:
        return self.image.shape[1]

    @property
    def image_height(self) -> int:
        return self.image.shape[0]
