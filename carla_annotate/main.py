import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Action
from enum import Enum
from pathlib import Path
from typing import Type

from carla_annotate.carla.simulation_recorder import SimulationRecorder
from carla_annotate.carla.simulation_replayer import SimulationReplayer
from carla_annotate.exporters.yolo_dataset_exporter import YoloDatasetExporter
from carla_annotate.types import Town, WeatherPreset, ServerConfig
from carla_annotate.visualizers.opencv_visualizer import OpencvVisualizer


class EnumAction(Action):

    def __init__(self, option_strings, dest, enum: Type[Enum], **kwargs):
        self._enum = enum
        # choices shown in help will be pretty CLI strings
        kwargs.setdefault("choices", [m.name.lower().replace("_", "-") for m in enum])
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # convert back from CLI string → Enum
        key = values.upper().replace("-", "_")
        setattr(namespace, self.dest, self._enum[key])


def print_args(mode: str, args) -> None:
    print(f"[{mode.capitalize()} mode]")
    for key, value in vars(args).items():
        if key == "func":
            continue
        if isinstance(value, Enum):
            value = value.value
        print(f"{key}: {value}")
    print()


def print_summary(frames: int, time_elapsed: float, recording_file: Path) -> None:
    print(f"Time elapsed: {time_elapsed:.2f}s")
    if frames > 0:
        avg = time_elapsed / frames
        print(f"Frames: {frames}  (avg {1000 * avg:.2f} ms/frame)")
    else:
        print("Frames: 0")
    print(f"File: {recording_file}")


def run_record(args):
    server_config = ServerConfig(host=args.host, port=args.port, timeout=args.timeout)
    print_args("record", args)
    time_start = time.time()
    with SimulationRecorder(args.recording_dir, args.town, server_config) as recorder:
        recorder.record()
        frames, recording_file = recorder.summary
    time_elapsed = time.time() - time_start
    print_summary(frames, time_elapsed, recording_file)


def run_annotate(args):
    server_config = ServerConfig(host=args.host, port=args.port, timeout=args.timeout)
    print_args("annotate", args)
    time_start = time.time()
    with SimulationReplayer(args.recording_file, args.weather_preset, server_config) as replayer:
        if args.visualize:
            with OpencvVisualizer(window_name="carla-annotate") as visualizer:
                for annotated_image in replayer.replay():
                    visualizer.visualize(annotated_image)
        else:
            with YoloDatasetExporter(args.dataset_dir) as exporter:
                for annotated_image in replayer.replay():
                    exporter.export(annotated_image)

        frames, recording_file = replayer.summary
    time_elapsed = time.time() - time_start
    print_summary(frames, time_elapsed, recording_file)


def main():
    parser = ArgumentParser(
        prog="carla-annotate",
        description="Generate Synthetic Autonomous Driving Image Datasets with CARLA",
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(
        title="modes",
        required=True,
        help="Available modes of operation"
    )

    server_config_parser = ArgumentParser(add_help=False)

    server_config_parser.add_argument(
        "--host",
        default="localhost",
        help="CARLA server host",
    )

    server_config_parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port",
    )

    server_config_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="CARLA server timeout",
    )

    # Record mode
    record_parser = subparsers.add_parser(
        "record",
        parents=[server_config_parser],
        help="Record CARLA simulation"
    )

    record_parser.set_defaults(func=run_record)

    record_parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory"
    )

    record_parser.add_argument(
        "--town",
        action=EnumAction,
        enum=Town,
        default=Town.TOWN_10HD,
        help="CARLA town",
    )

    # Annotate mode
    annotate_parser = subparsers.add_parser(
        "annotate",
        parents=[server_config_parser],
        help="Replay and annotate a CARLA simulation"
    )

    annotate_parser.set_defaults(func=run_annotate)

    annotate_parser.add_argument(
        "recording_file",
        type=Path,
        help="Simulation recording .log file"
    )

    annotate_parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory"
    )

    annotate_parser.add_argument(
        "--weather",
        action=EnumAction,
        enum=WeatherPreset,
        default=WeatherPreset.CLEAR_NOON,
        help="CARLA weather preset",
    )

    annotate_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of annotated images"
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
