# carla-annotate

Generate Synthetic Autonomous Driving Image Datasets from CARLA.

## Features

- **Scenario Selection**
    - Town
    - Weather
- **Simulation Setup**
    - One ego vehicle
    - One RGB camera attached to ego vehicle's roof (Waymo-style)
- **Route Planning & Driving**
    - On-the-spot route planning with full map coverage
    - Self-driving using CARLA Agents
- **Annotation**
    - Traffic Lights
- **Dataset Export**
    - Images + Annotations (category, 2D bounding boxes)
- **Visualization**
    - Optional using OpenCV

## Prerequisites

- Python 3.7
- CARLA 0.9.15

## Installation

### Step 1 — Clone carla-annotate

In separate directory from CARLA, clone `carla-annotate`:

```bash
git clone https://github.com/antonioskaparounakis/carla-annotate
```

Navigate into the `carla-annotate` directory:

```bash
cd carla-annotate
```

### Step 2 — Set up virtual environment

Create a virtual environment with Python 3.7:

```bash
python3.7 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Upgrade pip:

```bash
(venv) pip install --upgrade pip
```

### Step 3 — Install dependencies

#### CARLA Client library

Install the CARLA Client library from the `.whl` file in your CARLA 0.9.15 download:

```bash
(venv) pip install /path/to/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
```

#### CARLA Agents package

Copy the `agents` directory from your CARLA 0.9.15 download into the `carla_annotate` directory:

```bash
cp -r /path/to/CARLA_0.9.15/PythonAPI/carla/agents .
```

Copy the `requirements.txt` file from your CARLA 0.9.15 download into the `agents` directory:

```bash
cp /path/to/CARLA_0.9.15/PythonAPI/carla/requirements.txt ./agents/requirements.txt
```

Install the CARLA Agents package dependencies from `agents/requirements.txt`:

```bash
(venv) pip install -r agents/requirements.txt
```

#### carla-annotate dependencies

Install the carla-annotate dependencies from `requirements.txt`:

```bash
(venv) pip install -r requirements.txt
```

## Usage

`carla-annotate` provides two modes of operation: **record** and **annotate**.  
Both modes share common options for connecting to the CARLA server.

### Global Options

These options apply to all modes:

    --host HOST        CARLA server host (default: localhost)
    --port PORT        CARLA server port (default: 2000)
    --timeout TIMEOUT  CARLA server timeout in seconds (default: 5.0)

### Mode: record

Record a CARLA simulation to disk.

Positional arguments:

    output_dir          Output directory for recordings

Options:

    --town TOWN         CARLA town (default: town-10hd)
                        (choices are values from the Town enum)

Example:

    carla-annotate record ./output --town town-03

### Mode: annotate

Replay and annotate a recorded CARLA simulation.

Positional arguments:

    recording_file      Simulation recording .log file
    dataset_dir         Output dataset directory

Options:

    --weather-preset PRESET   CARLA weather preset (default: clear-noon)
                              (choices are values from the WeatherPreset enum)
    --visualize               Enable visualization of annotated images
                              (instead of exporting to dataset files)

Examples:

    carla-annotate annotate ./output/recording.log ./dataset --weather-preset wet-sunset

    carla-annotate annotate ./output/recording.log ./dataset --visualize