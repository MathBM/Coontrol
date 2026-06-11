# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Coontrol** is a truck cargo volume measurement system using 4 Pepperl+Fuchs R2000 LIDAR sensors. It captures 3D point clouds from the truck bed, aligns the loaded scan against an empty-bucket reference, isolates the cargo surface, and computes volume via a 2D heightmap integral.

## Setup

```bash
# Python dependencies
pip install -r requirements.txt

# Rust TCP client (required for real sensor data collection)
cd rust && cargo build --release
```

The compiled binary `rust/target/release/client_tcp` is spawned directly by `ScanManager` — no install step needed.

**Wayland sessions:** Open3D/GLFW requires X11. The `XDG_SESSION_TYPE=x11` env var is set automatically in `debug_pipeline.py`. For other scripts, export it manually or run via XWayland.

## Running

```bash
# Main GUI (PySide6)
python main.py

# Debug a specific scan folder step-by-step (interactive Open3D windows)
python debug_pipeline.py <scan_folder_name>
# e.g.: python debug_pipeline.py 2026-04-08_09h02min40s

# Test a single live front sensor
python test_front_sensor_live.py

# Generate synthetic scan data
python synthetic_data_generator.py
```

## Architecture

### Data Flow

```
Sensors (4x LIDAR) → rust/client_tcp → .bin files → PointCloudReconstructor → data.npz
                                                                                    ↓
                                                                            DataManager
                                                                                    ↓
                                                              Registration (RANSAC+ICP alignment)
                                                                                    ↓
                                                              SurfaceReconstructor.isolate_load_points
                                                                                    ↓
                                                              VolumeCalculator.volume_from_heightmap → mm³
```

### Key Classes

- **[src/DataManager.py](src/DataManager.py)** — Orchestrates the full pipeline. `process_data()` is the current method (heightmap); `process_data_legacy()` uses the older Poisson mesh approach. Auto-detects synthetic scans via presence of `SYNTHETIC_INFO.txt`.

- **[src/PointCloudReconstructor.py](src/PointCloudReconstructor.py)** — Parses raw `.bin` binary packets from the sensors (Pepperl+Fuchs R2000 protocol), converts polar→Cartesian, applies sensor-specific rigid body transforms (rotation + translation), filters outliers, and stitches the 3 profile sensors (right, left, top) using Z-axis position derived from the front sensor. Produces a combined `(x, y, z)` point list saved as `data.npz`.

- **[src/Registration.py](src/Registration.py)** — Aligns the loaded scan to the empty bucket reference using RANSAC feature matching (FPFH descriptors) followed by Generalized ICP. Only the Z translation (truck movement direction) is retained from ICP; X/Y translations and all rotations are discarded as artifacts.

- **[src/SurfaceReconstructor.py](src/SurfaceReconstructor.py)** — Removes bucket-wall points from the aligned scan using KD-tree nearest-neighbor distance thresholding, then statistical + radius outlier removal, then DBSCAN to keep only the largest cluster (the cargo).

- **[src/VolumeCalculator.py](src/VolumeCalculator.py)** — `volume_from_heightmap()`: bins points into a 2D grid (default 8mm cells), takes max Z per cell, sums `z_max × cell_area`. Robust to holes. Legacy `volume_calculation()` uses the divergence theorem on a closed mesh.

- **[src/ScanManager.py](src/ScanManager.py)** — Configures sensors via HTTP (through `SensorManager`), requests TCP handles, and launches the Rust binary to receive and write `.bin` files.

- **[src/Constants.py](src/Constants.py)** — Sensor IPs, physical mounting offsets/rotations, Z-axis boundary limits for the front sensor, and profile boundary clip values. **Edit here when recalibrating sensor positions.**

- **[src/Parameters.py](src/Parameters.py)** — Tunable algorithm parameters (RANSAC iterations, ICP settings, outlier removal thresholds, DBSCAN epsilon, heightmap cell size, etc.). **Edit here when tuning algorithm accuracy.**

### Coordinate System

All units are **millimeters**. After reconstruction, the coordinate frame has:
- **Z** = truck travel direction (used for alignment)
- **X/Y** = lateral/height axes
- Origin roughly at the front sensor mounting point

The front sensor's profile is used exclusively to derive the Z position of each scan line (via NTP timestamp correlation). Right, left, and top sensors provide the X/Y surface geometry.

### Scan Storage

```
pointcloud/
  caixa_vazia/          ← empty-bucket reference (must be named exactly this)
    192.168.1.{10-13}.bin
    data.npz
  2026-04-08_09h02min40s/   ← timestamped load scan
    192.168.1.{10-13}.bin
    data.npz
    SYNTHETIC_INFO.txt  ← present only for synthetic scans
```

`data.npz` is auto-generated by `DataManager` or `debug_pipeline.py` if missing. The `caixa_vazia` folder name is hardcoded in `Constants.BUCKET_PATH`.

### Synthetic Data

`synthetic_data_generator.py` / `SyntheticScanCreator` generate `.npz` test data (ramp shapes: linear, convex, concave, stepped, steep). Synthetic scans skip RANSAC+ICP and use centroid XY alignment instead, because they lack visible bucket walls for feature matching.

### GUI

`main.py` launches a PySide6 window ([src/interface/MainWindow.py](src/interface/MainWindow.py)) with a scan list table. Two processing methods are selectable: the current heightmap method (index 0) and the legacy Poisson method (index 1).

### Rust TCP Client

`rust/src/client_tcp.rs` connects to each sensor's TCP handle and writes binary scan data to `<output_folder>/<sensor_ip>.bin`. Called as a subprocess by `ScanManager.start_scan()`.

## Network

Sensors are on subnet `192.168.1.x`. Configure the host machine with a static IP in that range (e.g. `192.168.1.50`) before use. Sensor IPs: Front=.10, Right=.11, Left=.12, Top=.13.
