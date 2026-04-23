---
description: "Use when working on LIDAR sensor connection, data acquisition, point cloud reconstruction, or volume calculation. Covers the full pipeline: IP sensor HTTP/TCP protocol → binary PFSDP data → 3D point cloud → registration → load isolation → heightmap volume integral."
---

# Coontrol — LIDAR Pipeline Architecture

## System Overview

This project measures the volume of material loaded onto a truck bucket using 4 LIDAR sensors (Pepperl+Fuchs R2000). The pipeline is:

```
4 LIDAR sensors (HTTP API + TCP stream)
  → binary PFSDP packets (.bin files)
  → PointCloudReconstructor (3D xyz point cloud)
  → Registration (RANSAC + ICP alignment with empty bucket)
  → SurfaceReconstructor (isolate load points)
  → VolumeCalculator (heightmap integral)
  → volume in mm³
```

---

## 1. Sensor Network

Four sensors on a private Ethernet subnet (`192.168.1.x`). IPs are defined in `src/Constants.py`:

| Role  | IP           | Constant                    |
| ----- | ------------ | --------------------------- |
| Front | 192.168.1.10 | `Constants.SENSOR_FRONT_IP` |
| Right | 192.168.1.11 | `Constants.SENSOR_RIGHT_IP` |
| Left  | 192.168.1.12 | `Constants.SENSOR_LEFT_IP`  |
| Top   | 192.168.1.13 | `Constants.SENSOR_TOP_IP`   |

Each sensor exposes an HTTP REST API. All communication goes through `SensorManager` (Python) or the Rust TCP client.

---

## 2. Connection Protocol (HTTP + TCP)

### HTTP REST (Python — `src/SensorManager.py`)

All sensor commands are HTTP GET requests. A `SensorManager` instance is tied to one sensor IP.

```python
sm = SensorManager(sensor_ip="192.168.1.10", server_ip="192.168.1.50", server_port=6969)
```

**Connection lifecycle:**

1. `sm.request_handle_tcp()` — sensor returns a `handle` string and a `port`; the handle is stored as `sm.handle`
2. `sm.start_scanoutput()` — sensor starts streaming data on the returned TCP port
3. _(read TCP stream — done by Rust client or Python reader)_
4. `sm.stop_scanoutput()` — stop streaming
5. `sm.release_handle()` — free the sensor

**Watchdog:** If `watchdog="on"`, send `GET /cmd/feed_watchdog?handle=<handle>` every 15 s (default timeout is 60 s); missing it will drop the connection.

### TCP Data Stream (Rust — `rust/src/client_tcp.rs`)

Run once per sensor. Reads the raw binary PFSDP stream and saves it to disk:

```
cargo run --release -- 192.168.1.10
# saves sensor_192_168_1_10.bin
```

The main loop connects to `<sensor_ip>:<tcp_port>`, reads chunks with `tokio::io::AsyncReadExt`, and writes raw bytes with a `BufWriter<File>`.

---

## 3. Binary File Format (PFSDP packets)

Each `.bin` file is a concatenated stream of variable-length packets. Packets begin with the magic bytes `0xa25c` (little-endian `u16`).

Relevant header fields parsed in `PointCloudReconstructor.process_binary_file()`:

| Offset (after magic) | Type  | Field                       |
| -------------------- | ----- | --------------------------- |
| 2–5                  | `u32` | packet_size                 |
| 6–7                  | `u16` | header_size                 |
| 8–9                  | `u16` | scan_number                 |
| 12–19                | `u64` | timestamp_raw               |
| 42–45                | `i32` | first_angle (×10⁻⁴ °)       |
| 46–49                | `i32` | angular_increment (×10⁻⁴ °) |

Distance values follow after the header in the packet payload.

---

## 4. Point Cloud Reconstruction (`src/PointCloudReconstructor.py`)

`create_point_cloud(scan_path)` builds a merged 3D point cloud from the four `.bin` files:

1. **Front sensor** → `calculate_z_axis()` — derives a tilt-correction Z-axis from the front scanner's profile (front sensor acts as reference, not merged into the final cloud)
2. **Right + Left sensors** → `reconstruct_z_axis()` — converts 2D profile scans to 3D using the Z-axis reference, then applies rigid body transforms from `Constants` (translation + Euler rotation)
3. **Top sensor** → `reconstruct_z_axis()` — top-down scan
4. Boundary clipping via `remove_boundaries()` (X/Y limits in `Constants`)
5. Outlier removal via `filter_point_cloud()`
6. Final global transform: 90° rotation around Z + vertical offset (`Constants.SENSOR_TOP_HEIGHT = 2400 mm`)

All coordinates are in **mm**.

Scan data is cached as `{scan_path}data.npz` (key `"xyz"`) after the first reconstruction.

---

## 5. Scan Folder Structure

```
pointcloud/
  <timestamp>/           ← one folder per scan session
    192.168.1.10.bin     ← front sensor raw data
    192.168.1.11.bin     ← right sensor raw data
    192.168.1.12.bin     ← left sensor raw data
    192.168.1.13.bin     ← top sensor raw data
    data.npz             ← cached reconstructed xyz (created after first process)
    SYNTHETIC_INFO.txt   ← exists only for synthetic scans
  caixa_vazia/           ← empty bucket reference scan (same structure)
```

The bucket reference path is `Constants.BUCKET_PATH = "./pointcloud/caixa_vazia"`.

---

## 6. Registration (`src/Registration.py`)

Aligns the load scan with the empty bucket scan. `DataManager._align_auto()` selects the method:

- **Real scan**: RANSAC (global) + ICP (local refinement) via `registration.align_truck_bucket_and_load()`. Uses FPFH features. Parameters in `src/Parameters.py → Parameters.Registration`.
- **Synthetic scan** (detected by presence of `SYNTHETIC_INFO.txt`): centroid-based XY translation only (Z preserved). RANSAC fails on synthetic data because only the load surface is present, not the bucket walls.

---

## 7. Load Isolation (`src/SurfaceReconstructor.py`)

`isolate_load_points(truck_bucket, aligned_pcd, ...)` removes points belonging to the empty bucket from the aligned scan, leaving only the material load. Uses radius-based nearest-neighbour removal + DBSCAN clustering. Parameters in `Parameters.BucketRemoval`.

---

## 8. Volume Calculation (`src/VolumeCalculator.py`)

### Current method — Heightmap integral (use this)

```python
volume = VolumeCalculator().volume_from_heightmap(load_pcd, cell_size=8.0)
```

$$V = \sum_{x,y} z_{\max}(x, y) \cdot \Delta x \cdot \Delta y$$

- Builds a 2D grid (`cell_size` mm per cell, default 8 mm)
- For each cell, takes the maximum Z value of all points inside it
- Sums all cells — sparse/empty cells contribute 0
- **Returns volume in mm³**
- Robust to sparse data, holes, and any load shape

### Legacy method — Mesh + divergence theorem (avoid for new work)

```python
volume = VolumeCalculatorLegacy().volume_calculation(load_mesh)
```

Requires a closed triangulated mesh (Poisson reconstruction). Fails when the mesh has holes. Kept in `VolumeCalculatorLegacy` for comparison.

---

## 9. Main Orchestration (`src/DataManager.py`)

Full pipeline call:

```python
dm = DataManager()
volume_mm3 = dm.process_data("./pointcloud/2026-04-08_09h02min40s/")
```

`process_data()` runs: load → align → isolate → volume (heightmap).  
`process_data_legacy()` runs: load → align → isolate → mesh → volume (divergence).

---

## 10. Parameters and Constants

- `src/Constants.py`: hardware-defined values (IPs, physical offsets, boundary limits). Modify here when sensors are repositioned.
- `src/Parameters.py`: algorithm tuning (voxel size, RANSAC iterations, DBSCAN radius, heightmap cell size). Uses nested static classes, e.g., `Parameters.Registration.VOXEL_SIZE`.
