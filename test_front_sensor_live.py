"""
Live test for the front sensor.

Connects directly via TCP (no Rust binary), receives PFSDP packets in real
time, parses them with the same logic used in process_binary_file /
calculate_z_axis, and plots each 2D scan profile while printing the detected
z_axis value (max Y inside the boundary window).

Run:
    python test_front_sensor_live.py

Press Ctrl-C to stop.
"""

import socket
import signal
import sys
from math import cos, sin, pi
from struct import pack, unpack

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.Constants import Constants
from src.SensorManager import SensorManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLES_PER_SCAN = 600
SCAN_FREQUENCY = 50
MAX_SCANS_HISTORY = 5          # how many past scans to keep in the plot
RECV_CHUNK = 4096              # bytes per recv() call

MAGIC_BYTES = pack("H", 0xA25C)

# ---------------------------------------------------------------------------
# Sensor setup helpers (same pattern as ScanManager.start_scan)
# ---------------------------------------------------------------------------

def setup_sensor():
    sensor = SensorManager(
        Constants.SENSOR_FRONT_IP,
        Constants.SERVER_IP,
        Constants.SERVER_PORT,
    )

    print("[INFO] Setting sensor parameters...")
    res = sensor.set_parameters(
        samples_per_scan=SAMPLES_PER_SCAN,
        scan_frequency=SCAN_FREQUENCY,
        scan_direction=Constants.SCAN_DIRECTION,
    )
    print(f"  set_parameters -> {res}")

    print("[INFO] Requesting TCP handle...")
    res = sensor.request_handle_tcp(max_num_points_scan=SAMPLES_PER_SCAN)
    print(f"  request_handle_tcp -> {res}")

    port = res["data"].get("port", None)
    if port is None:
        raise RuntimeError("Could not obtain TCP port from sensor. Is the sensor reachable?")

    return sensor, port


def teardown_sensor(sensor):
    print("\n[INFO] Stopping scan output...")
    print(f"  stop_scanoutput -> {sensor.stop_scanoutput()}")
    print("[INFO] Releasing handle...")
    print(f"  release_handle  -> {sensor.release_handle()}")


# ---------------------------------------------------------------------------
# Packet parsing (mirrors process_binary_file, but for a single packet)
# ---------------------------------------------------------------------------

def ntp64_to_seconds(integer):
    seconds = integer >> 32
    frac = (integer & 0xFFFFFFFF) / 0x100000000
    return round(seconds + frac, 3)


def polar_to_xy(distances, first_angle_raw, angular_increment_raw):
    first_angle = first_angle_raw / 10000.0
    angular_increment = angular_increment_raw / 10000.0

    xy = []
    for i, distance in enumerate(distances):
        if distance == 4_294_967_295:   # invalid measurement
            continue
        angle = (first_angle + i * angular_increment) * pi / 180.0
        x = round(distance * cos(angle))
        y = round(distance * sin(angle))
        xy.append((x, y))
    return xy


def parse_packet(packet):
    """Parse a single PFSDP payload (after the magic bytes have been stripped).
    Returns (scan_number, timestamp, xy_list) or None on error."""
    if len(packet) <= 10:
        return None
    try:
        packet_size      = unpack("I", packet[2:6])[0] - len(MAGIC_BYTES)
        header_size      = unpack("H", packet[6:8])[0] - len(MAGIC_BYTES)
        scan_number      = unpack("H", packet[8:10])[0]
        timestamp_raw    = unpack("Q", packet[12:20])[0]
        first_angle      = unpack("i", packet[42:46])[0]
        angular_increment = unpack("i", packet[46:50])[0]
    except Exception:
        return None

    if len(packet) != packet_size:
        return None

    payload   = packet[header_size:]
    distances = unpack(f"{len(payload) // 4}I", payload[: len(payload) // 4 * 4])
    xy        = polar_to_xy(distances, first_angle, angular_increment)
    timestamp = ntp64_to_seconds(timestamp_raw)

    return scan_number, timestamp, xy


# ---------------------------------------------------------------------------
# z_axis calculation for a single scan (mirrors calculate_z_axis per scan)
# ---------------------------------------------------------------------------

def z_axis_for_scan(xy_list):
    """Return the minimum Y (closest to sensor) inside the boundary window,
    or None if no point falls inside the window."""
    x_min = Constants.BOUNDARIES_ZAXIS_X_MIN
    x_max = Constants.BOUNDARIES_ZAXIS_X_MAX
    y_min = Constants.BOUNDARIES_ZAXIS_Y_MIN
    y_max = Constants.BOUNDARIES_ZAXIS_Y_MAX

    best_y = None
    for x, y in xy_list:
        if x <= x_min or x >= x_max:
            continue
        if y <= y_max or y >= y_min:  # discard: too close (<=100) or beyond floor (>=4000)
            continue
        if best_y is None or y < best_y:
            best_y = y
    return best_y


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------

class LiveReceiver:
    def __init__(self):
        self.sensor, self.port = setup_sensor()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)

        print(f"[INFO] Connecting to {Constants.SENSOR_FRONT_IP}:{self.port} ...")
        self.sock.connect((Constants.SENSOR_FRONT_IP, self.port))
        print("[INFO] Connected.")

        print("[INFO] Starting scan output...")
        print(f"  start_scanoutput -> {self.sensor.start_scanoutput()}")

        self.buf = b""
        self.scan_accumulator = {}   # scan_number -> {timestamp, xy} — mirrors process_binary_file
        self.scan_history = []       # list of xy lists (last N scans)
        self.z_history    = []       # corresponding z_axis values
        self.scan_count   = 0

        # Matplotlib setup
        self.fig, (self.ax_profile, self.ax_z) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle("Front Sensor – Live View  (Ctrl-C to stop)")

        self.ax_profile.set_title("2D scan profile (XY)")
        self.ax_profile.set_xlabel("X (mm)")
        self.ax_profile.set_ylabel("Y (mm)")
        self.ax_profile.set_aspect("equal")

        self.ax_z.set_title("z_axis value per scan")
        self.ax_z.set_xlabel("Scan index")
        self.ax_z.set_ylabel("max Y in window (mm)")

        self._running = True
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *_):
        self._running = False

    def receive_scans(self):
        """Pull packets from the TCP stream and return completed scans.
        Merges multi-packet scans by scan_number, mirroring process_binary_file."""
        try:
            chunk = self.sock.recv(RECV_CHUNK)
        except socket.timeout:
            return []

        if not chunk:
            return []

        self.buf += chunk

        while MAGIC_BYTES in self.buf:
            idx = self.buf.index(MAGIC_BYTES)
            self.buf = self.buf[idx + len(MAGIC_BYTES):]

            next_idx = self.buf.find(MAGIC_BYTES)
            if next_idx == -1:
                break

            packet = self.buf[:next_idx]
            self.buf = self.buf[next_idx:]

            parsed = parse_packet(packet)
            if parsed is None:
                continue

            scan_number, timestamp, xy = parsed
            if scan_number not in self.scan_accumulator:
                self.scan_accumulator[scan_number] = {"timestamp": timestamp, "xy": []}
            self.scan_accumulator[scan_number]["xy"].extend(xy)

        # Emit all scan_numbers except the highest one (it may still be receiving packets)
        results = []
        if len(self.scan_accumulator) > 1:
            for k in sorted(self.scan_accumulator.keys())[:-1]:
                entry = self.scan_accumulator.pop(k)
                results.append((k, entry["timestamp"], entry["xy"]))
        return results

    def update_plot(self, _frame):
        if not self._running:
            plt.close(self.fig)
            return

        packets = self.receive_scans()
        for scan_number, timestamp, xy in packets:
            z = z_axis_for_scan(xy)
            self.scan_count += 1

            if z is None:
                print(
                    f"[SCAN #{self.scan_count:04d}] scan_number={scan_number:5d} "
                    f"timestamp={timestamp:.3f}s  points={len(xy):4d}  "
                    f"[NO DATA in window]"
                )
                continue  # don't pollute history with no-detection scans

            self.scan_history.append(xy)
            self.z_history.append(z)
            if len(self.scan_history) > MAX_SCANS_HISTORY:
                self.scan_history.pop(0)
                self.z_history.pop(0)

            print(
                f"[SCAN #{self.scan_count:04d}] scan_number={scan_number:5d} "
                f"timestamp={timestamp:.3f}s  points={len(xy):4d}  "
                f"z_axis(minY)={z} mm"
            )

        # ---- profile plot ----
        self.ax_profile.cla()
        self.ax_profile.set_title("2D scan profile (XY)")
        self.ax_profile.set_xlabel("X (mm)")
        self.ax_profile.set_ylabel("Y (mm)")

        # Boundary window rectangle
        self.ax_profile.axvline(Constants.BOUNDARIES_ZAXIS_X_MIN, color="r", lw=0.8, ls="--", label="X boundary")
        self.ax_profile.axvline(Constants.BOUNDARIES_ZAXIS_X_MAX, color="r", lw=0.8, ls="--")
        self.ax_profile.axhline(Constants.BOUNDARIES_ZAXIS_Y_MIN, color="g", lw=0.8, ls="--", label="Y boundary")
        self.ax_profile.axhline(Constants.BOUNDARIES_ZAXIS_Y_MAX, color="g", lw=0.8, ls="--")

        for k, xy in enumerate(self.scan_history):
            alpha = 0.3 + 0.7 * (k + 1) / max(len(self.scan_history), 1)
            if xy:
                xs, ys = zip(*xy)
                self.ax_profile.scatter(xs, ys, s=1, alpha=alpha, c="steelblue")

        if self.scan_history and self.scan_history[-1]:
            z_line = self.z_history[-1]
            self.ax_profile.axhline(z_line, color="orange", lw=1.2, label=f"z_axis={z_line} mm")

        self.ax_profile.legend(fontsize=7, loc="upper right")

        # ---- z_axis trend ----
        self.ax_z.cla()
        self.ax_z.set_title("z_axis value per scan")
        self.ax_z.set_xlabel("Scan index")
        self.ax_z.set_ylabel("max Y in window (mm)")
        if self.z_history:
            n = self.scan_count
            xs = range(n - len(self.z_history), n)
            self.ax_z.plot(xs, self.z_history, marker="o", ms=3, color="darkorange")
            self.ax_z.set_ylim(
                0,
                Constants.BOUNDARIES_ZAXIS_Y_MIN + 200,
            )

        self.fig.tight_layout()

    def run(self):
        ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=50,   # ms between frames
            cache_frame_data=False,
        )
        try:
            plt.show()
        finally:
            self.close()

    def close(self):
        print("[INFO] Shutting down...")
        try:
            teardown_sensor(self.sensor)
        except Exception as e:
            print(f"[WARN] Error during teardown: {e}")
        try:
            self.sock.close()
        except Exception:
            pass
        print("[INFO] Done.")


if __name__ == "__main__":
    receiver = LiveReceiver()
    receiver.run()
