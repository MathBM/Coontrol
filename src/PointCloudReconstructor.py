import os
import bisect
import numpy as np
import open3d as o3d
from math import cos, sin, pi
from struct import pack, unpack

from src.Constants import Constants
from src.Parameters import Parameters


class PointCloudReconstructor():

    def create_point_cloud(self, scan_path: str):
        scans_front = self.process_binary_file(os.path.join(scan_path, f"{Constants.SENSOR_FRONT_IP}.bin"))
        scans_right = self.process_binary_file(os.path.join(scan_path, f"{Constants.SENSOR_RIGHT_IP}.bin"))
        scans_left = self.process_binary_file(os.path.join(scan_path, f"{Constants.SENSOR_LEFT_IP}.bin"))
        scans_top = self.process_binary_file(os.path.join(scan_path, f"{Constants.SENSOR_TOP_IP}.bin"))

        z_axis, _, front_timestamps = self.calculate_z_axis(
            scans_front,
            Constants.BOUNDARIES_ZAXIS_X_MIN,
            Constants.BOUNDARIES_ZAXIS_X_MAX,
            Constants.BOUNDARIES_ZAXIS_Y_MIN,
            Constants.BOUNDARIES_ZAXIS_Y_MAX,
        )

        xyz_right = self.reconstruct_z_axis(scans_right, z_axis, front_timestamps)
        xyz_left = self.reconstruct_z_axis(scans_left, z_axis, front_timestamps)
        xyz_top = self.reconstruct_z_axis(scans_top, z_axis, front_timestamps)

        xyz_right = self.transform(xyz_right, Constants.SENSOR_RIGHT_ROTATION, Constants.SENSOR_RIGHT_TRANSLATION)
        xyz_left = self.transform(xyz_left, Constants.SENSOR_LEFT_ROTATION, Constants.SENSOR_LEFT_TRANSLATION)
        xyz_top = self.transform(xyz_top, (0, 0, 0), (0, 0, Constants.SENSOR_TOP_Z_OFFSET))

        xyz_right = self.remove_boundaries(
            xyz_right,
            Constants.BOUNDARIES_PROFILE_X_MIN,
            Constants.BOUNDARIES_PROFILE_X_MAX,
            Constants.BOUNDARIES_PROFILE_Y_MIN,
            Constants.BOUNDARIES_PROFILE_Y_MAX,
        )

        xyz_left = self.remove_boundaries(
            xyz_left,
            Constants.BOUNDARIES_PROFILE_X_MIN,
            Constants.BOUNDARIES_PROFILE_X_MAX,
            Constants.BOUNDARIES_PROFILE_Y_MIN,
            Constants.BOUNDARIES_PROFILE_Y_MAX,
        )

        xyz_top = self.remove_boundaries(
            xyz_top,
            Constants.BOUNDARIES_PROFILE_X_MIN,
            Constants.BOUNDARIES_PROFILE_X_MAX,
            Constants.BOUNDARIES_PROFILE_Y_MIN,
            Constants.BOUNDARIES_PROFILE_Y_MAX,
        )

        xyz_right = self.filter_point_cloud(xyz_right, 40, 0.1, 25, 50)
        xyz_left = self.filter_point_cloud(xyz_left, 40, 0.1, 25, 50)
        xyz_top = self.filter_point_cloud(xyz_top, 60, 0.06, 25, 120)

        # Transformação global: rotação -90° em Z + translação em Y (altura do sensor top)
        # O X_OFFSET do top é aplicado aqui, após a rotação, para mover no eixo X visual correto
        xyz_right = self.transform(xyz_right, (0, 0, -pi/2), (0, Constants.SENSOR_TOP_HEIGHT, 0))
        xyz_left  = self.transform(xyz_left,  (0, 0, -pi/2), (0, Constants.SENSOR_TOP_HEIGHT, 0))
        xyz_top   = self.transform(xyz_top,   (0, 0, -pi/2), (Constants.SENSOR_TOP_X_OFFSET, Constants.SENSOR_TOP_HEIGHT, 0))

        # Clip em coordenadas mundiais: remove paredes/teto do galpão que ficaram fora da caçamba
        p = Parameters.Registration
        xyz_right = self.remove_boundaries(xyz_right, p.CROP_X_MIN, p.CROP_X_MAX, p.CROP_Y_MIN, p.CROP_Y_MAX)
        xyz_left  = self.remove_boundaries(xyz_left,  p.CROP_X_MIN, p.CROP_X_MAX, p.CROP_Y_MIN, p.CROP_Y_MAX)
        xyz_top   = self.remove_boundaries(xyz_top,   p.CROP_X_MIN, p.CROP_X_MAX, p.CROP_Y_MIN, p.CROP_Y_MAX)

        # Recorte lateral só do top: tira as faces das paredes que o sensor top
        # enxerga de cima, deixando apenas chão/carga. Centrado no X_OFFSET (o mesmo
        # deslocamento aplicado ao top na linha acima), então acompanha o alinhamento
        # manual e a parede fica por conta exclusiva do left/right
        # (ver Constants.SENSOR_TOP_FLOOR_HALF_WIDTH).
        hw = Constants.SENSOR_TOP_FLOOR_HALF_WIDTH
        xyz_top = self.remove_boundaries(
            xyz_top,
            Constants.SENSOR_TOP_X_OFFSET - hw,
            Constants.SENSOR_TOP_X_OFFSET + hw,
            p.CROP_Y_MIN,
            p.CROP_Y_MAX,
        )

        # Correção de yaw do top: desfaz a diagonal da borda da frente causada pelo
        # sensor torto (shear em Z proporcional a X). Ver Constants.SENSOR_TOP_YAW_SLOPE.
        yaw = Constants.SENSOR_TOP_YAW_SLOPE
        if yaw:
            cx = Constants.SENSOR_TOP_X_OFFSET
            xyz_top = [(pt[0], pt[1], pt[2] + yaw * (pt[0] - cx)) for pt in xyz_top]

        # Split por lado: cada sensor lateral varre a seção inteira e enxerga também a
        # parede oposta / o interior, jogando pontos no lado errado (contaminação
        # cruzada, visível como L e R misturados nos dois lados sob a carga). Cada um
        # fica só com a sua metade — right à direita do centro, left à esquerda. O
        # centro é o X_OFFSET (mesmo centro do piso do top), então acompanha o
        # alinhamento manual. A parede fica por conta do sensor do respectivo lado e o
        # chão/carga por conta do top.
        center = Constants.SENSOR_TOP_X_OFFSET
        xyz_right = [pt for pt in xyz_right if pt[0] > center]
        xyz_left  = [pt for pt in xyz_left  if pt[0] < center]

        xyz = list()
        xyz.extend(xyz_right)
        xyz.extend(xyz_left)
        xyz.extend(xyz_top)

        return xyz

    def calculate_z_axis(self, scans_front: dict, x_min: int, x_max: int, y_min: int, y_max: int):
        z_axis = {}
        xyz_front = []
        front_timestamps = {}  # i -> timestamp do scan frontal

        for i, scan_key in enumerate(sorted(scans_front.keys())):
            # z_axis[i] = z_axis.get(i-1, y_min)
            z_axis[i] = y_min
            front_timestamps[i] = scans_front[scan_key]["timestamp"]

            for xy in scans_front[scan_key]["xy"]:
                x = xy[0]
                y = xy[1]
                z = i * 5

                if x <= x_min or x >= x_max or y <= y_max or y >= y_min:
                    continue

                if y < z_axis[i]:
                    z_axis[i] = y

                xyz_front.append((x, y, z))

        # Fix A: descarta o salto espúrio de fim de passagem.
        # Quando a traseira do caminhão sai da janela do sensor front, ele volta a
        # ler o fundo (~2440mm) e z_axis dispara (ex.: 170 -> 2440). Essas linhas
        # finais jogam a parede traseira dos sensores laterais pro Z errado, fazendo
        # a parede "sumir" no scan com carga. Depois que o caminhão é detectado perto
        # (z entra na metade inferior do range), congela z no último valor válido em
        # vez de deixar voltar pro fundo. O caso sem carga não tem o salto e fica
        # inalterado.
        keys = sorted(z_axis.keys())
        if keys:
            z_min = min(z_axis[i] for i in keys)
            z_max = max(z_axis[i] for i in keys)
            near_threshold = z_min + 0.5 * (z_max - z_min)
            truck_detected = False
            last_valid = z_axis[keys[0]]
            for i in keys:
                if z_axis[i] <= near_threshold:
                    truck_detected = True
                    last_valid = z_axis[i]
                elif truck_detected:
                    # caminhão já estava perto e z disparou pro fundo -> congela
                    z_axis[i] = last_valid

        return z_axis, xyz_front, front_timestamps

    def process_binary_file(self, file_path: str):
        file = open(file_path, "rb")
        data = file.read()
        file.close()

        scans = dict()
        magic_byte = pack("H", 0xa25c)

        for packet in data.split(magic_byte):

            if len(packet) <= 10:
                continue

            try:
                # packet_type = unpack("H", packet[:2])[0]
                packet_size = unpack("I", packet[2:6])[0] - len(magic_byte)
                header_size = unpack("H", packet[6:8])[0] - len(magic_byte)
                scan_number = unpack("H", packet[8:10])[0]
                # packet_number = unpack("H", packet[10:12])[0]
                timestamp_raw = unpack("Q", packet[12:20])[0]
                # timestamp_sync = ...
                # status_flags = unpack("I", packet[28:32])[0]
                # scan_frequency = unpack("I", packet[32:36])[0]
                # num_points_scan = unpack("H", packet[36:38])[0]
                # num_points_packet = unpack("H", packet[38:40])[0]
                # first_index = unpack("H", packet[40:42])[0]
                first_angle = unpack("i", packet[42:46])[0]
                angular_increment = unpack("i", packet[46:50])[0]
            except Exception:
                print("[Exception] corrupted package...")
                continue

            if len(packet) != packet_size:
                print("[packet_size] corrupted package...")
                continue

            if scan_number not in scans:
                scans[scan_number] = dict()
                scans[scan_number]["xy"] = list()
                scans[scan_number]["timestamp"] = self.ntp64_to_seconds(timestamp_raw)

            payload = packet[header_size:]  # list[uint32] - 4byte
            distances = unpack(f"{len(payload) // 4}I", payload[:len(payload) // 4 * 4])

            scans[scan_number]["xy"].extend(self.polar_to_xy(distances, first_angle, angular_increment))

        return scans

    def ntp64_to_seconds(self, integer):
        # Upper 32 bits for seconds
        seconds = integer >> 32

        # Lower 32 bits for fractional seconds
        fractional_seconds = integer & 0xFFFFFFFF
        fractional_seconds = fractional_seconds / 0x100000000

        return round(seconds + fractional_seconds, 3)

    def polar_to_xy(self, distances: list, first_angle: int, angular_increment: int):
        first_angle /= 10000
        angular_increment /= 10000

        xy = list()

        for i, distance in enumerate(distances):
            # Invalid measurements return 0xFFFFFFFF
            if distance == 4_294_967_295:
                continue

            angle = (first_angle + i * angular_increment) * pi / 180.0

            x = round(distance * cos(angle))
            y = round(distance * sin(angle))

            xy.append((x, y))

        return xy

    def reconstruct_z_axis(self, scans: dict, z_axis: dict, front_timestamps: dict = None) -> list[tuple[int, int, int]]:
        xyz = list()

        sorted_scan_keys = sorted(scans.keys())

        if front_timestamps is None:
            for key in zip(sorted_scan_keys, sorted(z_axis.keys())):
                for xy in scans[key[0]]["xy"]:
                    xyz.append((xy[0], xy[1], z_axis[key[1]]))
            return xyz

        sorted_indices = sorted(front_timestamps.keys(), key=lambda k: front_timestamps[k])
        sorted_ts = [front_timestamps[k] for k in sorted_indices]
        ts_front_start = sorted_ts[0]
        ts_front_end = sorted_ts[-1]

        # Fix B: correção uniforme de offset de relógio.
        # Cada sensor tem um skew NTP próprio (ex.: right −6.35s, left −0.98s). O
        # limiar antigo de 2s mandava o right pro alinhamento sequencial e o left pro
        # timestamp com ~1s de erro residual, posicionando as paredes em Z diferentes
        # entre si ("deslocada"). Os sensores varrem sincronizados (mesmo nº de scans
        # e mesma duração), então o desvio é um offset ~constante: subtraindo-o, todos
        # caem no mesmo critério (nearest-ts) no relógio do front.
        ts_sensor_start = scans[sorted_scan_keys[0]]["timestamp"]
        clock_offset = ts_sensor_start - ts_front_start

        # Faixa de Z válida = rampa monotônica do front, excluindo os platôs saturados
        # nos dois extremos (fundo e aproximação máxima), onde os pontos colapsariam num
        # plano falso. Ver Constants.ZAXIS_PLATEAU_MARGIN.
        zvals = list(z_axis.values())
        z_lo, z_hi = min(zvals), max(zvals)
        margin = Constants.ZAXIS_PLATEAU_MARGIN
        # Só aplica se houver faixa útil suficiente (evita zerar tudo em rampas curtas).
        drop_plateaus = (z_hi - z_lo) > 3 * margin
        lo_cut, hi_cut = z_lo + margin, z_hi - margin

        for scan_key in sorted_scan_keys:
            ts = scans[scan_key]["timestamp"] - clock_offset

            # Descarta scans fora do intervalo de tempo do sensor frontal
            if ts < ts_front_start or ts > ts_front_end:
                continue

            pos = bisect.bisect_left(sorted_ts, ts)
            if pos == 0:
                z_idx = sorted_indices[0]
            elif pos >= len(sorted_ts):
                z_idx = sorted_indices[-1]
            else:
                if abs(sorted_ts[pos] - ts) < abs(sorted_ts[pos - 1] - ts):
                    z_idx = sorted_indices[pos]
                else:
                    z_idx = sorted_indices[pos - 1]

            z = z_axis[z_idx]
            # Pula pontos que cairiam num platô saturado (plano falso)
            if drop_plateaus and (z <= lo_cut or z >= hi_cut):
                continue

            for xy in scans[scan_key]["xy"]:
                xyz.append((xy[0], xy[1], z))

        return xyz

    def transform(self, points, rotation: tuple[int, int, int], translation: tuple[int, int, int]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if rotation != (0, 0, 0):
            rotation_matrix = pcd.get_rotation_matrix_from_xyz(rotation)
            pcd.rotate(rotation_matrix, center=(0, 0, 0))

        if translation != (0, 0, 0):
            pcd.translate(translation)

        return np.asarray(pcd.points)

    def remove_boundaries(self, points, x_min: int, x_max: int, y_min: int, y_max: int):
        return [p for p in points if not (p[0] <= x_min or p[0] >= x_max or p[1] <= y_min or p[1] >= y_max)]

    def filter_point_cloud(self, points, nb_neighbors, std_ratio, nb_points, radius):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        xyz_1, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        xyz_2, _ = xyz_1.remove_radius_outlier(nb_points=nb_points, radius=radius)

        return np.asarray(xyz_2.points)