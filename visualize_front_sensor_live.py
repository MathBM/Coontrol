
import socket
import struct
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, pi
from struct import pack, unpack
from src.Constants import Constants
from src.SensorManager import SensorManager


# Se o sensor retorna apenas distâncias, não use polar_to_xy
def distances_to_profile(distances):
    # Remove valores inválidos (ex: 0xFFFFFFFF)
    return [d for d in distances if d != 4_294_967_295]

# Parâmetros do sensor (TOP)
sensor_ip = Constants.SENSOR_LEFT_IP
server_ip = Constants.SERVER_IP
server_port = Constants.SERVER_PORT

# Instancia o sensor TOP
sensor = SensorManager(sensor_ip, server_ip, server_port)
sensor.set_parameters(samples_per_scan=600, scan_frequency=50, scan_direction=Constants.SCAN_DIRECTION)
handle = sensor.request_handle_tcp(max_num_points_scan=600)
port = handle["data"].get("port", None)

if not port:
    print("Não foi possível obter a porta do sensor.")
    exit(1)

# Inicia o scan
sensor.start_scanoutput()

# Conecta ao socket TCP do sensor
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((sensor_ip, port))

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-')
ax.set_xlim(-5000, 5000)
ax.set_ylim(-5000, 5000)

buffer = b""
magic_byte = pack("H", 0xa25c)

try:
    while True:
        data = sock.recv(4096)
        if not data:
            break
        buffer += data
        while True:
            idx = buffer.find(magic_byte)
            if idx == -1:
                break
            if len(buffer) < idx + 50:
                # Cabeçalho mínimo + payload
                break
            # Encontrou magic_byte, tenta ler cabeçalho
            start = idx + len(magic_byte)
            try:
                packet_size = unpack("I", buffer[start+2:start+6])[0] - len(magic_byte)
                header_size = unpack("H", buffer[start+6:start+8])[0] - len(magic_byte)
                scan_number = unpack("H", buffer[start+8:start+10])[0]
                first_angle = unpack("i", buffer[start+42:start+46])[0]
                angular_increment = unpack("i", buffer[start+46:start+50])[0]
            except Exception as e:
                print(f"[Exception] Erro ao decodificar cabeçalho: {e}")
                break
            if len(buffer) < idx + packet_size:
                # Pacote ainda não completo
                break
            packet = buffer[idx:idx+packet_size]
            payload = packet[header_size:]
            if len(payload) == 0:
                buffer = buffer[idx+packet_size:]
                continue
            try:
                distances = unpack(f"{len(payload)//4}I", payload[:len(payload)//4*4])
            except Exception as e:
                print(f"[Exception] Erro ao decodificar payload: {e}")
                buffer = buffer[idx+packet_size:]
                continue
            profile = distances_to_profile(distances)
            if len(profile) == 0:
                buffer = buffer[idx+packet_size:]
                continue
            print("Perfil plotado:")
            print(profile)
            line.set_xdata(np.arange(len(profile)))
            line.set_ydata(profile)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            buffer = buffer[idx+packet_size:]
except KeyboardInterrupt:
    print("Interrompido pelo usuário.")
finally:
    sock.close()
    sensor.stop_scanoutput()
    sensor.release_handle()
    plt.ioff()
    plt.show()
