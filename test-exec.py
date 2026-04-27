from src.ScanManager import ScanManager
import time

sm = ScanManager()

scan_path = "./pointcloud/2026-04-24_caixa/"
sm.start_scan(scan_path)

time.sleep(10)  # mantém a gravação pelo tempo desejado

sm.stop_scan()  # stop → release_handle → aguarda Rust encerrar