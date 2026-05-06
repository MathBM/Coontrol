from src.PointCloudReconstructor import PointCloudReconstructor
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 run_reconstructor.py <scan_path>")
        sys.exit(1)
    scan_path = sys.argv[1]
    if not scan_path.endswith("/"):
        scan_path += "/"
    reconstructor = PointCloudReconstructor()
    xyz = reconstructor.create_point_cloud(scan_path)
    print(f"Total de pontos reconstruídos: {len(xyz)}")
    # Opcional: salvar ou visualizar pontos
    # print(xyz[:10])  # Mostra os 10 primeiros pontos