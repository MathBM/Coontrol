"""Debug do pipeline de processamento de volume"""
import os
import sys
import numpy as np
import open3d as o3d
from src.Constants import Constants
from src.Parameters import Parameters
from src.Registration import Registration
from src.SurfaceReconstructor import SurfaceReconstructor
from src.VolumeCalculator import VolumeCalculator

def print_stats(name, pcd):
    """Imprime estatísticas de uma nuvem de pontos"""
    if isinstance(pcd, o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
    else:
        points = pcd
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Pontos: {len(points)}")
    if len(points) > 0:
        print(f"X: min={points[:, 0].min():.1f} max={points[:, 0].max():.1f} range={points[:, 0].max() - points[:, 0].min():.1f} mm")
        print(f"Y: min={points[:, 1].min():.1f} max={points[:, 1].max():.1f} range={points[:, 1].max() - points[:, 1].min():.1f} mm")
        print(f"Z: min={points[:, 2].min():.1f} max={points[:, 2].max():.1f} range={points[:, 2].max() - points[:, 2].min():.1f} mm")
    else:
        print("❌ VAZIO!")

def visualize_step(pcds, window_name):
    """Visualiza nuvens de pontos"""
    o3d.visualization.draw_geometries(pcds, window_name=window_name)

if len(sys.argv) < 2:
    print("Uso: python debug_pipeline.py <pasta_scan>")
    print("Exemplo: python debug_pipeline.py 2026-04-02_09h49min48s_SYNTHETIC_convex")
    sys.exit(1)

scan_folder = sys.argv[1]
scan_path = f"{Constants.SCANS_DIRECTORY}{scan_folder}/"

print(f"\n{'#'*60}")
print(f"# DEBUG PIPELINE: {scan_folder}")
print(f"{'#'*60}")

# 1. CARREGAR DADOS
print("\n[1] CARREGANDO DADOS...")
xyz_array = np.load(f"{scan_path}data.npz")["xyz"]
xyz = o3d.geometry.PointCloud()
xyz.points = o3d.utility.Vector3dVector(xyz_array)
print_stats("RAMPA (xyz)", xyz)

bucket_array = np.load(f"{Constants.BUCKET_PATH}/data.npz")["xyz"]
truck_bucket = o3d.geometry.PointCloud()
truck_bucket.points = o3d.utility.Vector3dVector(bucket_array)
print_stats("CAÇAMBA (truck_bucket)", truck_bucket)

# Visualizar dados originais
visualize_step([xyz.paint_uniform_color([1, 0, 0]), 
                truck_bucket.paint_uniform_color([0, 1, 0])], 
               "1. Dados Originais: Rampa (vermelho) + Caçamba (verde)")

# 2. ALINHAMENTO
print("\n[2] ALINHAMENTO...")

is_synthetic = os.path.exists(f"{scan_path}SYNTHETIC_INFO.txt")

if is_synthetic:
    # Dados sintéticos: scan já está no mesmo sistema de coordenadas da caçamba.
    # Usamos translação por centróide (XY only) — idêntico ao que seria feito em
    # produção se o sensor estivesse perfeitamente calibrado.
    # RANSAC+ICP falha aqui porque o scan sintético só tem superfície de carga
    # (sem paredes da caçamba visíveis para matching de features).
    print("[ALINHAMENTO] Scan sintético — usando translação por centróide (XY)")
    xyz_points = np.asarray(xyz.points)
    bucket_points_arr = np.asarray(truck_bucket.points)
    translation = bucket_points_arr.mean(axis=0) - xyz_points.mean(axis=0)
    translation[2] = 0  # preservar Z
    aligned_arr = xyz_points + translation
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_arr)
else:
    # Dados reais: usar RANSAC + Generalized ICP — idêntico ao DataManager.process_data
    print("[ALINHAMENTO] Scan real — usando RANSAC + ICP (igual ao DataManager)")
    registration = Registration()
    aligned_pcd = registration.align_truck_bucket_and_load(
        xyz, truck_bucket,
        Parameters.Registration.VOXEL_SIZE,
        Parameters.Registration.MAX_ITERATION_RANSAC,
        Parameters.Registration.CONFIDENCE,
        Parameters.Registration.MAX_NN_NORMALS,
        Parameters.Registration.MAX_NN_FPFH,
        Parameters.Registration.EPSILON,
        Parameters.Registration.MAX_ITERATION_ICP,
        Parameters.Registration.RANSAC_LOOP_SIZE
    )

print_stats("RAMPA ALINHADA (aligned_pcd)", aligned_pcd)

visualize_step([aligned_pcd.paint_uniform_color([1, 0, 0]), 
                 truck_bucket.paint_uniform_color([0, 1, 0])], 
                "2. Após Alinhamento")

# 3. ISOLAMENTO DA CARGA
print("\n[3] ISOLANDO CARGA (removendo caçamba)...")
surface_reconstructor = SurfaceReconstructor()
load_pcd = surface_reconstructor.isolate_load_points(
    truck_bucket, aligned_pcd,
    Parameters.BucketRemoval.NB_NEIGHBORS,
    Parameters.BucketRemoval.STD_RATIO,
    Parameters.BucketRemoval.NB_POINTS,
    Parameters.BucketRemoval.RADIUS,
    Parameters.BucketRemoval.THRESHOLD_DISTANCE,
    Parameters.BucketRemoval.DBSCAN_EPS,
    Parameters.BucketRemoval.DBSCAN_MIN_SAMPLES
)
print_stats("CARGA ISOLADA (load_pcd)", load_pcd)

if len(np.asarray(load_pcd.points)) == 0:
    print("\n❌ ERRO: Carga isolada está vazia!")
    exit(1)

visualize_step([load_pcd], "3. Carga Isolada")

# 4. MERGE (apenas para visualização)
print("\n[4] MERGE (carga + base da caçamba — apenas visualização)...")
full_pcd = surface_reconstructor.merge_load_and_bucket_points(
    truck_bucket, load_pcd,
    Parameters.MergePoints.RAY_CAST_ORIGIN_X,
    Parameters.MergePoints.RAY_CAST_ORIGIN_Y,
    Parameters.MergePoints.RAY_CAST_ORIGIN_Z,
    Parameters.MergePoints.SIMPLE_MESH_RADIUS,
    Parameters.MergePoints.SIMPLE_MESH_MAX_NN,
    Parameters.MergePoints.SIMPLE_MESH_K,
    Parameters.MergePoints.NB_NEIGHBORS,
    Parameters.MergePoints.STD_RATIO
)
print_stats("PONTOS COMPLETOS (full_pcd)", full_pcd)

visualize_step([full_pcd], "4. Após Merge")

# 5. RECONSTRUÇÃO DA MALHA (visualização — não é usada para calcular volume)
print("\n[5] RECONSTRUÇÃO DA MALHA (Poisson — apenas visualização)...")
load_mesh = surface_reconstructor.reconstruct_load_mesh_poisson(
    full_pcd,
    depth=Parameters.MeshReconstruction.POISSON_DEPTH,
    density_quantile=Parameters.MeshReconstruction.DENSITY_QUANTILE
)

print(f"Vértices: {len(load_mesh.vertices)}")
print(f"Triângulos: {len(load_mesh.triangles)}")
print(f"Watertight: {load_mesh.is_watertight()}")
if not load_mesh.is_watertight():
    print("⚠️  Malha aberta — cálculo de volume via malha seria impreciso.")
    print("   Volume real é calculado via mapa de alturas (próximo passo).")

visualize_step([load_mesh], "5. Malha Reconstruída (visualização)")

# 6. CÁLCULO DO VOLUME — via mapa de alturas 2D sobre load_pcd
# V = ∑ z_max(x,y) × Δx × Δy  (integra a superfície escaneada acima de z=0)
# Robusto a buracos e malhas abertas: células sem dado contribuem z=0.
print("\n[6] CÁLCULO DO VOLUME (mapa de alturas)...")
volume_calculator = VolumeCalculator()
volume_mm3 = volume_calculator.volume_from_heightmap(
    load_pcd,
    cell_size=Parameters.VolumeCalculation.HEIGHTMAP_CELL_SIZE
)
volume_m3 = volume_mm3 / 1_000_000_000

# Volume esperado depende do tipo (lido do SYNTHETIC_INFO.txt se disponível)
# linear:  W * L * H / 2
# convex:  W * L * H * 2/3   (integral de sqrt(x/L))
# concave: W * L * H * 1/3   (integral de x^2/L^2)
# default: linear
import re as _re
_info_path = f"{scan_path}SYNTHETIC_INFO.txt"
_ramp_type = "linear"
_w, _l, _h = 1.8, 2.8, 0.8  # fallback (m)
if os.path.exists(_info_path):
    _info = open(_info_path).read()
    _m = _re.search(r"Tipo:\s*(\w+)", _info)
    if _m: _ramp_type = _m.group(1).lower()
    _mw = _re.search(r"Largura:\s*([\d.]+)", _info)
    _ml = _re.search(r"Comprimento:\s*([\d.]+)", _info)
    _mh = _re.search(r"Altura:\s*([\d.]+)", _info)
    if _mw: _w = float(_mw.group(1)) / 1000
    if _ml: _l = float(_ml.group(1)) / 1000
    if _mh: _h = float(_mh.group(1)) / 1000

_volume_factors = {"linear": 1/2, "convex": 2/3, "concave": 1/3}
_factor = _volume_factors.get(_ramp_type, 1/2)
expected_volume = _w * _l * _h * _factor

# Para sand_pile o volume esperado está diretamente nos metadados
_mev = _re.search(r"Volume esperado m3:\s*([\d.]+)", _info if os.path.exists(_info_path) else "")
if _mev:
    expected_volume = float(_mev.group(1))
    _factor = None

print(f"\nVolume calculado: {volume_mm3:.2f} mm³")
print(f"Volume calculado: {volume_m3:.4f} m³")
if _factor is not None:
    print(f"Volume esperado:  {expected_volume:.4f} m³ (tipo={_ramp_type}, {_w*1000:.0f}×{_l*1000:.0f}×{_h*1000:.0f}mm, fator={_factor:.3f})")
else:
    print(f"Volume esperado:  {expected_volume:.4f} m³ (tipo={_ramp_type}, integração numérica)")
print(f"Erro: {abs(volume_m3 - expected_volume):.4f} m³ ({abs(volume_m3 - expected_volume)/expected_volume*100:.2f}%)")

# DIAGNÓSTICO
print(f"\n{'='*60}")
print("DIAGNÓSTICO")
print(f"{'='*60}")

error_pct = abs(volume_m3 - expected_volume)/expected_volume*100

if error_pct < 5:
    print(f"✅ EXCELENTE: Erro de {error_pct:.2f}% está dentro da margem aceitável (<5%)")
elif error_pct < 15:
    print(f"⚠️  ACEITÁVEL: Erro de {error_pct:.2f}% é razoável, mas pode melhorar")
elif error_pct < 50:
    print(f"⚠️  ALTO: Erro de {error_pct:.2f}% indica problemas moderados")
else:
    print(f"❌ CRÍTICO: Erro de {error_pct:.2f}% muito alto")

if len(np.asarray(load_pcd.points)) < 1000:
    print("⚠️  ALERTA: Poucos pontos na carga isolada")

print(f"\n{'='*60}")
