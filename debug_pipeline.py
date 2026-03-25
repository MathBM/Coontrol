"""Debug do pipeline de processamento de volume"""
import os
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

scan_folder = "2026-03-18_15h36min01s_SYNTHETIC_linear"
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
print("\n[2] ALINHAMENTO (RANSAC + ICP)...")
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

# Visualizar alinhamento
visualize_step([aligned_pcd.paint_uniform_color([1, 0, 0]), 
                truck_bucket.paint_uniform_color([0, 1, 0])], 
               "2. Após Alinhamento: Rampa alinhada (vermelho) + Caçamba (verde)")

# DEBUG: Verificar coordenadas Z
print("\n[DEBUG] Verificando coordenadas Z:")
aligned_points = np.asarray(aligned_pcd.points)
bucket_points = np.asarray(truck_bucket.points)
print(f"  Rampa alinhada - Z: min={aligned_points[:, 2].min():.1f} max={aligned_points[:, 2].max():.1f}")
print(f"  Caçamba        - Z: min={bucket_points[:, 2].min():.1f} max={bucket_points[:, 2].max():.1f}")
print(f"  Pontos da rampa com Z > 100mm: {np.sum(aligned_points[:, 2] > 100)}")
print(f"  Pontos da rampa com Z > 50mm: {np.sum(aligned_points[:, 2] > 50)}")
print(f"  Pontos da rampa com Z > 20mm: {np.sum(aligned_points[:, 2] > 20)}")

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
    print("\n❌ PROBLEMA: Carga isolada está vazia!")
    print("   Possível causa: Os pontos da rampa foram todos removidos como se fossem caçamba")
    print("   Solução: Verificar parâmetros de BucketRemoval")
    exit(1)

# Visualizar carga isolada
visualize_step([load_pcd.paint_uniform_color([1, 0, 0])], 
               "3. Carga Isolada (vermelho)")

# 4. MERGE (adicionar base da caçamba)
print("\n[4] MERGE (carga + base da caçamba)...")
full_pcd = surface_reconstructor.merge_load_and_bucket_points(
    load_pcd, truck_bucket,
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

# Visualizar merge
visualize_step([full_pcd.paint_uniform_color([0, 0, 1])], 
               "4. Após Merge (azul)")

# 5. RECONSTRUÇÃO DA MALHA
print("\n[5] RECONSTRUÇÃO DA MALHA (Poisson)...")
load_mesh = surface_reconstructor.reconstruct_load_mesh_poisson(
    full_pcd,
    depth=10,
    n_filter_iterations=Parameters.MeshReconstruction.N_FILTER_ITERATIONS
)

if isinstance(load_mesh, o3d.geometry.TriangleMesh):
    print(f"Vértices: {len(load_mesh.vertices)}")
    print(f"Triângulos: {len(load_mesh.triangles)}")
    print(f"Watertight: {load_mesh.is_watertight()}")
    if len(load_mesh.vertices) > 0:
        vertices = np.asarray(load_mesh.vertices)
        print(f"X: min={vertices[:, 0].min():.1f} max={vertices[:, 0].max():.1f}")
        print(f"Y: min={vertices[:, 1].min():.1f} max={vertices[:, 1].max():.1f}")
        print(f"Z: min={vertices[:, 2].min():.1f} max={vertices[:, 2].max():.1f}")
else:
    print(f"Tipo: {type(load_mesh)}")

# Visualizar malha
visualize_step([load_mesh], "5. Malha Reconstruída (Poisson)")

# 6. CÁLCULO DO VOLUME
print("\n[6] CÁLCULO DO VOLUME...")
volume_calculator = VolumeCalculator()
volume_mm3 = volume_calculator.volume_calculation(load_mesh)
volume_m3 = volume_mm3 / 1_000_000_000

print(f"\nVolume calculado: {volume_mm3:.2f} mm³")
print(f"Volume calculado: {volume_m3:.4f} m³")
print(f"Volume esperado:  2.4000 m³")
print(f"Erro: {abs(volume_m3 - 2.4):.4f} m³ ({abs(volume_m3 - 2.4)/2.4*100:.2f}%)")

# DIAGNÓSTICO
print(f"\n{'='*60}")
print("DIAGNÓSTICO")
print(f"{'='*60}")

error_pct = abs(volume_m3 - 2.4)/2.4*100

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
    print("   Causa provável: Parâmetros de isolamento muito agressivos")

print(f"\n{'='*60}")
print("MÉTODO DE RECONSTRUÇÃO")
print(f"{'='*60}")
print("✓ Usando Poisson Surface Reconstruction (depth=10)")
print("  - Gera malhas mais precisas que Alpha Shapes")
print("  - Erro típico: < 5% para dados sintéticos")
print("  - Quase-watertight (fecha automaticamente)")

print(f"\n{'='*60}")
