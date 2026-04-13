import copy
import os
import numpy as np
import open3d as o3d

from src.PointCloudReconstructor import PointCloudReconstructor
from src.VolumeCalculatorLegacy import VolumeCalculatorLegacy
from src.VolumeCalculator import VolumeCalculator
from src.PointCloudPlotter import PointCloudPlotter
from src.Registration import Registration
from src.SurfaceReconstructor import SurfaceReconstructor
from src.Constants import Constants
from src.Parameters import Parameters


class DataManager():

    def __init__(self):
        self.point_cloud_plotter = PointCloudPlotter()
        self.volume_calculator_legacy = VolumeCalculatorLegacy()
        self.volume_calculator = VolumeCalculator()
        self.pcd_reconstructor = PointCloudReconstructor()
        self.registration = Registration()
        self.surface_reconstructor = SurfaceReconstructor()

    # ------------------------------------------------------------------
    # Helpers privados compartilhados pelos dois fluxos
    # ------------------------------------------------------------------

    def _load_scan_and_bucket(self, scan_path: str):
        """Carrega a nuvem de pontos do scan e da caçamba de referência."""
        if os.path.isfile(f"{scan_path}data.npz"):
            print(f"[INFO] Carregando dados de {scan_path}data.npz")
            xyz_array = np.load(f"{scan_path}data.npz")["xyz"]
            xyz = o3d.geometry.PointCloud()
            xyz.points = o3d.utility.Vector3dVector(xyz_array)
        else:
            print(f"[INFO] Processando arquivos binários de {scan_path}")
            xyz_list = self.pcd_reconstructor.create_point_cloud(scan_path)
            np.savez_compressed(f"{scan_path}data.npz", xyz=xyz_list)
            xyz = o3d.geometry.PointCloud()
            xyz.points = o3d.utility.Vector3dVector(np.array(xyz_list))

        bucket_data_path = f"{Constants.BUCKET_PATH}/data.npz"
        if os.path.isfile(bucket_data_path):
            print(f"[INFO] Carregando caçamba de {bucket_data_path}")
            bucket_array = np.load(bucket_data_path)["xyz"]
            truck_bucket = o3d.geometry.PointCloud()
            truck_bucket.points = o3d.utility.Vector3dVector(bucket_array)
        else:
            print(f"[INFO] Processando arquivos binários da caçamba de {Constants.BUCKET_PATH}")
            bucket_list = self.pcd_reconstructor.create_point_cloud(Constants.BUCKET_PATH)
            truck_bucket = o3d.geometry.PointCloud()
            truck_bucket.points = o3d.utility.Vector3dVector(np.array(bucket_list))

        return xyz, truck_bucket

    def _align_auto(self, scan_path: str, xyz: o3d.geometry.PointCloud,
                    truck_bucket: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Alinha o scan com a caçamba. Detecta automaticamente sintético vs real.

        Scans sintéticos já estão no sistema de coordenadas da caçamba; RANSAC+ICP
        falha neles pois só há superfície da carga (sem paredes visíveis para matching).
        """
        if os.path.isfile(f"{scan_path}SYNTHETIC_INFO.txt"):
            print(f"[INFO] Scan sintético detectado — alinhamento por centróide (XY)")
            xyz_pts = np.asarray(xyz.points)
            bucket_pts = np.asarray(truck_bucket.points)
            translation = bucket_pts.mean(axis=0) - xyz_pts.mean(axis=0)
            translation[2] = 0  # preservar Z
            aligned_pcd = copy.deepcopy(xyz)
            aligned_pcd.points = o3d.utility.Vector3dVector(xyz_pts + translation)
        else:
            aligned_pcd = self.registration.align_truck_bucket_and_load(
                xyz, truck_bucket,
                Parameters.Registration.VOXEL_SIZE,
                Parameters.Registration.MAX_ITERATION_RANSAC,
                Parameters.Registration.CONFIDENCE,
                Parameters.Registration.MAX_NN_NORMALS,
                Parameters.Registration.MAX_NN_FPFH,
                Parameters.Registration.EPSILON,
                Parameters.Registration.MAX_ITERATION_ICP,
                Parameters.Registration.RANSAC_LOOP_SIZE,
            )
        return aligned_pcd

    def _isolate_load(self, truck_bucket: o3d.geometry.PointCloud,
                      aligned_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Isola os pontos da carga removendo os pontos pertencentes à caçamba vazia."""
        return self.surface_reconstructor.isolate_load_points(
            truck_bucket, aligned_pcd,
            Parameters.BucketRemoval.NB_NEIGHBORS,
            Parameters.BucketRemoval.STD_RATIO,
            Parameters.BucketRemoval.NB_POINTS,
            Parameters.BucketRemoval.RADIUS,
            Parameters.BucketRemoval.THRESHOLD_DISTANCE,
            Parameters.BucketRemoval.DBSCAN_EPS,
            Parameters.BucketRemoval.DBSCAN_MIN_SAMPLES,
        )

    # ------------------------------------------------------------------
    # Fluxo novo — Mapa de Alturas 2D
    # ------------------------------------------------------------------

    def process_data(self, scan_path: str) -> float:
        """Fluxo novo: alinhamento adaptativo → isolamento → volume por mapa de alturas 2D.

        Calcula V = ∑ z_max(x,y) × Δx × Δy diretamente sobre a nuvem de pontos.
        Não requer malha fechada — robusto a buracos e regiões esparsas.
        """
        xyz, truck_bucket = self._load_scan_and_bucket(scan_path)
        aligned_pcd = self._align_auto(scan_path, xyz, truck_bucket)
        load_pcd = self._isolate_load(truck_bucket, aligned_pcd)

        volume = self.volume_calculator.volume_from_heightmap(
            load_pcd,
            cell_size=Parameters.VolumeCalculation.HEIGHTMAP_CELL_SIZE,
        )
        return volume

    # ------------------------------------------------------------------
    # Fluxo legado — Poisson + Teorema da Divergência
    # ------------------------------------------------------------------

    def process_data_legacy(self, scan_path: str) -> float:
        """Fluxo legado: RANSAC+ICP → isolamento → merge com caçamba → Poisson → volume.

        Constrói malha fechada (watertight) via Poisson e calcula o volume pelo
        teorema da divergência. Sensível a buracos e regiões sem dados escaneados.
        """
        xyz, truck_bucket = self._load_scan_and_bucket(scan_path)

        # Fluxo legado sempre usa RANSAC+ICP (sem auto-detect sintético)
        aligned_pcd = self.registration.align_truck_bucket_and_load(
            xyz, truck_bucket,
            Parameters.Registration.VOXEL_SIZE,
            Parameters.Registration.MAX_ITERATION_RANSAC,
            Parameters.Registration.CONFIDENCE,
            Parameters.Registration.MAX_NN_NORMALS,
            Parameters.Registration.MAX_NN_FPFH,
            Parameters.Registration.EPSILON,
            Parameters.Registration.MAX_ITERATION_ICP,
            Parameters.Registration.RANSAC_LOOP_SIZE,
        )

        load_pcd = self._isolate_load(truck_bucket, aligned_pcd)

        merged_pcd = self.surface_reconstructor.merge_load_and_bucket_points(
            truck_bucket, load_pcd,
            Parameters.MergePoints.RAY_CAST_ORIGIN_X,
            Parameters.MergePoints.RAY_CAST_ORIGIN_Y,
            Parameters.MergePoints.RAY_CAST_ORIGIN_Z,
            Parameters.MergePoints.SIMPLE_MESH_RADIUS,
            Parameters.MergePoints.SIMPLE_MESH_MAX_NN,
            Parameters.MergePoints.SIMPLE_MESH_K,
            Parameters.MergePoints.NB_NEIGHBORS,
            Parameters.MergePoints.STD_RATIO,
        )

        mesh = self.surface_reconstructor.reconstruct_load_mesh_poisson(
            merged_pcd,
            depth=Parameters.MeshReconstruction.POISSON_DEPTH,
            n_filter_iterations=Parameters.MeshReconstruction.N_FILTER_ITERATIONS,
            density_quantile=Parameters.MeshReconstruction.DENSITY_QUANTILE,
        )

        volume = self.volume_calculator.volume_calculation(mesh)
        return volume
