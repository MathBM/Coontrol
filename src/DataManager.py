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

    def process_data(self, scan_path: str) -> float:
        # Verificar se é scan sintético ou se data.npz já existe
        if os.path.isfile(f"{scan_path}data.npz"):
            print(f"[INFO] Carregando dados de {scan_path}data.npz")
            xyz_array = np.load(f"{scan_path}data.npz")["xyz"]
            # Converter array numpy para PointCloud Open3D
            xyz = o3d.geometry.PointCloud()
            xyz.points = o3d.utility.Vector3dVector(xyz_array)
        else:
            print(f"[INFO] Processando arquivos binários de {scan_path}")
            xyz_list = self.pcd_reconstructor.create_point_cloud(scan_path)
            np.savez_compressed(f"{scan_path}data.npz", xyz=xyz_list)
            # Converter lista para PointCloud Open3D
            xyz = o3d.geometry.PointCloud()
            xyz.points = o3d.utility.Vector3dVector(np.array(xyz_list))

        # Carregar caçamba de referência
        bucket_data_path = f"{Constants.BUCKET_PATH}/data.npz"
        if os.path.isfile(bucket_data_path):
            print(f"[INFO] Carregando caçamba de {bucket_data_path}")
            bucket_array = np.load(bucket_data_path)["xyz"]
            # Converter array numpy para PointCloud Open3D
            truck_bucket = o3d.geometry.PointCloud()
            truck_bucket.points = o3d.utility.Vector3dVector(bucket_array)
        else:
            print(f"[INFO] Processando arquivos binários da caçamba de {Constants.BUCKET_PATH}")
            bucket_list = self.pcd_reconstructor.create_point_cloud(Constants.BUCKET_PATH)
            # Converter lista para PointCloud Open3D
            truck_bucket = o3d.geometry.PointCloud()
            truck_bucket.points = o3d.utility.Vector3dVector(np.array(bucket_list))

        aligned_pcd = self.registration.align_truck_bucket_and_load(xyz, truck_bucket, Parameters.Registration.VOXEL_SIZE,
                                                                    Parameters.Registration.MAX_ITERATION_RANSAC,
                                                                    Parameters.Registration.CONFIDENCE,
                                                                    Parameters.Registration.MAX_NN_NORMALS,
                                                                    Parameters.Registration.MAX_NN_FPFH,
                                                                    Parameters.Registration.EPSILON,
                                                                    Parameters.Registration.MAX_ITERATION_ICP,
                                                                    Parameters.Registration.RANSAC_LOOP_SIZE)

        load_pcd = self.surface_reconstructor.isolate_load_points(truck_bucket, aligned_pcd,
                                                                  Parameters.BucketRemoval.NB_NEIGHBORS,
                                                                  Parameters.BucketRemoval.STD_RATIO,
                                                                  Parameters.BucketRemoval.NB_POINTS,
                                                                  Parameters.BucketRemoval.RADIUS,
                                                                  Parameters.BucketRemoval.THRESHOLD_DISTANCE,
                                                                  Parameters.BucketRemoval.DBSCAN_EPS,
                                                                  Parameters.BucketRemoval.DBSCAN_MIN_SAMPLES)

        full_pcd = self.surface_reconstructor.merge_load_and_bucket_points(load_pcd, truck_bucket,
                                                                           Parameters.MergePoints.RAY_CAST_ORIGIN_X,
                                                                           Parameters.MergePoints.RAY_CAST_ORIGIN_Y,
                                                                           Parameters.MergePoints.RAY_CAST_ORIGIN_Z,
                                                                           Parameters.MergePoints.SIMPLE_MESH_RADIUS,
                                                                           Parameters.MergePoints.SIMPLE_MESH_MAX_NN,
                                                                           Parameters.MergePoints.SIMPLE_MESH_K,
                                                                           Parameters.MergePoints.NB_NEIGHBORS,
                                                                           Parameters.MergePoints.STD_RATIO)
        
        load_mesh = self.surface_reconstructor.reconstruct_load_mesh(full_pcd, Parameters.MeshReconstruction.ALPHA,
                                                                     Parameters.MeshReconstruction.N_FILTER_ITERATIONS)
        
        volume = self.volume_calculator.volume_calculation(load_mesh)
        
        return volume
