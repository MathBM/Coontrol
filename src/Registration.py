import open3d as o3d
import numpy as np
from enum import StrEnum, auto
import copy

class ICPMethod(StrEnum):
  POINT_TO_POINT = auto()
  POINT_TO_PLANE = auto()
  GENERALIZED = auto()

class Registration():
  
  def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud, voxel_size: float, max_nn_normals: int,
                             max_nn_fpfh: int) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normals))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_fpfh))
    return pcd_down, pcd_fpfh
  
  def ransac_registration(self, source_down: o3d.geometry.PointCloud, target_down: o3d.geometry.PointCloud,
                          source_fpfh: o3d.pipelines.registration.Feature,
                          target_fpfh: o3d.pipelines.registration.Feature, voxel_size: float, max_iteration: int,
                          confidence: float) -> o3d.pipelines.registration.RegistrationResult:
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
      source_down, target_down, source_fpfh, target_fpfh, True,
      distance_threshold,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
      3, [
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
              0.9),
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
              distance_threshold)
      ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, confidence))
    
    return result
  
  def icp_registration(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, trans_init: np.ndarray,
                       voxel_size: float, method: ICPMethod=ICPMethod.POINT_TO_POINT, epsilon: float=1e-4,
                       max_iteration: int=30) -> o3d.pipelines.registration.RegistrationResult:
    distance_threshold = voxel_size * 0.4
    target.estimate_normals()
    convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
      relative_fitness=1e-6,
      relative_rmse=1e-6,
      max_iteration=max_iteration)

    if method == ICPMethod.GENERALIZED:
      estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(epsilon)
      result = o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        distance_threshold,
        trans_init,
        estimation_method,
        convergence_criteria)
      return result

    if method == ICPMethod.POINT_TO_POINT:
      estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif method == ICPMethod.POINT_TO_PLANE:
      estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    
    result = o3d.pipelines.registration.registration_icp(
      source, target, distance_threshold, trans_init,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
      convergence_criteria
    )
    
    return result
  
  def align_by_center_of_mass(self, load: o3d.geometry.PointCloud, bucket: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    try:
        bucket_center = bucket.get_center()
        load_center = load.get_center()

        offset_x = bucket_center[0] - load_center[0]
        offset_y = bucket_center[1] - load_center[1]
        offset_z = bucket_center[2] - load_center[2]
        
        print(f"[DEBUG CENTROIDE] Transladando carga: X = {offset_x:.2f} | Y = {offset_y:.2f} | Z = {offset_z:.2f}")

        transformation = np.eye(4)
        transformation[0, 3] = offset_x
        transformation[1, 3] = offset_y 
        transformation[2, 3] = offset_z

        aligned = copy.deepcopy(load)
        aligned.transform(transformation)

        return aligned
    except Exception as e:
        print(f'Error aligning bucket and load point clouds by center of mass: {e}')
        return load
  
  def align_truck_bucket_and_load(self, load: o3d.geometry.PointCloud, bucket: o3d.geometry.PointCloud, voxel_size: float,
                                max_iteration_ransac: int, confidence: float, max_nn_normals: int, max_nn_fpfh: int,
                                epsilon: float, max_iteration_icp: int, ransac_loop_size: int = 5, method: str = "OTHER") -> o3d.geometry.PointCloud:
    try:    
        result_ransac = None
        if method == "MASS":
            # 1. PRÉ-ALINHAMENTO POR CENTRO DE MASSA
            aligned = self.align_by_center_of_mass(load, bucket)

            # Otimização: Processa o target apenas uma vez fora do loop
            target_down, target_fpfh = self.preprocess_point_cloud(bucket, voxel_size, max_nn_normals, max_nn_fpfh)
            
            for _ in range(ransac_loop_size):
                source_down, source_fpfh = self.preprocess_point_cloud(aligned, voxel_size, max_nn_normals, max_nn_fpfh)
                  
                result = self.ransac_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, max_iteration_ransac, confidence)
                if not result_ransac or result.fitness > result_ransac.fitness:
                    result_ransac = result

            trans_init = result_ransac.transformation if result_ransac and result_ransac.fitness > 0 else np.eye(4)

            # 3. REFINAMENTO LOCAL (ICP)
            result_icp = self.icp_registration(aligned, bucket, trans_init, voxel_size, 'generalized', epsilon, max_iteration_icp)
            icp_t = np.array(result_icp.transformation)

            # 4. APLICAÇÃO DA MATRIZ DE TRAVA RÍGIDA (YAW PUROR)
            final_transformation = np.eye(4)
            
            # Preserva translações horizontais (X) e longitudinais (Z), zera a flutuação vertical/profundidade (Y)
            final_transformation[0][3] = icp_t[0][3]
            final_transformation[1][3] = 0.0  
            final_transformation[2][3] = icp_t[2][3]
            
            R = icp_t[:3, :3]
            yaw = np.arctan2(R[2, 0], R[0, 0])
            
            final_transformation[0, 0] = np.cos(yaw)
            final_transformation[0, 2] = np.sin(yaw)
            final_transformation[2, 0] = -np.sin(yaw)
            final_transformation[2, 2] = np.cos(yaw)
            
            # 5. TRANSFORMAÇÃO FINAL
            aligned.transform(final_transformation)
            return aligned

        else:
            load_roi, bucket_roi = load, bucket
            result_ransac = None
            target_down, target_fpfh = self.preprocess_point_cloud(bucket_roi, voxel_size, max_nn_normals, max_nn_fpfh)

            for _ in range(ransac_loop_size):
                source_down, source_fpfh = self.preprocess_point_cloud(load_roi, voxel_size, max_nn_normals, max_nn_fpfh)

                result = self.ransac_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, max_iteration_ransac, confidence)
                if not result_ransac or result.fitness > result_ransac.fitness:
                    result_ransac = result

            result_icp = self.icp_registration(load, bucket, result_ransac.transformation, voxel_size, 'generalized', epsilon, max_iteration_icp)
            
            transformation = np.array(result_icp.transformation)
            transformation[1][0] = 0
            transformation[2][0] = 0
            transformation[2][1] = 0
            transformation[1][2] = 0
            transformation[1][1] = 1
            transformation[2][2] = 1

            transformation[1][3] = 0
              
            aligned = copy.deepcopy(load)
            aligned.transform(transformation)
            return aligned

    except Exception as e:
        print(f'Error aligning bucket and load point clouds: {e}')
        return load
     
    
