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
              0.1),
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
  
  def align_truck_bucket_and_load(self, load: o3d.geometry.PointCloud, bucket: o3d.geometry.PointCloud, voxel_size: float,
                                  max_iteration_ransac: int, confidence: float, max_nn_normals: int, max_nn_fpfh: int,
                                  epsilon: float, max_iteration_icp: int, ransac_loop_size:int =5) -> o3d.geometry.PointCloud:
    try:     
      result_ransac = None

      for _ in range(ransac_loop_size):
        source_down, source_fpfh = self.preprocess_point_cloud(load, voxel_size, max_nn_normals, max_nn_fpfh)
        target_down, target_fpfh = self.preprocess_point_cloud(bucket, voxel_size, max_nn_normals, max_nn_fpfh)
          
        result = self.ransac_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, max_iteration_ransac, confidence)
        if not result_ransac or result.fitness > result_ransac.fitness:
          result_ransac = result

      result_icp = self.icp_registration(load, bucket, result_ransac.transformation, voxel_size, 'generalized', epsilon, max_iteration_icp)
      
      transformation = np.array(result_icp.transformation)

      # Cancel x rotation
      transformation[1][0] = 0
      transformation[2][0] = 0
      transformation[2][1] = 0
      transformation[1][2] = 0
      transformation[1][1] = 1
      transformation[2][2] = 1

      # Cancel y translation
      transformation[1][3] = 0
      
      aligned = copy.deepcopy(load)
      aligned.transform(transformation)

      return aligned
    except Exception as e:
      print(f'Error aligning bucket and load point clouds: {e}')
