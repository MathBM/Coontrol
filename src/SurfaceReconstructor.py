import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

from src.Parameters import Parameters

class SurfaceReconstructor():
  # Outline remover methods
  def remove_outliers(self, point_cloud, nb_neighbors, std_ratio, nb_points, radius):
    # Remoção de outliers estatísticos
    filtered_stat, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Remoção de outliers por raio
    filtered_radius, _ = filtered_stat.remove_radius_outlier(nb_points=nb_points, radius=radius)

    return filtered_radius

  def dbscan_clustering(self, point_cloud, eps, min_samples):
      points = np.asarray(point_cloud.points)
      clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
      labels = clustering.labels_

      # Identificar o maior cluster (ignorar outliers com label -1)
      unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
      if len(unique_labels) == 0:
          return point_cloud
      largest_cluster_label = unique_labels[np.argmax(counts)]

      # Filtrar pontos do maior cluster
      largest_cluster_points = points[labels == largest_cluster_label]
      filtered_pc = o3d.geometry.PointCloud()
      filtered_pc.points = o3d.utility.Vector3dVector(largest_cluster_points)

      return filtered_pc
  
  def isolate_load_points(self, bucket: o3d.geometry.PointCloud, load: o3d.geometry.PointCloud, nb_neighbors: int,
                          std_ratio: float, nb_points: int, radius: float, threshold_distance: float, eps: float, min_samples: int
                          ) -> o3d.geometry.PointCloud:
    print(f"\n[DEBUG isolate_load_points]")
    print(f"  bucket: {len(bucket.points)} pontos")
    bucket_pts = np.asarray(bucket.points)
    print(f"    Z: {bucket_pts[:, 2].min():.1f} a {bucket_pts[:, 2].max():.1f} mm")
    print(f"  load: {len(load.points)} pontos")
    load_pts = np.asarray(load.points)
    print(f"    Z: {load_pts[:, 2].min():.1f} a {load_pts[:, 2].max():.1f} mm")
    print(f"  threshold_distance: {threshold_distance} mm")
    
    kd_tree = o3d.geometry.KDTreeFlann(bucket)
    inner_load_points = []

    for point in load.points:
        [_, idx, _] = kd_tree.search_knn_vector_3d(point, 1)
        closest_point = bucket.points[idx[0]]
        if np.linalg.norm(np.array(point) - np.array(closest_point)) > threshold_distance:
            inner_load_points.append(point)

    print(f"  Pontos após filtro de distância: {len(inner_load_points)}")
    if len(inner_load_points) > 0:
        inner_pts = np.array(inner_load_points)
        print(f"    Z: {inner_pts[:, 2].min():.1f} a {inner_pts[:, 2].max():.1f} mm")

    removed_points = o3d.geometry.PointCloud()
    removed_points.points = o3d.utility.Vector3dVector(inner_load_points)

    inner_load, _ = removed_points.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                              std_ratio=std_ratio)
    print(f"  Pontos após statistical_outlier: {len(inner_load.points)}")
                                                            
    inner_load, _ = inner_load.remove_radius_outlier(nb_points=nb_points, radius=radius)
    print(f"  Pontos após radius_outlier: {len(inner_load.points)}")

    # Remover outliers
    filtered_pc = self.remove_outliers(inner_load, nb_neighbors, std_ratio, nb_points, radius)
    print(f"  Pontos após remove_outliers: {len(filtered_pc.points)}")

    # Aplicar DBSCAN para remover outliers adicionais
    clustered_pc = self.dbscan_clustering(filtered_pc, eps, min_samples)
    print(f"  Pontos após DBSCAN: {len(clustered_pc.points)}")

    return clustered_pc
  
  # Base reconstruction methods
  def point_to_line_distance(self, points, origin, direction):
    point_vecs = points - origin
    cross_prods = np.cross(direction, point_vecs)
    distances = np.linalg.norm(cross_prods, axis=1) / np.linalg.norm(direction)
    return distances
  
  def find_points_near_ray(self, load: o3d.geometry.PointCloud, bucket: o3d.geometry.PointCloud, ray_origin: list,
                           ray_direction: np.ndarray, detection_threshold: float):
      near_points = []
      for point in bucket.points:
          distance = self.point_to_ray_distance(point, ray_origin, ray_direction)
          if distance <= detection_threshold:
              for load_point in load.points:
                  distance = self.point_to_ray_distance(load_point, ray_origin, ray_direction)
                  if distance <= detection_threshold:
                      near_points.append(point)
                      break
      return near_points

  def generate_rays_with_slope(self, angular_step: float, slope: float, radius: float) -> list:
      rays = []
      angles = np.arange(0, 360, angular_step)
      for angle in angles:
          rad = np.deg2rad(angle)
          x = np.cos(rad) * radius
          z = np.sin(rad) * radius
          y = -slope
          direction = np.array([x, y, z])
          rays.append(direction)
      return rays

  def get_max_coordinate_in_plane(self, points, section, plane='xy', max_axis='y', tolerance=10):
    fixed_axis_idx = 2 if plane == 'xy' else 1 if plane == 'xz' else 0
    max_axis_idx = 0 if max_axis == 'x' else 1 if max_axis == 'y' else 2
    plane_coords = [p[max_axis_idx] for p in points if (section - tolerance) < p[fixed_axis_idx] < (section + tolerance)]
    
    return max(plane_coords)

  def get_min_coordinates(self, points):
    return min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1], min(points, key=lambda p: p[2])[2]

  def get_max_coordinates(self, points):
    return max(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[1])[1], max(points, key=lambda p: p[2])[2]

  def merge_load_and_bucket_points_legacy(self, bucket: o3d.geometry.PointCloud, load: o3d.geometry.PointCloud,
                                   detection_threshold: float, distance_threshold: float, angular_step: float,
                                   slope:float, nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:
    # Define the ray origin and direction
    min_x, _, min_z = self.get_min_coordinates(inner_load_points)
    max_x, _, max_z = self.get_max_coordinates(inner_load_points)
    center_x = (min_x + max_x) / 2

    delta_z = max_z - min_z
    lower_z = min_z + delta_z*0.15
    center_z = (min_z + max_z) / 2
    upper_z = max_z - delta_z*0.15

    lower_y = self.get_max_coordinate_in_plane(inner_load_points, lower_z, 'xy', 'y', 10)*1.2
    center_y = self.get_max_coordinate_in_plane(inner_load_points, center_z, 'xy', 'y', 10)*1.2
    upper_y = self.get_max_coordinate_in_plane(inner_load_points, upper_z, 'xy', 'y', 10)*1.2

    ray_origins = [
    np.array([center_x, lower_y, lower_z]),
    np.array([center_x, center_y, center_z]),
    np.array([center_x, upper_y, upper_z])]
    rays = []

    rays += self.generate_rays_with_slope(angular_step, slope, radius=50)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=200)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=300)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=500)
    near_points = []

    # Get direction as unit vector of each ray
    directions = [ray / np.linalg.norm(ray) for ray in rays]

    bucket_points = np.asarray(bucket.points)
    inner_load_points = np.asarray(load.points)

    # Iterate over each direction to find the lines that meet the threshold criteria
    valid_lines = []
    for origin in ray_origins:
        for direction in directions:
            bucket_distances = self.point_to_line_distance(bucket_points, origin, direction)
            load_distances = self.point_to_line_distance(inner_load_points, origin, direction)
            
            if np.any(bucket_distances < detection_threshold) and np.any(load_distances < detection_threshold):
                valid_lines.append(direction)
                
    near_points = []
    for origin in ray_origins:
        for direction in valid_lines:
            bucket_distances = self.point_to_line_distance(bucket.points, origin, direction)
            close_points = bucket_points[np.where(bucket_distances < detection_threshold)]
            near_points.extend(close_points)

    near_pcd = o3d.geometry.PointCloud()
    near_pcd.points = o3d.utility.Vector3dVector(near_points)
    kd_tree = o3d.geometry.KDTreeFlann(near_pcd)
    inner_bucket_points = []

    for point in bucket.points:
        [_, idx, _] = kd_tree.search_knn_vector_3d(point, 1)
        closest_point = near_pcd.points[idx[0]]
        if np.linalg.norm(np.array(point) - np.array(closest_point)) < distance_threshold:
            inner_bucket_points.append(point)
            
    points = np.concatentate((np.asarray(inner_bucket_points), np.asarray(load.points)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    return pcd
  
  def merge_load_and_bucket_points(self, bucket: o3d.geometry.PointCloud, load: o3d.geometry.PointCloud,
                                   ray_cast_origin_x: float, ray_cast_origin_y: float, ray_cast_origin_z: float,
                                   simple_mesh_radius: int, simple_mesh_max_nn: int, simple_mesh_k: int, 
                                   nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:  
    bucket_points = np.asarray(bucket.points)

    # Define origin of the ray casting
    origin = np.array([ray_cast_origin_x, ray_cast_origin_y, ray_cast_origin_z])

    # Define direction vector
    direction_vectors = bucket_points - origin
    magnitudes = np.linalg.norm(direction_vectors, axis=1)
    unitary_direction_vectors = direction_vectors / magnitudes[:, np.newaxis]
    rays_unit = np.hstack([[origin]*len(unitary_direction_vectors), unitary_direction_vectors])

    # Generate simple mash to help finding the points above load surface
    radius = simple_mesh_radius
    max_nn = simple_mesh_max_nn
    k = simple_mesh_k

    load.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    load.orient_normals_consistent_tangent_plane(k)

    mesh_real, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(load, depth=2) 
    mesh_real = mesh_real.filter_smooth_simple(number_of_iterations=1)
    mesh_real.paint_uniform_color([0.7, 0.7, 0.7])

    mesh_real_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_real)

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_real_legacy)

    rays = o3d.core.Tensor(rays_unit,
                          dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)

    direction_magnitudes = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in direction_vectors]
    colors = np.array(['green' if hit_distance < direction_magnitudes[i] else 'blue' for i, hit_distance in enumerate(ans['t_hit'].numpy())])

    points_caixa_brita = load.points
    points_to_select = colors == 'green'
    points_empty_base = np.asarray(bucket.points)
    points_empty_base = list(points_empty_base[points_to_select])
    points_caixa_brita.extend(points_empty_base)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_caixa_brita)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return pcd
    
  def reconstruct_load_mesh(self, load: o3d.geometry.PointCloud, alpha: float,
                            n_filter_iterations: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(load, alpha)
    bbox = load.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Refine the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=n_filter_iterations)
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_triangle_normals()

    return mesh
  
  def reconstruct_load_mesh_poisson(self, load: o3d.geometry.PointCloud, depth: int = 10,
                                    n_filter_iterations: int = 5, density_quantile: float = 0.01) -> o3d.geometry.TriangleMesh:
    """
    Reconstrói malha usando Poisson Surface Reconstruction.
    Gera malhas FECHADAS (watertight) automaticamente.
    
    Args:
        load: Nuvem de pontos
        depth: Profundidade da octree (maior = mais detalhes, 8-10 recomendado)
        n_filter_iterations: Iterações de suavização
        density_quantile: Quantil para remover vértices de baixa densidade (0.01-0.1)
    
    Returns:
        Malha triangular fechada
    """
    # Estimar normais (CRÍTICO para Poisson)
    # Raio maior para superfícies mais suaves e consistentes
    load.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=150, max_nn=40)
    )
    
    # Garantir consistência de normais
    load.orient_normals_consistent_tangent_plane(k=40)
    
    print(f"[Poisson] {len(load.points)} pontos, {len(load.normals)} normais")
    
    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        load, depth=depth, width=0, scale=1.0, linear_fit=False
    )
    
    print(f"[Poisson RAW] Watertight: {mesh.is_watertight()}, Vértices: {len(mesh.vertices)}, Triângulos: {len(mesh.triangles)}")
    
    print(f"[Poisson RAW] Watertight: {mesh.is_watertight()}, Vértices: {len(mesh.vertices)}, Triângulos: {len(mesh.triangles)}")
    
    # Se já é watertight, apenas limpar e retornar
    if mesh.is_watertight():
        print("[Poisson] ✓ Malha já é watertight!")
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        mesh.compute_triangle_normals()
        print(f"[Poisson FINAL] Watertight: {mesh.is_watertight()}")
        return mesh
    
    # Se não é watertight, NÃO remover vértices - apenas limpar
    print("[Poisson] ⚠ Malha não é watertight, mantendo malha bruta...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    
    print(f"[Poisson APÓS limpeza] Watertight: {mesh.is_watertight()}, Vértices: {len(mesh.vertices)}")
    
    # NÃO suavizar - pode quebrar watertight
    # if n_filter_iterations > 0:
    #     mesh = mesh.filter_smooth_simple(number_of_iterations=n_filter_iterations)
    
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_triangle_normals()
    
    print(f"[Poisson FINAL] Watertight: {mesh.is_watertight()}")
    
    return mesh
