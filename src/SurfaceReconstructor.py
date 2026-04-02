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
    print(f"[MERGE DEBUG] Iniciando com {len(load.points)} pontos de carga e {len(bucket.points)} pontos de caçamba")
    
    bucket_points = np.asarray(bucket.points)

    # Define origin of the ray casting
    origin = np.array([ray_cast_origin_x, ray_cast_origin_y, ray_cast_origin_z])
    print(f"[MERGE DEBUG] Ray casting origin: {origin}")

    # Define direction vector
    direction_vectors = bucket_points - origin
    magnitudes = np.linalg.norm(direction_vectors, axis=1)
    unitary_direction_vectors = direction_vectors / magnitudes[:, np.newaxis]
    rays_unit = np.hstack([[origin]*len(unitary_direction_vectors), unitary_direction_vectors])
    print(f"[MERGE DEBUG] Rays criados: {len(rays_unit)}")

    # Generate simple mash to help finding the points above load surface
    radius = simple_mesh_radius
    max_nn = simple_mesh_max_nn
    k = simple_mesh_k
    print(f"[MERGE DEBUG] Parâmetros: radius={radius}, max_nn={max_nn}, k={k}")

    # Remove pontos duplicados para evitar erro Qhull
    print(f"[MERGE DEBUG] Aplicando voxel_down_sample...")
    load_clean = load.voxel_down_sample(voxel_size=0.5)
    print(f"[MERGE DEBUG] Após voxel: {len(load_clean.points)} pontos")
    
    print(f"[MERGE DEBUG] Estimando normais...")
    load_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    
    print(f"[MERGE DEBUG] Orientando normais consistentes...")
    # Tratamento robusto para erro Qhull (pontos cocirculares/cospherical)
    try:
        load_clean.orient_normals_consistent_tangent_plane(k)
        print(f"[MERGE DEBUG] Normais orientadas com sucesso")
    except RuntimeError as e:
        if "QH6239" in str(e) or "Qhull" in str(e):
            print("⚠️  Aviso: Erro Qhull no merge. Usando método alternativo.")
            # Adiciona jitter mínimo
            points_array = np.asarray(load_clean.points)
            jitter = np.random.normal(0, 0.01, points_array.shape)
            load_clean.points = o3d.utility.Vector3dVector(points_array + jitter)
            load_clean.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
            try:
                load_clean.orient_normals_consistent_tangent_plane(k)
            except RuntimeError:
                load_clean.orient_normals_towards_camera_location(camera_location=origin)
        else:
            raise

    print(f"[MERGE DEBUG] Criando malha Poisson depth=2...")
    mesh_real, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(load_clean, depth=2) 
    print(f"[MERGE DEBUG] Criando malha Poisson depth=2...")
    mesh_real, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(load_clean, depth=2) 
    print(f"[MERGE DEBUG] Malha criada: {len(mesh_real.vertices)} vértices, {len(mesh_real.triangles)} triângulos")
    
    print(f"[MERGE DEBUG] Suavizando malha...")
    mesh_real = mesh_real.filter_smooth_simple(number_of_iterations=1)
    mesh_real.paint_uniform_color([0.7, 0.7, 0.7])
    print(f"[MERGE DEBUG] Malha suavizada")

    print(f"[MERGE DEBUG] Convertendo para tensor mesh...")
    mesh_real_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_real)

    print(f"[MERGE DEBUG] Criando cena de raycasting...")
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_real_legacy)

    print(f"[MERGE DEBUG] Preparando rays tensor...")
    rays = o3d.core.Tensor(rays_unit,
                          dtype=o3d.core.Dtype.Float32)

    print(f"[MERGE DEBUG] Lançando {len(rays)} rays...")
    ans = scene.cast_rays(rays)
    print(f"[MERGE DEBUG] Raycasting completo")

    print(f"[MERGE DEBUG] Processando resultados...")
    direction_magnitudes = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in direction_vectors]
    colors = np.array(['green' if hit_distance < direction_magnitudes[i] else 'blue' for i, hit_distance in enumerate(ans['t_hit'].numpy())])

    print(f"[MERGE DEBUG] Selecionando pontos com cores...")
    points_caixa_brita = load.points
    points_to_select = colors == 'green'
    points_empty_base = np.asarray(bucket.points)
    points_empty_base = list(points_empty_base[points_to_select])
    print(f"[MERGE DEBUG] Pontos selecionados da base: {len(points_empty_base)}")
    
    points_caixa_brita.extend(points_empty_base)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_caixa_brita)
    print(f"[MERGE DEBUG] PCD combinado: {len(pcd.points)} pontos")

    print(f"[MERGE DEBUG] Removendo outliers estatísticos...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[MERGE DEBUG] Após remoção de outliers: {len(pcd.points)} pontos")

    print(f"[MERGE DEBUG] Merge finalizado com sucesso!")
    return pcd
    
  def reconstruct_load_mesh(self, load: o3d.geometry.PointCloud, alpha: float,
                            n_filter_iterations: int) -> o3d.geometry.TriangleMesh:
    print(f"[Alpha Shapes] Criando malha com alpha={alpha}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(load, alpha)
    bbox = load.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    
    print(f"[Alpha Shapes RAW] Vértices: {len(mesh.vertices)}, Triângulos: {len(mesh.triangles)}, Watertight: {mesh.is_watertight()}")

    # Refine the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=n_filter_iterations)
    
    # Limpar degenerações
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    print(f"[Alpha Shapes APÓS LIMPEZA] Vértices: {len(mesh.vertices)}, Watertight: {mesh.is_watertight()}")
    
    # Calcular densidade para fechamento
    distances = load.compute_nearest_neighbor_distance()
    avg_density = np.mean(distances)
    
    # Tentar fechar se não for watertight
    if not mesh.is_watertight():
        print(f"[Alpha Shapes] Tentando fechar malha (densidade={avg_density:.1f}mm)...")
        mesh = self.close_mesh_holes(mesh, avg_density=avg_density)
    
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_triangle_normals()
    
    print(f"[Alpha Shapes FINAL] Watertight: {mesh.is_watertight()}")

    return mesh
  
  def close_mesh_holes(self, mesh: o3d.geometry.TriangleMesh, avg_density: float = 8.0) -> o3d.geometry.TriangleMesh:
    """
    Tenta fechar buracos em malhas não-watertight considerando a densidade dos pontos.
    Otimizado para densidade uniforme de 8mm (rampa + caçamba).
    
    Args:
        mesh: Malha a ser fechada
        avg_density: Densidade média dos pontos em mm (ex: 8mm para ambos rampa e caçamba)
    """
    if mesh.is_watertight():
        return mesh
    
    print(f"[FECHAR MALHA] Tentando fechar buracos (densidade={avg_density:.1f}mm)...")
    print(f"[FECHAR MALHA] Malha inicial: {len(mesh.triangles)} triângulos")
    
    initial_triangles = len(mesh.triangles)
    
    # ESTRATÉGIA 1: Simplificação moderada ajustada para densidade 8mm
    # Para 180k pontos com densidade 8mm: ~400k triângulos típico
    # Reduzir para 10k-20k mantém estrutura com margem para subdivisão
    target_triangles = max(10000, min(20000, initial_triangles // 10))
    print(f"[FECHAR MALHA] Target simplificação: {target_triangles} triângulos")
    
    mesh_simple = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )
    
    # Limpar agressivamente
    mesh_simple.remove_degenerate_triangles()
    mesh_simple.remove_duplicated_triangles()
    mesh_simple.remove_duplicated_vertices()
    mesh_simple.remove_non_manifold_edges()
    
    print(f"[FECHAR MALHA] Simplificado: {len(mesh_simple.triangles)} triângulos, Watertight={mesh_simple.is_watertight()}")
    
    # Se simplificação fechou, subdividir 1x apenas (mais conservador)
    if mesh_simple.is_watertight():
        print(f"[FECHAR MALHA] ✓ Estratégia 1 funcionou! Subdividindo 1x...")
        mesh_result = mesh_simple.subdivide_midpoint(number_of_iterations=1)
        mesh_result.remove_degenerate_triangles()
        mesh_result.remove_duplicated_triangles()
        mesh_result.remove_non_manifold_edges()
        
        print(f"[FECHAR MALHA] ✓ FECHADA! Watertight={mesh_result.is_watertight()}, Triângulos={len(mesh_result.triangles)}")
        return mesh_result
    
    # Estratégia 2: Mais agressiva
    target_triangles_ultra = max(5000, initial_triangles // 20)
    print(f"[FECHAR MALHA] Tentando estratégia 2: {target_triangles_ultra} triângulos")
    mesh_ultra = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles_ultra
    )
    
    mesh_ultra.remove_degenerate_triangles()
    mesh_ultra.remove_duplicated_triangles()
    mesh_ultra.remove_duplicated_vertices()
    mesh_ultra.remove_non_manifold_edges()
    
    print(f"[FECHAR MALHA] Simplificação 2: {len(mesh_ultra.triangles)} triângulos, Watertight={mesh_ultra.is_watertight()}")
    
    if mesh_ultra.is_watertight():
        print(f"[FECHAR MALHA] ✓ Estratégia 2 funcionou! Subdividindo 2x...")
        # Subdividir 2x para densidade 8mm
        mesh_result = mesh_ultra.subdivide_midpoint(number_of_iterations=2)
        mesh_result.remove_degenerate_triangles()
        mesh_result.remove_duplicated_triangles()
        mesh_result.remove_non_manifold_edges()
        print(f"[FECHAR MALHA] ✓ FECHADA! Watertight={mesh_result.is_watertight()}, Triângulos={len(mesh_result.triangles)}")
        return mesh_result
    
    # Estratégia 3: MUITO agressiva (último recurso)
    target_triangles_extreme = max(1500, min(3000, initial_triangles // 80))
    print(f"[FECHAR MALHA] Tentando estratégia 3 (extrema): {target_triangles_extreme} triângulos")
    mesh_extreme = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles_extreme
    )
    
    mesh_extreme.remove_degenerate_triangles()
    mesh_extreme.remove_duplicated_triangles()
    mesh_extreme.remove_duplicated_vertices()
    mesh_extreme.remove_non_manifold_edges()
    
    print(f"[FECHAR MALHA] Simplificação 3: {len(mesh_extreme.triangles)} triângulos, Watertight={mesh_extreme.is_watertight()}")
    
    if mesh_extreme.is_watertight():
        print(f"[FECHAR MALHA] ✓ Estratégia 3 funcionou! Subdividindo 3x...")
        mesh_result = mesh_extreme.subdivide_midpoint(number_of_iterations=3)
        mesh_result.remove_degenerate_triangles()
        mesh_result.remove_duplicated_triangles()
        mesh_result.remove_non_manifold_edges()
        print(f"[FECHAR MALHA] ✓ FECHADA! Watertight={mesh_result.is_watertight()}, Triângulos={len(mesh_result.triangles)}")
        return mesh_result
    
    print(f"[FECHAR MALHA] ⚠ Ainda aberta após todas estratégias. Retornando malha original.")
    return mesh
  
  def reconstruct_load_mesh_poisson(self, load: o3d.geometry.PointCloud, depth: int = 10,
                                    n_filter_iterations: int = 5, density_quantile: float = 0.01) -> o3d.geometry.TriangleMesh:
    """
    Reconstrói malha usando Poisson Surface Reconstruction.
    Gera malhas FECHADAS (watertight) automaticamente.
    
    Args:
        load: Nuvem de pontos
        depth: Profundidade da octree (maior = mais detalhes, 8-12 recomendado)
        n_filter_iterations: Iterações de suavização
        density_quantile: Quantil para remover vértices de baixa densidade (0.0-0.1)
    
    Returns:
        Malha triangular fechada
    """
    print(f"[POISSON DEBUG] Iniciando com {len(load.points)} pontos, depth={depth}")
    
    # Calcular densidade média dos pontos (deve ser ~8mm para dados sintéticos)
    print(f"[POISSON DEBUG] Calculando distâncias...")
    distances = load.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    
    print(f"[Poisson] Densidade de pontos: média={avg_distance:.2f}mm, mediana={median_distance:.2f}mm")
    
    # Para densidade de 8mm:
    # - Raio de normais: 24-32mm (3-4x a densidade)
    # - k neighbors: ~30-40 (cobre área de ~24mm²)
    normal_radius = avg_distance * 3.5  # 3.5x para captar contexto local
    
    # Estimar normais com raio adaptativo
    print(f"[POISSON DEBUG] Estimando normais (radius={normal_radius:.1f}mm)...")
    load.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=60)
    )
    
    # Orientar normais: para muitos pontos (>100k), orient_normals_consistent_tangent_plane é MUITO lento
    # Usar método baseado em câmera que é O(n) em vez de O(n·k)
    print(f"[POISSON DEBUG] Orientando normais (método camera)...")
    bbox = load.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    # Câmera acima do centro (Z positivo) - normais apontam para cima
    camera_location = center + np.array([0, 0, 1000])  # 1m acima
    load.orient_normals_towards_camera_location(camera_location)
    
    print(f"[Poisson] {len(load.points)} pontos, {len(load.normals)} normais (radius={normal_radius:.1f}mm)")
    
    # Get bounding box ANTES da reconstrução  
    bbox = load.get_axis_aligned_bounding_box()
    
    # Poisson surface reconstruction com scale=1.0 (sem extrapolação)
    print(f"[POISSON DEBUG] Iniciando create_from_point_cloud_poisson depth={depth}...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        load, depth=depth, width=0, scale=1.0, linear_fit=False
    )
    print(f"[POISSON DEBUG] Poisson completado!")
    
    print(f"[Poisson RAW] Watertight: {mesh.is_watertight()}, Vértices: {len(mesh.vertices)}, Triângulos: {len(mesh.triangles)}")
    
    # Limpar degenerações
    print(f"[POISSON DEBUG] Limpando malha...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    print(f"[Poisson APÓS LIMPEZA] Watertight: {mesh.is_watertight()}, Vértices: {len(mesh.vertices)}")
    
    # Se não for watertight, tentar fechar considerando densidade
    if not mesh.is_watertight():
        print(f"[POISSON DEBUG] Chamando close_mesh_holes...")
        mesh = self.close_mesh_holes(mesh, avg_density=avg_distance)
        print(f"[POISSON DEBUG] close_mesh_holes completado!")
    
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_triangle_normals()
    
    print(f"[Poisson FINAL] Watertight: {mesh.is_watertight()}")
    
    return mesh
