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

    # Única passagem de outlier removal (evita erosão por múltiplos passes)
    filtered_pc = self.remove_outliers(removed_points, nb_neighbors, std_ratio, nb_points, radius)
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

    load_pts = np.asarray(load.points)
    bucket_pts = np.asarray(bucket.points)

    # Abordagem: mapa de altura 2D (height map).
    # Para cada célula (x, y), armazena a z máxima da carga.
    # Um ponto da caçamba é selecionado (base sob a carga) quando:
    #   - existe carga na sua posição (x,y): z_carga > THRESHOLD
    #   - o ponto da caçamba está abaixo da carga: z_bucket < z_carga
    #
    # Isso substitui a abordagem de raycasting contra malha Poisson depth=2,
    # que criava uma malha fechada com base sintética abaixo de z=0 e selecionava
    # ~80% das paredes da caçamba erroneamente.
    FLOOR_THRESHOLD = 20.0  # mm — ignora ruído no piso (igual a THRESHOLD_DISTANCE)
    cell_size = 30.0  # mm — resolução do mapa (balanceia precisão vs velocidade)

    x_min = min(load_pts[:,0].min(), bucket_pts[:,0].min())
    y_min = min(load_pts[:,1].min(), bucket_pts[:,1].min())
    x_max = max(load_pts[:,0].max(), bucket_pts[:,0].max())
    y_max = max(load_pts[:,1].max(), bucket_pts[:,1].max())
    nx = int((x_max - x_min) / cell_size) + 2
    ny = int((y_max - y_min) / cell_size) + 2

    height_map = np.full((nx, ny), -np.inf)
    lxi = ((load_pts[:,0] - x_min) / cell_size).astype(int).clip(0, nx-1)
    lyi = ((load_pts[:,1] - y_min) / cell_size).astype(int).clip(0, ny-1)
    np.maximum.at(height_map, (lxi, lyi), load_pts[:,2])
    valid_cells = np.sum(height_map > FLOOR_THRESHOLD)
    print(f"[MERGE DEBUG] Height map {nx}x{ny} criado, {valid_cells} células com carga (z>{FLOOR_THRESHOLD}mm)")

    bxi = ((bucket_pts[:,0] - x_min) / cell_size).astype(int).clip(0, nx-1)
    byi = ((bucket_pts[:,1] - y_min) / cell_size).astype(int).clip(0, ny-1)
    load_z_at_bucket = height_map[bxi, byi]

    # Seleciona pontos da caçamba que estão abaixo da superfície da carga
    points_to_select = (load_z_at_bucket > FLOOR_THRESHOLD) & (bucket_pts[:,2] < load_z_at_bucket)
    selected_bucket_pts = bucket_pts[points_to_select]
    print(f"[MERGE DEBUG] Pontos selecionados da base: {len(selected_bucket_pts)}")

    points_combined = np.vstack([load_pts, selected_bucket_pts])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_combined)
    print(f"[MERGE DEBUG] PCD combinado: {len(pcd.points)} pontos")

    print(f"[MERGE DEBUG] Removendo outliers estatísticos...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[MERGE DEBUG] Após remoção de outliers: {len(pcd.points)} pontos")

    print(f"[MERGE DEBUG] Merge finalizado com sucesso!")
    return pcd

  def reconstruct_load_mesh_legacy(self, load: o3d.geometry.PointCloud, alpha: float,
                            n_filter_iterations: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(load, alpha)
    bbox = load.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Refine the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=n_filter_iterations)
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_triangle_normals()

    return mesh

    
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
    
    # Downsample para Poisson: Alpha/BPA exigem <5k pts, Poisson aguenta mais mas
    # 291k pontos ainda é lento. voxel=15mm mantém boa resolução (>densidade 8mm)
    # e reduz superfície de ~290k para ~40-60k pontos, tornando tudo O(10x) mais rápido.
    POISSON_VOXEL = 15.0  # mm — ajusta se quiser mais/menos detalhe
    load_ds = load.voxel_down_sample(voxel_size=POISSON_VOXEL)
    print(f"[POISSON DEBUG] Após voxel_down_sample({POISSON_VOXEL}mm): {len(load_ds.points)} pontos (de {len(load.points)})")

    # Calcular densidade média dos pontos no conjunto reduzido
    print(f"[POISSON DEBUG] Calculando distâncias...")
    distances = load_ds.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    
    print(f"[Poisson] Densidade de pontos: média={avg_distance:.2f}mm, mediana={median_distance:.2f}mm")
    
    normal_radius = avg_distance * 3.5  # 3.5x para captar contexto local
    
    # Estimar normais com raio adaptativo
    print(f"[POISSON DEBUG] Estimando normais (radius={normal_radius:.1f}mm)...")
    load_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=60)
    )
    
    print(f"[POISSON DEBUG] Orientando normais (método camera)...")
    bbox = load_ds.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    camera_location = center + np.array([0, 0, 1000])  # 1m acima
    load_ds.orient_normals_towards_camera_location(camera_location)

    # Pontos do piso da caçamba (z ≈ 0) têm normais apontando para cima após orient_towards_camera,
    # mas para Poisson correto precisam apontar para BAIXO (para fora do volume fechado).
    # Inverte normais cujo z está no nível do piso (z < 5mm acima do z_min da nuvem).
    pts_arr = np.asarray(load_ds.points)
    nrm_arr = np.asarray(load_ds.normals)
    z_floor_max = pts_arr[:,2].min() + 5.0  # 5mm acima do piso
    floor_mask = pts_arr[:,2] <= z_floor_max
    nrm_arr[floor_mask] *= -1.0  # flip: agora apontam para baixo (outward)
    load_ds.normals = o3d.utility.Vector3dVector(nrm_arr)
    print(f"[POISSON DEBUG] Normais de {floor_mask.sum()} pontos de piso invertidas (outward downward)")
    
    print(f"[Poisson] {len(load_ds.points)} pontos, {len(load_ds.normals)} normais (radius={normal_radius:.1f}mm)")
    
    # Get bounding box ANTES da reconstrução  
    bbox = load_ds.get_axis_aligned_bounding_box()
    
    # Poisson surface reconstruction
    # depth=8: resolução ~bbox/256 ≈ 12mm para caixa 3000mm — adequado para 8mm density
    print(f"[POISSON DEBUG] Iniciando create_from_point_cloud_poisson depth={depth}...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        load_ds, depth=depth, width=0, scale=1.1, linear_fit=False
    )
    print(f"[POISSON DEBUG] Poisson completado!")
    
    print(f"[Poisson RAW] Watertight: {mesh.is_watertight()}, Vértices: {len(mesh.vertices)}, Triângulos: {len(mesh.triangles)}")
    
    # --- Filtro 1: Remover vértices de baixa densidade ---
    # O Poisson retorna uma densidade por vértice que mede o suporte local dos pontos de entrada.
    # Vértices com densidade baixa correspondem a regiões sem dados (buracos, bordas escassas)
    # onde o Poisson extrapola → gera "balões". Removê-los elimina esses artefatos.
    if density_quantile > 0.0:
        densities_np = np.asarray(densities)
        threshold = np.quantile(densities_np, density_quantile)
        low_density_mask = densities_np < threshold
        mesh.remove_vertices_by_mask(low_density_mask)
        print(f"[Poisson] Density filter q={density_quantile}: removidos {low_density_mask.sum()} vértices "
              f"(threshold={threshold:.4f}), restam {len(mesh.vertices)}")
    
    # --- Filtro 2: Cortar ao bounding box dos pontos de entrada ---
    # scale=1.1 expande o octree do Poisson para além dos dados; o crop elimina essa extrapolação.
    mesh = mesh.crop(bbox)
    print(f"[Poisson] Após crop ao bbox: {len(mesh.vertices)} vértices, {len(mesh.triangles)} triângulos")
    
    # Limpar degenerações introduzidas pelo crop/remoção de vértices
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    print(f"[Poisson APÓS LIMPEZA] Watertight: {mesh.is_watertight()}, Vértices: {len(mesh.vertices)}")
    
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_triangle_normals()
    
    print(f"[Poisson FINAL] Watertight: {mesh.is_watertight()}")
    
    return mesh
