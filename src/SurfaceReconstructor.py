import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

class SurfaceReconstructor:
    
    # --- Métodos de Remoção de Outliers ---
    
    def remove_outliers(self, point_cloud, nb_neighbors, std_ratio, nb_points, radius):
        if len(point_cloud.points) == 0:
            return point_cloud
        # Remoção estatística
        filtered_stat, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        # Remoção por raio
        filtered_radius, _ = filtered_stat.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return filtered_radius

    def dbscan_clustering(self, point_cloud, eps, min_samples):
        points = np.asarray(point_cloud.points)
        if len(points) == 0:
            return point_cloud
            
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) == 0:
            return o3d.geometry.PointCloud() # Retorna vazio se só houver ruído
            
        largest_cluster_label = unique_labels[np.argmax(counts)]
        largest_cluster_points = points[labels == largest_cluster_label]
        
        filtered_pc = o3d.geometry.PointCloud()
        filtered_pc.points = o3d.utility.Vector3dVector(largest_cluster_points)
        return filtered_pc
    
    def isolate_load_points(self, bucket: o3d.geometry.PointCloud, load: o3d.geometry.PointCloud, 
                            nb_neighbors: int, std_ratio: float, nb_points: int, radius: float, 
                            threshold_distance: float, eps: float, min_samples: int) -> o3d.geometry.PointCloud:
        
        print(f"\n[DEBUG isolate_load_points]")
        # PERFORMANCE: Substituindo loop KDTree por função nativa do Open3D
        dists = np.asarray(load.compute_point_cloud_distance(bucket))
        inner_load_mask = dists > threshold_distance
        
        load_pts_np = np.asarray(load.points)
        inner_points = load_pts_np[inner_load_mask]
        
        print(f"  Pontos após filtro de distância: {len(inner_points)}")
        if len(inner_points) == 0:
            return o3d.geometry.PointCloud()

        removed_points = o3d.geometry.PointCloud()
        removed_points.points = o3d.utility.Vector3dVector(inner_points)

        # Filtros sequenciais
        filtered_pc = self.remove_outliers(removed_points, nb_neighbors, std_ratio, nb_points, radius)
        print(f"  Pontos após remove_outliers: {len(filtered_pc.points)}")

        # DBSCAN final
        clustered_pc = self.dbscan_clustering(filtered_pc, eps, min_samples)
        print(f"  Pontos após DBSCAN: {len(clustered_pc.points)}")

        return clustered_pc
  
    # --- Métodos de Reconstrução e Auxiliares ---

    def point_to_line_distance(self, points, origin, direction):
        direction = direction / np.linalg.norm(direction)
        point_vecs = points - origin
        cross_prods = np.cross(direction, point_vecs)
        distances = np.linalg.norm(cross_prods, axis=1)
        return distances

    def get_min_coordinates(self, points_np):
        return np.min(points_np, axis=0)

    def get_max_coordinates(self, points_np):
        return np.max(points_np, axis=0)

    def merge_load_and_bucket_points(self, bucket: o3d.geometry.PointCloud, load: o3d.geometry.PointCloud,
                                     ray_cast_origin_x: float, ray_cast_origin_y: float, ray_cast_origin_z: float,
                                     simple_mesh_radius: int, simple_mesh_max_nn: int, simple_mesh_k: int, 
                                     nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:  
        
        bucket_points = np.asarray(bucket.points)
        origin = np.array([ray_cast_origin_x, ray_cast_origin_y, ray_cast_origin_z])

        # Direções dos raios da origem para cada ponto do bucket
        direction_vectors = bucket_points - origin
        magnitudes = np.linalg.norm(direction_vectors, axis=1)
        unitary_direction_vectors = direction_vectors / magnitudes[:, np.newaxis]
        
        # Formato esperado pelo RaycastingScene: [N, 6] (origin_xyz, direction_xyz)
        origins = np.tile(origin, (len(unitary_direction_vectors), 1))
        rays_unit = np.hstack([origins, unitary_direction_vectors])

        # Criar malha temporária da carga para o Raycasting
        load_temp = o3d.geometry.PointCloud(load)
        load_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=simple_mesh_radius, max_nn=simple_mesh_max_nn))
        
        mesh_real, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(load_temp, depth=6) 
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_real)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)

        rays_tensor = o3d.core.Tensor(rays_unit, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_tensor)
        hit_distances = ans['t_hit'].numpy()

        # Lógica: se o raio bateu na malha ANTES de chegar no bucket, o ponto do bucket está "atrás/abaixo" da carga
        points_to_select = hit_distances < magnitudes
        
        # Filtragem e União
        selected_bucket_pts = bucket_points[points_to_select]
        load_pts = np.asarray(load.points)
        
        combined_points = np.concatenate((load_pts, selected_bucket_pts), axis=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        return pcd
    
    def reconstruct_load_mesh_poisson(self, load: o3d.geometry.PointCloud, depth: int = 10,
                                      n_filter_iterations: int = 5) -> o3d.geometry.TriangleMesh:
        if len(load.points) < 10:
            return o3d.geometry.TriangleMesh()

        load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=150, max_nn=40))
        load.orient_normals_consistent_tangent_plane(k=40)
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(load, depth=depth)
        
        # Limpeza de malha
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        if n_filter_iterations > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=n_filter_iterations)
            
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        mesh.compute_triangle_normals()
        
        return mesh