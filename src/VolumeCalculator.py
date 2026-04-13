import numpy as np
import open3d as o3d
from functools import reduce


class VolumeCalculator():
    def _new_volume_under_triangle(self, triangle):

        p1, p2, p3 = triangle
        x1, z1, y1 = p1
        x2, z2, y2 = p2
        x3, z3, y3 = p3

        return (((-x3*y2*z1) + (x2*y3*z1) + (x3*y1*z2) + (-x1*y3*z2) + (-x2*y1*z3) + (x1*y2*z3))/6)

    def _get_triangles_vertices(self, triangles, vertices):
        triangles_vertices = []

        for triangle in triangles:
            new_triangles_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]
            triangles_vertices.append(new_triangles_vertices)

        return np.array(triangles_vertices)
    
    def volume_calculation(self, load_mesh: o3d.geometry.PointCloud):
        try:
            volume = reduce(lambda a, b:  a + self._new_volume_under_triangle(b),
                            self._get_triangles_vertices(load_mesh.triangles, load_mesh.vertices), 0)
            volume = abs(volume)
        except Exception as e:
            volume = 0
        return volume

    def volume_from_heightmap(self, load: o3d.geometry.PointCloud, cell_size: float = 8.0) -> float:
        """
        Calcula o volume da carga integrando o mapa de alturas 2D.

        Método correto para escaneamento LIDAR de cima: V = ∑ z_max(x,y) × Δx × Δy.
        Não requer malha fechada — buracos e regiões esparsas contribuem z=0 (sem volume).
        Preciso para qualquer formato de carga (rampa, côncavo, convexo, pilha, etc.)

        Args:
            load: nuvem de pontos da carga isolada (superfície superior escaneada)
            cell_size: tamanho da célula do grid em mm (deve igualar a densidade do scan)
        Returns:
            Volume em mm³
        """
        pts = np.asarray(load.points)
        if len(pts) < 4:
            return 0.0

        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        nx = int((pts[:, 0].max() - x_min) / cell_size) + 2
        ny = int((pts[:, 1].max() - y_min) / cell_size) + 2

        height_map = np.zeros((nx, ny))
        xi = ((pts[:, 0] - x_min) / cell_size).astype(int).clip(0, nx - 1)
        yi = ((pts[:, 1] - y_min) / cell_size).astype(int).clip(0, ny - 1)
        np.maximum.at(height_map, (xi, yi), np.maximum(pts[:, 2], 0.0))

        return float(np.sum(height_map) * cell_size ** 2)