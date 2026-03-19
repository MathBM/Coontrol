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