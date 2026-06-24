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

    def detect_floor_level(self, pcd: o3d.geometry.PointCloud, height_axis: int = 1, bins: int = 40) -> float:
        """Detecta o nível do piso da caçamba como o pico de densidade ao longo do eixo vertical.

        Em uma caçamba (vazia ou com carga) o piso é a maior superfície plana, gerando
        um pico no histograma do eixo vertical. Usado como referência de altura zero.
        """
        v = np.asarray(pcd.points)[:, height_axis]
        hist, edges = np.histogram(v, bins=bins)
        peak = int(np.argmax(hist))
        return float((edges[peak] + edges[peak + 1]) / 2)

    def volume_from_heightmap(self, load: o3d.geometry.PointCloud, cell_size: float = 8.0,
                              plane_axes: tuple = (0, 1), height_axis: int = 2,
                              floor: float = 0.0, up_sign: float = 1.0) -> float:
        """
        Calcula o volume da carga integrando o mapa de alturas 2D.

        Método correto para escaneamento LIDAR de cima: V = ∑ altura(a,b) × Δa × Δb,
        onde (a, b) é o plano horizontal (footprint) e a altura é medida no eixo vertical
        a partir do piso. Não requer malha fechada — buracos e regiões esparsas contribuem
        altura 0 (sem volume). Preciso para qualquer formato de carga.

        Args:
            load: nuvem de pontos da carga isolada (superfície superior escaneada)
            cell_size: tamanho da célula do grid em mm (~2× a densidade do scan)
            plane_axes: índices dos dois eixos do plano horizontal (footprint).
                        Sintético: (0, 1) = X,Y.  Real: (0, 2) = X,Z.
            height_axis: índice do eixo vertical. Sintético: 2 = Z.  Real: 1 = Y.
            floor: coordenada do piso no eixo vertical (altura zero de referência).
            up_sign: +1 se "para cima" = eixo crescente; -1 se decrescente.
                     Sintético: +1 (Z cresce para cima).  Real: -1 (Y decresce para cima).
        Returns:
            Volume em mm³
        """
        pts = np.asarray(load.points)
        if len(pts) < 4:
            return 0.0

        a = pts[:, plane_axes[0]]
        b = pts[:, plane_axes[1]]
        height = up_sign * (pts[:, height_axis] - floor)  # altura acima do piso
        height = np.maximum(height, 0.0)

        a_min, b_min = a.min(), b.min()
        na = int((a.max() - a_min) / cell_size) + 2
        nb = int((b.max() - b_min) / cell_size) + 2

        height_map = np.zeros((na, nb))
        ai = ((a - a_min) / cell_size).astype(int).clip(0, na - 1)
        bi = ((b - b_min) / cell_size).astype(int).clip(0, nb - 1)
        np.maximum.at(height_map, (ai, bi), height)

        return float(np.sum(height_map) * cell_size ** 2)

    def volume_swept_sections(self, load: o3d.geometry.PointCloud, x_cell: float = 20.0,
                              floor: float = 0.0, lateral_axis: int = 0, vertical_axis: int = 1,
                              sweep_axis: int = 2, up_sign: float = -1.0) -> float:
        """
        Volume por integração de seções transversais (swept volume).

        Para o escaneamento LIDAR real, o caminhão se move ao longo de `sweep_axis` (Z) e
        cada linha de scan é uma SEÇÃO TRANSVERSAL medida densamente em `lateral_axis` (X).
        Entre linhas há esparsidade/buracos em Z — o método de mapa de alturas 2D trataria
        esses gaps como altura 0 (subestima). Aqui, ao contrário:

            V = Σ_seções  A(z) × dz

        onde A(z) é a área da seção (∫ altura dx no perfil) e dz é o espaçamento REAL até
        as linhas vizinhas. Um gap em Z não é volume zero: a seção representa todo o trecho
        dz. Resultado estável e robusto à esparsidade no eixo de varredura.

        Args:
            load: nuvem de pontos da carga isolada
            x_cell: tamanho da célula no eixo lateral em mm (~2× densidade do perfil)
            floor: coordenada do piso no eixo vertical (altura zero)
            lateral_axis/vertical_axis/sweep_axis: índices dos eixos (real: X=0, Y=1, Z=2)
            up_sign: +1 se "para cima" = eixo vertical crescente; -1 se decrescente
                     (real: -1, pois a carga sobe em direção a Y menor)
        Returns:
            Volume em mm³
        """
        pts = np.asarray(load.points)
        if len(pts) < 4:
            return 0.0

        sweep = pts[:, sweep_axis]
        lateral = pts[:, lateral_axis]
        height = np.maximum(up_sign * (pts[:, vertical_axis] - floor), 0.0)

        # Agrupa pontos por linha de scan (valor discreto do eixo de varredura)
        keys = np.round(sweep).astype(np.int64)
        z_vals = np.unique(keys)
        if len(z_vals) < 2:
            return 0.0

        # Área de cada seção transversal: ∫ altura dx (perfil binado em x_cell, max por célula)
        areas = np.zeros(len(z_vals))
        for i, z in enumerate(z_vals):
            sel = keys == z
            lat = lateral[sel]
            h = height[sel]
            if len(lat) < 2:
                continue
            lat_min = lat.min()
            nx = int((lat.max() - lat_min) / x_cell) + 2
            profile = np.zeros(nx)
            xi = ((lat - lat_min) / x_cell).astype(int).clip(0, nx - 1)
            np.maximum.at(profile, xi, h)
            areas[i] = profile.sum() * x_cell

        # dz = espaçamento real (meia distância aos vizinhos); extremos one-sided
        dz = np.empty(len(z_vals))
        dz[1:-1] = (z_vals[2:] - z_vals[:-2]) / 2.0
        dz[0] = z_vals[1] - z_vals[0]
        dz[-1] = z_vals[-1] - z_vals[-2]

        return float(np.sum(areas * dz))