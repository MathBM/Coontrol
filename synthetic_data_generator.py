import numpy as np
import open3d as o3d

class SyntheticDataGenerator:
    
    def __init__(self):
        pass

    def _add_noise(self, x: float, y: float, z: float, noise_level: float):
        """Aplica ruído gaussiano aos eixos X, Y e Z."""
        if noise_level > 0:
            nz = z + np.random.normal(0, noise_level)
            nx = x + np.random.normal(0, noise_level * 0.5)
            ny = y + np.random.normal(0, noise_level * 0.5)
            return (nx, ny, nz)
        return (x, y, z)

    def generate_ramp(
        self,
        width: float = 2000,
        length: float = 3000,
        height: float = 800,
        point_density: int = 20,
        noise_level: float = 2.0,
        add_ground: bool = True,
        curvature: str = "linear" # "linear", "concave" ou "convex"
    ):
        points = []
        x_values = np.arange(0, length, point_density)
        y_values = np.arange(-width/2, width/2, point_density)
        
        # 1. SUPERFÍCIE DA RAMPA
        for x in x_values:
            for y in y_values:
                # Cálculo da curvatura
                t = x / length
                if curvature == "concave":
                    z_top = (t ** 2) * height
                elif curvature == "convex":
                    z_top = (1 - (1 - t) ** 2) * height
                else: # linear
                    z_top = t * height
                    
                points.append(self._add_noise(x, y, z_top, noise_level))

        # 2. FECHAMENTOS (Laterais e Fundo)
        for x in x_values:
            t = x / length
            z_max = (t**2 * height) if curvature == "concave" else (t * height) # Simplificado
            for z in np.arange(0, z_max, point_density):
                points.append(self._add_noise(x, -width/2, z, noise_level))
                points.append(self._add_noise(x, width/2, z, noise_level))

        if add_ground:
            ground_margin = 500
            x_g = np.arange(-ground_margin, length + ground_margin, point_density * 2)
            y_g = np.arange(-width/2 - ground_margin, width/2 + ground_margin, point_density * 2)
            for x in x_g:
                for y in y_g:
                    z_g = height if x >= length else 0
                    points.append(self._add_noise(x, y, z_g, noise_level))
        
        return points

    def generate_hills(
        self,
        width: float = 4000,
        length: float = 4000,
        max_height: float = 500,
        frequency: float = 0.002, # Controla a quantidade de montes
        point_density: int = 25,
        noise_level: float = 5.0
    ):
        """Gera um terreno com montes (seno/cosseno) para teste de volume."""
        points = []
        x_range = np.arange(0, length, point_density)
        y_range = np.arange(0, width, point_density)
        
        for x in x_range:
            for y in y_range:
                # Combinação de Seno e Cosseno para criar picos e vales
                z = (np.sin(x * frequency) * np.cos(y * frequency)) * max_height
                # Garante que não haja valores negativos (base no chão)
                z = max(0, z) 
                points.append(self._add_noise(x, y, z, noise_level))
        return points

    def generate_stepped_ramp(
        self,
        width: float = 2000,
        num_steps: int = 5,
        step_length: float = 600,
        step_height: float = 150,
        point_density: int = 15,
        noise_level: float = 2.0,
    ):
        points = []
        y_values = np.arange(-width/2, width/2, point_density)
        for step in range(num_steps):
            x_start, x_end = step * step_length, (step + 1) * step_length
            z = step * step_height
            for x in np.arange(x_start, x_end, point_density):
                for y in y_values:
                    points.append(self._add_noise(x, y, z, noise_level))
        return points

    def visualize(self, points, window_name="Visualizador 3D"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        
        # Colorir por altura
        z_vals = np.asarray(pcd.points)[:, 2]
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-7)
        colors = np.zeros((len(z_norm), 3))
        colors[:, 0] = z_norm  # Vermelho para o topo
        colors[:, 2] = 1 - z_norm # Azul para a base
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500)
        print(f"Exibindo: {window_name} ({len(points)} pontos)")
        o3d.visualization.draw_geometries([pcd, coord], window_name=window_name)

# --- EXECUÇÃO ---
if __name__ == "__main__":
    gen = SyntheticDataGenerator()
    
    # 1. Rampa Linear
    ramp_lin = gen.generate_ramp(curvature="linear")
    gen.visualize(ramp_lin, "Rampa Linear")

    # 2. Rampa Curva (Côncava)
    ramp_curv = gen.generate_ramp(curvature="concave", noise_level=5.0)
    gen.visualize(ramp_curv, "Rampa Concava")

    # 3. Rampa com Degraus
    steps = gen.generate_stepped_ramp()
    gen.visualize(steps, "Escada/Degraus")

    # 4. Montes (Seno/Cosseno) para Volume
    hills = gen.generate_hills(max_height=600, frequency=0.003)
    gen.visualize(hills, "Montes Senoidais (Cálculo de Volume)")