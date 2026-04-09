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
        width: float = 2000,        # Largura da rampa (mm)
        length: float = 3000,       # Comprimento da rampa (mm)
        height: float = 800,        # Altura da rampa (mm)
        point_density: int = 5,     # Densidade de pontos (menor = mais denso)
        noise_level: float = 2.0,   # Nível de ruído (mm)
        add_ground: bool = False,    # Adicionar chão ao redor
    ):
        points = []
        x_values = np.arange(0, length, point_density)
        y_values = np.arange(-width/2, width/2, point_density)
        
        # 1. SUPERFÍCIE DA RAMPA
        for x in x_values:
            for y in y_values:
                # Altura proporcional ao comprimento (rampa linear)
                # IMPORTANTE: Z mínimo = 0.5mm (margem mínima para evitar conflito com bucket Z=0)
                z_base = 0.5  # mm - margem mínima acima da caçamba
                z = z_base + (x / length) * height
                
                # Adicionar ruído gaussiano
                if noise_level > 0:
                    z += np.random.normal(0, noise_level)
                    x_noise = x + np.random.normal(0, noise_level * 0.5)
                    y_noise = y + np.random.normal(0, noise_level * 0.5)
                else:
                    x_noise = x
                    y_noise = y
                
                # Garantir que Z nunca fique abaixo da margem mínima
                z = max(z_base, z)
                
                points.append((x_noise, y_noise, z))
        
        # Adicionar chão ao redor (z = 0)
        if add_ground:
            ground_margin = 500  # mm de margem ao redor
            
            # Chão à frente
            for x in np.arange(-ground_margin, 0, point_density * 2):
                for y in np.arange(-width/2 - ground_margin, width/2 + ground_margin, point_density * 2):
                    z_ground = np.random.normal(0, noise_level * 0.5) if noise_level > 0 else 0
                    z_ground = max(0.0, z_ground)  # Garantir Z >= 0
                    points.append((x, y, z_ground))
            
            # Chão atrás
            for x in np.arange(length, length + ground_margin, point_density * 2):
                for y in np.arange(-width/2 - ground_margin, width/2 + ground_margin, point_density * 2):
                    # Altura do final da rampa
                    z_ground = height + (np.random.normal(0, noise_level * 0.5) if noise_level > 0 else 0)
                    points.append((x, y, z_ground))
            
            # Chão nas laterais
            for x in np.arange(0, length, point_density * 2):
                # Lado esquerdo
                for y in np.arange(-width/2 - ground_margin, -width/2, point_density * 2):
                    z_ground = ((x / length) * height) + (np.random.normal(0, noise_level * 0.5) if noise_level > 0 else 0)
                    z_ground = max(0.0, z_ground)  # Garantir Z >= 0
                    points.append((x, y, z_ground))
                
                # Lado direito
                for y in np.arange(width/2, width/2 + ground_margin, point_density * 2):
                    z_ground = ((x / length) * height) + (np.random.normal(0, noise_level * 0.5) if noise_level > 0 else 0)
                    z_ground = max(0.0, z_ground)  # Garantir Z >= 0
                    points.append((x, y, z_ground))
        
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
            x_start = step * step_length
            x_end = (step + 1) * step_length
            z_base = 0.5  # mm - margem mínima acima da caçamba
            z = z_base + (step * step_height)
            
            x_values = np.arange(x_start, x_end, point_density)
            
            for x in x_values:
                for y in y_values:
                    z_noise = z + (np.random.normal(0, noise_level) if noise_level > 0 else 0)
                    x_noise = x + (np.random.normal(0, noise_level * 0.5) if noise_level > 0 else 0)
                    y_noise = y + (np.random.normal(0, noise_level * 0.5) if noise_level > 0 else 0)
                    
                    # Garantir que Z nunca fique abaixo da margem mínima
                    z_noise = max(z_base, z_noise)
                    
                    points.append((x_noise, y_noise, z_noise))
        
        return points
    
    def generate_curved_ramp(
        self,
        width: float = 2000,
        length: float = 3000,
        max_height: float = 800,
        point_density: int = 5,
        noise_level: float = 2.0,
        curvature: str = "concave"  # "concave" ou "convex"
    ):
        """
        Gera uma rampa curva (côncava ou convexa).
        
        Args:
            width: Largura da rampa
            length: Comprimento da rampa
            max_height: Altura máxima
            point_density: Espaçamento entre pontos
            noise_level: Quantidade de ruído
            curvature: Tipo de curvatura ("concave" ou "convex")
            
        Returns:
            Lista de tuplas (x, y, z)
        """
        points = []
        x_values = np.arange(0, length, point_density)
        y_values = np.arange(-width/2, width/2, point_density)
        
        for x in x_values:
            for y in y_values:
                # Normalizar x para [0, 1]
                x_norm = x / length
                
                # Z base acima da caçamba
                z_base = 0.5  # mm - margem mínima
                
                # Calcular altura baseado na curvatura
                if curvature == "concave":
                    # Parábola: começa suave, fica mais íngreme
                    z = z_base + (max_height * (x_norm ** 2))
                elif curvature == "convex":
                    # Raiz quadrada: começa íngreme, fica mais suave
                    z = z_base + (max_height * np.sqrt(x_norm))
                else:
                    # Linear por padrão
                    z = z_base + (max_height * x_norm)
                
                # Adicionar ruído
                if noise_level > 0:
                    z += np.random.normal(0, noise_level)
                    x_noise = x + np.random.normal(0, noise_level * 0.5)
                    y_noise = y + np.random.normal(0, noise_level * 0.5)
                else:
                    x_noise = x
                    y_noise = y
                
                # Garantir que Z nunca fique abaixo da margem mínima
                z = max(z_base, z)
                
                points.append((x_noise, y_noise, z))
        
        return points
    
    def generate_sand_pile(
        self,
        width: float = 2000,
        length: float = 3000,
        max_height: float = 600,
        point_density: int = 8,
        noise_level: float = 3.0,
        n_peaks: int = 3,
        seed: int = 42,
    ):
        """
        Gera um monte de areia com múltiplos picos Gaussianos.

        Forma: z(x,y) = sum_i A_i * exp(-((x-cx_i)^2/(2*sx_i^2) + (y-cy_i)^2/(2*sy_i^2)))

        Volume analítico (integração numérica em grade 1mm):
            V = integral_0^L integral_{-W/2}^{W/2} z(x,y) dy dx

        Args:
            width: Largura da caixa (mm)
            length: Comprimento da caixa (mm)
            max_height: Amplitude máxima dos picos (mm)
            point_density: Espaçamento entre pontos (mm)
            noise_level: Ruído gaussiano (mm)
            n_peaks: Número de picos
            seed: Seed para reproducibilidade

        Returns:
            (points, peaks, expected_volume_mm3)
            peaks: lista de (cx, cy, amplitude, sx, sy)
        """
        rng = np.random.default_rng(seed)

        # Definir picos garantindo que estejam bem dentro da caixa
        peaks = []
        for _ in range(n_peaks):
            cx = rng.uniform(length * 0.15, length * 0.85)
            cy = rng.uniform(-width * 0.35, width * 0.35)
            amplitude = rng.uniform(max_height * 0.5, max_height)
            sx = rng.uniform(length * 0.08, length * 0.20)
            sy = rng.uniform(width * 0.08, width * 0.20)
            peaks.append((float(cx), float(cy), float(amplitude), float(sx), float(sy)))

        z_base = 0.5  # mm - margem mínima acima do piso da caçamba

        def z_func(x, y):
            z = z_base
            for cx, cy, A, sx, sy in peaks:
                z += A * np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))
            return z

        # Gerar nuvem de pontos
        x_values = np.arange(0, length, point_density)
        y_values = np.arange(-width / 2, width / 2, point_density)
        points = []

        for x in x_values:
            for y in y_values:
                z = z_func(x, y)
                if noise_level > 0:
                    z += rng.normal(0, noise_level)
                    xn = x + rng.normal(0, noise_level * 0.5)
                    yn = y + rng.normal(0, noise_level * 0.5)
                else:
                    xn, yn = x, y
                z = max(z_base, z)
                points.append((float(xn), float(yn), float(z)))

        # Volume esperado por integração numérica (grade 1mm, sem ruído)
        xs = np.arange(0, length, 1.0)
        ys = np.arange(-width / 2, width / 2, 1.0)
        XX, YY = np.meshgrid(xs, ys)
        ZZ = np.full_like(XX, z_base)
        for cx, cy, A, sx, sy in peaks:
            ZZ = ZZ + A * np.exp(-((XX - cx) ** 2 / (2 * sx ** 2) + (YY - cy) ** 2 / (2 * sy ** 2)))
        expected_volume_mm3 = float(ZZ.sum())  # dx=dy=1mm, so area per cell = 1mm²

        return points, peaks, expected_volume_mm3

    def save_as_npz(self, points, filepath: str):
        """
        Salva os pontos no formato .npz (mesmo formato usado pelo código).
        
        Args:
            points: Lista de tuplas (x, y, z)
            filepath: Caminho completo do arquivo (ex: "./data/ramp_data.npz")
        """
        xyz = np.array(points)
        np.savez_compressed(filepath, xyz=xyz)
        print(f"Dados salvos em: {filepath}")
        print(f"Total de pontos: {len(points)}")
    
    def visualize(self, points):
        """
        Visualiza a nuvem de pontos usando Open3D.
        
        Args:
            points: Lista de tuplas (x, y, z)
        """
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
    
    print("=== Gerador de Dados Sintéticos 3D ===\n")
    
    # Opção 1: Rampa linear simples
    print("1. Gerando rampa linear...")
    ramp_points = generator.generate_ramp(
        width=2000,
        length=3000,
        height=800,
        point_density=10,
        noise_level=3.0,
        add_ground=False
    )
    
    stats = generator.get_stats(ramp_points)
    print(f"   Pontos gerados: {stats['num_points']}")
    print(f"   Range X: {stats['x_range']}")
    print(f"   Range Y: {stats['y_range']}")
    print(f"   Range Z: {stats['z_range']}\n")
    
    # Salvar
    generator.save_as_npz(ramp_points, "./synthetic_ramp_linear.npz")
    
    # Opção 2: Rampa com degraus
    print("\n2. Gerando rampa com degraus...")
    stepped_points = generator.generate_stepped_ramp(
        width=2000,
        num_steps=5,
        step_length=600,
        step_height=150,
        point_density=10,
        noise_level=3.0
    )
    
    generator.save_as_npz(stepped_points, "./synthetic_ramp_stepped.npz")
    
    # Opção 3: Rampa curva
    print("\n3. Gerando rampa curva (côncava)...")
    curved_points = generator.generate_curved_ramp(
        width=2000,
        length=3000,
        max_height=800,
        point_density=10,
        noise_level=3.0,
        curvature="concave"
    )
    
    generator.save_as_npz(curved_points, "./synthetic_ramp_curved.npz")
    
    # Visualizar (descomente para ver)
    print("\nVisualizando rampa linear...")
    generator.visualize(ramp_points)
