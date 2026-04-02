"""
Gerador de dados sintéticos 3D para simular sensor LIDAR.
Cria uma rampa com dimensões e inclinação configuráveis.
"""

import numpy as np
import open3d as o3d
from src.Constants import Constants


class SyntheticDataGenerator:
    
    def __init__(self):
        pass
    
    def generate_ramp(
        self,
        width: float = 2000,        # Largura da rampa (mm)
        length: float = 3000,       # Comprimento da rampa (mm)
        height: float = 800,        # Altura da rampa (mm)
        point_density: int = 5,     # Densidade de pontos (menor = mais denso)
        noise_level: float = 2.0,   # Nível de ruído (mm)
        add_ground: bool = False,    # Adicionar chão ao redor
    ):
        """
        Gera uma nuvem de pontos 3D representando uma rampa.
        
        Args:
            width: Largura da rampa em mm
            length: Comprimento da rampa em mm
            height: Altura máxima da rampa em mm
            point_density: Espaçamento entre pontos (menor = mais denso)
            noise_level: Quantidade de ruído a adicionar
            add_ground: Se deve adicionar pontos do chão ao redor
            
        Returns:
            Lista de tuplas (x, y, z) representando a nuvem de pontos
        """
        points = []
        
        # Gerar pontos da rampa
        x_values = np.arange(0, length, point_density)
        y_values = np.arange(-width/2, width/2, point_density)
        
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
    
    def generate_stepped_ramp(
        self,
        width: float = 2000,
        num_steps: int = 5,
        step_length: float = 600,
        step_height: float = 150,
        point_density: int = 5,
        noise_level: float = 2.0,
    ):
        """
        Gera uma rampa com degraus (escada).
        
        Args:
            width: Largura da rampa
            num_steps: Número de degraus
            step_length: Comprimento de cada degrau
            step_height: Altura de cada degrau
            point_density: Espaçamento entre pontos
            noise_level: Quantidade de ruído
            
        Returns:
            Lista de tuplas (x, y, z)
        """
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
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Colorir baseado na altura (Z)
        colors = np.asarray(pcd.points)
        z_values = colors[:, 2]
        z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())
        colors_rgb = np.zeros((len(z_normalized), 3))
        colors_rgb[:, 0] = z_normalized  # Red channel
        colors_rgb[:, 2] = 1 - z_normalized  # Blue channel
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        
        # Adicionar sistema de coordenadas
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
        
        o3d.visualization.draw_geometries([pcd, coord_frame])
    
    def get_stats(self, points):
        """
        Retorna estatísticas sobre a nuvem de pontos.
        """
        xyz = np.array(points)
        
        stats = {
            'num_points': len(points),
            'x_range': (xyz[:, 0].min(), xyz[:, 0].max()),
            'y_range': (xyz[:, 1].min(), xyz[:, 1].max()),
            'z_range': (xyz[:, 2].min(), xyz[:, 2].max()),
            'centroid': xyz.mean(axis=0)
        }
        
        return stats


# Exemplo de uso
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    
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
