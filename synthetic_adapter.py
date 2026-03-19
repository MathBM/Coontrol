"""
Adaptador para usar dados sintéticos no lugar dos dados reais do sensor.
Este módulo permite testar todo o pipeline sem modificar o código original.
"""

import numpy as np
from synthetic_data_generator import SyntheticDataGenerator
from src.PointCloudReconstructor import PointCloudReconstructor
from src.Constants import Constants


class SyntheticPointCloudReconstructor(PointCloudReconstructor):
    """
    Versão modificada do PointCloudReconstructor que usa dados sintéticos
    em vez de ler arquivos binários do sensor.
    
    Use esta classe no lugar de PointCloudReconstructor para testar
    o pipeline com dados sintéticos.
    """
    
    def __init__(self, use_synthetic=True):
        super().__init__()
        self.use_synthetic = use_synthetic
        self.generator = SyntheticDataGenerator()
    
    def create_point_cloud(self, scan_path: str = None):
        """
        Sobrescreve o método original para gerar dados sintéticos
        em vez de ler arquivos binários.
        
        Args:
            scan_path: Ignorado quando use_synthetic=True
            
        Returns:
            Lista de pontos 3D (x, y, z)
        """
        if not self.use_synthetic:
            # Usar método original se use_synthetic=False
            return super().create_point_cloud(scan_path)
        
        print("[MODO SINTÉTICO] Gerando dados de rampa...")
        
        # Gerar uma rampa sintética
        xyz = self.generator.generate_ramp(
            width=2000,
            length=3000,
            height=800,
            point_density=8,
            noise_level=3.0,
            add_ground=True
        )
        
        print(f"[MODO SINTÉTICO] {len(xyz)} pontos gerados")
        
        return xyz
    
    def create_synthetic_ramp(
        self,
        ramp_type: str = "linear",
        width: float = 2000,
        length: float = 3000,
        height: float = 800,
        point_density: int = 8,
        noise_level: float = 3.0,
        **kwargs
    ):
        """
        Cria uma rampa sintética com parâmetros personalizados.
        
        Args:
            ramp_type: Tipo de rampa ("linear", "stepped", "concave", "convex")
            width: Largura em mm
            length: Comprimento em mm
            height: Altura em mm
            point_density: Densidade de pontos (menor = mais denso)
            noise_level: Nível de ruído em mm
            **kwargs: Argumentos adicionais específicos do tipo
            
        Returns:
            Lista de pontos 3D
        """
        if ramp_type == "linear":
            return self.generator.generate_ramp(
                width=width,
                length=length,
                height=height,
                point_density=point_density,
                noise_level=noise_level,
                add_ground=kwargs.get('add_ground', True)
            )
        
        elif ramp_type == "stepped":
            return self.generator.generate_stepped_ramp(
                width=width,
                num_steps=kwargs.get('num_steps', 5),
                step_length=kwargs.get('step_length', length / 5),
                step_height=kwargs.get('step_height', height / 5),
                point_density=point_density,
                noise_level=noise_level
            )
        
        elif ramp_type in ["concave", "convex"]:
            return self.generator.generate_curved_ramp(
                width=width,
                length=length,
                max_height=height,
                point_density=point_density,
                noise_level=noise_level,
                curvature=ramp_type
            )
        
        else:
            raise ValueError(f"Tipo de rampa inválido: {ramp_type}")


def replace_reconstructor_with_synthetic():
    """
    Helper function para substituir o PointCloudReconstructor
    por SyntheticPointCloudReconstructor no DataManager.
    
    Exemplo de uso:
        from synthetic_adapter import replace_reconstructor_with_synthetic
        
        # No seu código, antes de processar:
        synthetic_reconstructor = replace_reconstructor_with_synthetic()
        
        # Agora use synthetic_reconstructor no lugar do original
    """
    return SyntheticPointCloudReconstructor(use_synthetic=True)


# ============================================================================
# Exemplo de uso direto
# ============================================================================

def example_usage_in_data_manager():
    """
    Exemplo de como modificar o DataManager para usar dados sintéticos.
    
    NO SEU CÓDIGO ORIGINAL (DataManager.py):
    
    # Antes:
    # self.pcd_reconstructor = PointCloudReconstructor()
    
    # Depois (para testar com dados sintéticos):
    from synthetic_adapter import SyntheticPointCloudReconstructor
    self.pcd_reconstructor = SyntheticPointCloudReconstructor(use_synthetic=True)
    
    # Ou, para gerar um tipo específico de rampa:
    self.pcd_reconstructor = SyntheticPointCloudReconstructor(use_synthetic=True)
    xyz = self.pcd_reconstructor.create_synthetic_ramp(
        ramp_type="concave",
        width=1800,
        length=2800,
        height=900,
        point_density=6,
        noise_level=2.5
    )
    """
    pass


def quick_test_pipeline():
    """
    Teste rápido: simula o processo completo como se estivesse
    usando dados reais do sensor.
    """
    print("=== Teste Rápido do Pipeline com Dados Sintéticos ===\n")
    
    # 1. Criar reconstrutor sintético
    reconstructor = SyntheticPointCloudReconstructor(use_synthetic=True)
    
    # 2. "Processar" dados (na verdade, gera sintéticos)
    xyz = reconstructor.create_point_cloud()
    
    print(f"Total de pontos: {len(xyz)}\n")
    
    # 3. Aplicar transformação (exemplo)
    xyz_transformed = reconstructor.transform(
        xyz,
        rotation=(0, 0, 0),
        translation=(100, 0, 0)  # Mover 100mm no eixo X
    )
    
    print(f"Pontos após transformação: {len(xyz_transformed)}\n")
    
    # 4. Aplicar filtros
    xyz_filtered = reconstructor.filter_point_cloud(
        xyz_transformed,
        nb_neighbors=40,
        std_ratio=0.1,
        nb_points=25,
        radius=50
    )
    
    print(f"Pontos após filtros: {len(xyz_filtered)}")
    print(f"Removidos: {len(xyz_transformed) - len(xyz_filtered)} pontos\n")
    
    # 5. Salvar resultado
    import os
    os.makedirs("./synthetic_data", exist_ok=True)
    np.savez_compressed("./synthetic_data/pipeline_test.npz", xyz=xyz_filtered)
    
    print("✓ Dados salvos em: ./synthetic_data/pipeline_test.npz")
    print("✓ Pipeline testado com sucesso!\n")
    
    # 6. Visualizar
    generator = SyntheticDataGenerator()
    print("Visualizando resultado final...")
    generator.visualize(xyz_filtered)
    
    return xyz_filtered


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Adaptador de Dados Sintéticos                           ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    quick_test_pipeline()
