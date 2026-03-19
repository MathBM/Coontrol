"""
Módulo para criar scans sintéticos compatíveis com o sistema.
Permite testar o pipeline completo sem o sensor LIDAR real.
"""

import os
import numpy as np
from datetime import datetime
from synthetic_data_generator import SyntheticDataGenerator
from src.Constants import Constants


class SyntheticScanCreator:
    """
    Cria scans sintéticos que podem ser processados pelo DataManager
    exatamente como se fossem scans reais dos sensores LIDAR.
    """
    
    def __init__(self):
        self.generator = SyntheticDataGenerator()
    
    def create_synthetic_scan(
        self,
        ramp_type: str = "linear",
        width: float = 2000,
        length: float = 3000,
        height: float = 800,
        point_density: int = 8,
        noise_level: float = 3.0,
        custom_name: str = None,
        **kwargs
    ) -> str:
        """
        Cria um scan sintético completo e salva no diretório de scans.
        
        Args:
            ramp_type: Tipo de rampa ("linear", "stepped", "concave", "convex")
            width: Largura em mm
            length: Comprimento em mm  
            height: Altura em mm
            point_density: Densidade de pontos (menor = mais denso)
            noise_level: Nível de ruído em mm
            custom_name: Nome customizado (opcional)
            **kwargs: Argumentos adicionais para tipos específicos de rampa
            
        Returns:
            Caminho da pasta criada
        """
        # Criar nome da pasta
        if custom_name:
            folder_name = f"{custom_name}_SYNTHETIC"
        else:
            date = datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
            folder_name = f"{date}_SYNTHETIC_{ramp_type}"
        
        scan_path = f"{Constants.SCANS_DIRECTORY}{folder_name}/"
        
        # Criar diretório
        if not os.path.exists(scan_path):
            os.makedirs(scan_path)
        
        # Gerar dados sintéticos baseado no tipo
        print(f"[SYNTHETIC] Gerando rampa tipo '{ramp_type}'...")
        
        if ramp_type == "linear":
            xyz = self.generator.generate_ramp(
                width=width,
                length=length,
                height=height,
                point_density=point_density,
                noise_level=noise_level,
                add_ground=kwargs.get('add_ground', True)
            )
        
        elif ramp_type == "stepped":
            xyz = self.generator.generate_stepped_ramp(
                width=width,
                num_steps=kwargs.get('num_steps', 6),
                step_length=kwargs.get('step_length', length / 6),
                step_height=kwargs.get('step_height', height / 6),
                point_density=point_density,
                noise_level=noise_level
            )
        
        elif ramp_type == "concave":
            xyz = self.generator.generate_curved_ramp(
                width=width,
                length=length,
                max_height=height,
                point_density=point_density,
                noise_level=noise_level,
                curvature="concave"
            )
        
        elif ramp_type == "convex":
            xyz = self.generator.generate_curved_ramp(
                width=width,
                length=length,
                max_height=height,
                point_density=point_density,
                noise_level=noise_level,
                curvature="convex"
            )
        
        else:
            raise ValueError(f"Tipo de rampa inválido: {ramp_type}")
        
        # Salvar como data.npz (formato esperado pelo DataManager)
        # Converter lista de tuplas para array numpy
        if isinstance(xyz, list):
            xyz_array = np.array(xyz, dtype=np.float64)
        else:
            xyz_array = xyz
        
        np.savez_compressed(f"{scan_path}data.npz", xyz=xyz_array)
        
        print(f"[SYNTHETIC] Scan criado com {len(xyz)} pontos")
        print(f"[SYNTHETIC] Salvo em: {scan_path}")
        
        # Criar arquivo de metadados (opcional, para referência)
        self._save_metadata(scan_path, ramp_type, width, length, height, 
                           point_density, noise_level, len(xyz))
        
        return scan_path
    
    def _save_metadata(self, scan_path: str, ramp_type: str, width: float,
                      length: float, height: float, density: int, 
                      noise: float, num_points: int):
        """Salva metadados do scan sintético para referência"""
        metadata = f"""=== SCAN SINTÉTICO ===
Tipo: {ramp_type}
Largura: {width} mm
Comprimento: {length} mm
Altura: {height} mm
Densidade: {density} mm
Ruído: {noise} mm
Pontos gerados: {num_points}
Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        with open(f"{scan_path}SYNTHETIC_INFO.txt", "w") as f:
            f.write(metadata)
    
    def create_quick_test_scan(self) -> str:
        """
        Cria um scan sintético rápido para testes.
        Rampa linear com parâmetros padrão.
        """
        return self.create_synthetic_scan(
            ramp_type="linear",
            custom_name=datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
        )
    
    def create_varied_test_scans(self):
        """
        Cria vários scans sintéticos de diferentes tipos
        para testes completos do pipeline.
        """
        scan_types = [
            ("linear", {"width": 1800, "length": 2500, "height": 600}),
            ("steep", {"width": 1500, "length": 2000, "height": 1200}),
            ("stepped", {"width": 1500, "length": 3000, "height": 720, "num_steps": 6}),
            ("concave", {"width": 1600, "length": 2800, "height": 700}),
            ("convex", {"width": 1600, "length": 2800, "height": 700}),
        ]
        
        created_scans = []
        
        for scan_type, params in scan_types:
            scan_path = self.create_synthetic_scan(
                ramp_type=scan_type,
                **params
            )
            created_scans.append(scan_path)
            print()  # Linha em branco entre scans
        
        print(f"[SYNTHETIC] Criados {len(created_scans)} scans de teste!")
        return created_scans


def quick_create_synthetic_scan():
    """Helper function para criar rapidamente um scan sintético"""
    creator = SyntheticScanCreator()
    return creator.create_quick_test_scan()


if __name__ == "__main__":
    # Teste direto do módulo
    print("=== Criador de Scans Sintéticos ===\n")
    
    creator = SyntheticScanCreator()
    
    # Criar um scan de teste
    scan_path = creator.create_quick_test_scan()
    
    print(f"\n✓ Scan sintético criado com sucesso!")
    print(f"  Você pode processá-lo pela interface do sistema.")
