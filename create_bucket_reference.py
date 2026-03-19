"""
Script para criar uma caçamba de referência sintética vazia.
Use isto quando não tiver dados reais de caçamba vazia.
"""

import os
import numpy as np


def create_empty_bucket_reference():
    """
    Cria uma caçamba de referência sintética vazia (plano no chão).
    Usado como baseline para cálculo de volume.
    """
    print("=== Criando Caçamba de Referência Sintética ===\n")
    
    # Dimensões aproximadas de uma caçamba de caminhão
    width = 2400   # 2.4m
    length = 3000  # 3m
    height = 0     # Plano no chão (Z=0)
    point_density = 15  # mm
    
    print("Gerando plano de referência (caçamba vazia)...")
    
    # Gerar pontos em uma grade plana (Z=0)
    points = []
    for x in range(0, int(length), point_density):
        for y in range(int(-width/2), int(width/2), point_density):
            z = 0  # Plano no chão
            # Adicionar pequeno ruído
            noise = np.random.normal(0, 1.0, 3)
            points.append((x + noise[0], y + noise[1], z + noise[2]))
    
    bucket_array = np.array(points, dtype=np.float64)
    
    # Criar diretório
    bucket_path = "./pointcloud/caixa_vazia/"
    if not os.path.exists(bucket_path):
        os.makedirs(bucket_path)
    
    # Salvar
    np.savez_compressed(f"{bucket_path}data.npz", xyz=bucket_array)
    
    print(f"✓ Caçamba de referência criada com {len(points)} pontos")
    print(f"  Salvo em: {bucket_path}data.npz")
    print(f"  Dimensões: {width}mm x {length}mm x {height}mm")
    
    # Salvar metadados
    metadata = f"""=== CAÇAMBA DE REFERÊNCIA SINTÉTICA ===
Tipo: Plano vazio (baseline)
Largura: {width} mm
Comprimento: {length} mm
Altura: {height} mm (plano)
Pontos: {len(points)}
Descrição: Caçamba vazia usada como referência para cálculo de volume.
"""
    with open(f"{bucket_path}BUCKET_INFO.txt", "w") as f:
        f.write(metadata)
    
    print("\n✓ Caçamba de referência pronta para uso!")
    return bucket_path


if __name__ == "__main__":
    create_empty_bucket_reference()
