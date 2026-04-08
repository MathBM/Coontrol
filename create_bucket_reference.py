"""
Script para criar uma caçamba de referência sintética vazia.
Use isto quando não tiver dados reais de caçamba vazia.
"""

import os
import numpy as np


def create_empty_bucket_reference():
    """
    Cria uma caçamba de referência sintética vazia com piso + 4 paredes.
    Usado como baseline para cálculo de volume.
    """
    print("=== Criando Caçamba de Referência Sintética ===\n")
    
    # Dimensões aproximadas de uma caçamba de caminhão
    width = 2000    # 2m (eixo Y)
    length = 3000   # 3m (eixo X)
    wall_height = 900  # altura das paredes (mm) - margem acima da carga máx
    point_density = 8  # 8mm - mesma densidade da rampa sintética
    
    print("Gerando piso e paredes da caçamba...")
    
    points = []

    # --- PISO (z=0) ---
    for x in range(0, int(length), point_density):
        for y in range(int(-width / 2), int(width / 2), point_density):
            points.append((float(x), float(y), 0.0))

    # --- PAREDES LATERAIS (y = ±width/2, ao longo de X) ---
    for x in range(0, int(length), point_density):
        for z in range(0, int(wall_height), point_density):
            points.append((float(x), float(-width / 2), float(z)))
            points.append((float(x), float( width / 2), float(z)))

    # --- PAREDES FRONTAIS (x=0 e x=length, ao longo de Y) ---
    for y in range(int(-width / 2), int(width / 2), point_density):
        for z in range(0, int(wall_height), point_density):
            points.append((0.0,         float(y), float(z)))
            points.append((float(length), float(y), float(z)))

    bucket_array = np.array(points, dtype=np.float64)
    
    # Criar diretório
    bucket_path = "./pointcloud/caixa_vazia/"
    if not os.path.exists(bucket_path):
        os.makedirs(bucket_path)
    
    # Salvar
    np.savez_compressed(f"{bucket_path}data.npz", xyz=bucket_array)
    
    print(f"✓ Caçamba de referência criada com {len(points)} pontos")
    print(f"  Salvo em: {bucket_path}data.npz")
    print(f"  Dimensões: {width}mm x {length}mm, paredes: {wall_height}mm")
    
    # Salvar metadados
    metadata = f"""=== CAÇAMBA DE REFERÊNCIA SINTÉTICA (PERFEITA) ===
Tipo: Plano vazio perfeito (baseline)
Largura: {width} mm
Comprimento: {length} mm
Altura: 0.0 mm (plano perfeito, SEM ruído)
Altura paredes: {wall_height} mm
Pontos: {len(points)}
Descrição: Caçamba vazia com piso plano e 4 paredes, sem ruído.
"""
    with open(f"{bucket_path}BUCKET_INFO.txt", "w") as f:
        f.write(metadata)
    
    print("\n✓ Caçamba de referência pronta para uso!")
    return bucket_path


if __name__ == "__main__":
    create_empty_bucket_reference()
