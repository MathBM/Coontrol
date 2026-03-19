"""
Script de exemplo para testar o pipeline com dados sintéticos.
Use este script para testar filtros, registro, reconstrução e cálculo de volume
sem precisar do sensor LIDAR.
"""

import sys
import numpy as np
from synthetic_data_generator import SyntheticDataGenerator
from src.PointCloudPlotter import PointCloudPlotter
from src.VolumeCalculatorLegacy import VolumeCalculatorLegacy
from src.Registration import Registration
from src.SurfaceReconstructor import SurfaceReconstructor
from src.Parameters import Parameters
from src.PointCloudReconstructor import PointCloudReconstructor

sys.path.append("src")


def test_basic_generation():
    """Teste básico: gera e visualiza uma rampa"""
    print("=== Teste 1: Geração e Visualização Básica ===\n")
    
    generator = SyntheticDataGenerator()
    
    # Gerar rampa linear
    points = generator.generate_ramp(
        width=2000,          # 2 metros de largura
        length=3000,         # 3 metros de comprimento
        height=800,          # 80 cm de altura
        point_density=8,     # Pontos a cada 8mm
        noise_level=2.5,     # Ruído de 2.5mm (simula imprecisão do sensor)
        add_ground=True
    )
    
    # Mostrar estatísticas
    stats = generator.get_stats(points)
    print(f"Pontos gerados: {stats['num_points']}")
    print(f"Faixa X: {stats['x_range'][0]:.1f} a {stats['x_range'][1]:.1f} mm")
    print(f"Faixa Y: {stats['y_range'][0]:.1f} a {stats['y_range'][1]:.1f} mm")
    print(f"Faixa Z: {stats['z_range'][0]:.1f} a {stats['z_range'][1]:.1f} mm")
    print(f"Centroide: ({stats['centroid'][0]:.1f}, {stats['centroid'][1]:.1f}, {stats['centroid'][2]:.1f})\n")
    
    # Salvar para uso posterior
    generator.save_as_npz(points, "./synthetic_data/ramp_test.npz")
    
    # Visualizar
    print("Abrindo visualização 3D...")
    print("(Pressione Q para fechar a janela)\n")
    generator.visualize(points)
    
    return points


def test_with_filters():
    """Teste 2: Aplica filtros do pipeline nos dados sintéticos"""
    print("\n=== Teste 2: Aplicação de Filtros ===\n")
    
    generator = SyntheticDataGenerator()
    reconstructor = PointCloudReconstructor()
    
    # Gerar dados com mais ruído para testar filtros
    points = generator.generate_ramp(
        width=2000,
        length=3000,
        height=800,
        point_density=8,
        noise_level=5.0,  # Mais ruído para testar filtros
        add_ground=True
    )
    
    print(f"Pontos antes do filtro: {len(points)}")
    
    # Aplicar filtros (mesmos parâmetros do código original)
    filtered_points = reconstructor.filter_point_cloud(
        points,
        nb_neighbors=40,
        std_ratio=0.1,
        nb_points=25,
        radius=50
    )
    
    print(f"Pontos depois do filtro: {len(filtered_points)}")
    print(f"Pontos removidos: {len(points) - len(filtered_points)} ({100*(1-len(filtered_points)/len(points)):.1f}%)\n")
    
    # Visualizar resultado
    print("Visualizando pontos filtrados...")
    generator.visualize(filtered_points)
    
    return filtered_points


def test_different_ramp_types():
    """Teste 3: Gera diferentes tipos de rampas"""
    print("\n=== Teste 3: Diferentes Tipos de Rampas ===\n")
    
    generator = SyntheticDataGenerator()
    
    # 1. Rampa linear suave
    print("1. Rampa Linear Suave")
    linear = generator.generate_ramp(
        width=1500, length=2500, height=600,
        point_density=10, noise_level=1.0
    )
    generator.save_as_npz(linear, "./synthetic_data/ramp_linear.npz")
    print(f"   Salvo: {len(linear)} pontos\n")
    
    # 2. Rampa íngreme
    print("2. Rampa Íngreme")
    steep = generator.generate_ramp(
        width=1500, length=2000, height=1200,  # Mais íngreme
        point_density=10, noise_level=2.0
    )
    generator.save_as_npz(steep, "./synthetic_data/ramp_steep.npz")
    print(f"   Salvo: {len(steep)} pontos\n")
    
    # 3. Rampa com degraus (escada)
    print("3. Rampa com Degraus (Escada)")
    stepped = generator.generate_stepped_ramp(
        width=1500, num_steps=6, step_length=500,
        step_height=120, point_density=10, noise_level=2.0
    )
    generator.save_as_npz(stepped, "./synthetic_data/ramp_stepped.npz")
    print(f"   Salvo: {len(stepped)} pontos\n")
    
    # 4. Rampa curva côncava
    print("4. Rampa Curva Côncava")
    concave = generator.generate_curved_ramp(
        width=1500, length=2500, max_height=700,
        point_density=10, noise_level=2.0, curvature="concave"
    )
    generator.save_as_npz(concave, "./synthetic_data/ramp_concave.npz")
    print(f"   Salvo: {len(concave)} pontos\n")
    
    # 5. Rampa curva convexa
    print("5. Rampa Curva Convexa")
    convex = generator.generate_curved_ramp(
        width=1500, length=2500, max_height=700,
        point_density=10, noise_level=2.0, curvature="convex"
    )
    generator.save_as_npz(convex, "./synthetic_data/ramp_convex.npz")
    print(f"   Salvo: {len(convex)} pontos\n")
    
    # Visualizar cada uma
    print("Visualizando todas as rampas...")
    print("(1/5) Linear")
    generator.visualize(linear)
    print("(2/5) Íngreme")
    generator.visualize(steep)
    print("(3/5) Degraus")
    generator.visualize(stepped)
    print("(4/5) Côncava")
    generator.visualize(concave)
    print("(5/5) Convexa")
    generator.visualize(convex)


def load_and_use_synthetic_data():
    """Exemplo de como carregar dados salvos e usar no pipeline"""
    print("\n=== Carregando Dados Sintéticos Salvos ===\n")
    
    # Carregar dados salvos
    data = np.load("./synthetic_data/ramp_test.npz")
    xyz = data['xyz']
    
    print(f"Dados carregados: {len(xyz)} pontos")
    print(f"Shape: {xyz.shape}")
    print(f"Tipo: {type(xyz)}\n")
    
    # Converter para lista de tuplas (formato esperado pelo código)
    points_list = [tuple(point) for point in xyz]
    
    print("Dados prontos para usar no pipeline!")
    print("Você pode usar 'points_list' diretamente nos métodos de:")
    print("  - filter_point_cloud()")
    print("  - transform()")
    print("  - remove_boundaries()")
    print("  - E todo o resto do pipeline!\n")
    
    return points_list


def main():
    """Menu principal"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Testador de Dados Sintéticos - Projeto LIDAR            ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    print("Escolha um teste para executar:\n")
    print("1. Geração e visualização básica")
    print("2. Teste de filtros")
    print("3. Gerar diferentes tipos de rampas")
    print("4. Carregar dados salvos")
    print("0. Executar todos os testes\n")
    
    choice = input("Digite sua escolha (0-4): ").strip()
    
    # Criar diretório para dados sintéticos
    import os
    os.makedirs("./synthetic_data", exist_ok=True)
    
    if choice == "1":
        test_basic_generation()
    elif choice == "2":
        test_with_filters()
    elif choice == "3":
        test_different_ramp_types()
    elif choice == "4":
        load_and_use_synthetic_data()
    elif choice == "0":
        print("\n>>> Executando todos os testes...\n")
        test_basic_generation()
        test_with_filters()
        test_different_ramp_types()
        load_and_use_synthetic_data()
    else:
        print("Opção inválida!")
    
    print("\n✓ Concluído!")


if __name__ == "__main__":
    main()
