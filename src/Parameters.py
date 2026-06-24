class Parameters():
  # Registration algorithm ----------------------------------------------------------------
  class Registration():
    VOXEL_SIZE = 30          # ~4× mediana NN (7.6mm): 30mm → normal_radius=60mm, feature_radius=150mm; melhora discriminação Z em paredes repetitivas
    MAX_NN_NORMALS = 40
    MAX_NN_FPFH = 150
    CONFIDENCE = 0.999
    MAX_ITERATION_RANSAC = 4000000
    EPSILON = 1e-6
    MAX_ITERATION_ICP = 200
    RANSAC_LOOP_SIZE = 30    # mais tentativas estocásticas para escapar do mínimo local identidade
    # Limites absolutos de coordenadas para pré-processamento do RANSAC
    # Descarta paredes/teto do galpão antes de calcular features FPFH.
    # Baseado nas posições dos sensores (SENSOR_*_TRANSLATION ≈ ±1130mm em X; SENSOR_TOP_HEIGHT=2400mm em Y).
    CROP_X_MIN = -1150  # mm — exclui parede lateral esquerda do galpão
    CROP_X_MAX =  1150  # mm — exclui parede lateral direita do galpão
    CROP_Y_MIN =  300   # mm — exclui piso do galpão abaixo da caçamba
    CROP_Y_MAX =  1600  # mm — exclui teto do galpão (sensor top está em 2400mm)
  
  # Bucket point removal algorithm --------------------------------------------------------
  class BucketRemoval():
    THRESHOLD_DISTANCE = 20  # ~2.6× mediana NN (7.6mm): absorve erro residual de alinhamento
    NB_NEIGHBORS = 20
    STD_RATIO = 5.0          # permissivo: necessário pela densidade heterogênea (mediana 7 pts/linha)
    NB_POINTS = 5            # mínimo seguro para regiões esparsas (linhas com poucos pontos)
    RADIUS = 50.0            # cobre P90 do espaçamento Z (20mm) com margem; marginal em linhas esparsas
    DBSCAN_EPS = 49.619      # conecta 3-4 linhas consecutivas (Z spacing P75=13mm); coerente com RADIUS
    DBSCAN_MIN_SAMPLES = 7

  # Load and bucket points merge algorithm ----------------------------------------------
  class MergePoints():
    RAY_CAST_ORIGIN_X = 11.5
    RAY_CAST_ORIGIN_Y = 1000
    RAY_CAST_ORIGIN_Z = -1800
    SIMPLE_MESH_RADIUS = 25  # ~3× mediana NN (7.6mm): conecta pontos dentro do perfil
    SIMPLE_MESH_MAX_NN = 60
    SIMPLE_MESH_K = 15
    NB_NEIGHBORS = 20
    STD_RATIO = 12

  class MergePointsLegacy():
    DISTANCE_THRESHOLD = 120 
    DETECTION_THRESHOLD = 20
    ANGULAR_STEP = 25
    SLOPE = 500
    NB_NEIGHBORS = 10
    STD_RATIO = 20
    
  class MeshReconstruction():
    ALPHA = 100  # Alpha Shapes: 80-120 (menor = mais detalhes, tenta fechar malha)
    N_FILTER_ITERATIONS = 8  # Suavização: 5-10
    # Parâmetros Poisson (método recomendado)
    POISSON_DEPTH = 8   # depth=8 → ~12mm para caixa 3000mm, rápido com 40-60k pts
    DENSITY_QUANTILE = 0.1  # Remove 10% de vértices de menor suporte → elimina balões em buracos/bordas

  class VolumeCalculation():
    HEIGHTMAP_CELL_SIZE = 20.0  # mm — ≥ P90 do espaçamento Z real (20mm) garante célula preenchida mesmo em regiões esparsas

