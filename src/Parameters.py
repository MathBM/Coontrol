class Parameters():
  # Registration algorithm ----------------------------------------------------------------
  class Registration():
    VOXEL_SIZE = 20  # Reduzido para mais detalhes no alinhamento
    MAX_NN_NORMALS = 40  # Mais vizinhos para normais mais estáveis
    MAX_NN_FPFH = 150  # Mais features para matching
    CONFIDENCE = 0.999  # Slightly lower confidence
    MAX_ITERATION_RANSAC = 2000000  # Mais iterações
    EPSILON = 1e-6
    MAX_ITERATION_ICP = 100  # Mais iterações ICP
    RANSAC_LOOP_SIZE = 10  # Mais tentativas de RANSAC
  
  # Bucket point removal algorithm --------------------------------------------------------
  class BucketRemoval():
    THRESHOLD_DISTANCE = 20
    NB_NEIGHBORS = 20       # Local o suficiente para formas complexas (density 8mm)
    STD_RATIO = 5.0         # Menos agressivo: preserva vales/selas entre picos
    NB_POINTS = 5           # Mínimo de vizinhos no raio (fácil satisfazer a 8mm)
    RADIUS = 50.0           # ~6x densidade: elimina outliers reais sem erodir bordas
    DBSCAN_EPS = 49.619
    DBSCAN_MIN_SAMPLES = 7

  # Load and bucket points merge algorithm ----------------------------------------------
  class MergePoints():
    RAY_CAST_ORIGIN_X = 11.5
    RAY_CAST_ORIGIN_Y = 1000
    RAY_CAST_ORIGIN_Z = -1800
    # Parâmetros otimizados para densidade uniforme 8mm (rampa + caçamba)
    SIMPLE_MESH_RADIUS = 25  # ~3x densidade (24mm)
    SIMPLE_MESH_MAX_NN = 60  # Mais vizinhos para densidade 8mm
    SIMPLE_MESH_K = 15  # Orientação de normais estável
    NB_NEIGHBORS = 20  # Filtro robusto para densidade 8mm
    STD_RATIO = 12  # Ajustado para densidade uniforme

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
    DENSITY_QUANTILE = 0.0  # 0.0 = não remove vértices (preserva volume)
