class Parameters():
  # Registration algorithm ----------------------------------------------------------------
  class Registration():
    VOXEL_SIZE = 20          # recalibrado p/ nova config: nuvem mais densa (mediana NN ~5mm vazia / ~3.5mm carga) e objeto pequeno (~550mm). 20mm → normal_radius=40mm, feature_radius=100mm (~18% da largura)
    MAX_NN_NORMALS = 40
    MAX_NN_FPFH = 150
    CONFIDENCE = 0.999
    MAX_ITERATION_RANSAC = 4000000
    EPSILON = 1e-6
    MAX_ITERATION_ICP = 200
    RANSAC_LOOP_SIZE = 30    # mais tentativas estocásticas para escapar do mínimo local identidade
    # NOTA: CROP_* são usados por PointCloudReconstructor.create_point_cloud (recorte
    # mundial das 3 nuvens), NÃO por align_truck_bucket_and_load (que roda sem recorte).
    # Geometria atual: sensor top em 1200mm, caixa ~±280mm em X, piso ~656-688mm em Y.
    CROP_X_MIN = -400   # mm
    CROP_X_MAX =  400   # mm
    # CROP_Y_MIN sobe até logo abaixo do piso (piso do top ~656mm). Abaixo disso os
    # sensores left/right registram só reflexões/ruído sub-piso (faixa Y 250-550 que
    # aparecia embaixo da carga) — nada real da caçamba existe abaixo do piso.
    CROP_Y_MIN =  600   # mm — remove ruído sub-piso; piso real fica acima
    CROP_Y_MAX =  1150  # mm — sensor top agora em 1200mm
  
  # Bucket point removal algorithm --------------------------------------------------------
  class BucketRemoval():
    THRESHOLD_DISTANCE = 20  # absorve erro residual de alinhamento (mediana NN agora ~3.5-5.8mm); mantido em 20 — thr maior (30) removeu carga junto nos testes
    NB_NEIGHBORS = 20
    STD_RATIO = 5.0          # permissivo: necessário pela densidade heterogênea (P10 ~15 pts/linha)
    NB_POINTS = 5            # mínimo seguro para regiões esparsas (linhas com poucos pontos)
    RADIUS = 30.0            # recalibrado: P90 do espaçamento Z caiu de ~20mm p/ ~5-8mm; 30mm cobre com folga e mantém vizinhança densa
    DBSCAN_EPS = 30.0        # recalibrado: P75 do espaçamento Z caiu de ~13mm p/ ~4-6mm; 30mm conecta ~5 linhas sem fragmentar a carga
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
    HEIGHTMAP_CELL_SIZE = 10.0  # mm — recalibrado: P90 do espaçamento Z caiu de ~20mm p/ ~5-8mm; célula 10mm dá mais resolução lateral mantendo células preenchidas

