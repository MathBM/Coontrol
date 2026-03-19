class Parameters():
  # Registration algorithm ----------------------------------------------------------------
  class Registration():
    VOXEL_SIZE = 30
    MAX_NN_NORMALS = 30
    MAX_NN_FPFH = 100
    CONFIDENCE = 1.0
    MAX_ITERATION_RANSAC = 1000000
    EPSILON = 1e-6
    MAX_ITERATION_ICP = 50
    RANSAC_LOOP_SIZE = 5
  
  # Bucket point removal algorithm --------------------------------------------------------
  class BucketRemoval():
    THRESHOLD_DISTANCE = 20
    NB_NEIGHBORS = 40
    STD_RATIO = 2.5  # Melhor resultado anterior
    NB_POINTS = 50
    RADIUS = 250.0
    DBSCAN_EPS = 49.619
    DBSCAN_MIN_SAMPLES = 7

  # Load and bucket points merge algorithm ----------------------------------------------
  class MergePoints():
    RAY_CAST_ORIGIN_X = 11.5
    RAY_CAST_ORIGIN_Y = 1000
    RAY_CAST_ORIGIN_Z = -1800
    SIMPLE_MESH_RADIUS = 100
    SIMPLE_MESH_MAX_NN = 100
    SIMPLE_MESH_K = 5
    NB_NEIGHBORS = 10
    STD_RATIO = 20

  class MergePointsLegacy():
    DISTANCE_THRESHOLD = 120 
    DETECTION_THRESHOLD = 20
    ANGULAR_STEP = 25
    SLOPE = 500
    NB_NEIGHBORS = 10
    STD_RATIO = 20
    
  class MeshReconstruction():
    ALPHA = 150  # Valor original
    N_FILTER_ITERATIONS = 5
