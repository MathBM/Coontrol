class Constants():
    # Verificar as imagens em ../assets/montagem/

    # Ethernet ----------------------------------------------------------------
    SERVER_IP = "192.168.1.50"  # Não necessário quando usado handle TCP
    SERVER_PORT = 6969          # Não necessário quando usado handle TCP

    SENSOR_FRONT_IP = "192.168.1.10"
    SENSOR_RIGHT_IP = "192.168.1.12"
    SENSOR_LEFT_IP = "192.168.1.11"
    SENSOR_TOP_IP = "192.168.1.13"

    # Scans -------------------------------------------------------------------
    SCANS_DIRECTORY = "./pointcloud/"
    SCAN_DIRECTION = "ccw"
    BUCKET_PATH = "./pointcloud/caixa_vazia"

    # Sensors -----------------------------------------------------------------
    SENSOR_TOP_HEIGHT = 1200

    # [sensor_x_offset, sensor_y_offset, sensor_z_offset]
    SENSOR_RIGHT_TRANSLATION = (350, 1130, 0)
    SENSOR_LEFT_TRANSLATION = (350, -1130, 0)

    # Euler angles [x, y, z]
    SENSOR_RIGHT_ROTATION = (0, 0, 0)
    SENSOR_LEFT_ROTATION = (0, 0, 0)
    SENSOR_TOP_Z_OFFSET = 0  # ajuste fino de posição Z do sensor top (mm); positivo = avança, negativo = recua
    SENSOR_TOP_X_OFFSET = -15   # ajuste fino de posição X do sensor top (mm); positivo = desloca para direita, negativo = para esquerda

    # Recorte lateral do xyz_top, aplicado após o X_OFFSET, para tirar as paredes
    # que o sensor top enxerga de cima (silhueta das faces internas) e deixar só o
    # chão/carga. O sensor top, varrendo de cima, projeta a face vertical de cada
    # parede num pico denso de pontos no mesmo X (~±180mm do centro); o chão fica
    # no interior. Mantém-se apenas |x - X_OFFSET| < HALF_WIDTH.
    # É RELATIVO ao X_OFFSET de propósito: o X_OFFSET translada o top inteiro (chão
    # E as paredes que o top vê), então centrar o recorte nele faz o recorte
    # acompanhar o alinhamento manual em vez de descalibrar quando se ajusta o offset.
    # Assim a parede passa a vir só dos sensores left/right e o X_OFFSET desloca
    # exclusivamente o chão/carga.
    # Não usar recorte em Y: parede e carga sobem juntas em altura, separá-las por
    # Y apagaria a carga nos scans carregados. Reduzir se ainda sobrar parede;
    # aumentar se estiver comendo borda do chão (paredes começam em ~±150-180mm).
    SENSOR_TOP_FLOOR_HALF_WIDTH = 170

    BOUNDARIES_PROFILE_X_MIN = 100
    BOUNDARIES_PROFILE_X_MAX = 1000
    # Janela lateral (eixo Y intermediário, vira X mundial após rotação -90°).
    # Precisa ser MAIOR que o offset dos sensores laterais (SENSOR_*_TRANSLATION Y=±1130),
    # senão os sensores left/right — que capturam as PAREDES da caçamba — são 100% clipados
    # e sobra só o sensor top (que não enxerga as paredes quando há carga ocluindo).
    BOUNDARIES_PROFILE_Y_MIN = -1000
    BOUNDARIES_PROFILE_Y_MAX = 1000


    BOUNDARIES_ZAXIS_X_MIN = -2200
    BOUNDARIES_ZAXIS_X_MAX = -400

    BOUNDARIES_ZAXIS_Y_MIN = 3000  # distância mínima do sensor para o chão/esteira (clip longe)
    BOUNDARIES_ZAXIS_Y_MAX = 100   # distância máxima próxima ao sensor (clip perto, filtra ruído)


