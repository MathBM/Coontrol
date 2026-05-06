<img src="./assets/images/splash.png" alt="COONTROL-UFSC" />

# Coontrol

Sistema de medição de volume de carga em caçamba de caminhão usando 4 sensores LIDAR Pepperl+Fuchs R2000.

---

## Requisitos

- Python 3.10+
- Rust (para compilar o cliente TCP)
- Dependências Python:

```bash
pip install -r requirements.txt
```

- Compilar o cliente TCP (necessário antes da primeira coleta real):

```bash
cd rust && cargo build --release && cd ..
```

> O binário gerado em `rust/target/release/client_tcp` é chamado diretamente pelo `ScanManager` ao iniciar uma coleta. O `cargo build --release` dentro da pasta `rust/` é suficiente — nenhum `install` adicional é necessário.
>
> **Windows:** o compilador Rust gera `rust\target\release\client_tcp.exe`. O caminho no `src/ScanManager.py` está hardcoded para Linux (`./rust/target/release/client_tcp`). Caso utilize Windows, altere a linha correspondente em `ScanManager.py`:
>
> ```python
> rust_bin = "./rust/target/release/client_tcp.exe"
> ```

---

## Configuração de rede

Os sensores estão em uma subnet Ethernet privada `192.168.1.x`. Configure a interface de rede da máquina com um IP estático nessa faixa (ex.: `192.168.1.50`).

IPs definidos em `src/Constants.py`:

| Sensor | IP           |
| ------ | ------------ |
| Front  | 192.168.1.10 |
| Right  | 192.168.1.11 |
| Left   | 192.168.1.12 |
| Top    | 192.168.1.13 |

**Validar conexão antes de usar:** teste a conectividade com cada sensor via `ping`:

```bash
ping -c 3 192.168.1.10
ping -c 3 192.168.1.11
ping -c 3 192.168.1.12
ping -c 3 192.168.1.13
```

Se algum sensor não responder ao ping, verifique o cabo Ethernet, o IP estático configurado na interface de rede da máquina e o IP do sensor.

---

## Fluxo de uso

### 1. Iniciar a interface

```bash
python main.py
```

### 2. Coletar as pointclouds necessárias

São necessárias **duas medições**:

**Medição da caçamba vazia (referência):**

1. Posicione o caminhão com a caçamba vazia sob os sensores
2. Clique em **Start Scan** na interface
3. Aguarde alguns segundos de coleta
4. Clique em **Stop Scan**
5. Localize a pasta gerada em `pointcloud/` (nome com timestamp)
6. **Renomeie manualmente** a pasta para `caixa_vazia`:
   ```
   pointcloud/caixa_vazia/
   ```
   Esta pasta é o ponto de referência para todos os cálculos de volume.

**Medição da carga:**

1. Posicione o caminhão com a carga sob os sensores
2. Clique em **Start Scan**
3. Aguarde a coleta
4. Clique em **Stop Scan**
5. A pasta com timestamp é criada automaticamente em `pointcloud/`

### 3. Gerar o data.npz (reconstrução 3D)

Para cada pasta de pointcloud (caixa_vazia e carga):

1. Selecione a pasta na interface do `main.py`
2. Clique em **Process Data**

Isso executa o `PointCloudReconstructor`, que lê os 4 arquivos `.bin` dos sensores e gera o arquivo `data.npz` com a nuvem de pontos 3D reconstruída. O `data.npz` é necessário para todas as etapas seguintes.

Estrutura esperada após o processo:

```
pointcloud/
  caixa_vazia/
    192.168.1.10.bin
    192.168.1.11.bin
    192.168.1.12.bin
    192.168.1.13.bin
    data.npz          ← gerado pelo Process Data
  2026-04-08_09h02min40s/
    192.168.1.10.bin
    ...
    data.npz          ← gerado pelo Process Data
```

### 4. Visualizar e depurar o pipeline

Use o `debug_pipeline.py` para visualizar a nuvem de pontos 3D, acompanhar cada etapa do algoritmo e verificar o volume calculado:

```bash
python debug_pipeline.py <nome_da_pasta>
```

Exemplo:

```bash
python debug_pipeline.py 2026-04-08_09h02min40s
```

O script executa e exibe interativamente:

1. **Dados originais** — carga (vermelho) + caçamba vazia (verde) sobrepostos
2. **Alinhamento** — resultado do RANSAC + ICP
3. **Isolamento da carga** — pontos removidos da caçamba, restando apenas o material
4. **Volume final** — calculado pelo heightmap integral (em mm³ e litros)

> O `debug_pipeline.py` reconstrói o `data.npz` automaticamente caso ele não exista na pasta indicada.

---

## Teste rápido com sensor individual

Para verificar se um sensor específico está entregando dados corretamente:

```bash
python test_front_sensor_live.py
```

Exibe em tempo real o perfil 2D do sensor frontal e o valor de distância detectado dentro da janela de interesse, permitindo validar o funcionamento e ajustar os limites em `Constants.py` (`BOUNDARIES_ZAXIS_*`).
