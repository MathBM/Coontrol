# Integração de Dados Sintéticos - Guia de Uso

## 📋 Visão Geral

Os dados sintéticos foram **totalmente integrados** na interface principal do projeto. Agora você pode criar e processar scans sintéticos diretamente pela interface gráfica, sem precisar do sensor LIDAR real!

## 🚀 Como Usar

### Método 1: Interface Gráfica (Recomendado)

1. **Inicie a aplicação**:

   ```bash
   source venv/bin/activate
   python main.py
   ```

2. **Crie um scan sintético**:
   - Clique no botão verde **"Create Synthetic Scan"**
   - Escolha o tipo de rampa:
     - **Linear**: Rampa reta tradicional
     - **Stepped**: Escada com degraus
     - **Concave**: Rampa curva côncava
     - **Convex**: Rampa curva convexa
   - Clique em OK

3. **Processe o scan**:
   - O scan sintético aparecerá automaticamente na tabela com sufixo `_SYNTHETIC`
   - Selecione o scan na tabela
   - Clique em **"Process Data"**
   - O volume será calculado e exibido na coluna "Volume"

### Método 2: Via Código Python

```python
from src.SyntheticScanCreator import SyntheticScanCreator

# Criar instância
creator = SyntheticScanCreator()

# Criar scan sintético rápido com parâmetros padrão
scan_path = creator.create_quick_test_scan()

# Ou criar com parâmetros personalizados
scan_path = creator.create_synthetic_scan(
    ramp_type="concave",      # linear, stepped, concave, convex
    width=2000,               # mm
    length=3000,              # mm
    height=800,               # mm
    point_density=8,          # mm (menor = mais denso)
    noise_level=3.0,          # mm
    custom_name="meu_teste"   # nome personalizado (opcional)
)

print(f"Scan criado em: {scan_path}")
```

### Método 3: Script de Teste Standalone

```bash
source venv/bin/activate
python test_synthetic_data.py
```

Escolha uma das opções:

- **1**: Geração e visualização básica
- **2**: Teste de filtros
- **3**: Gerar diferentes tipos de rampas
- **4**: Carregar dados salvos
- **0**: Executar todos os testes

## 📁 Estrutura de Arquivos

Quando você cria um scan sintético, os seguintes arquivos são gerados:

```
pointcloud/
└── 2026-03-18_15h30min45s_SYNTHETIC_linear/
    ├── data.npz              # Nuvem de pontos (formato compatível)
    └── SYNTHETIC_INFO.txt    # Metadados do scan
```

O arquivo `data.npz` está no **formato exato** esperado pelo `DataManager`, permitindo que o processamento funcione sem modificações no código original.

## 🔧 Parâmetros Customizáveis

### Tipos de Rampa

#### 1. Linear (Rampa Reta)

```python
creator.create_synthetic_scan(
    ramp_type="linear",
    width=2000,        # largura
    length=3000,       # comprimento
    height=800,        # altura
    add_ground=True    # incluir plano do chão
)
```

#### 2. Stepped (Escada)

```python
creator.create_synthetic_scan(
    ramp_type="stepped",
    width=1500,
    length=3000,
    height=720,
    num_steps=6,                    # número de degraus
    step_length=500,                # comprimento de cada degrau
    step_height=120                 # altura de cada degrau
)
```

#### 3. Concave/Convex (Curva)

```python
creator.create_synthetic_scan(
    ramp_type="concave",  # ou "convex"
    width=1600,
    length=2800,
    max_height=700
)
```

### Parâmetros Gerais

- **width**: Largura em milímetros (padrão: 2000)
- **length**: Comprimento em milímetros (padrão: 3000)
- **height**: Altura em milímetros (padrão: 800)
- **point_density**: Densidade de pontos em mm - quanto menor, mais denso (padrão: 8)
- **noise_level**: Ruído em mm para simular imprecisão do sensor (padrão: 3.0)

## 🎯 Fluxo Completo de Teste

```python
# 1. Criar scan sintético
from src.SyntheticScanCreator import SyntheticScanCreator
creator = SyntheticScanCreator()
scan_path = creator.create_synthetic_scan(ramp_type="linear")

# 2. Processar na interface
# - Abra a aplicação (python main.py)
# - Selecione o scan com sufixo _SYNTHETIC
# - Clique em "Process Data"

# 3. O DataManager processa automaticamente:
#    ✓ Carrega data.npz
#    ✓ Aplica filtros
#    ✓ Faz registro com caçamba
#    ✓ Reconstrói superfície
#    ✓ Calcula volume
```

## 🔍 Diferenças entre Real e Sintético

| Aspecto       | Scan Real      | Scan Sintético          |
| ------------- | -------------- | ----------------------- |
| Origem        | Sensores LIDAR | Gerador matemático      |
| Formato       | Arquivos .bin  | data.npz direto         |
| Processamento | Idêntico       | Idêntico                |
| Ruído         | Real do sensor | Simulado (configurável) |
| Velocidade    | ~30s captura   | Instantâneo             |

## 💡 Casos de Uso

### 1. Desenvolvimento sem Hardware

Desenvolva e teste o pipeline completo sem acesso aos sensores LIDAR.

### 2. Testes de Regressão

Crie scans sintéticos consistentes para testar mudanças no código:

```python
# Sempre gera a mesma geometria
creator.create_synthetic_scan(
    ramp_type="linear",
    custom_name="regression_test_baseline"
)
```

### 3. Validação de Algoritmos

Teste filtros e reconstrução com geometrias conhecidas:

```python
# Crie vários tipos de teste
creator.create_varied_test_scans()
```

### 4. Demonstrações

Mostre o sistema funcionando sem necessidade de setup de hardware.

## 📝 Arquivos Relacionados

- **src/SyntheticScanCreator.py**: Criador de scans sintéticos (integrado na interface)
- **synthetic_data_generator.py**: Gerador base de geometrias
- **synthetic_adapter.py**: Adaptador para substituir PointCloudReconstructor
- **test_synthetic_data.py**: Script de teste standalone
- **SYNTHETIC_DATA_GUIDE.md**: Guia detalhado do gerador

## ⚙️ Configuração

Os scans sintéticos respeitam as constantes do projeto definidas em `src/Constants.py`:

```python
SCANS_DIRECTORY = "./pointcloud/"  # Local onde scans são salvos
BUCKET_PATH = "./pointcloud/caixa_vazia"  # Caçamba de referência
```

## 🐛 Troubleshooting

### Problema: Erro ao criar scan sintético

**Solução**: Certifique-se de que o diretório `./pointcloud/` existe:

```bash
mkdir -p pointcloud
```

### Problema: Scan não aparece na tabela

**Solução**: Clique em "Refresh Table" após criar o scan.

### Problema: Erro ao processar

**Solução**: Verifique se a caçamba de referência existe em `./pointcloud/caixa_vazia/`.

## 🎓 Próximos Passos

1. **Teste o sistema**: Crie scans de diferentes tipos e processe-os
2. **Compare resultados**: Veja como diferentes geometrias afetam o volume calculado
3. **Ajuste parâmetros**: Experimente com densidade de pontos e ruído
4. **Desenvolva novos recursos**: Use scans sintéticos para testar mudanças no código

## 📞 Suporte

Para mais informações sobre o gerador de dados sintéticos, consulte:

- `SYNTHETIC_DATA_GUIDE.md` - Guia detalhado do gerador
- `test_synthetic_data.py` - Exemplos de uso

---

**Criado por**: Sistema de Integração de Dados Sintéticos  
**Data**: Março 2026  
**Versão**: 1.0
