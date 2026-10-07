# Plano de Implementação: Configuração Declarativa via YAML (Componente I5)

Este documento descreve o plano para externalizar todos os parâmetros de ajuste do pipeline de áudio, VAD e Whisper em um arquivo de configuração `YAML` (`config/pipeline_config.yaml`). Isso permitirá aos pesquisadores e desenvolvedores calibrar o componente para diferentes cenários de ruído e hardware (ex: Jetson Nano vs Desktop) sem alterar uma única linha de código Python.

---

## User Review Required

> [!IMPORTANT]
> **Hierarquia de Parâmetros:** A regra padrão proposta para valores de configuração é:
> **Linha de Comando (CLI)** > **Arquivo YAML** > **Valores Padrão no Código**.
> Ou seja, valores definidos no YAML serão carregados por padrão, mas qualquer flag passada via terminal (ex: `--threshold 0.65`) terá prioridade e sobrescreverá o valor do YAML para aquela execução.

---

## Open Questions

Nenhuma dúvida bloqueante no momento. Os valores padrões recomendados serão calibrados para o cenário da UFSJ/ODS (português, 16kHz, inferência CUDA/CPU automática).

---

## Proposed Changes

```
i5/
├── config/
│   └── pipeline_config.yaml         # [NEW] Arquivo central de parametrização
├── src/
│   ├── config_loader.py             # [NEW] Módulo tipado para carga e validação do YAML
│   ├── TranscriptionWhisper.py      # [MODIFY] Suporte a parâmetros vindos da config
│   └── pipeline.py                  # [MODIFY] Integração com config_loader e flag --config
└── requirements.txt                 # [MODIFY] Inclusão explícita de pyyaml
```

---

### Componente 1: Arquivo de Configuração YAML

#### [NEW] `config/pipeline_config.yaml`
Estrutura hierárquica e comentada para facilitar a calibração por qualquer membro da equipe:

```yaml
# =====================================================================
# Configurações do Componente I5 (Áudio, VAD e Transcrição)
# =====================================================================

# Parâmetros de áudio e streaming (Ingestão / P5 Mock)
audio:
  sample_rate: 16000              # Taxa padrão exigida pelo Silero VAD e Whisper (Hz)
  chunk_duration_ms: 100          # Tamanho dos blocos de streaming (ms)
  simulate_realtime: false        # Se verdadeiro, insere pausas proporcionais de microfone

# Front-End de Áudio na CPU (Filtros e Normalização)
preprocessor:
  apply_noise_filter: true        # Aplica filtro passa-banda
  enable_spectral_subtraction: false # Subtração espectral (desativada para blocos curtos)
  lowcut_hz: 80.0                 # Remove ruído subgrave, DC offset e vento
  highcut_hz: 7500.0              # Limite superior da fala humana
  filter_order: 5                 # Ordem do filtro Butterworth

# Voice Activity Detection (Silero VAD na CPU via ONNX)
vad:
  speech_threshold: 0.50          # Limiar de ativação de fala (0.0 a 1.0)
  negative_threshold_offset: 0.15 # Histerese para encerramento de locução
  silence_threshold_ms: 700       # Silêncio consecutivo para fechar o batch (ms)
  pre_speech_pad_ms: 250          # Margem de áudio antes do início da fala (ms)
  post_speech_pad_ms: 200         # Margem de áudio após término da fala (ms)
  min_speech_duration_ms: 250     # Duração mínima para aceitar o lote (ignora estalos)
  use_onnx: true                  # Otimização CPU com ONNX Runtime

# Modelo ASR (Whisper na GPU)
whisper:
  model_name: "openai/whisper-small" # "tiny", "base", "small", etc.
  language: "portuguese"          # Idioma forçado para transcrição
  task: "transcribe"              # "transcribe" (não traduzir)
  device: "auto"                  # "auto" (detecta CUDA/CPU), "cuda" ou "cpu"
  torch_dtype: "float16"          # "float16" (GPU) ou "float32" (CPU)

# Políticas de Publicação e Logs (Camada 4 / A7)
output:
  log_level: "INFO"
  print_inference_time: true
  min_confidence_warning: 0.50    # Alerta se confiança for menor que esse limiar
```

---

### Componente 2: Módulo de Carga e Validação

#### [NEW] `src/config_loader.py`
Carregador resiliente com fallback inteligente:
* Se o arquivo `pipeline_config.yaml` não existir ou alguma chave estiver ausente, mescla automaticamente com dicionário de valores padrão seguros.
* Converte caminhos relativos para absolutos referenciados na raiz do projeto.

---

### Componente 3: Atualizações no Pipeline e Transcritor

#### [MODIFY] `src/TranscriptionWhisper.py`
* Adaptar o construtor `__init__` para aceitar `language`, `task`, `torch_dtype` e `device="auto"`, eliminando valores *hardcoded*.

#### [MODIFY] `src/pipeline.py`
* Adicionar suporte ao argumento `--config config/pipeline_config.yaml`.
* Instanciar `AudioPreprocessor`, `VADSegmenter` e `TranscritorWhisper` consumindo o objeto de configuração carregado.

#### [MODIFY] `requirements.txt`
* Adicionar `pyyaml>=6.0.0` para rastreamento formal das dependências do ambiente.

---

## Verification Plan

### Testes Automatizados e Execuções de Linha de Comando

1. **Validação da Carga Padrão:**
   ```bash
   python3 src/pipeline.py data/audio6.ogg
   ```
   *Verificar se carrega automaticamente o arquivo `config/pipeline_config.yaml` e transcreve em português.*

2. **Validação de Sobrescrita via CLI:**
   ```bash
   python3 src/pipeline.py data/audio6.ogg --threshold 0.65
   ```
   *Verificar se o valor `0.65` substitui o `0.50` do YAML apenas durante essa execução.*

3. **Validação de Teste com Arquivo YAML Customizado:**
   Criar um arquivo temporário `config/test_config.yaml` com parâmetros diferentes (ex: `silence_threshold_ms: 1200`) e rodar:
   ```bash
   python3 src/pipeline.py data/audio5.ogg --config config/test_config.yaml
   ```

### Verificação Manual pelo Usuário
* O usuário poderá alterar livremente valores em `config/pipeline_config.yaml` (como trocar o `model_name` para `"openai/whisper-tiny"`, ou aumentar `speech_threshold` para `0.65`) e conferir o comportamento imediatamente no terminal.

