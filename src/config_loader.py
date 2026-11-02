"""
Módulo de carregamento e validação de configurações do Componente I5.
Gerencia a leitura do arquivo YAML com fallback seguro e fusão de parâmetros CLI.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import copy
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "audio": {
        "sample_rate": 16000,
        "chunk_duration_ms": 100,
        "simulate_realtime": False,
    },
    "kafka": {
        "bootstrap_servers": "34.66.235.91:9093",
        "group_id": "i5-transcritor",
        "topic_input": "ods.p5.audio_chunk",
        "auto_offset_reset": "latest",
        "poll_timeout_s": 1.0,
        "enable_auto_commit": True,
    },
    "preprocessor": {
        "apply_noise_filter": True,
        "enable_spectral_subtraction": False,
        "lowcut_hz": 80.0,
        "highcut_hz": 7500.0,
        "filter_order": 5,
    },
    "vad": {
        "speech_threshold": 0.50,
        "negative_threshold_offset": 0.15,
        "silence_threshold_ms": 700,
        "pre_speech_pad_ms": 250,
        "post_speech_pad_ms": 200,
        "min_speech_duration_ms": 250,
        "use_onnx": True,
    },
    "whisper": {
        "model_name": "openai/whisper-small",
        "language": "portuguese",
        "task": "transcribe",
        "device": "auto",
        "torch_dtype": "float16",
    },
    "output": {
        "log_level": "INFO",
        "print_inference_time": True,
        "min_confidence_warning": 0.50,
        "save_json": True,
        "json_dir": "jsons_gerados",
    },
}


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mescla recursivamente dicionários preservando as chaves padrão caso omitidas.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def load_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega o arquivo YAML de configuração. Caso não seja encontrado ou haja chaves
    faltantes, completa automaticamente com os defaults pré-estabelecidos.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    resolved_path: Optional[Path] = None
    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        resolved_path = p
    elif DEFAULT_CONFIG_PATH.exists():
        resolved_path = DEFAULT_CONFIG_PATH

    if resolved_path and resolved_path.exists():
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                user_yaml = yaml.safe_load(f)
                if isinstance(user_yaml, dict):
                    _deep_merge(config, user_yaml)
                    config["_config_file"] = str(resolved_path)
        except Exception as e:
            print(f"[I5 - Config] Aviso ao ler '{resolved_path}': {e}. Usando defaults.")
            config["_config_file"] = None
    else:
        config["_config_file"] = None

    return config
