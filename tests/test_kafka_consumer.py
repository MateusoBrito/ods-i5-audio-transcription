"""
Suíte de testes unitários para o consumidor Kafka do Componente I5 (ods.p5.audio_chunk).
Valida envelope B1/B2, decodificação PCM/Base64 e cálculo temporal com captured_at.
"""

import base64
import datetime
import json
import sys
from pathlib import Path
import uuid
import numpy as np

# Garante inclusão da raiz do projeto no PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_pipeline_config
from src.kafka_audio_consumer import KafkaAudioConsumer, CAMPOS_OBRIGATORIOS_ENVELOPE, SCHEMA_ESPERADO
from src.vad_segmenter import VADSegmenter


def test_kafka_config_loading():
    """Valida se a seção kafka é carregada corretamente do YAML."""
    config = load_pipeline_config()
    assert "kafka" in config
    kafka_cfg = config["kafka"]
    assert kafka_cfg["bootstrap_servers"] == "34.66.235.91:9093"
    assert kafka_cfg["group_id"] == "i5-transcritor"
    assert kafka_cfg["topic_input"] == "ods.p5.audio_chunk"


def test_validar_envelope_valido():
    """Valida aceitação de um envelope canônico válido conforme o Guia B2."""
    consumer = KafkaAudioConsumer.__new__(KafkaAudioConsumer)
    consumer.target_sample_rate = 16000

    evento_valido = {
        "message_id": str(uuid.uuid4()),
        "message_type": "event",
        "schema": "ods.p5.audio_chunk",
        "schema_version": "1.0",
        "producer": "P5",
        "occurred_at": "2026-10-21T12:00:00.000Z",
        "payload": {
            "captured_at": "2026-10-21T12:00:00.000Z",
            "audio_data": "AAABAA==",
            "sample_rate": 16000,
        },
    }

    valido, motivo = consumer.validar_envelope(evento_valido)
    assert valido is True
    assert motivo is None


def test_validar_envelope_campos_ausentes():
    """Valida rejeição de mensagens com campos obrigatórios ausentes."""
    consumer = KafkaAudioConsumer.__new__(KafkaAudioConsumer)

    evento_incompleto = {
        "message_id": str(uuid.uuid4()),
        "schema": "ods.p5.audio_chunk",
        "payload": {"audio_data": "abc"},
    }

    valido, motivo = consumer.validar_envelope(evento_incompleto)
    assert valido is False
    assert "Campo obrigatório ausente" in motivo


def test_validar_envelope_schema_invalido():
    """Valida rejeição de mensagens com schema diferente de ods.p5.audio_chunk."""
    consumer = KafkaAudioConsumer.__new__(KafkaAudioConsumer)

    evento_errado = {
        "message_id": str(uuid.uuid4()),
        "message_type": "event",
        "schema": "ods.outro.evento",
        "schema_version": "1.0",
        "producer": "Outro",
        "occurred_at": "2026-10-21T12:00:00.000Z",
        "payload": {"audio_data": "abc"},
    }

    valido, motivo = consumer.validar_envelope(evento_errado)
    assert valido is False
    assert "Schema inesperado" in motivo


def test_decodificar_audio_base64_int16():
    """Valida a decodificação de PCM 16-bit codificado em Base64 para NumPy float32."""
    consumer = KafkaAudioConsumer.__new__(KafkaAudioConsumer)
    consumer.target_sample_rate = 16000

    # Cria onda senoidal simples PCM 16-bit
    samples = 1600
    t = np.linspace(0, 0.1, samples, endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 32767.0).astype(np.int16)
    b64_audio = base64.b64encode(sine.tobytes()).decode("utf-8")

    payload = {
        "audio_data": b64_audio,
        "sample_rate": 16000,
        "dtype": "int16",
    }

    arr, sr = consumer.decodificar_audio(payload)
    assert sr == 16000
    assert len(arr) == samples
    assert arr.dtype == np.float32
    assert -1.0 <= arr.min() and arr.max() <= 1.0


def test_propagacao_captured_at_para_start_at():
    """Valida que o captured_at injetado no VAD é propagado com precisão para o start_at do AudioBatch."""
    import librosa
    from pathlib import Path

    segmenter = VADSegmenter(sample_rate=16000, speech_threshold=0.5, silence_threshold_ms=700)

    audio_file = Path("data/audio6.ogg")
    if not audio_file.exists():
        pytest.skip("audio6.ogg não encontrado")

    data, sr = librosa.load(str(audio_file), sr=16000, mono=True)
    base_captured_iso = "2026-10-21T15:30:00.000000+00:00"

    # Envia o áudio em chunks de 100ms
    chunk_size = 1600
    todos = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        b = segmenter.push_audio(chunk, captured_at=base_captured_iso)
        if b:
            todos.extend(b)

    final_b = segmenter.flush()
    if final_b:
        todos.append(final_b)

    assert len(todos) >= 1
    first_batch = todos[0]
    assert first_batch.start_at is not None
    assert "2026-10-21T15:30:0" in first_batch.start_at
    assert first_batch.end_at is not None
