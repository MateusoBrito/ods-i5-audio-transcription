#!/usr/bin/env python3
"""
Produtor Mock do Serviço P5 para testes locais e validação do fluxo Kafka (Componente I5).
Lê um arquivo de áudio, divide em fatias e emite eventos canônicos no tópico ods.p5.audio_chunk.
"""

import argparse
import base64
import datetime
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from confluent_kafka import Producer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_pipeline_config


def publish_mock_p5_audio(
    audio_path: str,
    bootstrap_servers: str = "34.66.235.91:9093",
    topic: str = "ods.p5.audio_chunk",
    chunk_ms: int = 100,
    simulate_realtime: bool = True,
    max_chunks: Optional[int] = None,
):
    """
    Publica chunks de áudio de teste no Kafka simulando o hardware do P5.
    """
    file_p = Path(audio_path)
    if not file_p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")

    print(f"[P5 Mock] Lendo e reamostrando áudio de '{file_p.name}' a 16kHz mono...")
    data, sr = librosa.load(str(file_p), sr=16000, mono=True)
    total_samples = len(data)
    chunk_samples = int(sr * (chunk_ms / 1000.0))

    conf = {
        "bootstrap.servers": bootstrap_servers,
        "client.id": "mock-p5-producer",
    }
    print(f"[P5 Mock] Conectando ao broker Kafka: {bootstrap_servers}...")
    producer = Producer(conf)

    print(f"[P5 Mock] Transmitindo para tópico '{topic}' (chunks de {chunk_ms}ms)...")
    chunk_idx = 0
    start_capture_dt = datetime.datetime.now(datetime.timezone.utc)

    try:
        for offset in range(0, total_samples, chunk_samples):
            chunk = data[offset : offset + chunk_samples]
            if len(chunk) == 0:
                break

            # Calcula o timestamp de captura exato deste chunk (tempo real ou relativo)
            chunk_time_offset = offset / sr
            captured_dt = start_capture_dt + datetime.timedelta(seconds=chunk_time_offset)
            captured_at_iso = captured_dt.isoformat()

            # Converte float32 [-1.0, 1.0] para PCM 16-bit com sinal
            int16_chunk = (chunk * 32767.0).astype(np.int16)
            b64_audio = base64.b64encode(int16_chunk.tobytes()).decode("utf-8")

            # Monta o envelope canônico B1/B2
            evento = {
                "message_id": str(uuid.uuid4()),
                "message_type": "event",
                "schema": "ods.p5.audio_chunk",
                "schema_version": "1.0",
                "producer": "P5",
                "occurred_at": captured_at_iso,
                "payload": {
                    "captured_at": captured_at_iso,
                    "audio_data": b64_audio,
                    "sample_rate": sr,
                    "dtype": "int16",
                    "channels": 1,
                    "chunk_index": chunk_idx,
                    "num_samples": len(chunk),
                },
            }

            json_bytes = json.dumps(evento, ensure_ascii=False).encode("utf-8")
            producer.produce(topic=topic, value=json_bytes)
            producer.poll(0)

            chunk_idx += 1
            if max_chunks and chunk_idx >= max_chunks:
                break

            if simulate_realtime:
                time.sleep(len(chunk) / sr)

        producer.flush(timeout=5)
        print(f"[P5 Mock] Envio concluído com sucesso: {chunk_idx} chunks publicados em '{topic}'.")

    except KeyboardInterrupt:
        print("\n[P5 Mock] Envio interrompido pelo usuário.")
        producer.flush(timeout=2)


def main():
    parser = argparse.ArgumentParser(description="Produtor Mock P5 - Transmissão de Áudio para o Kafka")
    parser.add_argument(
        "--file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "audio6.ogg"),
        help="Caminho do arquivo de áudio para transmitir",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Arquivo YAML de configuração",
    )
    parser.add_argument(
        "--broker",
        type=str,
        default=None,
        help="Endereço do broker Kafka (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Tópico de destino (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Publica na velocidade máxima sem simular pausas reais",
    )

    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    kafka_cfg = config.get("kafka", {})

    broker = args.broker or kafka_cfg.get("bootstrap_servers", "34.66.235.91:9093")
    topic = args.topic or kafka_cfg.get("topic_input", "ods.p5.audio_chunk")

    publish_mock_p5_audio(
        audio_path=args.file,
        bootstrap_servers=broker,
        topic=topic,
        simulate_realtime=not args.no_realtime,
    )


if __name__ == "__main__":
    main()

