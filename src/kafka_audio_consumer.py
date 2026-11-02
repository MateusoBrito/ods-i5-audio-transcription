"""
Módulo consumidor do barramento Kafka para o Componente I5.
Consome o fluxo de áudio contínuo emitido por P5 (ods.p5.audio_chunk),
valida o envelope canônico B1/B2 e extrai a base temporal captured_at.
"""

import base64
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from confluent_kafka import Consumer, KafkaError, KafkaException

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_pipeline_config

logger = logging.getLogger("I5.KafkaConsumer")

# Campos canônicos obrigatórios do envelope B1/B2 (Guia B2, Seção 4.3)
CAMPOS_OBRIGATORIOS_ENVELOPE = [
    "message_id",
    "message_type",
    "schema",
    "schema_version",
    "producer",
    "occurred_at",
    "payload",
]

SCHEMA_ESPERADO = "ods.p5.audio_chunk"
VERSAO_SUPORTADA_PREFIX = "1."


class KafkaAudioConsumer:
    """
    Consumidor Kafka dedicado à ingestão de áudio da Camada 1 (P5) para o componente I5.
    """

    def __init__(
        self,
        bootstrap_servers: str = "34.66.235.91:9093",
        group_id: str = "i5-transcritor",
        topic: str = "ods.p5.audio_chunk",
        auto_offset_reset: str = "latest",
        poll_timeout_s: float = 1.0,
        enable_auto_commit: bool = True,
        target_sample_rate: int = 16000,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topic = topic
        self.poll_timeout_s = poll_timeout_s
        self.target_sample_rate = target_sample_rate
        self.running = False

        conf = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": enable_auto_commit,
            # Configurações de resiliência e reconexão responsável (Guia B2, Seção 4.5)
            "reconnect.backoff.ms": 1000,
            "reconnect.backoff.max.ms": 10000,
        }

        logger.info(
            f"Inicializando Kafka Consumer: broker={self.bootstrap_servers}, "
            f"group.id={self.group_id}, topico={self.topic}"
        )
        self.consumer = Consumer(conf)
        self.consumer.subscribe([self.topic])

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Instancia o consumidor a partir da configuração central YAML.
        """
        if config is None:
            config = load_pipeline_config(config_path)

        kafka_cfg = config.get("kafka", {})
        audio_cfg = config.get("audio", {})

        return cls(
            bootstrap_servers=kafka_cfg.get("bootstrap_servers", "34.66.235.91:9093"),
            group_id=kafka_cfg.get("group_id", "i5-transcritor"),
            topic=kafka_cfg.get("topic_input", "ods.p5.audio_chunk"),
            auto_offset_reset=kafka_cfg.get("auto_offset_reset", "latest"),
            poll_timeout_s=float(kafka_cfg.get("poll_timeout_s", 1.0)),
            enable_auto_commit=bool(kafka_cfg.get("enable_auto_commit", True)),
            target_sample_rate=int(audio_cfg.get("sample_rate", 16000)),
        )

    def validar_envelope(self, evento: Any) -> Tuple[bool, Optional[str]]:
        """
        Validação do envelope em 3 níveis conforme Seção 4.3 do Guia B2.
        """
        if not isinstance(evento, dict):
            return False, "Mensagem desserializada não é um dicionário/objeto JSON"

        # 1. Checagem de campos obrigatórios
        for campo in CAMPOS_OBRIGATORIOS_ENVELOPE:
            if campo not in evento:
                return False, f"Campo obrigatório ausente no envelope: '{campo}'"

        # 2. Schema esperado e versão suportada
        schema = evento.get("schema")
        if schema != SCHEMA_ESPERADO:
            return False, f"Schema inesperado: '{schema}' (esperado: '{SCHEMA_ESPERADO}')"

        versao = str(evento.get("schema_version", ""))
        if not versao.startswith(VERSAO_SUPORTADA_PREFIX):
            return False, f"Versão do schema não suportada: '{versao}' (esperada: {VERSAO_SUPORTADA_PREFIX}*)"

        # 3. Validação do conteúdo específico do payload P5
        payload = evento.get("payload")
        if not isinstance(payload, dict):
            return False, "Payload não é um objeto JSON"

        # Verifica se possui dado de áudio em algum formato aceito
        audio_keys = ["audio_data", "audio_bytes", "audio", "data"]
        if not any(k in payload for k in audio_keys):
            return False, f"Payload sem campo de dados de áudio (esperado um de: {audio_keys})"

        return True, None

    def decodificar_audio(self, payload: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """
        Decodifica o chunk de áudio contido no payload para array float32 em [-1.0, 1.0].
        Suporta Base64 (PCM 16-bit ou float32) e lista numérica.
        """
        # Localiza o campo de áudio
        raw_val = None
        for k in ["audio_data", "audio_bytes", "audio", "data"]:
            if k in payload:
                raw_val = payload[k]
                break

        dtype_str = str(payload.get("dtype", "int16")).lower()
        sample_rate = int(payload.get("sample_rate", self.target_sample_rate))

        if isinstance(raw_val, str):
            # String codificada em Base64
            byte_data = base64.b64decode(raw_val)
            if "float32" in dtype_str:
                audio_arr = np.frombuffer(byte_data, dtype=np.float32)
            else:
                # Default PCM 16-bit com sinal
                int16_arr = np.frombuffer(byte_data, dtype=np.int16)
                audio_arr = int16_arr.astype(np.float32) / 32768.0

        elif isinstance(raw_val, (list, tuple)):
            audio_arr = np.asarray(raw_val, dtype=np.float32)
        elif isinstance(raw_val, bytes):
            int16_arr = np.frombuffer(raw_val, dtype=np.int16)
            audio_arr = int16_arr.astype(np.float32) / 32768.0
        else:
            raise ValueError(f"Tipo de dado de áudio não suportado: {type(raw_val)}")

        # Reamostragem se o P5 transmitir em taxa diferente de 16kHz
        if sample_rate != self.target_sample_rate and len(audio_arr) > 0:
            audio_arr = librosa.resample(
                audio_arr, orig_sr=sample_rate, target_sr=self.target_sample_rate
            )
            sample_rate = self.target_sample_rate

        return audio_arr.astype(np.float32), sample_rate

    def extrair_metadados_tempo(self, evento: Dict[str, Any]) -> str:
        """
        Extrai o timestamp de captura original (captured_at) com fallback para occurred_at.
        """
        payload = evento.get("payload", {})
        captured_at = payload.get("captured_at") or evento.get("occurred_at")
        return str(captured_at)

    def listen_and_feed(self, pipeline, max_messages: Optional[int] = None):
        """
        Laço de escuta do Kafka que alimenta o pipeline em tempo real com preservação de captured_at.
        """
        self.running = True
        logger.info(f"[I5 - Kafka] Iniciando escuta no tópico '{self.topic}'...")
        print(f"[I5 - Kafka] Conectado a '{self.bootstrap_servers}'. Aguardando chunks de '{self.topic}'...")

        msg_count = 0
        try:
            while self.running:
                # Sondagem com timeout para não sobrecarregar CPU nem créditos da nuvem (Guia B2, 4.5)
                msg = self.consumer.poll(self.poll_timeout_s)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"[I5 - Kafka] Erro de mensagem Kafka: {msg.error()}")
                        continue

                # 1. Desserialização JSON UTF-8
                try:
                    raw_text = msg.value().decode("utf-8")
                    evento = json.loads(raw_text)
                except Exception as e:
                    logger.warning(f"[I5 - Kafka] Mensagem descartada (JSON inválido): {e}")
                    continue

                # 2. Validação do envelope B1/B2
                valido, motivo = self.validar_envelope(evento)
                if not valido:
                    logger.warning(f"[I5 - Kafka] Mensagem descartada: {motivo}")
                    continue

                # 3. Extração dos dados de áudio e do timestamp captured_at
                try:
                    payload = evento["payload"]
                    audio_chunk, sr = self.decodificar_audio(payload)
                    captured_at = self.extrair_metadados_tempo(evento)
                except Exception as e:
                    logger.warning(f"[I5 - Kafka] Falha ao decodificar áudio do payload: {e}")
                    continue

                # 4. Injeta no pipeline do I5 propagando captured_at
                pipeline.process_chunk(audio_chunk, captured_at=captured_at)

                msg_count += 1
                if max_messages and msg_count >= max_messages:
                    logger.info(f"[I5 - Kafka] Limite de {max_messages} mensagens atingido.")
                    break

        except KeyboardInterrupt:
            print("\n[I5 - Kafka] Interrupção de teclado recebida. Encerrando consumidor...")
        finally:
            self.close()

    def close(self):
        """
        Encerramento seguro do cliente Kafka (desinscreve e fecha commitando offsets).
        """
        self.running = False
        try:
            logger.info("[I5 - Kafka] Fechando conexão do consumidor...")
            self.consumer.close()
            logger.info("[I5 - Kafka] Conexão fechada com sucesso.")
        except Exception as e:
            logger.warning(f"[I5 - Kafka] Aviso ao fechar consumidor: {e}")

