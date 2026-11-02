#!/usr/bin/env python3
"""
Pipeline de Streaming e Transcrição em Tempo Real (Componente I5).

Orquestra o fluxo contínuo de áudio configurável via YAML e CLI:
1. Ingestão de áudio em streaming (P5 Mock / Microfone) a 16.000 Hz.
2. Pré-processamento e filtragem passa-banda (AudioPreprocessor na CPU).
3. Detecção de atividade de voz e segmentação contínua (VADSegmenter na CPU).
4. Fila desacoplada em memória RAM (Queue Produtor-Consumidor).
5. Transcrição forçada em Português com timestamps preservados (Whisper na GPU).
6. Emissão de payloads estruturados para A6/A7 (via callback, terminal e JSON).
"""

import argparse
import datetime
import json
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

# Suprime avisos de depreciação do Transformers/PyTorch
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import torch

# Adiciona o diretório raiz ao sys.path para imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_preprocessor import AudioPreprocessor
from src.config_loader import load_pipeline_config
from src.vad_segmenter import AudioBatch, VADSegmenter

try:
    from src.TranscriptionWhisper import TranscritorWhisper
except ImportError:
    from TranscriptionWhisper import TranscritorWhisper


class RealtimeSpeechPipeline:
    """
    Orquestrador do Componente I5 para processamento contínuo de áudio e transcrição.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        on_transcription: Optional[Callable[[dict], None]] = None,
        **overrides,
    ):
        """
        Inicializa o pipeline a partir de dicionário de configuração, arquivo YAML
        ou parâmetros diretos com precedência para overrides.
        """
        # Carrega configuração base
        if config is not None:
            self.config = config
        else:
            self.config = load_pipeline_config(config_path)

        # Aplica overrides pontuais (ex: passados diretamente via código ou CLI)
        if "sample_rate" in overrides and overrides["sample_rate"] is not None:
            self.config["audio"]["sample_rate"] = overrides["sample_rate"]
        if "speech_threshold" in overrides and overrides["speech_threshold"] is not None:
            self.config["vad"]["speech_threshold"] = overrides["speech_threshold"]
        if "silence_threshold_ms" in overrides and overrides["silence_threshold_ms"] is not None:
            self.config["vad"]["silence_threshold_ms"] = overrides["silence_threshold_ms"]
        if "apply_noise_filter" in overrides and overrides["apply_noise_filter"] is not None:
            self.config["preprocessor"]["apply_noise_filter"] = overrides["apply_noise_filter"]
        if "model_name" in overrides and overrides["model_name"] is not None:
            self.config["whisper"]["model_name"] = overrides["model_name"]
        if "device" in overrides and overrides["device"] is not None:
            self.config["whisper"]["device"] = overrides["device"]
        if "language" in overrides and overrides["language"] is not None:
            self.config["whisper"]["language"] = overrides["language"]

        self.on_transcription = on_transcription

        audio_cfg = self.config["audio"]
        prep_cfg = self.config["preprocessor"]
        vad_cfg = self.config["vad"]
        whisper_cfg = self.config["whisper"]
        output_cfg = self.config["output"]

        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.apply_noise_filter = prep_cfg.get("apply_noise_filter", True)

        # Configurações de saída
        self.save_json = output_cfg.get("save_json", True)
        self.json_dir = output_cfg.get("json_dir", "jsons_gerados")
        self.min_confidence_warning = output_cfg.get("min_confidence_warning", 0.50)

        config_source = self.config.get("_config_file") or "Defaults embutidos"
        print(f"[I5 - Pipeline] Configuração carregada de: {config_source}")

        # Front-End de Áudio (CPU)
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=self.sample_rate,
            lowcut=prep_cfg.get("lowcut_hz", 80.0),
            highcut=prep_cfg.get("highcut_hz", 7500.0),
            filter_order=prep_cfg.get("filter_order", 5),
            enable_spectral_subtraction=prep_cfg.get("enable_spectral_subtraction", False),
        )

        # Voice Activity Detection (Silero VAD na CPU via ONNX)
        self.segmenter = VADSegmenter(
            sample_rate=self.sample_rate,
            speech_threshold=vad_cfg.get("speech_threshold", 0.50),
            silence_threshold_ms=vad_cfg.get("silence_threshold_ms", 700),
            pre_speech_pad_ms=vad_cfg.get("pre_speech_pad_ms", 250),
            post_speech_pad_ms=vad_cfg.get("post_speech_pad_ms", 200),
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 250),
            use_onnx=vad_cfg.get("use_onnx", True),
        )

        # Transcritor ASR (Whisper na GPU)
        self.transcritor = TranscritorWhisper(
            model_name=whisper_cfg.get("model_name", "openai/whisper-small"),
            device=whisper_cfg.get("device", "auto"),
            language=whisper_cfg.get("language", "portuguese"),
            task=whisper_cfg.get("task", "transcribe"),
            torch_dtype=whisper_cfg.get("torch_dtype", "float16"),
        )
        print(f"[I5 - Pipeline] Whisper iniciado no device: {self.transcritor.device} (modelo: {whisper_cfg.get('model_name')})")

        # Fila desacoplada em RAM entre VAD (CPU) e Whisper (GPU)
        self.batch_queue: queue.Queue = queue.Queue()
        self.results: List[dict] = []
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None

    def _transcription_worker(self) -> None:
        """
        Consumidor executado em thread dedicada para inferência no Whisper.
        Lê batches de fala da fila e processa diretamente da memória RAM.
        """
        if self.save_json:
            caminho_pasta = PROJECT_ROOT / self.json_dir
            caminho_pasta.mkdir(parents=True, exist_ok=True)
        else:
            caminho_pasta = None

        while self._running or not self.batch_queue.empty():
            try:
                item = self.batch_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                self.batch_queue.task_done()
                break

            batch: AudioBatch = item
            inference_start = time.perf_counter()

            # 1. Base temporal da transcrição ancorada no captured_at de P5
            timestamp_inicio = batch.start_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
            origem = batch.metadata.get("origem", "ods.p5.audio_chunk")

            # 2. Chama a IA que retorna o JSON oficial (String)
            json_contrato_string = self.transcritor.transcrever_audio(
                audio_array=batch.audio,
                sampling_rate=batch.sample_rate,
                origem=origem,
                timestamp_inicio=timestamp_inicio,
            )

            inference_time_ms = (time.perf_counter() - inference_start) * 1000

            # 3. Lê o JSON de volta para extrair dados
            dados_contrato = json.loads(json_contrato_string)
            payload_interno = dados_contrato["payload"]

            texto_extraido = payload_interno["texto_transcrito"]
            confianca_extraida = payload_interno["confianca_geral"]

            self.results.append(dados_contrato)

            # 4. Salva o JSON oficial na pasta configurada
            caminho_arquivo_str = ""
            if self.save_json and caminho_pasta:
                timestamp_arquivo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                caminho_arquivo = caminho_pasta / f"evento_i5_{timestamp_arquivo}.json"
                with open(caminho_arquivo, "w", encoding="utf-8") as f:
                    f.write(json_contrato_string)
                caminho_arquivo_str = f" [Salvo em {caminho_arquivo.relative_to(PROJECT_ROOT)}]"

            # 5. Notificação via callback ou stdout
            if self.on_transcription:
                self.on_transcription(dados_contrato)
            else:
                alerta = " ⚠️ (baixa confiança)" if confianca_extraida < self.min_confidence_warning else ""
                print(
                    f"[{batch.start_time:6.2f}s -> {batch.end_time:6.2f}s] "
                    f"(conf: {confianca_extraida:.2f}{alerta} | {inference_time_ms:.0f}ms) "
                    f"=> \"{texto_extraido}\"{caminho_arquivo_str}"
                )

            self.batch_queue.task_done()

    def start(self) -> None:
        """
        Inicia a thread consumidora da GPU.
        """
        self._running = True
        self.results.clear()
        self._consumer_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="WhisperWorker",
        )
        self._consumer_thread.start()

    def stop(self) -> None:
        """
        Encerra a thread consumidora e aguarda a finalização dos batches pendentes.
        """
        self._running = False
        self.batch_queue.put(None)  # Envia sinalizador de término
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join()

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        captured_at: Optional[Union[str, datetime.datetime, float, int]] = None,
    ) -> None:
        """
        Processa um bloco contínuo de áudio recebido em streaming (Kafka ou microfone).
        
        Args:
            audio_chunk: Array numpy 1D com áudio capturado (ex: a cada 32ms - 100ms).
            captured_at: Timestamp de captura emitido pelo hardware/P5 (ISO 8601 ou epoch).
        """
        # 1. Filtro passa-banda leve na CPU
        clean_chunk = self.preprocessor.process(
            audio_chunk,
            apply_noise_reduction=self.apply_noise_filter,
        )

        # 2. VAD: empacota quando detecta pausa/silêncio ancorando captured_at
        batches = self.segmenter.push_audio(clean_chunk, captured_at=captured_at)
        for batch in batches:
            self.batch_queue.put(batch)

    def flush(self) -> None:
        """
        Força a finalização de qualquer fala que esteja retida no buffer do VAD.
        """
        last_batch = self.segmenter.flush()
        if last_batch:
            self.batch_queue.put(last_batch)

    def process_stream(
        self,
        stream_generator: Generator[np.ndarray, None, None],
    ) -> List[dict]:
        """
        Consome um gerador de chunks de áudio contínuo até o final e retorna todos os resultados.
        """
        self.start()
        try:
            for chunk in stream_generator:
                self.process_chunk(chunk)
            self.flush()
        finally:
            self.stop()

        return self.results


def simulate_audio_stream(
    file_path: str,
    target_sample_rate: int = 16000,
    chunk_size_samples: int = 1600,  # 100ms a 16kHz
    simulate_realtime: bool = False,
) -> Generator[np.ndarray, None, None]:
    """
    Simula uma fonte contínua de áudio (P5) lendo um arquivo e gerando pedaços sequenciais.
    Garante reamostragem a 16.000 Hz mono para compatibilidade com Silero VAD e Whisper.
    """
    data, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)

    total_samples = len(data)
    for i in range(0, total_samples, chunk_size_samples):
        chunk = data[i : i + chunk_size_samples]
        if simulate_realtime:
            # Pausa para simular o tempo real de captura do microfone
            chunk_duration = len(chunk) / target_sample_rate
            time.sleep(chunk_duration)
        yield chunk


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline I5: VAD Contínuo + Transcrição Whisper em Tempo Real",
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "audio6.ogg"),
        help="Caminho do arquivo de áudio para teste (default: data/audio6.ogg)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Caminho para o arquivo YAML de configuração (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo Whisper (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device de inferência: auto, cuda ou cpu (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Idioma forçado do Whisper (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Limiar do VAD para ativação de fala (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--silence-threshold-ms",
        type=int,
        default=None,
        help="Limiar de silêncio para encerramento de batch em ms (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=None,
        help="Tamanho do chunk de streaming em ms (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        default=None,
        help="Simula taxa de transmissão em tempo real inserindo pausas proporcionais",
    )
    parser.add_argument(
        "--no-noise-filter",
        action="store_true",
        default=False,
        help="Desativa filtro passa-banda de ruído",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        default=False,
        help="Desativa a gravação de arquivos JSON de evento",
    )
    parser.add_argument(
        "--kafka",
        action="store_true",
        help="Consome áudio em streaming diretamente do barramento Kafka (P5 - ods.p5.audio_chunk)",
    )

    args = parser.parse_args()

    # 1. Carrega configuração do YAML
    config = load_pipeline_config(args.config)

    # 2. Aplica overrides da CLI (CLI > YAML)
    if args.threshold is not None:
        config["vad"]["speech_threshold"] = args.threshold
    if args.silence_threshold_ms is not None:
        config["vad"]["silence_threshold_ms"] = args.silence_threshold_ms
    if args.model is not None:
        config["whisper"]["model_name"] = args.model
    if args.device is not None:
        config["whisper"]["device"] = args.device
    if args.language is not None:
        config["whisper"]["language"] = args.language
    if args.chunk_ms is not None:
        config["audio"]["chunk_duration_ms"] = args.chunk_ms
    if args.simulate_realtime:
        config["audio"]["simulate_realtime"] = True
    if args.no_noise_filter:
        config["preprocessor"]["apply_noise_filter"] = False
    if args.no_save_json:
        config["output"]["save_json"] = False

    pipeline = RealtimeSpeechPipeline(config=config)

    # 3. Modo KAFKA: escuta do barramento de eventos (P5)
    if args.kafka:
        from src.kafka_audio_consumer import KafkaAudioConsumer

        kafka_cfg = config.get("kafka", {})
        print(f"\n=======================================================")
        print(f"   Iniciando Pipeline I5 via KAFKA (Streaming P5)")
        print(f"   Broker: {kafka_cfg.get('bootstrap_servers')}")
        print(f"   Tópico: {kafka_cfg.get('topic_input')}")
        print(f"   Group ID: {kafka_cfg.get('group_id')}")
        print(f"   VAD Speech Threshold: {config['vad'].get('speech_threshold')}")
        print(f"=======================================================\n")

        pipeline.start()
        consumer = KafkaAudioConsumer.from_config(config=config)
        try:
            consumer.listen_and_feed(pipeline)
        finally:
            pipeline.flush()
            pipeline.stop()
            print(f"\n[I5 - Pipeline] Finalizado. Total de frases/batches transcritos: {len(pipeline.results)}")
        return

    # 4. Modo Local/Arquivo
    audio_file = Path(args.audio_path)
    if not audio_file.exists():
        print(f"Erro: Arquivo '{audio_file}' não encontrado.")
        sys.exit(1)

    sample_rate = config["audio"].get("sample_rate", 16000)
    chunk_ms = config["audio"].get("chunk_duration_ms", 100)
    simulate_realtime = config["audio"].get("simulate_realtime", False)

    print(f"\n=======================================================")
    print(f"   Iniciando Pipeline I5: VAD Contínuo + Whisper")
    print(f"   Arquivo: {audio_file.name}")
    print(f"   Modo tempo real: {simulate_realtime}")
    print(f"   VAD Speech Threshold: {config['vad'].get('speech_threshold')}")
    print(f"   VAD Silence Threshold: {config['vad'].get('silence_threshold_ms')}ms")
    print(f"=======================================================\n")

    pipeline = RealtimeSpeechPipeline(config=config)

    chunk_size = int(sample_rate * (chunk_ms / 1000.0))
    stream = simulate_audio_stream(
        str(audio_file),
        target_sample_rate=sample_rate,
        chunk_size_samples=chunk_size,
        simulate_realtime=simulate_realtime,
    )

    start_total = time.perf_counter()
    resultados = pipeline.process_stream(stream)
    elapsed = time.perf_counter() - start_total

    print(f"\n=======================================================")
    print(f"   Processamento Concluído!")
    print(f"   Tempo total: {elapsed:.2f}s")
    print(f"   Total de frases/batches transcritos: {len(resultados)}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
