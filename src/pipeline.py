#!/usr/bin/env python3
"""
Pipeline de Streaming e Transcrição em Tempo Real (Componente I5).

Orquestra o fluxo contínuo de áudio:
1. Ingestão de áudio em streaming (P5 Mock / Microfone) a 16.000 Hz.
2. Pré-processamento e filtragem passa-banda (AudioPreprocessor na CPU).
3. Detecção de atividade de voz e segmentação contínua (VADSegmenter na CPU).
4. Fila desacoplada em memória RAM (Queue Produtor-Consumidor).
5. Transcrição forçada em Português com timestamps preservados (Whisper na GPU).
6. Emissão de payloads estruturados para A6/A7 (via callback ou stdout).
"""

import argparse
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Generator, List, Optional

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
        model_name: str = "openai/whisper-small",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        silence_threshold_ms: int = 700,
        apply_noise_filter: bool = True,
        on_transcription: Optional[Callable[[dict], None]] = None,
    ):
        """
        Inicializa o pipeline com os componentes de VAD e Transcrição.
        """
        self.sample_rate = sample_rate
        self.apply_noise_filter = apply_noise_filter
        self.on_transcription = on_transcription

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[I5 - Pipeline] Inicializando componentes...")
        print(f"[I5 - Pipeline] Device de inferência Whisper: {self.device}")

        # Front-End de Áudio (CPU)
        # Desativa subtração espectral em streaming curto para evitar canibalização da fala
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=self.sample_rate,
            enable_spectral_subtraction=False,
        )
        self.segmenter = VADSegmenter(
            sample_rate=self.sample_rate,
            speech_threshold=speech_threshold,
            silence_threshold_ms=silence_threshold_ms,
            use_onnx=True,
        )

        # Transcritor ASR (GPU)
        self.transcritor = TranscritorWhisper(
            model_name=model_name,
            device=self.device,
        )

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
        while self._running or not self.batch_queue.empty():
            try:
                item = self.batch_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                # Sinalizador de término (poison pill)
                self.batch_queue.task_done()
                break

            batch: AudioBatch = item
            inference_start = time.perf_counter()

            # Transcreve o array em memória (sem I/O de disco)
            resultado = self.transcritor.transcrever_audio(
                batch.audio,
                sampling_rate=batch.sample_rate,
            )
            inference_time_ms = (time.perf_counter() - inference_start) * 1000

            # Monta o contrato de saída alinhado ao CONTEXT.md
            payload = {
                "batch_index": batch.batch_index,
                "start_time": round(batch.start_time, 3),
                "end_time": round(batch.end_time, 3),
                "duration": round(batch.duration, 3),
                "texto": resultado["texto"],
                "confianca": resultado["confianca"],
                "inference_time_ms": round(inference_time_ms, 2),
                "timestamp_criacao": time.time(),
                "metadata": batch.metadata,
            }

            self.results.append(payload)

            if self.on_transcription:
                self.on_transcription(payload)
            else:
                print(
                    f"[{payload['start_time']:6.2f}s -> {payload['end_time']:6.2f}s] "
                    f"(conf: {payload['confianca']:.2f} | {payload['inference_time_ms']:.0f}ms) "
                    f"=> \"{payload['texto']}\""
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

    def process_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Processa um bloco contínuo de áudio recebido em streaming.
        
        Args:
            audio_chunk: Array numpy 1D com áudio capturado (ex: a cada 32ms - 100ms).
        """
        # 1. Filtro passa-banda leve na CPU
        clean_chunk = self.preprocessor.process(
            audio_chunk,
            apply_noise_reduction=self.apply_noise_filter,
        )

        # 2. VAD: empacota quando detecta pausa/silêncio
        batches = self.segmenter.push_audio(clean_chunk)
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
        "--model",
        type=str,
        default="openai/whisper-small",
        help="Modelo Whisper a ser carregado (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        help="Simula taxa de transmissão em tempo real inserindo pausas proporcionais",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Tamanho do chunk de streaming em milissegundos (default: 100ms)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Limiar do VAD para ativação de fala (default: 0.5)",
    )
    parser.add_argument(
        "--silence-threshold-ms",
        type=int,
        default=700,
        help="Limiar de silêncio para encerramento de batch (default: 700ms)",
    )
    parser.add_argument(
        "--no-noise-filter",
        action="store_true",
        help="Desativa filtro de ruído",
    )

    args = parser.parse_args()

    audio_file = Path(args.audio_path)
    if not audio_file.exists():
        print(f"Erro: Arquivo '{audio_file}' não encontrado.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"   Iniciando Pipeline I5: VAD Contínuo + Whisper")
    print(f"   Arquivo: {audio_file.name}")
    print(f"   Modo tempo real: {args.simulate_realtime}")
    print(f"=======================================================\n")

    pipeline = RealtimeSpeechPipeline(
        model_name=args.model,
        speech_threshold=args.threshold,
        silence_threshold_ms=args.silence_threshold_ms,
        apply_noise_filter=not args.no_noise_filter,
    )

    chunk_size = int(16000 * (args.chunk_ms / 1000.0))
    stream = simulate_audio_stream(
        str(audio_file),
        target_sample_rate=16000,
        chunk_size_samples=chunk_size,
        simulate_realtime=args.simulate_realtime,
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

