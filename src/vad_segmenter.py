"""
Módulo de segmentação de áudio e Voice Activity Detection (VAD) para o componente I5.
Integra o Silero VAD operando 100% offline em CPU via ONNX Runtime para proteger a GPU
e fatiar fluxos contínuos de áudio em batches de fala com timestamps preservados.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Generator, List, Optional
import numpy as np
import torch
from silero_vad import load_silero_vad


class VADState(Enum):
    SILENCE = "SILENCE"
    SPEAKING = "SPEAKING"


@dataclass
class AudioBatch:
    """
    Representa um lote isolado de fala pronto para inferência no ASR (Whisper).
    """
    batch_index: int
    audio: np.ndarray  # Array 1D float32 a 16kHz
    start_time: float  # Timestamp de início em segundos (alinhado com a captura)
    end_time: float    # Timestamp de término em segundos
    duration: float    # Duração total em segundos
    sample_rate: int = 16000
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AudioBatch(index={self.batch_index}, "
            f"start={self.start_time:.3f}s, end={self.end_time:.3f}s, "
            f"duration={self.duration:.3f}s, samples={len(self.audio)})"
        )


class VADSegmenter:
    """
    Segmentador de fluxo contínuo de áudio baseado em Silero VAD e máquina de estados.
    """

    CHUNK_SIZE = 512  # Amostras por frame exigido pelo Silero VAD a 16kHz (32ms)

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        silence_threshold_ms: int = 700,
        pre_speech_pad_ms: int = 250,
        post_speech_pad_ms: int = 200,
        min_speech_duration_ms: int = 250,
        use_onnx: bool = True,
    ):
        """
        Inicializa o segmentador VAD.

        Args:
            sample_rate: Taxa de amostragem (padrão 16000 Hz).
            speech_threshold: Limiar de probabilidade do Silero VAD para detectar fala.
            silence_threshold_ms: Tempo de silêncio consecutivo (ms) para fechar o batch.
            pre_speech_pad_ms: Duração de áudio pré-fala (ms) mantida no buffer de pre-roll.
            post_speech_pad_ms: Duração de áudio pós-fala (ms) para evitar cortes abruptos.
            min_speech_duration_ms: Duração mínima de fala para aceitar o batch e descartar ruídos curtos.
            use_onnx: Se verdadeiro, usa ONNX Runtime (recomendado para máxima performance em CPU).
        """
        if sample_rate != 16000:
            raise ValueError(f"Silero VAD requer sample_rate=16000 Hz, recebido: {sample_rate}")

        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.negative_threshold = max(0.1, speech_threshold - 0.15)  # Histerese
        self.silence_threshold_samples = int(sample_rate * silence_threshold_ms / 1000)
        self.pre_pad_samples = int(sample_rate * pre_speech_pad_ms / 1000)
        self.post_pad_samples = int(sample_rate * post_speech_pad_ms / 1000)
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)

        # Número de chunks no buffer circular de pre-roll
        self.max_pre_roll_chunks = max(1, int(np.ceil(self.pre_pad_samples / self.CHUNK_SIZE)))
        self.post_pad_chunks = int(np.ceil(self.post_pad_samples / self.CHUNK_SIZE))

        # Carrega modelo offline
        self.use_onnx = use_onnx
        self.model = load_silero_vad(onnx=use_onnx)

        # Estado da máquina
        self.state = VADState.SILENCE
        self.pre_roll_buffer: Deque[np.ndarray] = deque(maxlen=self.max_pre_roll_chunks)
        self.current_speech_chunks: List[np.ndarray] = []
        self.last_speech_chunk_idx: int = 0
        self.consecutive_silence_samples: int = 0

        # Base de tempo e contagem de amostras
        self.total_processed_samples: int = 0
        self.batch_start_sample: int = 0
        self.batch_counter: int = 0
        self.base_timestamp: float = 0.0

        # Buffer para dados que não completam múltiplos de CHUNK_SIZE
        self.leftover_samples: np.ndarray = np.empty(0, dtype=np.float32)

    def reset(self, base_timestamp: float = 0.0) -> None:
        """
        Reinicia o estado interno do segmentador e do modelo Silero VAD.
        """
        self.model.reset_states()
        self.state = VADState.SILENCE
        self.pre_roll_buffer.clear()
        self.current_speech_chunks.clear()
        self.last_speech_chunk_idx = 0
        self.consecutive_silence_samples = 0
        self.total_processed_samples = 0
        self.batch_start_sample = 0
        self.batch_counter = 0
        self.base_timestamp = base_timestamp
        self.leftover_samples = np.empty(0, dtype=np.float32)

    def _evaluate_chunk(self, chunk: np.ndarray) -> float:
        """
        Submete um chunk de 512 amostras ao Silero VAD e retorna a probabilidade de fala.
        """
        chunk_tensor = torch.from_numpy(chunk)
        prob = self.model(chunk_tensor, self.sample_rate).item()
        return prob

    def _create_batch(self) -> Optional[AudioBatch]:
        """
        Fecha o batch de áudio acumulado e valida o critério de duração mínima.
        """
        if not self.current_speech_chunks:
            return None

        # Trunca o silêncio excedente preservando o padding pós-fala configurado
        keep_chunks = min(
            len(self.current_speech_chunks),
            self.last_speech_chunk_idx + self.post_pad_chunks,
        )
        kept_chunks = self.current_speech_chunks[:keep_chunks]
        if not kept_chunks:
            return None

        batch_audio = np.concatenate(kept_chunks)

        # Descarta se a fala útil for menor que a duração mínima (proteção de GPU contra estalos)
        if len(batch_audio) < self.min_speech_samples:
            return None

        self.batch_counter += 1
        start_time = self.base_timestamp + (self.batch_start_sample / self.sample_rate)
        duration = len(batch_audio) / self.sample_rate
        end_time = start_time + duration

        return AudioBatch(
            batch_index=self.batch_counter,
            audio=batch_audio,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            sample_rate=self.sample_rate,
            metadata={
                "trimmed_silence_samples": self.consecutive_silence_samples,
                "chunks_count": len(kept_chunks),
            },
        )

    def push_audio(self, audio_chunk: np.ndarray) -> List[AudioBatch]:
        """
        Ingere um bloco de áudio de qualquer tamanho (streaming), divide em frames de 512
        amostras e retorna batches finalizados quando o silêncio atinge o limiar.
        """
        if not isinstance(audio_chunk, np.ndarray) or audio_chunk.dtype != np.float32:
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

        # Concatena com sobra de chamada anterior se houver
        if len(self.leftover_samples) > 0:
            audio_chunk = np.concatenate([self.leftover_samples, audio_chunk])
            self.leftover_samples = np.empty(0, dtype=np.float32)

        n_samples = len(audio_chunk)
        batches: List[AudioBatch] = []

        if n_samples < self.CHUNK_SIZE:
            self.leftover_samples = audio_chunk
            return batches

        # Processa chunks de 512 amostras
        n_chunks = n_samples // self.CHUNK_SIZE
        remainder = n_samples % self.CHUNK_SIZE
        if remainder > 0:
            self.leftover_samples = audio_chunk[-remainder:]

        for i in range(n_chunks):
            start_idx = i * self.CHUNK_SIZE
            end_idx = start_idx + self.CHUNK_SIZE
            chunk = audio_chunk[start_idx:end_idx]

            chunk_sample_pos = self.total_processed_samples
            self.total_processed_samples += self.CHUNK_SIZE

            prob = self._evaluate_chunk(chunk)

            if self.state == VADState.SILENCE:
                if prob >= self.speech_threshold:
                    # Início da fala detectada!
                    self.state = VADState.SPEAKING
                    # O início do batch inclui os frames do pre-roll buffer
                    pre_roll_samples = len(self.pre_roll_buffer) * self.CHUNK_SIZE
                    self.batch_start_sample = chunk_sample_pos - pre_roll_samples
                    self.current_speech_chunks = list(self.pre_roll_buffer)
                    self.current_speech_chunks.append(chunk)
                    self.last_speech_chunk_idx = len(self.current_speech_chunks)
                    self.consecutive_silence_samples = 0
                    self.pre_roll_buffer.clear()
                else:
                    # Permanece em silêncio: alimenta buffer circular de pre-roll
                    self.pre_roll_buffer.append(chunk)

            elif self.state == VADState.SPEAKING:
                self.current_speech_chunks.append(chunk)

                if prob >= self.negative_threshold:
                    # Locução ativa continuando
                    self.last_speech_chunk_idx = len(self.current_speech_chunks)
                    self.consecutive_silence_samples = 0
                else:
                    # Frame silencioso durante locução
                    self.consecutive_silence_samples += self.CHUNK_SIZE
                    if self.consecutive_silence_samples >= self.silence_threshold_samples:
                        # Limiar de silêncio atingido: encerra e emite o batch
                        batch = self._create_batch()
                        if batch is not None:
                            batches.append(batch)

                        # Transição de volta para SILENCE
                        self.state = VADState.SILENCE
                        self.current_speech_chunks.clear()
                        self.last_speech_chunk_idx = 0
                        self.consecutive_silence_samples = 0

        return batches

    def flush(self) -> Optional[AudioBatch]:
        """
        Finaliza qualquer batch pendente na máquina de estados (ex: término de arquivo ou interrupção de stream).
        """
        if self.state == VADState.SPEAKING and self.current_speech_chunks:
            batch = self._create_batch()
            self.state = VADState.SILENCE
            self.current_speech_chunks.clear()
            self.last_speech_chunk_idx = 0
            self.consecutive_silence_samples = 0
            return batch
        return None

    def process_full_audio(
        self,
        audio: np.ndarray,
        base_timestamp: float = 0.0,
        chunk_step_samples: int = 1024,
    ) -> List[AudioBatch]:
        """
        Processa um array de áudio completo simulando ingestão em streaming por blocos.
        """
        self.reset(base_timestamp=base_timestamp)
        all_batches: List[AudioBatch] = []

        total_len = len(audio)
        for offset in range(0, total_len, chunk_step_samples):
            chunk = audio[offset : offset + chunk_step_samples]
            batches = self.push_audio(chunk)
            all_batches.extend(batches)

        final_batch = self.flush()
        if final_batch is not None:
            all_batches.append(final_batch)

        return all_batches

