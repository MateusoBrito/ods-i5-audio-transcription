"""
Módulo de pré-processamento de áudio e atenuação de ruído ambiente para o componente I5.
Executado 100% em CPU com baixo custo computacional para respeitar restrições da Jetson Nano.
"""

from typing import Tuple
import numpy as np
from scipy import signal


class AudioPreprocessor:
    """
    Pré-processador de sinal de áudio responsável por normalização,
    conversão mono, filtragem passa-banda e atenuação de ruído ambiente.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        lowcut: float = 80.0,
        highcut: float = 7500.0,
        filter_order: int = 5,
        enable_spectral_subtraction: bool = True,
        spectral_floor: float = 0.15,
    ):
        """
        Inicializa o pré-processador.

        Args:
            target_sample_rate: Taxa de amostragem padrão (16000 Hz para Silero VAD / Whisper).
            lowcut: Frequência de corte inferior em Hz (remove DC offset, hum 50/60Hz e vento).
            highcut: Frequência de corte superior em Hz (remove chiados e frequências fora do espectro vocal humano).
            filter_order: Ordem do filtro Butterworth.
            enable_spectral_subtraction: Se verdadeiro, aplica subtração espectral suave para reduzir ruído estático de fundo.
            spectral_floor: Piso espectral mínimo para evitar artefatos musicais na subtração espectral.
        """
        self.target_sample_rate = target_sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.enable_spectral_subtraction = enable_spectral_subtraction
        self.spectral_floor = spectral_floor

        # Pré-computa coeficientes do filtro Butterworth passa-banda
        nyquist = 0.5 * target_sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        self.sos = signal.butter(filter_order, [low, high], btype="band", output="sos")

    def to_mono_float32(self, audio: np.ndarray) -> np.ndarray:
        """
        Converte o array de áudio para formato mono em float32 normalizado (-1.0 a 1.0).
        """
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)

        # Se for estéreo/multicanal, calcula a média dos canais
        if audio.ndim > 1:
            if audio.shape[0] < audio.shape[1]:  # canais na primeira dimensão
                audio = np.mean(audio, axis=0)
            else:  # canais na segunda dimensão
                audio = np.mean(audio, axis=1)

        # Conversão de tipos inteiros para float32 no intervalo [-1.0, 1.0]
        if np.issubdtype(audio.dtype, np.integer):
            max_val = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)

        # Remove NaN ou Inf se presentes
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio

    def apply_bandpass(self, audio: np.ndarray) -> np.ndarray:
        """
        Aplica filtro passa-banda Butterworth usando SOS (Second-Order Sections) para estabilidade numérica.
        """
        if len(audio) == 0:
            return audio

        # sosfiltfilt aplica o filtro para frente e para trás, resultando em distorção de fase zero
        filtered = signal.sosfiltfilt(self.sos, audio)
        return filtered.astype(np.float32)

    def apply_spectral_subtraction(
        self,
        audio: np.ndarray,
        noise_samples_estimate: int = 1600,
        oversubtraction: float = 1.3,
    ) -> np.ndarray:
        """
        Atenuação de ruído estacionário por subtração espectral baseada nas amostras iniciais de ruído.

        Args:
            audio: Sinal de áudio 1D float32.
            noise_samples_estimate: Quantidade de amostras iniciais estimadas como ruído de fundo (default: 1600 amostras = 100ms a 16kHz).
            oversubtraction: Fator de sobre-subtração para garantir maior limpeza.
        """
        n_samples = len(audio)
        n_fft = 512
        hop_length = 256

        if n_samples < n_fft:
            return audio

        # STFT (Short-Time Fourier Transform)
        frequencies, times, stft_matrix = signal.stft(
            audio,
            fs=self.target_sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
        )

        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)

        # Estimar o perfil de ruído a partir dos primeiros frames (ou do menor percentil se áudio curto)
        n_noise_frames = max(1, min(magnitude.shape[1], noise_samples_estimate // hop_length))
        noise_profile = np.mean(magnitude[:, :n_noise_frames], axis=1, keepdims=True)

        # Subtração espectral com piso mínimo para evitar distorção da fala
        subtracted = magnitude - (oversubtraction * noise_profile)
        subtracted = np.maximum(subtracted, self.spectral_floor * magnitude)

        # Reconstrução do sinal com a fase original
        clean_stft = subtracted * np.exp(1j * phase)
        _, clean_audio = signal.istft(
            clean_stft,
            fs=self.target_sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
        )

        # Ajusta comprimento para bater exatamente com o original
        clean_audio = clean_audio[:n_samples]
        if len(clean_audio) < n_samples:
            clean_audio = np.pad(clean_audio, (0, n_samples - len(clean_audio)))

        return clean_audio.astype(np.float32)

    def process(self, audio: np.ndarray, apply_noise_reduction: bool = True) -> np.ndarray:
        """
        Pipeline completo de pré-processamento de áudio.

        1. Conversão para mono float32 [-1, 1]
        2. Filtro passa-banda (80Hz - 7500Hz)
        3. Subtração espectral de ruído ambiente (se habilitada)
        """
        audio = self.to_mono_float32(audio)
        audio = self.apply_bandpass(audio)

        if apply_noise_reduction and self.enable_spectral_subtraction and len(audio) >= 512:
            audio = self.apply_spectral_subtraction(audio)

        return audio
