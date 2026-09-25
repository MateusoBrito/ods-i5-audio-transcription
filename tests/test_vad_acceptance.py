"""
Testes automatizados do Critério de Aceitação para o componente I5 (VAD e Segmentação).

Critério de Aceitação:
"Injetando um arquivo de áudio longo contendo 1 minuto de silêncio e 5 segundos de fala,
o componente descarta o silêncio e gera um batch isolado contendo apenas a locução."
"""

import os
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio_preprocessor import AudioPreprocessor
from src.vad_segmenter import VADSegmenter


class TestVADAcceptanceCriteria(unittest.TestCase):
    """
    Suíte de testes de aceitação e robustez do VAD e Pré-processamento.
    """

    @classmethod
    def setUpClass(cls):
        cls.sample_rate = 16000
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.data_file = cls.project_root / "data" / "WhatsApp Ptt 2026-09-16 at 11.29.24.ogg"
        assert cls.data_file.exists(), f"Arquivo de teste não encontrado: {cls.data_file}"

        # Carrega áudio real contendo fala
        raw_audio, sr = sf.read(str(cls.data_file))
        assert sr == cls.sample_rate

        # Isola um segmento real de fala (~2.1 segundos)
        # No arquivo 11.29.24, entre 3.3s e 5.3s há fala contínua clara
        start_speech = int(3.3 * cls.sample_rate)
        end_speech = int(5.3 * cls.sample_rate)
        cls.single_speech_snippet = raw_audio[start_speech:end_speech]

        # Constrói exatamente 5.0 segundos de fala contínua repetindo e encadeando o snippet
        target_5s_samples = int(5.0 * cls.sample_rate)
        tiles_needed = int(np.ceil(target_5s_samples / len(cls.single_speech_snippet)))
        cls.speech_5s = np.tile(cls.single_speech_snippet, tiles_needed)[:target_5s_samples]

    def setUp(self):
        self.preprocessor = AudioPreprocessor(target_sample_rate=self.sample_rate)
        self.segmenter = VADSegmenter(
            sample_rate=self.sample_rate,
            speech_threshold=0.5,
            silence_threshold_ms=700,
            pre_speech_pad_ms=250,
            post_speech_pad_ms=200,
            min_speech_duration_ms=250,
            use_onnx=True,
        )

    def test_acceptance_criteria_one_minute_silence_five_seconds_speech(self):
        """
        Critério de Aceitação:
        Injetando 1 minuto de silêncio (30s antes + 30s depois) com 5 segundos de fala no centro:
        - O componente descarta o silêncio (> 90% do áudio).
        - Gera exatamente 1 batch isolado contendo apenas a locução.
        - Timestamps e duração preservados com precisão.
        """
        silence_before = np.zeros(30 * self.sample_rate, dtype=np.float32)
        silence_after = np.zeros(30 * self.sample_rate, dtype=np.float32)

        # Monta áudio contínuo de 65 segundos: 30s silêncio + 5s fala + 30s silêncio
        long_audio = np.concatenate([silence_before, self.speech_5s, silence_after])
        total_duration = len(long_audio) / self.sample_rate
        self.assertEqual(total_duration, 65.0)

        # Processamento pelo pipeline
        clean_audio = self.preprocessor.process(long_audio)
        batches = self.segmenter.process_full_audio(clean_audio)

        # Validação 1: Exatamente 1 batch emitido
        self.assertEqual(
            len(batches),
            1,
            f"Esperava exatamente 1 batch isolado, mas foram gerados {len(batches)} batches.",
        )

        batch = batches[0]

        # Validação 2: O início do batch deve estar próximo de 30s (considerando pre-roll pad)
        # O pre-roll pad adiciona até ~250ms antes do gatilho
        self.assertAlmostEqual(batch.start_time, 30.0, delta=0.5)

        # Validação 3: Duração do batch próxima de 5.0s (considerando pads pré e pós-fala)
        self.assertAlmostEqual(batch.duration, 5.0, delta=0.6)

        # Validação 4: Mais de 90% do áudio descartado (proteção da GPU)
        total_speech_len = batch.duration
        silence_discarded = total_duration - total_speech_len
        saving_ratio = silence_discarded / total_duration
        self.assertGreater(saving_ratio, 0.90)

    def test_pure_silence_yields_zero_batches(self):
        """
        Garante que 60 segundos de silêncio absoluto não acionem nenhuma inferência na GPU.
        """
        pure_silence = np.zeros(60 * self.sample_rate, dtype=np.float32)
        clean = self.preprocessor.process(pure_silence)
        batches = self.segmenter.process_full_audio(clean)
        self.assertEqual(len(batches), 0)

    def test_low_frequency_hum_filtered_out(self):
        """
        Garante que ruído de rede elétrica (mains hum 60Hz) seja atenuado pelo filtro passa-banda
        e não dispare o VAD indevidamente.
        """
        t = np.linspace(0, 10, 10 * self.sample_rate, endpoint=False)
        # Sinal de hum senoidal em 60Hz com amplitude moderada
        hum_noise = 0.3 * np.sin(2 * np.pi * 60 * t).astype(np.float32)

        clean = self.preprocessor.process(hum_noise)
        batches = self.segmenter.process_full_audio(clean)
        self.assertEqual(len(batches), 0)

    def test_short_transient_click_discarded(self):
        """
        Garante que estalos rápidos (< 100ms) sejam descartados pelo filtro de duração mínima,
        protegendo a GPU.
        """
        silence = np.zeros(5 * self.sample_rate, dtype=np.float32)
        # Injeta um pico curto de 50ms
        click_duration = int(0.05 * self.sample_rate)
        silence[self.sample_rate : self.sample_rate + click_duration] = 0.8

        clean = self.preprocessor.process(silence)
        batches = self.segmenter.process_full_audio(clean)
        self.assertEqual(len(batches), 0)


if __name__ == "__main__":
    unittest.main()

