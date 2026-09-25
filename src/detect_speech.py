#!/usr/bin/env python3
"""
Script executável para detecção de fala e segmentação de áudio usando Silero VAD (Componente I5).

Uso:
    python src/detect_speech.py data/audio.ogg
    python src/detect_speech.py --save-batches --out-dir ./output
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

# Adiciona o diretório raiz ao sys.path para permitir imports diretos de src
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_preprocessor import AudioPreprocessor
from src.vad_segmenter import AudioBatch, VADSegmenter


def detect_speech_in_audio(
    audio_path: str,
    speech_threshold: float = 0.5,
    silence_threshold_ms: int = 700,
    pre_speech_pad_ms: int = 250,
    post_speech_pad_ms: int = 200,
    apply_noise_filter: bool = True,
    save_batches: bool = False,
    output_dir: str = "output_batches",
) -> List[AudioBatch]:
    """
    Carrega o arquivo de áudio, aplica filtragem de ruído, segmenta com Silero VAD
    e opcionalmente salva os batches isolados de fala.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    print(f"\n[I5 - VAD] Lendo arquivo: {path.name}")
    audio_data, sample_rate = sf.read(str(path))
    original_duration = len(audio_data) / sample_rate
    print(f"[I5 - VAD] Duração original: {original_duration:.2f}s | Sample rate: {sample_rate} Hz")

    # Pré-processamento e filtragem de ruído ambiente
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    start_prep_time = time.perf_counter()
    clean_audio = preprocessor.process(audio_data, apply_noise_reduction=apply_noise_filter)
    prep_elapsed_ms = (time.perf_counter() - start_prep_time) * 1000

    # Segmentação VAD
    segmenter = VADSegmenter(
        sample_rate=16000,
        speech_threshold=speech_threshold,
        silence_threshold_ms=silence_threshold_ms,
        pre_speech_pad_ms=pre_speech_pad_ms,
        post_speech_pad_ms=post_speech_pad_ms,
        use_onnx=True,
    )

    start_vad_time = time.perf_counter()
    batches = segmenter.process_full_audio(clean_audio)
    vad_elapsed_ms = (time.perf_counter() - start_vad_time) * 1000

    total_speech_duration = sum(b.duration for b in batches)
    silence_discarded = max(0.0, original_duration - total_speech_duration)
    gpu_reduction_pct = (silence_discarded / original_duration * 100) if original_duration > 0 else 0.0

    print(f"[I5 - VAD] Pré-processamento concluído em {prep_elapsed_ms:.1f}ms")
    print(f"[I5 - VAD] Inferência VAD concluída em {vad_elapsed_ms:.1f}ms (CPU)")
    print(f"[I5 - VAD] Batches de fala detectados: {len(batches)}")
    print(f"[I5 - VAD] Duração de fala útil: {total_speech_duration:.2f}s | Silêncio descartado: {silence_discarded:.2f}s ({gpu_reduction_pct:.1f}% de economia na GPU)")

    if batches:
        print("\n" + "=" * 70)
        print(f"{'Batch #':<8} | {'Início (s)':<12} | {'Fim (s)':<12} | {'Duração (s)':<12} | {'Amostras':<10}")
        print("-" * 70)
        for b in batches:
            print(f"{b.batch_index:<8} | {b.start_time:<12.3f} | {b.end_time:<12.3f} | {b.duration:<12.3f} | {len(b.audio):<10}")
        print("=" * 70 + "\n")
    else:
        print("[I5 - VAD] Nenhuma fala detectada. Áudio classificado inteiramente como ruído/silêncio.")

    # Salva batches se solicitado
    if save_batches and batches:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        base_name = path.stem
        for b in batches:
            batch_file = out_path / f"{base_name}_batch_{b.batch_index:03d}_{b.start_time:.2f}s-{b.end_time:.2f}s.wav"
            sf.write(str(batch_file), b.audio, 16000)
            print(f"[I5 - VAD] Batch {b.batch_index} salvo em: {batch_file}")

    return batches


def main():
    parser = argparse.ArgumentParser(description="Detecção de Fala e Segmentação VAD - Componente I5")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "WhatsApp Ptt 2026-09-16 at 11.29.24.ogg"),
        help="Caminho para o arquivo de áudio (.wav, .ogg, etc.)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Limiar de probabilidade do VAD para ativação de fala (default: 0.5)",
    )
    parser.add_argument(
        "--silence-threshold-ms",
        type=int,
        default=700,
        help="Duração de silêncio contínuo em ms para encerrar o batch (default: 700)",
    )
    parser.add_argument(
        "--no-noise-filter",
        action="store_true",
        help="Desativa o filtro de atenuação de ruído ambiente",
    )
    parser.add_argument(
        "--save-batches",
        action="store_true",
        help="Exporta os batches isolados em formato .wav",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output_batches",
        help="Diretório de destino dos batches salvos (default: output_batches)",
    )

    args = parser.parse_args()

    detect_speech_in_audio(
        audio_path=args.audio_path,
        speech_threshold=args.threshold,
        silence_threshold_ms=args.silence_threshold_ms,
        apply_noise_filter=not args.no_noise_filter,
        save_batches=args.save_batches,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

