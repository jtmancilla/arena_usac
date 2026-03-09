"""
ARENA — Utilidades de I/O (v2)
================================
Funciones para cargar audio, segmentar en chunks, exportar timeline
y generar resúmenes por llamada.

Cambios v2:
  - chunk_sec default: 3.0 (batch optimized)
  - buffer_seconds default: 4.0
  - Nuevos umbrales por defecto
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np

from .monitor import MonitorAudioUnificado

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def iter_audio_files(audio_dir: Path, recursive: bool = False) -> List[Path]:
    """Lista archivos de audio en un directorio."""
    audio_dir = Path(audio_dir).expanduser().resolve()
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise ValueError(f"audio_dir inválido: {audio_dir}")
    pattern = "**/*" if recursive else "*"
    out = [p for p in audio_dir.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    out.sort()
    return out


def load_audio_mono(path: Path, sr_original: int = 0) -> Tuple[np.ndarray, int]:
    """Carga audio como mono (mezcla canales si es estéreo)."""
    sr_load: Optional[int] = None if int(sr_original) == 0 else int(sr_original)
    y, sr = librosa.load(str(path), sr=sr_load, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr)


def iter_chunks(y: np.ndarray, sr: int, chunk_sec: float) -> Iterable[Tuple[int, int, np.ndarray]]:
    """Segmenta audio en chunks de duración fija."""
    if chunk_sec <= 0:
        raise ValueError("chunk_sec inválido")
    samples_per_chunk = chunk_sec * float(sr)
    n_chunks = int(np.ceil(len(y) / samples_per_chunk)) if samples_per_chunk > 0 else 0
    for k in range(n_chunks):
        start = int(round(k * samples_per_chunk))
        end = int(round((k + 1) * samples_per_chunk))
        if start >= len(y):
            break
        end = min(end, len(y))
        chunk = y[start:end]
        if len(chunk) < int(0.5 * sr):  # mínimo 0.5s (ajustado para chunk de 3s)
            break
        yield start, end, chunk


def export_timeline_json(
    *,
    input_audio: Path,
    output_json: Path,
    input_label: Optional[str] = None,
    device: str = "cpu",
    chunk_sec: float = 3.0,           # v2: 1.0 → 3.0
    sr_original: int = 0,
    buffer_seconds: float = 4.0,      # v2: 6.0 → 4.0
    umbral_silencio_rms: float = 0.008,  # v2: 0.01 → 0.008
    umbral_drift: float = 0.42,       # v2: 0.35 → 0.42
    umbral_estres_min: float = 0.18,  # v2: 0.12 → 0.18
    umbral_tension_ratio: float = 1.5,
) -> Dict[str, Any]:
    """Procesa un audio completo con ARENA y exporta timeline JSON."""
    y, sr = load_audio_mono(input_audio, sr_original=sr_original)

    monitor = MonitorAudioUnificado(device=str(device), buffer_seconds=float(buffer_seconds))
    monitor.umbral_silencio_rms = float(umbral_silencio_rms)
    monitor.umbral_drift = float(umbral_drift)
    monitor.umbral_estres_min = float(umbral_estres_min)
    monitor.umbral_tension_ratio = float(umbral_tension_ratio)

    points: List[Dict[str, Any]] = []
    for k, (start, end, chunk) in enumerate(iter_chunks(y, sr, chunk_sec)):
        out = monitor.procesar(chunk, sr_original=int(sr))
        points.append(
            {
                "chunk_index": k,
                "t_sec": round(start / sr, 4),
                "t_start_sec": round(start / sr, 4),
                "t_end_sec": round(end / sr, 4),
                "sr_loaded": int(sr),
                **out,
            }
        )

    payload: Dict[str, Any] = {
        "meta": {
            "input": str(input_label or input_audio.name),
            "sr": int(sr),
            "chunk_sec": float(chunk_sec),
            "points": len(points),
            "buffer_seconds": float(buffer_seconds),
            "umbral_silencio_rms": float(umbral_silencio_rms),
            "umbral_drift": float(umbral_drift),
            "umbral_estres_min": float(umbral_estres_min),
            "umbral_tension_ratio": float(umbral_tension_ratio),
            "version": "v2",
        },
        "points": points,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload
