"""
Motor autocontenido: Monitor de Audio Unificado (ARENA)
Versión 2 — Optimizado para USAC batch processing (audio mono, 2-3 hablantes)

Cambios vs v1:
  - chunk_sec: 1.0 → 3.0 (batch, no real-time — embeddings más estables)
  - umbral_drift: 0.35 → 0.42 (evitar que cambios de turno normales disparen drift)
  - umbral_estres_min: 0.12 → 0.18 (filtrar distancia natural entre hablantes mono)
  - umbral_silencio_rms: 0.01 → 0.008 (capturar voces bajitas)
  - buffer_seconds: 6.0 → 4.0 (calibración más rápida con chunks de 3s)

NOTA SOBRE AUDIO MONO CON MÚLTIPLES HABLANTES
-----------------------------------------------
En llamadas USAC el audio es mono con 2-3 hablantes mezclados:
  - ES (Ejecutivo de Servicio) — generalista, suele abrir la llamada
  - C (Cliente) — busca contactar a su EC
  - EC (Ejecutivo de Cuenta) — personal, puede entrar al final
  - BOT BBVA — voz sintetizada, alta y estable

El anchor se calibra con los primeros ~4s de voz activa (generalmente el ES).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import librosa
import numpy as np
from resemblyzer import VoiceEncoder
from scipy.spatial.distance import cosine


class MonitorAudioUnificado:
    def __init__(self, device: str = "cpu", buffer_seconds: float = 4.0):
        self.encoder = VoiceEncoder(device)
        self.sr_model = 16000

        # Configuración v2 (optimizada para USAC batch)
        self.buffer_seconds = float(buffer_seconds)
        self.umbral_silencio_rms = 0.008       # v1: 0.01 → capturar voces bajitas
        self.umbral_drift = 0.42               # v1: 0.35 → menos falsos positivos en mono
        self.umbral_estres_min = 0.18          # v1: 0.12 → filtrar distancia natural mono
        self.umbral_tension_ratio = 1.5

        # Estado Interno
        self.buffer_audio: list[np.ndarray] = []
        self.samples_acumulados = 0
        self.calibrado = False

        # Baseline
        self.anchor_emb: Optional[np.ndarray] = None
        self.baseline_teo = 0.0

        # Observabilidad
        self._dist_hist: deque[float] = deque(maxlen=30)

    def _calc_expressive_metrics(self, distancia: float) -> Dict[str, Any]:
        self._dist_hist.append(float(distancia))
        denom = float(self.umbral_drift) if float(self.umbral_drift) > 0 else 0.42
        expressive_score = float(min(100.0, max(0.0, (distancia / denom) * 100.0)))
        ratio = float(distancia / denom) if denom else 0.0

        if ratio < 0.35:
            band = "LEVE"
        elif ratio < 0.70:
            band = "MEDIA"
        else:
            band = "ALTA"

        trend = 0.0
        if len(self._dist_hist) >= 10:
            last = list(self._dist_hist)[-5:]
            prev = list(self._dist_hist)[-10:-5]
            trend = float(np.mean(last)) - float(np.mean(prev))

        sustained = 0
        if len(self._dist_hist) >= 6:
            recent = list(self._dist_hist)[-6:]
            sustained = sum(1 for d in recent if d > float(self.umbral_estres_min))

        drift_risk = "BAJO"
        if ratio >= 0.85 or sustained >= 5:
            drift_risk = "ALTO"
        elif ratio >= 0.60 or sustained >= 3:
            drift_risk = "MEDIO"

        return {
            "expressive_score": round(expressive_score, 2),
            "expressive_band": band,
            "expressive_trend": round(float(trend), 5),
            "drift_risk": drift_risk,
            "dist_ratio_to_drift": round(ratio, 4),
            "recent_expressive_hits": int(sustained),
        }

    def _es_silencio(self, audio: np.ndarray) -> bool:
        rms = float(np.mean(librosa.feature.rms(y=audio)))
        return rms < float(self.umbral_silencio_rms)

    def _calcular_teo_rapido(self, audio: np.ndarray) -> float:
        if audio is None or len(audio) < 3:
            return 0.0
        val = np.abs(audio[1:-1] ** 2 - audio[:-2] * audio[2:])
        return float(np.mean(val) * 1000.0)

    def _calcular_rms(self, audio: np.ndarray) -> float:
        """RMS del chunk para métricas de calidad y detección de BOT."""
        return float(np.mean(librosa.feature.rms(y=audio)))

    def _intentar_calibrar(self, audio_chunk_16k: np.ndarray, teo_chunk: float) -> bool:
        self.buffer_audio.append(audio_chunk_16k)
        self.samples_acumulados += int(len(audio_chunk_16k))
        segundos_acumulados = float(self.samples_acumulados) / float(self.sr_model)
        if segundos_acumulados >= float(self.buffer_seconds):
            audio_full = (
                np.concatenate(self.buffer_audio)
                if len(self.buffer_audio) > 1
                else self.buffer_audio[0]
            )
            self.anchor_emb = self.encoder.embed_utterance(audio_full)
            self.baseline_teo = float(teo_chunk)
            self.buffer_audio = []
            self.calibrado = True
            return True
        return False

    def procesar(self, audio_chunk_raw: np.ndarray, sr_original: int = 22050) -> Dict[str, Any]:
        rms_valor = self._calcular_rms(audio_chunk_raw)

        # 0) Noise Gate
        if rms_valor < float(self.umbral_silencio_rms):
            return {
                "status": "silencio",
                "mensaje": "Esperando voz...",
                "score_estres": 0.0,
                "alerta_drift": False,
                "rms": round(rms_valor, 6),
            }

        # 1) Resample a 16kHz
        if int(sr_original) != int(self.sr_model):
            audio_16k = librosa.resample(
                audio_chunk_raw, orig_sr=int(sr_original), target_sr=int(self.sr_model)
            )
        else:
            audio_16k = audio_chunk_raw

        teo_actual = self._calcular_teo_rapido(audio_16k)

        # 2) Calibración
        if not self.calibrado:
            self._intentar_calibrar(audio_16k, teo_actual)
            progreso = float(self.samples_acumulados) / float(
                self.sr_model * self.buffer_seconds
            )
            return {
                "status": "calibrando_buffer",
                "progreso": min(1.0, round(progreso, 2)),
                "mensaje": "Aprendiendo patrón de voz...",
                "rms": round(rms_valor, 6),
            }

        # 3) Monitoreo
        assert self.anchor_emb is not None
        current_emb = self.encoder.embed_utterance(audio_16k)
        distancia = float(cosine(self.anchor_emb, current_emb))
        ratio_teo = float(teo_actual / (self.baseline_teo + 0.001))

        # Calcular std del TEO en este chunk (para detección de BOT)
        # El BOT tiene TEO muy estable → std bajo dentro del chunk
        teo_std = 0.0
        if len(audio_16k) > 100:
            # Calcular TEO en sub-ventanas dentro del chunk
            window = len(audio_16k) // 4
            teo_windows = []
            for i in range(4):
                sub = audio_16k[i * window:(i + 1) * window]
                if len(sub) >= 3:
                    teo_windows.append(self._calcular_teo_rapido(sub))
            if teo_windows:
                teo_std = float(np.std(teo_windows))

        expressive_metrics = self._calc_expressive_metrics(distancia)

        resultado: Dict[str, Any] = {
            "status": "monitoreando",
            "score_estres": 0.0,
            "alerta_drift": False,
            "tipo_evento": "normal",
            "rms": round(rms_valor, 6),
            "metricas": {
                "distancia_vectorial": round(distancia, 4),
                "teo_ratio": round(ratio_teo, 2),
                "teo_std": round(teo_std, 4),
                **expressive_metrics,
            },
        }

        # Árbol de decisión
        if distancia > float(self.umbral_drift):
            # ¿Es BOT? RMS alto + TEO estable + distancia alta
            if rms_valor > 0.05 and teo_std < 0.3 and ratio_teo < 3.0:
                resultado["tipo_evento"] = "posible_bot"
                resultado["alerta_drift"] = False
            else:
                resultado["alerta_drift"] = True
                resultado["tipo_evento"] = "drift_severo"
        elif distancia > float(self.umbral_estres_min):
            if ratio_teo > float(self.umbral_tension_ratio):
                score = (distancia * 100.0) + (ratio_teo * 10.0)
                resultado["score_estres"] = float(min(100.0, score))
                resultado["tipo_evento"] = "estres_detectado"
            else:
                resultado["tipo_evento"] = "variacion_expresiva"

        return resultado
