"""
ARENA — Audio Real-time Emotion & Anomaly Detection
=====================================================
Módulo autocontenido para análisis biométrico de voz.

Componentes:
    - MonitorAudioUnificado  : Motor central de detección (embeddings + TEO)
    - io_utils               : Carga de audio, chunking, export y resumen
"""

from .monitor import MonitorAudioUnificado

__all__ = ["MonitorAudioUnificado"]
