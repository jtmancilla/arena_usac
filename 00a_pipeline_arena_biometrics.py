"""
00a — Extraccion ARENA (dato crudo)
=====================================
Procesa cada audio WAV con ARENA y genera un timeline JSON por audio.
No genera resumen ni interpretacion — eso lo hace 00b.

Uso:
    python 00a_pipeline_arena_biometrics.py

Output:
    data/arena_timelines/*.timeline.json
"""

import os
import time
import logging
from pathlib import Path
from datetime import date

from arena.io_utils import export_timeline_json

# ============================================================================
# CONFIGURACION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = SCRIPT_DIR / "audios_speech2text"
TIMELINES_DIR = SCRIPT_DIR / "data" / "arena_timelines"
LOG_DIR = SCRIPT_DIR / "logs"

# Parametros ARENA v2 (optimizados para USAC batch, mono, 2-3 hablantes)
CHUNK_SEC = 3.0
BUFFER_SECONDS = 4.0
UMBRAL_SILENCIO_RMS = 0.008
UMBRAL_DRIFT = 0.42
UMBRAL_ESTRES_MIN = 0.18
UMBRAL_TENSION_RATIO = 1.5

# Logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / f"arena_biometrics_{date.today()}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(console)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logging.info("=" * 60)
    logging.info("00a — Extraccion ARENA (timeline JSONs)")
    logging.info("=" * 60)
    logging.info(
        f"Parametros: chunk={CHUNK_SEC}s | drift={UMBRAL_DRIFT} | "
        f"estres={UMBRAL_ESTRES_MIN} | silencio={UMBRAL_SILENCIO_RMS}"
    )

    TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        logging.error(f"No se encontraron WAV en {AUDIO_DIR}")
        return

    logging.info(f"{len(wav_files)} audios en {AUDIO_DIR}")
    logging.info("")

    start_total = time.time()
    ok = 0

    for i, wav_path in enumerate(wav_files, 1):
        fname = wav_path.stem
        fecha = f"{fname[0:4]}-{fname[4:6]}-{fname[6:8]} {fname[8:10]}:{fname[10:12]}"

        logging.info(f"[{i}/{len(wav_files)}] A{i:02d} ({fecha})")

        try:
            t0 = time.time()
            timeline_path = TIMELINES_DIR / f"{fname}.timeline.json"
            payload = export_timeline_json(
                input_audio=wav_path,
                output_json=timeline_path,
                chunk_sec=CHUNK_SEC,
                buffer_seconds=BUFFER_SECONDS,
                umbral_silencio_rms=UMBRAL_SILENCIO_RMS,
                umbral_drift=UMBRAL_DRIFT,
                umbral_estres_min=UMBRAL_ESTRES_MIN,
                umbral_tension_ratio=UMBRAL_TENSION_RATIO,
            )
            n_pts = len(payload.get("points", []))
            elapsed = time.time() - t0
            logging.info(f"    {n_pts} chunks -> {timeline_path.name} ({elapsed:.1f}s)")
            ok += 1
        except Exception as e:
            logging.error(f"    Error: {e}")

    elapsed_min = (time.time() - start_total) / 60
    logging.info("")
    logging.info(f"{ok}/{len(wav_files)} audios procesados en {elapsed_min:.1f} min")
    logging.info(f"Timelines en: {TIMELINES_DIR}")
    logging.info("Siguiente paso: python 00b_arena_summary.py")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
