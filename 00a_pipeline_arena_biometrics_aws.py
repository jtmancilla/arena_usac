"""
00a — Extraccion ARENA (Version AWS S3 + Multihilo)
===================================================
Procesa audios WAV directamente desde S3 usando concurrencia.
Genera un timeline JSON por audio y lo sube a la ruta de S3 destino.

Optimizado para:
- Evitar OOM (Out Of Memory) descargando y borrando los Audios locales, y limpiando (gc).
- Ignorar archivos ya procesados (Resume automático).
- Usar ThreadPoolExecutor para paralelismo.
- Soporta GPU si está disponible mediante las librerías base de Resemblyzer/Torch.
"""

import os
import gc
import json
import time
import boto3
import logging
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

# Asegurar que importamos del modulo local
from arena.io_utils import export_timeline_json

# ============================================================================
# CONFIGURACION AWS
# ============================================================================

CUTOFF_DATES = [
    "2026-01-02"
]
BUCKET = "ada-us-east-1-sbx-live-mx-scxa-data"
PATH_PROJECT = "proyectos/coe_advanced_analytics_cx/usac_speech2text"
PATH_INPUT = f"{PATH_PROJECT}/data/01_raw/audios_usac/inbound"
PATH_OUTPUT = f"{PATH_PROJECT}/data/02_processed/00a_arena_timelines/inbound"

MAX_WORKERS = 4  # Ajustar según la RAM (Procesar audio requiere más RAM que LLM)

# Parametros ARENA v2
CHUNK_SEC = 3.0
BUFFER_SECONDS = 4.0
UMBRAL_SILENCIO_RMS = 0.008
UMBRAL_DRIFT = 0.42
UMBRAL_ESTRES_MIN = 0.18
UMBRAL_TENSION_RATIO = 1.5

# ============================================================================
# LOGGING
# ============================================================================
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=f"logs/arena_00a_{date.today()}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(console)

s3 = boto3.client("s3")

# ============================================================================
# FUNCIONES NUCLEO
# ============================================================================

def process_single_audio(s3_key: str, s3_output_path: str) -> bool:
    """Descarga de S3, procesa el audio, sube el JSON resultante, y limpia."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename = os.path.basename(s3_key)
    local_audio = f"tmp/{filename}"
    
    # Nombre del archivo json final
    json_filename = filename.replace(".wav", ".timeline.json")
    local_json = f"tmp/{json_filename}"
    s3_json_key = f"{s3_output_path}{json_filename}"

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    try:
        # 1. Descargar Audio
        s3_thread = boto3.client("s3")
        s3_thread.download_file(BUCKET, s3_key, local_audio)

        # 2. Procesar (Usa CPU o GPU automáticamente si torch+cuda están instalados)
        payload = export_timeline_json(
            input_audio=Path(local_audio),
            output_json=Path(local_json),      # lo guardamos temporalmente local
            device=device,
            chunk_sec=CHUNK_SEC,
            buffer_seconds=BUFFER_SECONDS,
            umbral_silencio_rms=UMBRAL_SILENCIO_RMS,
            umbral_drift=UMBRAL_DRIFT,
            umbral_estres_min=UMBRAL_ESTRES_MIN,
            umbral_tension_ratio=UMBRAL_TENSION_RATIO,
        )

        n_pts = len(payload.get("points", []))

        # 3. Subir el JSON a S3
        s3_thread.upload_file(local_json, BUCKET, s3_json_key)

        return True

    except Exception as e:
        logging.error(f"❌ Error procesando {filename}: {e}")
        return False

    finally:
        # 4. Limpieza estricta de memoria para no tirar la instancia (OOM)
        if os.path.exists(local_audio):
            os.remove(local_audio)
        if os.path.exists(local_json):
            os.remove(local_json)
        
        # Eliminar referencias en memoria para el GC
        if "payload" in locals():
            del payload
        gc.collect()


def batch_process_day(cutoff_date: str):
    start_time = time.time()
    logging.info(f"📅 --- INICIANDO DÍA: {cutoff_date} ---")

    prefix_in = f"{PATH_INPUT}/{cutoff_date}/"
    prefix_out = f"{PATH_OUTPUT}/{cutoff_date}/"

    # Obtener audios (.wav)
    resp_in = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix_in)
    audios = [x["Key"] for x in resp_in.get("Contents", []) if x["Key"].endswith(".wav")]

    # Obtener procesados (.json) para saltarlos
    resp_out = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix_out)
    jsons_out = [
        x["Key"].split("/")[-1].replace(".timeline.json", "") 
        for x in resp_out.get("Contents", []) 
        if x["Key"].endswith(".json")
    ]

    # Filtrar solo pendientes
    pendientes = [x for x in audios if x.split("/")[-1].replace(".wav", "") not in jsons_out]
    
    if not pendientes:
        logging.info(f"✅ Todos los audios (n={len(audios)}) de la fecha {cutoff_date} ya están procesados.")
        return

    logging.info(f"🔎 Procesando {len(pendientes)}/{len(audios)} audios pendientes (Workers={MAX_WORKERS})...")

    # Ejecucion paralela
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_audio, key, prefix_out): key for key in pendientes}
        
        for future in as_completed(futures):
            es_exito = future.result()
            if es_exito:
                success_count += 1
            
            # Simple log de avance
            if success_count > 0 and success_count % 100 == 0:
                logging.info(f"   -> Progreso: {success_count}/{len(pendientes)} procesados.")

    duration_min = (time.time() - start_time) / 60
    logging.info(f"✅ Día {cutoff_date} completado. Exitosos: {success_count}/{len(pendientes)}. Tiempo: {duration_min:.2f} min.")
    logging.info("-" * 50)


# ============================================================================
# ENTRYPOINT
# ============================================================================
def main():
    logging.info("=" * 60)
    logging.info("🚀 00a — Extraccion ARENA (AWS S3)")
    logging.info("=" * 60)
    
    t0 = time.time()
    for d in CUTOFF_DATES:
        try:
            batch_process_day(d)
        except Exception as e:
            logging.critical(f"❌ Error fatal en lote {d}: {e}")
    
    tf_min = (time.time() - t0) / 60
    logging.info(f"⏱️ PROCESO GLOBAL COMPLETADO EN {tf_min:.2f} MINUTOS.")
    logging.info("Siguiente paso: python 00b_arena_summary_aws.py")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
