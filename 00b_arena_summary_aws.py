"""
00b — Resumen ARENA + Clasificacion por reglas (Version AWS S3)
================================================================
Lee los timeline JSONs generados por 00a_aws directamente desde S3.
Aplica reglas estrictas de SNR, silencios y voces.
Genera arena_summary.csv y lo sube a la ruta S3 correspondiente.

Optimizado para:
- Lectura en paralelo desde S3 (en memoria, sin archivos locales)
- Consolidacion de lote general para el/los dïas solicitados
"""

import json
import time
import boto3
import logging
from pathlib import Path
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURACION AWS
# ============================================================================
CUTOFF_DATES = [
    "2026-01-02"
]
BUCKET = "ada-us-east-1-sbx-live-mx-scxa-data"
PATH_PROJECT = "proyectos/coe_advanced_analytics_cx/usac_speech2text"
PATH_INPUT_TIMELINES = f"{PATH_PROJECT}/data/02_processed/00a_arena_timelines/inbound"
PATH_OUTPUT_CSV = f"{PATH_PROJECT}/data/02_processed/00b_arena_summary/arena_summary.csv"

MAX_WORKERS = 10 # Menos restricciones porque la descarga de JSON es ultraligera

# ============================================================================
# REGLAS DE NEGOCIO (Idénticas al script local 00b)
# ============================================================================

# Silencios
REGLA_ESPERA_PROLONGADA_SEC = 30       # Silencio > N segundos = espera prolongada
REGLA_ESPERA_LARGA_SEC = 15            # Umbral para contar esperas "largas"

# Calidad de audio (para transcripcion Whisper)
REGLA_CALIDAD_RMS_MUY_BAJO = 0.035    # RMS promedio debajo = voz bajita
REGLA_CALIDAD_SNR_BAJO = 3.0          # Ratio voz/ruido debajo = ruido interfiere
REGLA_CALIDAD_RMS_VARIACION = 2.0     # RMS std/avg ratio encima = eco o voces dispares
REGLA_CALIDAD_TEO_MIN = 2.0           # TEO avg encima + plano = artefacto / tono de espera
REGLA_CALIDAD_SLOPE_MAX = 0.15        # Slope debajo = tendencia plana

# Voces
REGLA_VOZ_LEJANA_DIST = 0.35          # Distancia al anchor encima = voz muy diferente
REGLA_VOZ_LEJANA_MIN_CHUNKS = 1       # Minimo de chunks lejanos para considerar 3ra voz

# Triage (estres inicial del cliente)
REGLA_TRIAGE_DIST_MIN = 0.18          # Distancia minima para considerar que es el cliente (no ES)
REGLA_TRIAGE_CHUNKS = 3               # Cuantos chunks post-calibracion analizar
REGLA_TRIAGE_TEO_UMBRAL = 4.0         # TEO inicio cliente encima = flag de triage (fijo, auditable)


# ============================================================================
# LOGGING
# ============================================================================
import os
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=f"logs/arena_00b_{date.today()}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(console)

s3 = boto3.client("s3")

# ============================================================================
# FUNCIONES
# ============================================================================

def read_s3_json(bucket: str, key: str) -> dict:
    """Descarga JSON de S3 directo a memoria, extrae KPIs y devuelve fila diccionario."""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        
        filename = key.split("/")[-1].replace(".timeline.json", "")
        points = data.get("points", [])
        
        return calcular_fila_kpis(filename, points)
    
    except Exception as e:
        logging.error(f"Error procesando {key}: {e}")
        return None


def calcular_fila_kpis(filename: str, points: list) -> dict:
    """Implementa la logica de conversion a KPI a partir de la lista de puntos."""
    monitoreando = [p for p in points if p.get("status") == "monitoreando"]
    silencios = [p for p in points if p.get("status") == "silencio"]
    total = len(points)
    
    duracion_sec = max((p.get("t_end_sec", 0) for p in points), default=0)
    duracion_min = round(duracion_sec / 60, 1)

    # ==================================================================
    # D1: SILENCIOS
    # ==================================================================
    pct_silencio = round(len(silencios) / total * 100, 1) if total > 0 else 0

    bloques_silencio = []
    blk_start, blk_end = None, None
    for p in points:
        if p.get("status") == "silencio":
            if blk_start is None:
                blk_start = p["t_start_sec"]
            blk_end = p["t_end_sec"]
        else:
            if blk_start is not None:
                bloques_silencio.append(blk_end - blk_start)
                blk_start = None
    if blk_start is not None:
        bloques_silencio.append(blk_end - blk_start)

    silencio_max_sec = round(max(bloques_silencio), 0) if bloques_silencio else 0
    n_esperas_largas = sum(1 for d in bloques_silencio if d > REGLA_ESPERA_LARGA_SEC)

    # ==================================================================
    # D2: CALIDAD DE AUDIO (SNR para Whisper)
    # ==================================================================
    rms_voz = [p.get("rms", 0) for p in monitoreando if p.get("rms", 0) > 0]
    rms_sil = [p.get("rms", 0) for p in silencios if p.get("rms", 0) > 0]

    rms_avg = float(np.mean(rms_voz)) if rms_voz else 0
    rms_std = float(np.std(rms_voz)) if rms_voz else 0
    rms_sil_avg = float(np.mean(rms_sil)) if rms_sil else 0

    rms_variacion = rms_std / (rms_avg + 1e-8)
    snr = rms_avg / rms_sil_avg if rms_sil_avg > 0 else 999.0

    if snr < REGLA_CALIDAD_SNR_BAJO:
        calidad_audio = "baja"
    elif rms_avg < REGLA_CALIDAD_RMS_MUY_BAJO or rms_variacion > REGLA_CALIDAD_RMS_VARIACION:
        calidad_audio = "media"
    else:
        calidad_audio = "buena"

    # Deteccion de artefacto musical / etática
    teos = [p["metricas"]["teo_ratio"] for p in monitoreando if isinstance(p.get("metricas"), dict) and "teo_ratio" in p["metricas"]]
    teo_avg = float(np.mean(teos)) if teos else 0
    times = [p["t_sec"] for p in monitoreando if isinstance(p.get("metricas"), dict) and "teo_ratio" in p["metricas"]]
    slope = float(np.polyfit(times, teos, 1)[0]) * 60 if len(times) > 2 else 0.0
    teo_artefacto = teo_avg > REGLA_CALIDAD_TEO_MIN and abs(slope) < REGLA_CALIDAD_SLOPE_MAX

    if teo_artefacto and calidad_audio == "buena":
        calidad_audio = "verificar"

    # ==================================================================
    # D3: VOCES
    # ==================================================================
    dists = [p.get("metricas", {}).get("distancia_vectorial", 0) for p in monitoreando if isinstance(p.get("metricas"), dict)]
    chunks_voz_lejana = sum(1 for d in dists if d > REGLA_VOZ_LEJANA_DIST)
    mas_de_2_voces = chunks_voz_lejana >= REGLA_VOZ_LEJANA_MIN_CHUNKS

    # ==================================================================
    # D4: TRIAGE
    # ==================================================================
    teo_inicio_cliente = None
    if len(monitoreando) >= REGLA_TRIAGE_CHUNKS:
        primeros = monitoreando[:REGLA_TRIAGE_CHUNKS]
        primer_dist = primeros[0].get("metricas", {}).get("distancia_vectorial", 0)

        if primer_dist > REGLA_TRIAGE_DIST_MIN:
            teos_inicio = [p.get("metricas", {}).get("teo_ratio", 0) for p in primeros]
            teo_inicio_cliente = round(float(np.mean(teos_inicio)), 1)

    return {
        "filename": filename,
        "duracion_min": duracion_min,
        "calidad_audio": calidad_audio,
        "pct_silencio": pct_silencio,
        "silencio_max_sec": int(silencio_max_sec),
        "n_esperas_largas": n_esperas_largas,
        "mas_de_2_voces": mas_de_2_voces,
        "teo_inicio_cliente": teo_inicio_cliente,
    }


def clasificar(row: dict) -> list:
    categorias = []
    if row["silencio_max_sec"] > REGLA_ESPERA_PROLONGADA_SEC:
        categorias.append("espera_prolongada")
    if row["calidad_audio"] in ("baja", "verificar"):
        categorias.append("revisar_audio")
    if row["mas_de_2_voces"]:
        categorias.append("multiples_voces")
    if row["triage_posible_estres"]:
        categorias.append("triage_estres_cliente")
    return categorias if categorias else ["normal"]


# ============================================================================
# MAIN
# ============================================================================
def main():
    logging.info("=" * 60)
    logging.info("🚀 00b — Resumen ARENA + Clasificacion (AWS S3)")
    logging.info("=" * 60)

    # 1. Obtener json files a lo largo de todos los cutoff_dates
    all_keys = []
    
    for d in CUTOFF_DATES:
        prefix = f"{PATH_INPUT_TIMELINES}/{d}/"
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        keys = [x["Key"] for x in resp.get("Contents", []) if x["Key"].endswith(".json")]
        all_keys.extend(keys)

    if not all_keys:
        logging.error(f"❌ No se encontraron timelines en los dias solicitados.")
        return

    logging.info(f"🔎 Procesando {len(all_keys)} timelines JSON en memoria con {MAX_WORKERS} hilos...")

    # 2. Descarga y cálculo en paralelo
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(read_s3_json, BUCKET, k) for k in all_keys]
        for idx, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                # Agregar un identificador A01, A02 secuencial
                res["audio_id"] = f"A{idx:05d}"
                rows.append(res)
                
            if idx % 100 == 0:
                logging.info(f"   -> {idx}/{len(all_keys)} json procesados.")

    df = pd.DataFrame(rows)

    # 3. Aplicar Triage (Umbral Fijo)
    df["triage_posible_estres"] = df.apply(
        lambda r: (
            r["teo_inicio_cliente"] is not None
            and not pd.isna(r["teo_inicio_cliente"])
            and r["teo_inicio_cliente"] > REGLA_TRIAGE_TEO_UMBRAL
            and r["calidad_audio"] not in ("baja", "verificar")
        ),
        axis=1,
    )

    # Categorias
    for idx, row in df.iterrows():
        df.at[idx, "categorias"] = ", ".join(clasificar(row.to_dict()))

    # Columnas finales
    col_order = [
        "audio_id", "filename", "duracion_min",
        "calidad_audio", "pct_silencio", "silencio_max_sec", "n_esperas_largas",
        "mas_de_2_voces", "teo_inicio_cliente", "triage_posible_estres", "categorias"
    ]
    df = df[col_order]

    # 4. Guardar local y subir a S3
    local_csv = "tmp/arena_summary.csv"
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    
    df.to_csv(local_csv, index=False, encoding="utf-8")
    s3.upload_file(local_csv, BUCKET, PATH_OUTPUT_CSV)
    
    os.remove(local_csv)
    
    logging.info(f"✅ CSV finalizado y subido a S3: s3://{BUCKET}/{PATH_OUTPUT_CSV}")
    
    # 5. Imprimir conteo
    logging.info("\nConteo General:")
    todas_cats = []
    for cats in df["categorias"]:
        todas_cats.extend(cats.split(", "))
    
    for cat in ["espera_prolongada", "revisar_audio", "multiples_voces", "triage_estres_cliente", "normal"]:
        n = todas_cats.count(cat)
        logging.info(f"  {cat}: {n} / {len(df)}")
    
    tf_min = (time.time() - t0) / 60
    logging.info(f"\n⏱️ PROCESO GLOBAL COMPLETADO EN {tf_min:.2f} MINUTOS.")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
