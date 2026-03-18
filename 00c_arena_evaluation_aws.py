"""
00c — Evaluacion e Impacto de ARENA vs LLM (Version AWS S3)
===========================================================
Cruza los resumenes fisicos de ARENA con las extracciones linguisticas
del LLM (Bedrock) generadas en el paso 03. 

Su proposito es dual:
1. Extraer una muestra curada de audios extremos para QA (Comprobacion de ARENA).
2. Revelar los puntos ciegos (Falsos Positivos) donde la IA fallo por falta de contexto fisico.

Exporta metricas y consolidados como archivos .CSV a S3.
"""

import os
import json
import boto3
import logging
import pandas as pd
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CONFIGURACION
# ============================================================================
CUTOFF_DATES = ["2026-01-02"]
BUCKET = "ada-us-east-1-sbx-live-mx-scxa-data"
PATH_PROJECT = "proyectos/coe_advanced_analytics_cx/usac_speech2text"

# Entradas
PATH_ARENA_SUMMARY = f"{PATH_PROJECT}/data/02_processed/00b_arena_summary/arena_summary.csv"
PATH_LLM_JSONS = f"{PATH_PROJECT}/data/02_processed/02_bedrock_extraction/inbound"

# Salidas (Reporting)
PATH_OUTPUT_REPORTING = f"{PATH_PROJECT}/data/03_reporting/arena_evaluation"

# Nombres de archivos de salida parametrizados
FILENAME_QA_SAMPLE = "arena_qa_validation_sample.csv"
FILENAME_BLINDSPOTS = "arena_llm_blindspots_report.csv"

# Parametros de evaluacion
UMBRAL_EXCELENCIA_LLM = 90
MAX_WORKERS_S3 = 10

# ============================================================================
# LOGGING
# ============================================================================
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=f"logs/arena_00c_{date.today()}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(console)

s3 = boto3.client("s3")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def flatten_json(y, parent_key='', sep='_'):
    """Aplana recursivamente el JSON del LLM para convertirlo en tabular."""
    items = []
    for k, v in y.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def read_llm_json_from_s3(bucket: str, key: str) -> dict:
    """Descarga e interpreta un JSON de Bedrock desde S3 a memoria."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        content = resp['Body'].read().decode('utf-8')
        raw_json = json.loads(content)
        
        flat = flatten_json(raw_json)
        flat["filename_clean"] = key.split("/")[-1].replace(".json", "")
        return flat
    except Exception as e:
        logging.error(f"Error procesando LLM JSON en {key}: {e}")
        return None

def fetch_all_llm_data() -> pd.DataFrame:
    """Obtiene y aplana todos los JSON de Bedrock de las fechas seleccionadas."""
    logging.info(f"Buscando extracciones de Bedrock en S3 para cortes: {CUTOFF_DATES}")
    
    all_keys = []
    paginator = s3.get_paginator('list_objects_v2')
    
    for cutoff in CUTOFF_DATES:
        prefix = f"{PATH_LLM_JSONS}/{cutoff}/"
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    all_keys.append(obj['Key'])
                    
    if not all_keys:
        return pd.DataFrame()

    data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_S3) as executor:
        futures = [executor.submit(read_llm_json_from_s3, BUCKET, k) for k in all_keys]
        for f in as_completed(futures):
            res = f.result()
            if res:
                data.append(res)
                
    return pd.DataFrame(data)

def subir_csv_a_s3(df: pd.DataFrame, nombre_archivo: str):
    """Guarda un DataFrame temporalmente en local y lo sube a la carpeta de Reporting en S3."""
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    
    local_path = f"tmp/{nombre_archivo}"
    s3_path = f"{PATH_OUTPUT_REPORTING}/{nombre_archivo}"
    
    df.to_csv(local_path, index=False, encoding="utf-8")
    s3.upload_file(local_path, BUCKET, s3_path)
    os.remove(local_path)
    logging.info(f"Reporte subido a S3: s3://{BUCKET}/{s3_path}")

# ============================================================================
# CORE LOGICO
# ============================================================================
def main():
    logging.info("=" * 60)
    logging.info("00c — Evaluacion de ARENA vs Inteligencia Subjetiva (AWS)")
    logging.info("=" * 60)
    
    # 1. CARGAR DATOS ARENA
    logging.info("Descargando arena_summary.csv de S3...")
    if not os.path.exists("tmp"): os.makedirs("tmp")
    local_arena = "tmp/arena_summary_temp.csv"
    
    try:
        s3.download_file(BUCKET, PATH_ARENA_SUMMARY, local_arena)
        df_arena = pd.read_csv(local_arena)
        os.remove(local_arena)
    except Exception as e:
        logging.error(f"No se pudo descargar el Summary de ARENA: {e}")
        return

    df_arena["filename_clean"] = df_arena["filename"].astype(str).str.replace(".wav", "", regex=False)

    # 2. CARGAR Y APLANAR DATOS LLM BEDROCK
    df_llm = fetch_all_llm_data()
    if df_llm.empty:
        logging.error("No existen datos de texto/LLM en S3 para cruzar.")
        return

    # 3. CRUCE (INNER JOIN)
    df_merged = pd.merge(df_arena, df_llm, on="filename_clean", how="inner")
    logging.info(f"Cruce exitoso. Total de llamadas emparejadas: {len(df_merged)}")

    # =========================================================================
    # PARTE 1: METRICAS Y MUESTRA QA PARA COMPROBAR CALIDAD DEL SISTEMA ARENA
    # =========================================================================
    logging.info("--- PASO 1: EXTRACCION METRICAS Y MUESTRA DE VALIDACION DE ARENA ---")
    
    casos_ruido = len(df_merged[df_merged["calidad_audio"] == "baja"])
    casos_esperas = len(df_merged[df_merged["silencio_max_sec"] > 30])
    casos_tension = len(df_merged[df_merged["triage_posible_estres"] == True])
    
    logging.info(f"Metricas puras ARENA sobre el total ({len(df_merged)} llamadas):")
    logging.info(f"  -> Con ruido critico (inelegibles para STT): {casos_ruido}")
    logging.info(f"  -> Con silencios prolongados (>30s): {casos_esperas}")
    logging.info(f"  -> Clientes que inician con alta tension (Estres): {casos_tension}")

    # Extraer la muestra curada (30 audios) marcando claramente el motivo de la seleccion
    
    m_ruido = df_merged[df_merged["calidad_audio"] == "baja"].head(10).copy()
    if not m_ruido.empty:
        m_ruido["motivo_auditoria"] = "Ruido Critico"
        
    m_espera = df_merged[df_merged["silencio_max_sec"] > 30].sort_values(by="silencio_max_sec", ascending=False).head(10).copy()
    if not m_espera.empty:
        m_espera["motivo_auditoria"] = "Espera Prolongada (>30s)"
        
    m_tension = df_merged[df_merged["triage_posible_estres"] == True].sort_values(by="teo_inicio_cliente", ascending=False).head(10).copy()
    if not m_tension.empty:
        m_tension["motivo_auditoria"] = "Tension de Cliente"
    
    # Consolidar muestra QA
    frames_qa = [f for f in [m_ruido, m_espera, m_tension] if not f.empty]
    
    if frames_qa:
        df_qa_sample = pd.concat(frames_qa).drop_duplicates(subset=["filename_clean"])
        
        # Reordenar columnas para que la auditoria sea mas legible
        cols_base = ["motivo_auditoria", "filename_clean", "calidad_audio", "silencio_max_sec", "triage_posible_estres", "teo_inicio_cliente"]
        otras_cols = [c for c in df_qa_sample.columns if c not in cols_base]
        df_qa_sample = df_qa_sample[cols_base + otras_cols]
        
        subir_csv_a_s3(df_qa_sample, FILENAME_QA_SAMPLE)
    
    # =========================================================================
    # PARTE 2: COMPARATIVO DURO (LLM FALLANDO VS ARENA ACERTANDO)
    # =========================================================================
    logging.info("--- PASO 2: IMPACTO Y PUNTOS CIEGOS DEL LLM ---")
    
    col_puntajes = [c for c in df_merged.columns if "puntaje" in c.lower() or "score" in c.lower() or "calificacion" in c.lower()]
    
    if col_puntajes:
        calificacion_ia = col_puntajes[0]
        logging.info(f"Utilizando '{calificacion_ia}' como columna de evaluacion del LLM.")
        
        df_blindspots = df_merged[
            (pd.to_numeric(df_merged[calificacion_ia], errors='coerce') >= UMBRAL_EXCELENCIA_LLM) & 
            (
                (df_merged["silencio_max_sec"] > 30) | 
                (df_merged["calidad_audio"].isin(["baja", "verificar"])) |
                (df_merged["triage_posible_estres"] == True)
            )
        ]
        
        casos_ciegos = len(df_blindspots)
        pct = round(casos_ciegos/len(df_merged)*100, 1) if len(df_merged) > 0 else 0
        logging.info(f"Falsos Positivos del LLM: {casos_ciegos} llamadas ({pct}% del pool evaluado).")
        
        if casos_ciegos > 0:
            df_blindspots = df_blindspots.sort_values(by="silencio_max_sec", ascending=False)
            subir_csv_a_s3(df_blindspots, FILENAME_BLINDSPOTS)
    else:
        logging.warning("No se encontro columna de puntaje en el LLM para procesar puntos ciegos.")

    logging.info("=" * 60)
    logging.info("PROCESO 00c FINALIZADO.")

if __name__ == "__main__":
    main()
