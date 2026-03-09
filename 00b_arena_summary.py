"""
00b — Resumen ARENA + Clasificacion por reglas
=================================================
Lee los timeline JSONs generados por 00a y produce:
  - arena_summary.csv con metricas confiables y clasificacion automatica

Dimensiones:
  1. Silencios          — Esperas reales con timestamps
  2. Calidad de audio   — Que tan procesable es para Whisper
  3. Voces              — Flag si hay mas de 2 voces (bot, otro agente, etc.)
  4. Triage             — Cliente con posible estres desde el inicio

Uso:
    python 00b_arena_summary.py

Input:
    data/arena_timelines/*.timeline.json

Output:
    data/arena_summary.csv
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURACION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
TIMELINES_DIR = SCRIPT_DIR / "data" / "arena_timelines"
OUTPUT_CSV = SCRIPT_DIR / "data" / "arena_summary.csv"

# --- Reglas de negocio (ajustables) ---

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================================
# FUNCIONES
# ============================================================================

def leer_timeline(path: Path) -> dict:
    """Lee un timeline JSON y extrae metricas confiables."""
    with open(path) as f:
        data = json.load(f)

    points = data["points"]
    meta = data.get("meta", {})
    filename = Path(meta.get("input", path.stem)).stem

    monitoreando = [p for p in points if p.get("status") == "monitoreando"]
    silencios = [p for p in points if p.get("status") == "silencio"]
    total = len(points)

    # Duracion
    duracion_sec = max((p.get("t_end_sec", 0) for p in points), default=0)
    duracion_min = round(duracion_sec / 60, 1)

    # ==================================================================
    # D1: SILENCIOS
    # ==================================================================
    pct_silencio = round(len(silencios) / total * 100, 1) if total > 0 else 0

    bloques_silencio = []
    blk_start = None
    blk_end = None
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
    # D2: CALIDAD DE AUDIO (para Whisper)
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

    # Deteccion de artefacto: TEO alto + plano = problema de grabacion
    teos = [
        p["metricas"]["teo_ratio"]
        for p in monitoreando
        if isinstance(p.get("metricas"), dict) and "teo_ratio" in p["metricas"]
    ]
    teo_avg = float(np.mean(teos)) if teos else 0
    times = [
        p["t_sec"]
        for p in monitoreando
        if isinstance(p.get("metricas"), dict) and "teo_ratio" in p["metricas"]
    ]
    slope = float(np.polyfit(times, teos, 1)[0]) * 60 if len(times) > 2 else 0.0
    teo_artefacto = teo_avg > REGLA_CALIDAD_TEO_MIN and abs(slope) < REGLA_CALIDAD_SLOPE_MAX

    if teo_artefacto and calidad_audio == "buena":
        calidad_audio = "verificar"

    # ==================================================================
    # D3: VOCES (flag si mas de 2)
    # ==================================================================
    dists = [
        p.get("metricas", {}).get("distancia_vectorial", 0)
        for p in monitoreando
        if isinstance(p.get("metricas"), dict)
    ]
    chunks_voz_lejana = sum(1 for d in dists if d > REGLA_VOZ_LEJANA_DIST)
    mas_de_2_voces = chunks_voz_lejana >= REGLA_VOZ_LEJANA_MIN_CHUNKS

    # ==================================================================
    # D4: TRIAGE (estres inicial del cliente)
    # ==================================================================
    teo_inicio_cliente = None

    if len(monitoreando) >= REGLA_TRIAGE_CHUNKS:
        primeros = monitoreando[:REGLA_TRIAGE_CHUNKS]
        primer_dist = primeros[0].get("metricas", {}).get("distancia_vectorial", 0)

        if primer_dist > REGLA_TRIAGE_DIST_MIN:
            teos_inicio = [
                p.get("metricas", {}).get("teo_ratio", 0)
                for p in primeros
            ]
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
    """Aplica reglas de negocio. Retorna lista de categorias."""
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
    logging.info("00b — Resumen ARENA + Clasificacion")
    logging.info("=" * 60)

    json_files = sorted(TIMELINES_DIR.glob("*.timeline.json"))
    if not json_files:
        logging.error(f"No se encontraron timelines en {TIMELINES_DIR}")
        logging.error("Ejecuta primero: python 00a_pipeline_arena_biometrics.py")
        return

    logging.info(f"{len(json_files)} timelines en {TIMELINES_DIR}")

    # Procesar
    rows = []
    for i, jf in enumerate(json_files, 1):
        summary = leer_timeline(jf)
        summary["audio_id"] = f"A{i:02d}"
        rows.append(summary)

    df = pd.DataFrame(rows)

    # Triage con umbral fijo
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

    # Ordenar columnas
    col_order = [
        "audio_id", "filename", "duracion_min",
        "calidad_audio",
        "pct_silencio", "silencio_max_sec", "n_esperas_largas",
        "mas_de_2_voces",
        "teo_inicio_cliente", "triage_posible_estres",
        "categorias",
    ]
    df = df[col_order]

    # Exportar
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # Mostrar
    logging.info("")
    header = (
        f"{'ID':<5} {'Min':>5} {'CalAudio':<10} {'%Sil':>5} {'MaxSil':>6} "
        f"{'Esp>15':>6} {'3+voz':>5} {'TEOcli':>6} {'Triage':>6} {'Categorias'}"
    )
    logging.info(header)
    logging.info("-" * len(header))

    for _, r in df.iterrows():
        teo_cli = f"{r['teo_inicio_cliente']:.1f}" if pd.notna(r["teo_inicio_cliente"]) else "  -"
        triage = "SI" if r["triage_posible_estres"] else "-"
        voces = "SI" if r["mas_de_2_voces"] else "-"
        logging.info(
            f"{r['audio_id']:<5} {r['duracion_min']:>5} {r['calidad_audio']:<10} "
            f"{r['pct_silencio']:>5} {r['silencio_max_sec']:>6} "
            f"{r['n_esperas_largas']:>6} {voces:>5} "
            f"{teo_cli:>6} {triage:>6} {r['categorias']}"
        )

    # Conteo
    logging.info("")
    logging.info("Clasificacion:")
    todas_cats = []
    for cats in df["categorias"]:
        todas_cats.extend(cats.split(", "))
    for cat in ["espera_prolongada", "revisar_audio", "multiples_voces",
                 "triage_estres_cliente", "normal"]:
        n = todas_cats.count(cat)
        if n > 0:
            logging.info(f"  {cat}: {n}/{len(df)}")

    logging.info("")
    logging.info(f"CSV: {OUTPUT_CSV}")
    logging.info(f"Columnas: {len(df.columns)}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
