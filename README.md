# ARENA Biometrics — Analisis acustico de llamadas

Modulo de analisis biometrico de voz integrado al pipeline USAC Speech2Text.
Procesa audios WAV y genera metricas acusticas que complementan el analisis
de texto realizado por el LLM (Whisper + Bedrock).

---

## Que hace ARENA

Analiza el audio crudo (no la transcripcion) en 4 dimensiones:

| Dimension | Que mide | El LLM no puede |
|---|---|---|
| Silencios | Duracion real de cada espera con timestamps | Solo estima por texto |
| Calidad de audio | Que tan procesable es para Whisper | Procesa igual un audio malo |
| Voces (mas de 2) | Si hay bot, otro agente, o EC | No tiene acceso al audio |
| Triage | Si el cliente llego estresado al inicio | No tiene acceso al audio |

---

## Uso

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Extraer datos crudos (genera 1 JSON por audio)
python 00a_pipeline_arena_biometrics.py

# 3. Generar resumen + clasificacion (genera arena_summary.csv)
python 00b_arena_summary.py
```

Requisitos: audios WAV en `audios_speech2text/`, Python 3.9+

---

## Salida: arena_summary.csv (11 columnas)

| Columna | Ejemplo | Descripcion |
|---|---|---|
| `audio_id` | A12 | Identificador legible |
| `filename` | 20250714... | Nombre del archivo (para merge con DataFrame LLM) |
| `duracion_min` | 3.1 | Duracion en minutos |
| `calidad_audio` | buena | Que tan procesable es para Whisper |
| `pct_silencio` | 47.6 | Porcentaje del audio sin voz |
| `silencio_max_sec` | 42 | Espera mas larga en segundos |
| `n_esperas_largas` | 2 | Esperas mayores a 15 segundos |
| `mas_de_2_voces` | True | Mas de 2 hablantes detectados (bot, EC, otro) |
| `teo_inicio_cliente` | 5.4 | TEO del cliente en los primeros segundos (NaN si no aplica) |
| `triage_posible_estres` | True | Flag: el cliente parece estresado desde el inicio |
| `categorias` | espera_prolongada | Clasificacion automatica |

---

## Categorias

Cada audio recibe una o mas categorias:

**espera_prolongada**
- Regla: `silencio_max_sec > 30`
- El cliente tuvo una espera mayor a 30 segundos.
- Validar contra el item `gestiona_espera` del checklist USAC.

**revisar_audio**
- Regla: calidad `baja` o `verificar`
- Audio con problemas de grabacion. La transcripcion Whisper podria tener errores.
- `baja`: voces casi inaudibles (RMS < 0.01).
- `verificar`: TEO uniformemente alto sin cambios. La grabacion tiene artefactos.

**multiples_voces**
- Regla: chunks con distancia al anchor > 0.35
- Mas de 2 hablantes. Puede ser bot BBVA, ejecutivo de cuenta, u otro agente.

**triage_estres_cliente**
- Regla: `teo_inicio_cliente > 4.0` y `calidad_audio` es buena
- El cliente tiene TEO elevado desde los primeros segundos de la llamada.
- No confirma estres. Dice "vale la pena revisarlo con mas atencion".
- Umbral fijo (4.0): mismo resultado siempre, auditable, no depende del batch.

**normal**
- Ninguna regla aplica. Sin hallazgos.

---

## Como leer los resultados

Abrir `arena_summary.csv` y ver la columna `categorias`:

| Categoria | Accion |
|---|---|
| `espera_prolongada` | Revisar `silencio_max_sec` |
| `revisar_audio` | Transcripcion Whisper podria tener errores |
| `multiples_voces` | Hubo bot, EC, u otro agente |
| `triage_estres_cliente` | Revisar con prioridad |
| `normal` | Sin hallazgos |

Ejemplo con los 20 audios de prueba:

```
ID    Min   CalAudio    %Sil  MaxSil  Esp>15 3+voz TEOcli Triage Categorias
A01   10.4  verificar    8.2    30       1    SI    24.6    -     revisar_audio, multiples_voces
A05   11.6  buena       25.9    66       3     -       -    -     espera_prolongada
A08    4.1  buena       76.8    75       3     -     1.8    -     espera_prolongada
A12    3.1  buena       47.6    42       2    SI     5.4   SI     espera_prolongada, multiples_voces, triage_estres_cliente
A04    1.7  buena        2.9     2       0     -     0.7    -     normal
```

Lectura:
- A12: La llamada mas critica. Espera de 42s, mas de 2 voces, y el cliente
  llego con TEO de 5.4 (arriba del umbral 4.0). Revisar con prioridad.
- A08: 76.8% silencio con espera maxima de 75s. Casi toda la llamada es espera.
- A01: Audio con artefactos. No confiar en metricas de TEO.
- A04: Normal, sin hallazgos.

---

## Merge con DataFrame LLM

```python
import pandas as pd

df_llm = pd.read_csv("data/llm_attributes.csv")
df_arena = pd.read_csv("data/arena_summary.csv")
df_final = df_llm.merge(df_arena, on="filename", how="left")
```

---

## Parametros

### Motor ARENA (01b)

Configurables al inicio de `00a_pipeline_arena_biometrics.py`:

| Parametro | Valor | Descripcion |
|---|---|---|
| `CHUNK_SEC` | 3.0 | Tamano del chunk en segundos |
| `BUFFER_SECONDS` | 4.0 | Segundos de calibracion con primer hablante |
| `UMBRAL_DRIFT` | 0.42 | Distancia coseno para drift severo |
| `UMBRAL_ESTRES_MIN` | 0.18 | Distancia minima para variacion expresiva |
| `UMBRAL_SILENCIO_RMS` | 0.008 | RMS debajo = silencio |

### Reglas de negocio (01c)

Configurables al inicio de `00b_arena_summary.py`:

| Parametro | Valor | Descripcion |
|---|---|---|
| `REGLA_ESPERA_PROLONGADA_SEC` | 30 | Silencio encima = espera prolongada |
| `REGLA_ESPERA_LARGA_SEC` | 15 | Umbral para contar esperas largas |
| `REGLA_CALIDAD_RMS_MUY_BAJO` | 0.01 | RMS debajo = voz inaudible |
| `REGLA_CALIDAD_RMS_VARIACION` | 2.0 | Ratio RMS std/avg encima = eco |
| `REGLA_CALIDAD_RUIDO_FONDO` | 0.005 | RMS de silencios encima = ruido alto |
| `REGLA_CALIDAD_TEO_MIN` | 2.0 | TEO avg encima + plano = artefacto |
| `REGLA_VOZ_LEJANA_DIST` | 0.35 | Distancia encima = tercera voz |
| `REGLA_VOZ_LEJANA_MIN_CHUNKS` | 1 | Minimo chunks para confirmar 3ra voz |
| `REGLA_TRIAGE_DIST_MIN` | 0.18 | Distancia minima para identificar al cliente |
| `REGLA_TRIAGE_CHUNKS` | 3 | Chunks post-calibracion para evaluar |
| `REGLA_TRIAGE_TEO_UMBRAL` | 4.0 | TEO encima = flag triage (fijo, auditable) |

---

## Limitaciones

- Audio mono: ARENA calibra con el primer hablante (ES). Cuando habla el
  cliente o EC, la distancia sube. Es comportamiento esperado.
- Voces en mono: ARENA detecta que hay mas de 2 voces pero no identifica
  quien es cada una (bot, EC, u otro).
- Triage: Es un flag de triaje, no un diagnostico. Indica que vale la pena
  revisar la llamada con mas atencion.
- Calidad verificar: Audios con TEO uniformemente alto tienen metricas de
  TEO no confiables. Silencios y voces si aplican.

---

## Estructura

```
arena/                              Modulo ARENA (motor, reutilizable)
  monitor.py                        Motor de deteccion v2
  io_utils.py                       Carga, segmentacion, export

00a_pipeline_arena_biometrics.py    Extraccion: WAV -> timeline JSONs
00b_arena_summary.py                Resumen: JSONs -> arena_summary.csv

data/
  arena_summary.csv                 Reporte final (11 columnas)
  arena_timelines/                  Dato crudo (1 JSON por audio)
```
