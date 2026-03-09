# ARENA USAC (Arquitectura AWS S3)

**Análisis de Riesgos, Esperas y Niveles Acústicos** adaptado para la nube.

Esta versión de ARENA está diseñada para operar de forma masiva en entornos AWS (como instancias EC2 o notebooks de SageMaker), procesando directamente desde S3.

## Resumen del Proyecto

ARENA actúa como una **Capa Cero** en el pipeline de analítica de voz (Speech Analytics). Extrae la "física" de la llamada (tiempos de silencio reales, calidad de la señal, estrés en las cuerdas vocales) para que los análisis hechos por Inteligencia Artificial (LLMs) sobre la transcripción tengan contexto acústico, evitando falsos positivos.

### Archivos del Repositorio

*   `arena/`: Módulo core con los algoritmos biométricos impulsados por Resemblyzer y librosa.
*   `00a_pipeline_arena_biometrics_aws.py`: Script de extracción masiva. Descarga audios temporalmente de S3, extrae su huella biométrica (`timeline.json`) y luego limpia la memoria.
*   `00b_arena_summary_aws.py`: Script de consolidación. Descarga los JSON de S3 directo a memoria, evalúa las reglas de negocio y expulsa un `arena_summary.csv` único.
*   `requirements.txt`: Dependencias del motor. 
*   `arena_business_slides.md`: Slides preparadas para explicar el valor del proyecto a áreas de negocio.
*   `README_aws.md`: Este documento de contexto técnico.

---

## Configuración S3 e Infraestructura

Dado que las ubicaciones de S3 de entrada y salida pueden variar por ambiente (Sandbox, Prod, etc.), los scripts AWS están pensados para ser lo más generales y configurables posible. 

Al comienzo de `00a` y `00b` encontrarás el bloque `CONFIGURACION AWS`. Ahí puedes modificar:
- Las fechas (`CUTOFF_DATES`)
- El bucket raíz (`BUCKET`)
- Rutas dinámicas S3 para leer `.wav` y depositar los `.json` y el `.csv` final.

### Recursos Computacionales (Instancias AWS)
Este pipeline es agnóstico al nivel de la instancia que utilices, pero se comporta diferente según el hardware:

*   **Procesamiento Inteligente (OOM Prevention):** El script `00a` procesa archivos multimedia. Para asegurar que instancias pequeñas con poca memoria RAM no colapsen (OOM), el script libera memoria iterativamente ejecutando `gc.collect()`.
*   **Paralelismo Variable (`MAX_WORKERS`):** Puedes aumentar los hilos concurrentes si usas una instancia grande, procesando audios en un par de segundos de manera simultánea.
*   **Aceleración Nativa GPU:** Si la infraestructura de AWS en la que resides posee GPU (por ejemplo, si corren ahí el modelo de Whisper y posee `cuda` activado), el pipeline biométrico de ARENA la detectará y utilizará automáticamente para el Voice Encoder, acelerando drásticamente el proceso. De lo contrario, utilizará el CPU de modo normal.

---

## Métricas Clave y Lógica de Calidad

Los scripts evalúan 4 dimensiones crudas antes de que el audio se envíe a transcripción, y han sido afinadas respecto al verdadero impacto sobre herramientas de conversión Speech-To-Text como Whisper:

1. **Silencios (Operatividad Real)**
   *   Mide el tiempo consolidado en el que ni el cliente ni el ejecutivo emitieron sonido. Esto cuantifica los "Hold" ciegos, que en las transcripciones puras no existen, desvelando fricciones omitidas por el texto.
2. **Riesgo de Transcripción (La verdadera "Calidad Sensible")**
   *   *¿Está el audio en condiciones de ser transcrito correctamente por una IA?*
   *   **Buena:** Relación Señal/Ruido (**SNR**) sana. La voz aplasta claramente al ruido de fondo.
   *   **Media:** El volumen es demasiado bajito o cuenta con ecos fuertes que harán que Whisper pierda algunas sílabas.
   *   **Baja:** El ruido es casi tan fuerte como la voz (SNR bajo). Riesgo severísimo de que el modelo comience a transcribir palabras inventadas basándose en el ruido.
   *   **Verificar:** El nivel electromagnético/Tensión está disparado (ej. música electrónica de espera o estática dura en la línea local). Causa N°1 de alucinaciones sintéticas de Whisper.
3. **Múltiples Voces (Calidad de Turnos)**
   *   Identifica cuándo demasiadas distancias vectoriales ocurren en la misma llamada, alertando de clientes/ejecutivos pisándose las voces o ruido en piso.
4. **Triage de Estrés del Cliente**
   *   Tomando una huella acústica durante las primeras interacciones, ARENA marca a quien inicia el contacto con las cuerdas vocales tensas (agresividad/desesperación natural humana), para que podamos saber si el ejecutivo mitigó un cliente enojado, o hizo enojar a un cliente neutral.

## Instalación Minimalista

1. Ubica el directorio del proyecto en el ambiente (SageMaker/EC2).
2. Ejecuta: `pip install -r requirements.txt`
3. Lanzas los archivos `00a` y `00b` secuencialmente según la cronología del Job.
