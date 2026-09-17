# Estado de la remediación

> **Esta sección se añadió el 2026-09-17.** El resto del documento es la
> auditoría tal como se entregó el 2026-09-16 y **no se ha modificado**: describe
> el proyecto como estaba entonces. Léelo con esta sección delante, porque en dos
> puntos la auditoría se quedó corta y aquí se dice cuáles.

Los identificadores de este documento (`P0-1`, `P1-02`, `REF-03`, `PERF-04`,
`SPEC-DM-001`…) se citan desde mensajes de commit y desde comentarios del código.
Por eso el documento vive en el repositorio y no fuera de él.

## Resumen

**38 de los 43 ítems del plan de la sección 37 están cerrados.** La suite pasó de
205 a 374 tests. Ningún cierre se dio por bueno sin verificación: cada corrección
se comprobó revirtiéndola en aislamiento y confirmando que un test falla, y los
tres refactors grandes se verificaron con arneses diferenciales contra el código
anterior.

| Fase | Ítems | Estado |
|---|---|---|
| 0 — P0 bloqueantes | 1-5 | completa |
| 1 — Correctness | 6-14 | completa |
| 2 — Fiabilidad | 15-21 | completa |
| 3 — Infraestructura | 22-28 | 6,5 de 7 |
| 4 — Arquitectura | 29-34 | 5 de 6 |
| 5 — Performance | 35-38 | 1,5 de 4 |
| 6 — Mantenibilidad | 39-43 | 4 de 5 |

## Ítem por ítem

| # | ID | Estado | Commit |
|---|---|---|---|
| 1 | P0-1 marco de coordenadas | cerrado | `cb13a30` |
| 2 | P0-2 buffer del CSV sin vaciar | cerrado | `cb13a30` |
| 3 | P0-3 off-by-one al reanudar | cerrado | `cb13a30` |
| 4 | P0-4 checkpoint tras chunk fallido | cerrado | `cb13a30` |
| 5 | Tests de regresión de los P0 | cerrado | `b21a833` |
| 6 | P1-01 solape derecho ignorado | cerrado | `a50bd52` |
| 7 | P1-12 `snr_pre_dedisp` sobre el eje DM | cerrado | `cb13a30` |
| 8 | P1-02 inversión de frecuencia | **cerrado, ver corrección 1** | `a062378`, `98ed151` |
| 9 | P1-03 rejilla DM corrupta al trocear | cerrado | `a50bd52` |
| 10 | P1-04/05/06 geometría de chunks | cerrado | `a50bd52` |
| 11 | P1-11 polarización en SPEC-HF-002 | cerrado | `a50bd52` |
| 12 | P1-10 centinelas de `is_burst` | cerrado | `977a2ba` |
| 13 | P1-07, P1-08 MJD baricéntrico | cerrado | `977a2ba` |
| 14 | P1-09 contaminación de `TSTART_MJD_CORR` | cerrado | `a50bd52` |
| 15-20 | P1-13…P1-19, P2-01, P2-02 | cerrado | `a76ca85` |
| 21 | Tests de los caminos críticos | cerrado | `69a38ac`, `c0c75c9` |
| 22 | `docker-compose.yml` | cerrado | `14c721d` |
| 23 | P1-21, P1-22 Dockerfile y CI | cerrado, **build sin ejecutar** | `14c721d` |
| 24 | P2-32, P2-33 rotación y persistencia de logs | cerrado | `14c721d` |
| 25 | P2-34…P2-36 CI | cerrado | `14c721d` |
| 26 | REF-19, REF-20 código muerto y `src/logging` | cerrado | `780c492` |
| 27 | P2-29, REF-07 `advanced-config/` | **parcial** | `9eb891a` |
| 28 | P2-28 coherencia de `config.yaml` | cerrado | `14c721d` |
| 29 | REF-01 driver de archivo unificado | cerrado | `4e54d8b` |
| 30 | REF-02 subir el dispatch LF/HF | cerrado | `dd41c43` |
| 31 | REF-09 romper los ciclos | cerrado | `8256bb5` |
| 32 | REF-05 función de 884 líneas del HF | cerrado | `dd41c43` |
| 33 | REF-10 adoptar los contratos | **pendiente** | — |
| 34 | REF-03 separar los lectores de `stream_fits` | cerrado | `f8ef15a` |
| 35 | P1-23 `force_plots` | cerrado | `9eb891a` |
| 36 | PERF-04 GC, batching, `cudnn.benchmark` | **parcial** | `9eb891a` |
| 37 | PERF-03, PERF-05 copias de arrays | **pendiente** | — |
| 38 | PERF-02 dedispersión torch vectorizada | **pendiente** | — |
| 39 | REF-11, REF-13, REF-16 duplicación | cerrado | `a76ca85`, `4e54d8b` |
| 40 | REF-04 `create_composite_plot` | cerrado | `d1c3856` |
| 41 | REF-14, REF-15, REF-18 scripts | cerrado | `6f38a95` |
| 42 | Reorganizar `src/scripts/` y `src/tests/` | **pendiente** | — |
| 43 | CHANGELOG, README y las 2 SPECs falsas | cerrado | `8256bb5` |

## Dos correcciones a la auditoría

### 1. P1-02 tenía tres sitios, no dos, y ya no es una incógnita

La auditoría lo dejó sin resolver porque exigía comprobar el comportamiento de
`your`. Se comprobó, y el criterio `foff > 0` era el invertido:

- `your/formats/psrfits.py` **no normaliza nada**: el flip está comentado y
  `need_flipband` es `False` fijo. Verificado sobre PSRFITS sintético en ambas
  orientaciones, con `blimpy` como tercera opinión.
- Un burst inyectado a DM 500 en un fichero descendente se recupera en DM 500
  exacto bajo `DATA_NEEDS_REVERSAL`, y en **DM 748 con S/N 10,4** bajo `foff > 0`.
  El modo de fallo no es perder el burst: es reportarlo con un DM fabricado que
  supera el umbral de detección.

Y había un tercer sitio que la auditoría no listó: **la ruta astropy primaria de
`stream_fits` no invertía canales en absoluto**, mientras la copia duplicada de
más abajo sí lo hacía. Cualquier instalación sin `your` procesaba PSRFITS
descendentes con el eje de canales intacto.

### 2. D7: el fichero se emitía dos veces. La auditoría no lo detectó

`stream_fits` es un generador, y el `except Exception` que elegía el fallback
estaba fuera de un bucle que ya había hecho `yield`. El manejador de cola
desreferenciaba `out_buf` después de que el bucle lo hubiera puesto a `None`,
cosa que ocurre **siempre que `FILE_LENG % chunk_samples == 0` sin solape** —
divisibilidad, no un caso raro. La excepción nunca llegaba al llamador: el lector
duplicado reabría el fichero y reemitía desde el sample 0.

Medido sobre 512 samples con `chunk=256`, el consumidor recibía
`[(0,256),(256,512),(0,256),(256,512)]`: cada sample dos veces, con solo un aviso
genérico de *falling back to astropy* en el log. Todo candidato de ese fichero se
encontraba y se escribía por duplicado.

Salió de los tests de caracterización construidos **después** de la auditoría, no
de la auditoría misma. Es el argumento más fuerte a favor de REF-03: la decisión
primario-vs-fallback tiene que tomarse antes del primer `yield`, porque lo
emitido no se puede retirar.

## Lo que queda, y por qué

### Necesita hardware que aquí no hay

PERF-02 y la parte restante de PERF-04 —batching de inferencia, AMP,
`cudnn.benchmark`, `channels_last`— tocan la ruta torch-GPU. La máquina donde se
hizo este trabajo no tiene GPU, así que no se pueden verificar aquí.

### Necesita una decisión del responsable del proyecto

- **Reprocesar los catálogos existentes.** P0-1 invalidó todos los DM y tiempos
  de llegada del pipeline LF; P1-02 los de PSRFITS. Esta es la consecuencia
  práctica de toda la auditoría.
- **`advanced-config/models.yaml`**: 177 líneas que se cargan y nadie lee
  (ítem 27). `visualization.yaml` y `logging.yaml` ya están cableados. Cablear o
  borrar.
- **Arrancar Docker** para verificar el build del ítem 23.

### Deuda técnica generada por la propia remediación

- **Cuatro tests de la fase 0 afirman sobre el texto fuente** de los drivers
  (`src.count("samples_to_remove = actual_chunk_size") == 2`, un `try` cuya última
  sentencia sea `chunk_succeeded = True`, …). Se escribieron así porque entonces
  nada podía ejercitar esas rutas, y han bloqueado las últimas líneas de
  duplicación tanto en REF-01 como en REF-03. Hay que reescribirlos contra
  comportamiento; los tests que lo permiten ya existen.
- **Cuatro defectos del pipeline HF** destapados por REF-01, documentados en el
  código y sin corregir. El más serio: HF relanza en vez de devolver un
  resultado, y su único llamador está dentro del `try` de LF, que lo convierte en
  `_error_result` usando las `DetectionStats` **vacías de LF**. Una corrida que
  escribe 500 candidatos y luego falla reporta `n_candidates: 0` con las filas ya
  en disco — justo el fallo que `_error_result` dice prevenir.
- **REF-12**: `off_regions`, parámetro documentado como no usado, con 49
  referencias todavía.

### Pendiente de plan

REF-10 (ítem 33) es el cambio arquitectónicamente más valioso y el más caro; la
auditoría pide hacerlo incremental y con tests de paridad. El ítem 42,
reorganizar `src/scripts/` y `src/tests/`, sigue abierto.

## Entorno

La suite no corría al empezar esta remediación: faltaban `psutil`, `matplotlib`,
`astropy` y `torch`. Se creó `.venv` sobre Python 3.12 —la misma versión que fija
el Dockerfile— con `requirements.txt` completo.

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

---

# Auditoría técnica — DRAFTS-UC

**Repositorio:** DRAFTS++ (pipeline de detección de FRB: CenterNet + ResNet sobre FITS/filterbank)
**Rama auditada:** `dev` @ `65419c9`
**Fecha:** 2026-09-16
**Alcance:** 151 archivos versionados, ~32.000 líneas Python, 593 commits
**Método:** 10 subagentes especializados en paralelo (arquitectura, correctness, seguridad, performance, mantenibilidad, testing, datos/concurrencia/errores, DevOps/configuración, patrones/SOLID, higiene), seguidos de consolidación y verificación cruzada manual de todos los P0 y de una muestra de los P1.

**Restricción cumplida:** no se modificó, creó ni borró ningún archivo del repositorio. Las únicas escrituras fueron a un directorio temporal de sesión.

---

## 1. Executive Summary

### Estado general

El proyecto tiene una base de ingeniería considerablemente mejor que la media de un repositorio científico: existe un documento de contratos físicos verificables (`SPECS-physics.md`) con identificadores SPEC-* referenciados desde los commits y **forzados por tests automáticos**, hay un lockfile con 1273 hashes, hay mutation testing configurado, hay CI, hay Docker y hay un sistema de checkpoint con escritura atómica. Nada de eso es habitual.

Sin embargo, la auditoría encontró **cuatro defectos P0 confirmados**, y el más grave de ellos invalida silenciosamente los resultados científicos del pipeline principal. El patrón común de los cuatro no es descuido puntual, sino la ausencia de dos invariantes explícitas: *en qué marco de coordenadas vive cada magnitud* y *qué significa exactamente "completado"*.

El repositorio **no está listo para producción científica** en su estado actual. Los P0 son corregibles con cambios pequeños y localizados —ninguno requiere rediseño— pero hasta que se corrijan, cualquier catálogo de candidatos producido por el pipeline de baja frecuencia debe considerarse inválido.

### Recuento

| Severidad | Cantidad | Naturaleza |
|---|---|---|
| **P0** | 4 | 1 de corrección científica, 3 de pérdida silenciosa de datos |
| **P1** | 24 | Correctness condicionada, integridad de datos, seguridad de cadena de suministro, performance a escala |
| **P2** | 38 | Impacto real pero acotado o condicionado |
| **P3** | 31 | Deuda técnica, higiene, inconsistencias documentales |

### Los cuatro P0

| ID | Título | Efecto |
|---|---|---|
| **P0-1** | Las cajas de CenterNet se interpretan en el marco del cubo DM en vez del marco 512×512 del CNN | DM y tiempo de **todos** los candidatos del pipeline LF son sistemáticamente incorrectos |
| **P0-2** | El pipeline HF nunca vacía el buffer del CSV | Se pierden hasta 49 candidatos por archivo; con menos de 50, se pierden todos |
| **P0-3** | La reanudación por checkpoint salta un chunk de más | Pérdida determinista de un chunk por cada reanudación |
| **P0-4** | El checkpoint se guarda también tras un chunk fallido | El mecanismo de recuperación garantiza que lo perdido no se recupere nunca |

Los tres últimos se componen entre sí: un chunk falla (P0-4 lo marca completo), se reanuda (P0-3 salta además el siguiente), y en HF el buffer nunca se vacía (P0-2). La corrida de recuperación reporta `SUCCESS_CHUNKED` limpio habiendo perdido dos chunks y sus candidatos.

### Riesgos principales

1. **Resultados plausibles pero incorrectos.** Ninguno de los cuatro P0 produce una excepción. El pipeline corre, genera CSV, genera plots y reporta éxito. P0-1 en particular produce DMs dentro del rango configurado, indistinguibles de detecciones legítimas.
2. **La imagen Docker no es el entorno auditado.** El `Dockerfile` copia `requirements.txt` y nunca lo instala; en su lugar instala una lista escrita a mano con `torch==2.1.0`, `numpy==1.24.3`, Python 3.10 y 15 paquetes sin versión, mientras el proyecto declara `requires-python >=3.11`, `numpy>=2.4` y `torch>=2.11`. El lockfile con hashes —el activo de seguridad real del repositorio— no protege la imagen.
3. **El coste de ejecución está dominado por un flag de configuración.** `debug.force_plots: true` en `config.yaml` obliga a renderizar 4 PNG a 300 dpi por cada slice, tenga candidatos o no. A escala de terabytes son millones de imágenes y órdenes de magnitud más tiempo que la propia dedispersión.
4. **Configuración que promete lo que no cumple.** Tres de los cuatro YAML de `advanced-config/` (579 líneas) se cargan y nunca se leen. Un operador que ajuste el modelo, el logging o la visualización desde esos archivos no está ajustando nada.

### Fortalezas reales

- **`SPECS-physics.md` y el sistema de SPEC-IDs.** Contratos físicos verificables, cada uno con test asociado, con enforcement automático (`test_spec_constants.py::test_no_hardcoded_kdm_literals` escanea `src/` prohibiendo literales de la constante de dispersión). Es el mejor mecanismo de gobernanza del proyecto y debe ampliarse, no sustituirse.
- **`requirements.lock.txt`** con 88 paquetes, 1273 hashes y el comando de regeneración documentado en su cabecera.
- **Streaming real.** El archivo no se carga completo: `stream_fil` usa `np.memmap` y `stream_fits` emite por subints. El presupuesto de memoria está explícitamente calculado.
- **`candidate_finalization.finalize_patch()`** (SPEC-CAND-001) es la extracción correcta, hecha correctamente. Es el modelo a seguir para el resto del refactor.
- **Fallbacks graceful** GPU→CPU, Numba→Torch→NumPy, torch ausente→clasificador SNR, matplotlib ausente. Permiten correr en CI sin GPU y en HPC sin display.
- **Higiene de marcadores:** cero TODO/FIXME/HACK reales, cero wildcard imports, cero `__pycache__` versionado, y los tres manifiestos de dependencias (`pyproject.toml`, `requirements.txt`, `requirements.lock.txt`) son 100 % consistentes entre sí.

### Blockers de producción

1. P0-1 — corrección científica del pipeline LF.
2. P0-2, P0-3, P0-4 — durabilidad de los candidatos.
3. `docker-compose.yml` monta `D:/Your/Data/Path`: el proyecto no arranca tras un `git clone` sin editar el compose.
4. El Dockerfile ejecuta un stack de dependencias que CI nunca prueba.
5. Los logs se pierden con el modo de uso documentado (`docker compose run --rm`) y no tienen rotación.

---

## 2. System Map

```
                         main.py  (CLI → dict → inject_config)
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ src/core/pipeline.py   run_pipeline() → _process_file_chunked()       │
 │   · descubre ficheros, carga modelos, bucle target → file → chunk     │
 │   ├── decisión bow-tie [pipeline.py:668-700] ───────────┐             │
 │   ▼ LF                                           HF ▼   │             │
 │ _process_block()                      high_freq_pipeline.py           │
 │   │                                   _process_file_chunked_high_freq │
 │   ▼                                              │                    │
 │ detection_engine.py                 snr_detect_and_classify...()      │
 │ (CenterNet → ResNet)                (SNR peak → L-pol → ResNet)       │
 └────────┬─────────────────────────────────────────┬────────────────────┘
          │        candidate_finalization.py        │   ← única pieza
          │        finalize_patch() (SPEC-CAND-001) │     realmente compartida
          ▼                                         ▼
 ╔══════════════════════════════════════════════════════════════════════╗
 ║ ESTADO GLOBAL MUTABLE:  src/config/config.py  (módulo-singleton)     ║
 ║ leído por 31 módulos · mutado desde 6 en runtime · sin reset         ║
 ╚══════════════════════════════════════════════════════════════════════╝
      ▲          ▲            ▲             ▲            ▲
  input/    preprocessing/  output/   visualization/  analysis/
  fits,fil  dedisp, slice   CSV,      matplotlib      SNR, K_DM
  params    planner, dsamp  metrics
```

**Flujo de datos real:**

```
file → stream(chunk+overlap) → downsample → cubo DM → trim(overlap)
     → slices → detect → classify → finalize_patch → Candidate → CSV + PNG
```

**Estilo arquitectónico real:** monolito batch, un solo proceso, en estilo *pipes-and-filters* imperativo, con dos ramas de ejecución paralelas y estructuralmente duplicadas (LF/HF) y un módulo de configuración global mutable actuando como bus de datos implícito.

**No** es una arquitectura por capas: `core` importa `visualization` a nivel de módulo y `visualization` importa `core` a nivel de módulo. No hay puertos ni adaptadores pese al nombre `contracts.py`.

---

## 3. Repository Map

```
DRAFTS-UC/
├── main.py                    entrypoint CLI                          ACTIVO
├── config.yaml                configuración principal (36 claves)     ACTIVO
├── advanced-config/           4 YAML; solo 1 de 23 claves se consume  CASI MUERTA
├── src/
│   ├── core/          (12)    orquestación LF/HF, checkpoint,         ACTIVO
│   │                          contracts, mjd, hardware
│   ├── input/         (10)    lectores FITS/fil, parámetros, pol      ACTIVO
│   ├── preprocessing/ (6)     dedispersión, chunking, downsampling    ACTIVO
│   ├── detection/     (2)     interfaz de inferencia                  ACTIVO
│   ├── models/        (3+2)   redes + 120 MB de pesos .pth            ACTIVO
│   ├── analysis/      (3)     K_DM, SNR, métricas físicas             ACTIVO
│   ├── output/        (6)     CSV, métricas, resúmenes                ACTIVO (1 muerto)
│   ├── visualization/ (11)    8 módulos de plotting                   ACTIVO (1 deshabilitado)
│   ├── logging/       (6)     logger global                    ACTIVO (1 módulo muerto, NOMBRE PELIGROSO)
│   ├── config/        (2)     carga YAML + singleton global           ACTIVO
│   ├── domain/        (2)     re-export puro                     PASSTHROUGH, solo tests
│   ├── training/      (7)     entrenamiento offline                   FUERA DE LA RUTA
│   ├── scripts/       (22)    herramientas de tesis + one-shots       MIXTO, 0 documentados
│   └── tests/         (4)     2 scripts de análisis, NO son tests     CONFUSO
├── tests/             (17)    suite pytest real                       ACTIVO
├── mutation/          (9)     cosmic-ray: 4 toml + 4 sqlite (2,9 MB)  ARTEFACTOS
├── .github/workflows/         CI: solo pytest                         ACTIVO
├── Dockerfile                 stack divergente del declarado          ACTIVO, DIVERGENTE
├── docker-compose.yml         ruta de desarrollador hardcodeada       ROTO TRAS CLONE
├── SPECS-physics.md           contratos físicos verificables          ACTIVO, EXCELENTE
└── CHANGELOG.md               última entrada 2025-11-14               OBSOLETO
```

### Responsabilidad, consumidores y estado por carpeta

| Carpeta | Responsabilidad | Consumidores | Estado |
|---|---|---|---|
| `src/core/` | Orquestación, contratos, checkpoint, tiempos | `main.py` | Activa. Contiene los 4 P0 |
| `src/input/` | Lectura de formatos y extracción de metadatos | `core` | Activa. `fits_handler.py` = 1851 líneas, 3 % de cobertura |
| `src/preprocessing/` | Dedispersión, planificación de chunks y slices | `core` | Activa. Núcleo científico |
| `src/analysis/` | Constante de dispersión, SNR, significancia | 12 módulos | Activa. Magneto sano |
| `src/output/` | Persistencia de candidatos y métricas | `core`, `detection` | Activa. `summary_manager.py` (217 L) muerto |
| `src/visualization/` | Generación de figuras | `core` | Activa. ~19 % duplicado, `plot_patches.py` deshabilitado hace 10 meses |
| `src/logging/` | Logger global | 9 módulos | Activa. **Sombrea el módulo estándar `logging`** |
| `src/domain/` | Re-export de física | Solo tests | Passthrough sin adoptar |
| `src/scripts/` | Herramientas de tesis y diagnóstico | Ninguno automático | Mixto: legítimas, one-shots consumidos, 3 con imports rotos |
| `src/tests/` | 2 scripts de simulación | Ninguno | Confuso: no son tests, el README raíz manda ejecutarlos con pytest |
| `src/training/` | Entrenamiento de las dos redes | Manual | Fuera de la ruta de producción |
| `advanced-config/` | Configuración avanzada | `user_config.py` (carga) | 3 de 4 archivos nunca se leen |
| `mutation/` | Sesiones de cosmic-ray | CI (`workflow_dispatch`) | Artefactos generados versionados |

---

## 4. Arquitectura actual

### Fan-in / fan-out

| Módulo | Responsabilidad real | fan-in | fan-out | Observación |
|---|---|---|---|---|
| `src/config/config.py` | Config de usuario + metadata por archivo + device | **31** | 1 | Magneto patológico: mutable |
| `src/analysis/science_metrics.py` | `K_DM_MS`, delays, sigma post-trials | **12** | 0 | Magneto sano: puro e inmutable |
| `src/analysis/snr_utils.py` | Perfil SNR matched-filter | **12** | 1 | Magneto sano |
| `src/core/pipeline.py` | Orquestador LF + entrypoint | 2 | **23** | Hub |
| `src/core/high_freq_pipeline.py` | Orquestador HF + detección SNR | 2 | **18** | Hub |
| `src/output/summary_manager.py` | — | **0** | 1 | Muerto (217 líneas) |
| `src/visualization/plot_patches.py` | — | **0** | 2 | Deshabilitado (281 líneas) |
| `src/domain/physics.py` | Re-export | 0 desde `src/` | 1 | Solo lo usan los tests |

### Ciclos de dependencias confirmados

**Componente fuertemente conexo de 5 paquetes:** `{core, input, output, preprocessing, visualization}`.

Aristas que lo cierran:

| Arista | Ubicación | Tipo |
|---|---|---|
| `core → visualization` | `detection_engine.py:23` | Nivel módulo |
| `visualization → core` | `plot_composite.py:21` | Nivel módulo |
| `output → core` | `validation_metrics.py:73,98` | Tardío |
| `preprocessing → core` | `dedispersion.py:264`, `slice_len_calculator.py:181` | Tardío |

**Ciclo C1 — `pipeline ↔ high_freq_pipeline`:**

```
pipeline.py:44                → high_freq_pipeline   [NIVEL MÓDULO]
high_freq_pipeline.py:1570    → pipeline._optimize_memory      [TARDÍO]
high_freq_pipeline.py:1606    → pipeline.finalize_file_status  [TARDÍO]
```

Los dos imports tardíos existen exclusivamente para que el ciclo no reviente en tiempo de import.

**Ciclo C2 — `pipeline_parameters ↔ slice_len_calculator`:**

```
pipeline_parameters.py:10     → slice_len_calculator            [NIVEL MÓDULO]
slice_len_calculator.py:124   → pipeline_parameters             [TARDÍO]
slice_len_calculator.py:408   → pipeline_parameters             [TARDÍO]
slice_len_calculator.py:480   → pipeline_parameters             [TARDÍO]
slice_len_calculator.py:566   → pipeline_parameters             [TARDÍO]
```

Cuatro imports tardíos del mismo símbolo en el mismo archivo: síntoma inequívoco de ciclo evadido, no de carga perezosa intencional.

**Causa raíz de ambos ciclos:** dos módulos de cálculo puro (`pipeline_parameters.py`, `mjd_utils.py`) viven en `core/`, que es el paquete de orquestación, cuando los necesitan `preprocessing`, `output` y `visualization`.

### Arquitectura declarada frente a arquitectura real

| Afirmación | Fuente | Realidad |
|---|---|---|
| "todos los módulos del pipeline importan desde `src/domain/physics.py`" | `SPECS-physics.md` SPEC-DM-001 | **Cero** módulos de producción lo hacen; importan directo de `analysis/science_metrics.py` |
| "Ambas rutas de detección usan `finalize_patch()`" | `SPECS-physics.md` SPEC-CAND-001 | La Fase 3b (Linear) reimplementa la secuencia a mano en `high_freq_pipeline.py:593-601` |
| "Smart chunking system for handling large files" | `CHANGELOG.md` | El checkpoint/resume solo existe en LF; HF reinicia desde cero |
| "Python 3.8+ compatibility" | `CHANGELOG.md` | `pyproject.toml:9` exige `>=3.11`; el código usa sintaxis 3.10+ |
| `pytest src/tests -q` | `README.md:228` | Recolecta **0 tests**; la suite real está en `tests/` |
| Badge "Python 3.10+" | `README.md:4` | Contradice `pyproject` (≥3.11), `.python-version` (3.12) y CI (3.11/3.12) |

`SPECS-physics.md` es el mejor documento del repositorio, pero 2 de sus 13 SPECs describen un estado deseado y no el actual.

---

## 5. Estructura actual de carpetas

### Problemas verificados

**5.1 — `src/logging/` sombrea el módulo estándar de Python.** No es teórico; se reprodujo:

```
$ python -c "import sys; sys.path.insert(0,'<repo>/src'); import logging; logging.Formatter"
AttributeError: partially initialized module 'logging' from
'<repo>/src/logging/__init__.py' has no attribute 'Formatter'
(most likely due to a circular import)
```

Con `src/` en `sys.path`, cualquier librería que haga `import logging` —pandas, por ejemplo— recibe el paquete del proyecto, cuyo `logging_config.py:47` hace `class DRAFTSFormatter(logging.Formatter)` sobre sí mismo.

Siete archivos del repositorio insertan `src/` en `sys.path`:

| Archivo | Línea | Por qué sobrevive hoy |
|---|---|---|
| `src/scripts/analyze_case_alma_psr1745.py` | 31 | `import logging` en línea 26, antes del insert |
| `src/scripts/analyze_case_b0355.py` | 30 | `import logging` línea 25 |
| `src/scripts/analyze_case_fast_frex.py` | 29 | `import logging` línea 24 |
| `src/scripts/analyze_case_frb121102.py` | 30 | `import logging` línea 25 |
| `src/scripts/catalog_data_files.py` | 34 | `import logging` línea 26 |
| `src/scripts/fits_header_analyzer.py` | 60 | `import logging` línea 54 |
| `src/tests/analisis_chunk_reduccion.py` | 11 | Nunca importa `logging` |

Los siete sobreviven por el orden de los imports. Cualquier reordenación, o un `import pandas` añadido después del insert, rompe el script. El pipeline de producción no está afectado porque se ejecuta desde la raíz con imports `src.*`, de modo que `src/` no entra en `sys.path`.

**5.2 — `src/tests/` coexiste con `tests/`.** La suite real está en `tests/` (17 archivos, fijada por `pyproject.toml:44 testpaths=["tests"]`). `src/tests/` contiene 2 scripts de análisis y un README que correctamente avisa *"Pytest suite lives in /tests"*. Pero `README.md:228` manda ejecutar `pytest src/tests -q`, que recolecta 0 tests. Un usuario nuevo ejecuta nada y cree que la suite pasa.

Agravante: `src/scripts/test_slicing_alignment.py` empieza por `test_` sin ser pytest.

**5.3 — `src/scripts/` como cajón de sastre.** 22 archivos, ninguno referenciado en README, CHANGELOG, SPECS, Dockerfile, CI ni `config.yaml`. No existe `src/scripts/README.md`. Mezcla tres categorías: herramientas de validación de tesis legítimas, herramientas de diagnóstico, y one-shots ya consumidos con rutas hardcodeadas a directorios que están en `.gitignore`.

**5.4 — `src/models/` mezcla código y artefactos.** Tres módulos Python conviven con 120 MB de pesos `.pth`. `.gitignore` necesita tres reglas de negación para gestionarlo.

### Lo que NO es un problema

**`src/input/utils.py`** — pese al nombre genérico, contiene 6 helpers cohesivos y todos usados (149 líneas). Está incluido en `mutation/cr-utils.toml` como módulo crítico. Renombrarlo sería ruido puro.

**La cadena de `src/visualization/`** — `visualization_unified` → `plot_composite` → componentes es arquitectura por capas correcta, no duplicación. Sí hay duplicación *dentro* de los módulos (§23), pero la cadena está bien.

**La estructura `src/core|input|output|preprocessing|analysis|detection`** — la física del pipeline se mapea bien a esas carpetas. No tocar.

---

## 6. P0 — Corrupción, pérdida y resultados fundamentalmente incorrectos

### [P0-1] Las cajas de CenterNet se interpretan en el marco del cubo DM, no en el marco 512×512 del CNN

**Severidad:** P0 · **Confianza:** Alta · **Estado:** Confirmado (verificado manualmente) · **Categoría:** Bug / Correctness científica

#### Ubicación

- `src/core/detection_engine.py:147-148` (`detect_and_classify_candidates_in_band`)
- `src/core/detection_engine.py:161` (mismo error de marco)
- Contrato violado: `src/preprocessing/dm_candidate_extractor.py:11-18`

#### Problema

```python
# detection_engine.py:147-156
img_h = int(band_img.shape[0]) if band_img is not None and band_img.ndim >= 1 else 512
img_w = int(band_img.shape[1]) if band_img is not None and band_img.ndim >= 2 else 512
dm_val, t_sec, t_sample = extract_candidate_dm(
    (box[0] + box[2]) / 2,
    (box[1] + box[3]) / 2,
    slice_len,
    img_height=img_h,
    img_width=img_w,
    dm_values=dm_values,
)
```

`img_height` e `img_width` deben ser las dimensiones **de la imagen en la que viven las coordenadas de la caja**. Las cajas vienen en coordenadas 512×512. Aquí se pasan las dimensiones del cubo DM.

#### Causa raíz

Cadena completa verificada:

1. `detection_engine.py:512` — `band_img = slice_cube[band_idx]`. Forma `(height_dm, ancho_slice)`. Con el `config.yaml` del repositorio (DM 480–640, `smear_limited`, 800 canales, tsamp 256 µs, `time_rate: 4`): aproximadamente 147 × 977.
2. `detection_engine.py:74` — `img_tensor = preprocess_img(band_img)`, y `visualization_unified.py:90` hace `cv2.resize(img, (512, 512))`. **Redimensiona una copia**; `band_img` conserva su forma original.
3. `models/ObjectDet/centernet_utils.py:132` — `output[i][:, :4] *= input_shape`, con `input_shape = 512` fijado en `centernet_utils.py:141`. **Las cajas salen en coordenadas [0, 512].**
4. `detection_engine.py:147-148` declara el marco equivocado.

El contrato de `extract_candidate_dm` es explícito (`dm_candidate_extractor.py:11-18`): los parámetros tienen default `512` y el docstring describe el mapeo desde "a CNN detection box". **Todos los demás llamadores usan ese default** (`plot_composite.py:170,485,776,939`, `plot_dm_time.py:43,183`, `visualization_unified.py:55`), igual que `tests/test_contracts.py:102`. Solo el módulo que escribe el CSV lo sobreescribe.

#### Escenario y aritmética

Dentro de `extract_candidate_dm`:

```python
denom = max(int(img_height) - 1, 1)
frac = min(max(float(py) / denom, 0.0), 1.0)      # clamp a [0,1]
row = int(round(frac * (dm_arr.size - 1)))
```

Con `H = 147` y `py` en el rango [0, 512]:

| `py` (marco 512) | DM con el código actual | DM correcto |
|---|---|---|
| 0 | 480.00 | 480.00 |
| 64 | 550.59 | 499.85 |
| 128 | 621.19 | 520.81 |
| 256 | **640.00 (saturado)** | 560.52 |
| 383 | **640.00 (saturado)** | 600.23 |

`frac` se satura en 1.0 para todo `py > 146`: **el 71 % del eje devuelve exactamente `DM_max`**.

Con el caso opuesto (DM 0–1000, `H = 1001`): `py` nunca supera 512, luego `frac` no pasa de 0.512 y el DM reportado no llega nunca a la mitad del rango.

En el eje temporal: `scale_time = slice_len / img_width`. Con `img_width = ancho_slice`, `scale_time = 1.0`; el valor correcto sería `slice_len / 512`, aproximadamente 1.9. Un `px = 511` produce `t_sample = 511` en vez de unos 974.

#### Impacto

Ese `t_sample` alimenta `global_sample` en `detection_engine.py:171-175`, que a su vez alimenta `finalize_patch`. El parche dedispersado se extrae **centrado en el instante equivocado**, de modo que el SNR y la probabilidad de ResNet se calculan sobre datos que no contienen el pulso.

Resultado: DM incorrecto, tiempo incorrecto, SNR degradado y clasificación degradada, para **todos** los candidatos del pipeline LF, siempre que las dimensiones del cubo no sean exactamente 512×512 — que es el caso normal.

El defecto es silencioso: los DM producidos están dentro del rango configurado y son indistinguibles de detecciones legítimas.

#### Evidencia observable

El CSV contradice las etiquetas de los propios plots, porque los módulos de visualización sí usan el default 512. Comparar el DM de una fila del CSV con el DM rotulado en el plot compuesto del mismo candidato es la confirmación empírica más barata.

#### Solución recomendada

Usar `img_height=512, img_width=512` en `detection_engine.py:147-148`, es decir, eliminar el override.

Corregir también `detection_engine.py:161`: `candidate_region = band_img[y1:y2, x1:x2]` recorta el cubo original con coordenadas del marco 512.

#### Alternativas y trade-offs

Devolver las cajas ya desescaladas desde `detect()` sería más limpio conceptualmente, pero cambia el contrato de una función usada también por la visualización; el riesgo de introducir una segunda inconsistencia es mayor que el beneficio. La corrección mínima es preferible.

#### Test de regresión

Inyectar un FRB sintético con DM y tiempo conocidos, forzar `height_dm != 512` y `slice_len != 512`, y comprobar que el DM del CSV cae dentro de dos pasos de DM del valor real y el tiempo dentro de dos muestras decimadas. Añadir una aserción de coherencia: para la misma caja, `detection_engine` y `plot_composite` deben devolver el mismo DM.

---

### [P0-2] El pipeline HF nunca vacía el buffer del CSV de candidatos

**Severidad:** P0 · **Confianza:** Alta · **Estado:** Confirmado (verificado manualmente) · **Categoría:** Pérdida de datos

Reportado de forma independiente por cuatro subagentes: arquitectura, correctness, datos y patrones.

#### Ubicación

- `src/output/candidate_manager.py:105,119-124,131-134`
- `src/core/pipeline.py:690-697` (retorno anticipado), `:813` (única llamada a flush)
- `src/core/high_freq_pipeline.py:869` (escritura de candidatos)

#### Problema

```python
# candidate_manager.py:131-134
def write(self, row: list) -> None:
    self._buffer.append(row)
    if len(self._buffer) >= self._flush_interval:   # flush_interval = 50
        self.flush()
```

Las filas se acumulan en una lista Python. Solo `flush()` o `close()` las escriben a disco. La única llamada a `flush_all()` en todo el repositorio está en `pipeline.py:813`, después del bucle de chunks del pipeline LF.

La rama HF retorna antes:

```python
# pipeline.py:690-697
result = _process_file_chunked_high_freq(...)
return result
```

`_process_file_chunked_high_freq` escribe candidatos (`high_freq_pipeline.py:869`) y retorna en `:1615` sin vaciar. No existe `atexit` ni `__del__` en todo `src/`: grep verificado, cero resultados.

#### Causa raíz

El ciclo de vida de un recurso global (`CandidateWriter._instances`) es propiedad de una sola de las dos ramas de ejecución. Consecuencia directa de la duplicación estructural LF/HF (problema sistémico S-2, sección 10): al clonar el orquestador, esta línea no se clonó.

#### Escenario

Observación de alta frecuencia con 137 candidatos. Se escriben 100 en dos vaciados automáticos; los 37 restantes quedan en RAM. El proceso termina. El JSON de resumen dice `n_candidates: 137`; el CSV tiene 100 filas.

Con menos de 50 candidatos por archivo —el caso típico en alta frecuencia— **se pierden todos**, y el log sigue diciendo `SAVED: DM=...` (`high_freq_pipeline.py:876`) y el estado final `SUCCESS_CHUNKED_HIGH_FREQ`.

#### Impacto

Pérdida silenciosa de candidatos científicos. Descuadre permanente entre resumen y CSV. Handle de archivo sin cerrar; en Windows, CSV bloqueado. Rompe de facto SPEC-CAND-001, que exige que el CSV de candidatos sea byte-idéntico entre rutas.

#### Extensión del defecto al pipeline LF

`flush_all()` no está en un `finally`. Los cinco manejadores de error por archivo (`pipeline.py:860-904`) retornan sin vaciar **y** devuelven `n_candidates: 0` aunque ya se hubieran escrito candidatos. Una falla de I/O tras 300 candidatos produce un resumen que dice "0 candidatos" y un CSV con entre 250 y 299 filas.

#### Solución recomendada

Tres medidas, no una:

1. `CandidateWriter.flush_all()` en un `finally` en `_process_file_chunked` y en `_process_file_chunked_high_freq`.
2. `atexit.register(CandidateWriter.flush_all)` en `candidate_manager.py` como red de seguridad.
3. Reportar los conteos reales acumulados (`file_stats`) en las rutas de error, no ceros.

Idealmente, convertir `CandidateWriter` en context manager y usar `with` en ambos orquestadores.

#### Trade-offs

`atexit` no cubre `SIGKILL` ni un OOM del kernel; por eso hacen falta las tres medidas. Bajar `flush_interval` a 1 elimina el riesgo pero pierde el beneficio de I/O bufferizada, que es el motivo declarado del diseño (`candidate_manager.py:90-91`).

#### Test de regresión

Procesar un archivo forzando la ruta HF con N candidatos donde N no sea múltiplo de 50; afirmar que el número de filas del CSV coincide con `resultado["n_candidates"]`.

---

### [P0-3] La reanudación por checkpoint salta un chunk de más

**Severidad:** P0 · **Confianza:** Alta · **Estado:** Confirmado (verificado manualmente) · **Categoría:** Pérdida de datos

#### Ubicación

`src/core/pipeline.py:717-720` y `:802`; `src/core/checkpoint.py:56-70`

#### Problema

```python
# pipeline.py:717-720
for chunk_idx, (block, metadata) in enumerate(streaming_func(...), 1):   # base 1
    if chunk_idx <= resume_after + 1:
        logger.debug("Skipping chunk %d (already completed)", chunk_idx)
        continue
# pipeline.py:802
save_checkpoint(save_dir, fits_path.stem, chunk_idx, chunk_count)        # guarda base 1
```

`enumerate(..., 1)` hace que `chunk_idx` sea base 1. `save_checkpoint` guarda ese mismo índice como `last_completed_chunk`. `load_checkpoint` (`checkpoint.py:65`) lo devuelve sin transformar.

Si el último chunk completado fue el 5, `resume_after = 5` y la condición salta `chunk_idx <= 6`. **El chunk 6 nunca se procesó.** La condición correcta es `chunk_idx <= resume_after`.

#### Causa raíz

Confusión de convención de índices, delatada por el propio código: `checkpoint.py:67` registra "Resuming %s from chunk %d" con `last + 1`, mientras `pipeline.py:712` registra "Resuming after chunk {resume_after + 1}". Dos interpretaciones opuestas del mismo número, y la condición implementa la equivocada.

#### Escenario

Corte de energía tras el chunk 5. Al reanudar, el pipeline procesa desde el 7. El chunk 6 —decenas de segundos de observación— desaparece del CSV sin ningún aviso.

Con `resume_after = -1` (sin checkpoint) la condición salta `chunk_idx <= 0`, es decir, nada. Por eso el defecto solo se manifiesta al reanudar y nunca se ha observado en una corrida normal.

#### Impacto

Pérdida determinista de un chunk por cada reanudación. Reanudar **no** produce resultados idénticos a una corrida completa, lo que invalida científicamente cualquier búsqueda reanudada.

#### Solución recomendada

`if chunk_idx <= resume_after: continue`. Unificar además el mensaje de log de ambos sitios para que describan la misma semántica.

#### Test de regresión

Procesar un archivo de 5 chunks completo y guardar el CSV. Repetir interrumpiendo en el 3 y reanudando. Afirmar que ambos CSV tienen el mismo número de filas y las mismas tuplas `(chunk_id, slice_id, t_sample)`.

---

### [P0-4] El checkpoint se guarda también cuando el chunk ha fallado

**Severidad:** P0 · **Confianza:** Alta · **Estado:** Confirmado (verificado con la indentación exacta) · **Categoría:** Pérdida de datos

#### Ubicación

`src/core/pipeline.py:750` (`try`), `:768-773` (`except`), `:802` (`save_checkpoint`)

#### Problema

El `try` de la línea 750 y el `except Exception as chunk_error` de la 768 están a 12 espacios de indentación. `save_checkpoint` en la línea 802 está al mismo nivel, en el cuerpo del bucle, **fuera del `try`**.

```python
            except Exception as chunk_error:
                # SPEC-IO-002: do not silently drop chunks; count failures so the
                # file is reported as PARTIAL instead of SUCCESS.
                failed_chunk_count += 1
                logger.exception(f"Error processing chunk ...")

            # ... 28 lineas despues, mismo nivel de indentacion ...
            save_checkpoint(save_dir, fits_path.stem, chunk_idx, chunk_count)
```

Un chunk que revienta incrementa `failed_chunk_count` **y** se marca como completado.

#### Causa raíz

La mitigación de SPEC-IO-002 (`finalize_file_status`, `pipeline.py:103-115`) es correcta y está bien intencionada, pero solo protege la corrida actual: el estado `PARTIAL` vive en el JSON de resumen. El checkpoint borra la evidencia para la siguiente corrida.

No existe una definición de "chunk completado" atada a salida durable: el checkpoint marca progreso del bucle, no progreso persistido.

#### Escenario

El chunk 42 falla al generar plots. `failed_chunk_count = 1`, el archivo termina como `SUCCESS_CHUNKED_PARTIAL`, que es correcto. El operador ve `PARTIAL` y relanza. `load_checkpoint` devuelve 42, la condición salta hasta el 43 (P0-3 añade uno más), y la corrida de recuperación reporta `SUCCESS_CHUNKED` limpio sin haber recuperado nada. `failed_chunk_count` se reinicia a 0 en la nueva ejecución.

#### Impacto

El mecanismo de recuperación garantiza que los datos perdidos no se recuperen nunca.

#### Composición con P0-2 y P0-3

```
[chunk 41] OK          -> save_checkpoint(41)
[chunk 42] ERROR       -> failed_chunk_count += 1
                       -> save_checkpoint(42)      <- P0-4: se ejecuta igual
[chunk 43..] OK        -> status = "SUCCESS_CHUNKED_PARTIAL"

--- el operador relanza ---

load_checkpoint() = 42
skip: chunk_idx <= 42 + 1                          <- P0-3: salta 1..43

RESULTADO: chunk 42 perdido por el error
           chunk 43 perdido por el off-by-one
           segunda corrida reporta SUCCESS_CHUNKED limpio
           (y en HF, ademas, el buffer nunca se vacio -- P0-2)
```

#### Solución recomendada

Mover `save_checkpoint` dentro del `try`, o condicionarlo a que no haya habido excepción. Vaciar el `CandidateWriter` con `fsync` **antes** de guardar el checkpoint. Persistir además el conjunto de índices de chunks fallidos en el campo `extra` —que ya existe en la firma (`checkpoint.py:28`) y ningún llamador usa— y reprocesarlos al reanudar.

#### Test de regresión

Mockear `_process_block` para lanzar en el chunk 2 de 5; afirmar que el checkpoint conserva `last_completed_chunk == 1`.

---

## 7. P1 — Impacto alto y probable

### Índice

| ID | Título | Categoría | Estado |
|---|---|---|---|
| P1-01 | `trim_valid_window` ignora el solape derecho | Correctness / datos | Confirmado |
| P1-02 | Tres criterios contradictorios de inversión del eje de frecuencia en `stream_fits` | Correctness | Confirmado |
| P1-03 | La rejilla DM se corrompe cuando se activa el troceado en DM | Correctness | Confirmado |
| P1-04 | `start_sample` incorrecto en la ruta astropy de fallback | Correctness | Confirmado |
| P1-05 | La ruta de emergencia "buffer too large" salta 2× solape entre chunks | Correctness | Probable |
| P1-06 | La primera ventana de solape de cada FITS nunca se busca | Correctness | Confirmado |
| P1-07 | MJD baricéntrico con coordenadas de FRB 121102 y Effelsberg cableadas | Datos | Confirmado |
| P1-08 | El MJD baricéntrico degrada a topocéntrico sin marcarlo | Datos | Confirmado |
| P1-09 | `TSTART_MJD_CORR` contamina el archivo siguiente | Datos | Confirmado |
| P1-10 | `is_burst` contradice la decisión de guardado si la Fase 3a está deshabilitada | Correctness | Confirmado |
| P1-11 | Desalineación de polarización en la rama SPEC-HF-002 | Correctness | Confirmado |
| P1-12 | `snr_pre_dedisp` se calcula recorriendo el eje DM | Correctness | Confirmado |
| P1-13 | El pipeline no es idempotente: reejecutar duplica filas | Datos | Confirmado |
| P1-14 | Colisión de salidas entre archivos homónimos | Datos | Confirmado |
| P1-15 | Las rutas de error reportan 0 candidatos habiendo escrito candidatos | Datos | Confirmado |
| P1-16 | El checkpoint no valida configuración ni identidad del input | Datos | Confirmado |
| P1-17 | El pipeline HF no tiene checkpoint | Datos | Confirmado |
| P1-18 | Fuga de archivos temporales memmap multi-GB | Recursos | Confirmado |
| P1-19 | Cero reintentos en todo el I/O | Fiabilidad | Confirmado |
| P1-20 | `config` global mutable sin reset entre archivos | Arquitectura | Confirmado |
| P1-21 | La imagen Docker ejecuta un stack que CI nunca prueba | Seguridad / reproducibilidad | Confirmado |
| P1-22 | Builds Docker no reproducibles | Seguridad | Confirmado |
| P1-23 | `force_plots: true` domina el coste de ejecución | Performance | Confirmado |
| P1-24 | Duplicación estructural del orquestador LF/HF (~50 %) | Arquitectura | Confirmado |

---

### [P1-01] `trim_valid_window` recibe `overlap_right_ds` y lo ignora

**Confianza:** Alta · **Estado:** Confirmado (verificado manualmente) · **Categoría:** Correctness / integridad de datos

**Ubicación:** `src/core/data_flow_manager.py:453-471`

```python
def trim_valid_window(block_ds, dm_time_full, overlap_left_ds, overlap_right_ds):
    valid_start_ds = max(0, overlap_left_ds)
                                                         # <- linea en blanco (comentario borrado)
    valid_end_ds = block_ds.shape[0]                     # nunca resta overlap_right_ds
```

El parámetro se calcula (`:562`), se pasa (`pipeline.py:267-268`), se registra en métricas (`:276`) y no se usa.

**Prueba de la semántica pretendida:** la rama SPEC-HF-002, cincuenta líneas más allá, sí lo hace bien:

```python
# high_freq_pipeline.py:1430-1432
n_valid = max(0, block_ds.shape[0] - overlap_left_ds - overlap_right_ds)
block_ds = block_ds[overlap_left_ds: block_ds.shape[0] - overlap_right_ds]
```

**Impacto:** la cola de solape derecho se procesa como región válida y vuelve a procesarse como región válida del chunk siguiente, produciendo **candidatos duplicados** con el mismo tiempo absoluto e inflando los conteos. Además, en esa cola el cubo DM se construye con suma parcial de canales (`count_series` incompleto, `dedispersion.py:184-188`), lo que deprime el SNR y deja los planos `mid` y `diff` sin sentido, favoreciendo detecciones espurias de borde. Con `DM_max = 640` en banda L el solape ronda 0,9 s, unos 3 o 4 slices por chunk.

**Solución:** `valid_end_ds = max(valid_start_ds, block_ds.shape[0] - overlap_right_ds)`.

**Test:** chunk sintético con un pulso colocado en la zona de solape derecho; comprobar que se reporta exactamente una vez en todo el archivo.

---

### [P1-02] Tres criterios contradictorios de inversión del eje de frecuencia dentro de `stream_fits`

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Correctness científica

**Ubicación:** `src/input/fits_handler.py`

| Ruta | Condición de inversión | Línea |
|---|---|---|
| `your` (principal) | `if getattr(pf,'foff',0.0) > 0` | 840-841 |
| astropy SUBINT dentro del `try` | **ninguna** — `yield` en 1257 y 1339 sin invertir | 889-1342 |
| astropy SUBINT tras excepción | `if config.DATA_NEEDS_REVERSAL` | 1718-1719, 1803-1804 |
| `_load_fits_non_subint` | `if config.DATA_NEEDS_REVERSAL` | 218-219 |

`config.DATA_NEEDS_REVERSAL` lo fija `normalize_frequency_axis` (`input/utils.py:19-27`) y vale `True` cuando `DAT_FREQ` es descendente, es decir `foff < 0`: **la condición opuesta** a la de la ruta `your`.

**Impacto:** si el orden de canales del bloque no coincide con `config.FREQ` (ascendente), la dedispersión aplica a cada canal el retardo del canal espejo. El barrido dispersivo se invierte, el bow-tie no colapsa en ningún DM y un FRB real no se recupera. Es el fallo más silencioso posible: el pipeline corre, produce cubos y detecta ruido estructurado.

**Evidencia de la convención correcta:** `filterbank_handler.py:512-513` usa `if config.DATA_NEEDS_REVERSAL`.

**Solución:** unificar las cuatro rutas a `config.DATA_NEEDS_REVERSAL`, tras verificar de forma independiente qué orden devuelve `your.get_data`.

**Test:** PSRFITS sintético con `DAT_FREQ` descendente y un pulso dispersado a DM conocido; afirmar que el cubo tiene su máximo en la fila del DM correcto, ejercitando las dos ramas de `stream_fits`.

---

### [P1-03] La rejilla de DM se corrompe cuando se activa el troceado en DM

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Correctness científica

**Ubicación:** `src/core/data_flow_manager.py:299-316`

```python
dm_range = dm_max - dm_min
chunk_dm_min = dm_min + (start_dm / height) * dm_range   # deberia ser /(height-1)
chunk_dm_max = dm_min + (end_dm   / height) * dm_range
chunk_cube = d_dm_time_g(block_ds, height=chunk_height,
                         dm_min=chunk_dm_min, dm_max=chunk_dm_max)
```

Dentro de `d_dm_time_g` (`dedispersion.py:265-269`) el tamaño no coincide con el grid global, así que se cae a `np.linspace(chunk_dm_min, chunk_dm_max, chunk_height)`: cada trozo vuelve a incluir ambos extremos, de modo que el paso interno es `rango_trozo/(C-1)` en vez de `rango_total/(H-1)`.

Con DM 0–1000, `height = 1001` y `dm_chunk_height = 250` (5 trozos):

| fila | DM realmente dedispersado | DM asumido aguas abajo | error |
|---|---|---|---|
| 249 | 249.750 | 249.000 | +0.750 |
| 250 | 249.750 | 250.000 | −0.250 |
| 750 | 749.251 | 750.000 | −0.749 |
| 1000 | **999.001** | 1000.000 | −0.999 |

Hay 4 filas con DM duplicado en las fronteras, y **`DM_max` nunca se busca**.

**Impacto:** el eje DM deja de ser monótono-uniforme, que es justamente lo que asumen `extract_candidate_dm` (`dm_candidate_extractor.py:49-57`) y `_dm_from_image_at_time` (`high_freq_pipeline.py:89`). Error sistemático de hasta una unidad de DM, con discontinuidades, y pérdida del trial `DM_max`.

**Por qué los tests no lo detectan:** `tests/test_cube_windowed_parity.py:78-85` pasa `threshold = cube_gb * 1.01`, lo que fuerza `num_dm_chunks == 1`, el único caso en que la fórmula es exacta.

**Solución:** pasar el vector de DM del trozo explícitamente (`dm_values_full[start_dm:end_dm]`) en lugar de re-derivar `(dm_min, dm_max)` por trozo.

**Test:** `_build_dm_time_cube_chunked` con un `threshold` que fuerce al menos 3 trozos, comparado contra `d_dm_time_g` directo.

---

### [P1-04] Ruta astropy de fallback: `start_sample` es el inicio del bloque, no de la región válida

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Correctness / tiempos

**Ubicación:** `src/input/fits_handler.py:1723,1739,1744-1745`

```python
start_sample_idx = emitted - out_buf.shape[0]      # inicio del BLOQUE (con solape izq.)
```

La ruta equivalente (`:1220`) sí lo hace bien: `emitted - out_buf.shape[0] + valid_start`. `stream_fil` también (`filterbank_handler.py:522`).

`_process_block` asume la semántica correcta: `chunk_start_time_sec = start_sample * config.TIME_RESO` (`pipeline.py:187`), y el bloque se recorta por `overlap_left_ds` en `:267`.

**Impacto:** todos los tiempos absolutos (`t_sec_dm_time`, `mjd_utc`, `mjd_bary_*`) quedan adelantados en `overlap_raw × TIME_RESO`, que es el retardo dispersivo máximo completo: unos 0,9 s para DM 640 en banda L, y segundos o minutos para DM altos o bandas bajas. El offset es constante y plausible, por lo que es invisible salvo por cruce con catálogo.

**Solución:** `start_sample_idx = emitted - out_buf.shape[0] + valid_start`.

**Test:** paridad de metadatos entre `stream_fil` y ambas rutas de `stream_fits`, afirmando `start_sample - block_start_sample == overlap_left`.

---

### [P1-05] La ruta de emergencia "buffer too large" salta 2× solape entre chunks

**Confianza:** Media-Alta · **Estado:** Probable · **Categoría:** Correctness / pérdida de datos

**Ubicación:** `src/input/fits_handler.py:1186-1189,1260` y su clon en `1687-1690,1764`

```python
samples_to_remove = actual_chunk_size if not buffer_too_large else end_with_overlap
```

Con `buffer_too_large`, se descartan `actual_chunk_size + 2·overlap` muestras. El siguiente buffer empieza en `valid_end + overlap`, y el siguiente chunk vuelve a declarar `valid_start = overlap`. El hueco resultante es de `2 × overlap_samples` de datos nunca buscados, en cada emisión de emergencia.

**Por qué es el caso normal y no excepcional:** con `chunk_samples > 1.000.000` (`:973-975`), `max_buffer_samples = chunk + 2·overlap`, y la emisión se dispara justo cuando el buffer alcanza ese valor. Como el buffer crece en bloques de `NSBLK`, casi siempre lo supera, de modo que la condición `buffer_too_large` es cierta.

**Impacto:** pérdida silenciosa de aproximadamente 1,8 s por chunk con la configuración por defecto, y más con DM alto o banda baja. Nada verifica que `end_sample[n] == start_sample[n+1]`.

**Solución:** `samples_to_remove = actual_chunk_size` también en la rama de emergencia, y añadir una aserción de continuidad en el consumidor.

---

### [P1-06] La primera ventana de solape de cada archivo PSRFITS nunca se busca

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Correctness / pérdida de datos

**Ubicación:** `src/input/fits_handler.py:1208-1211` y su clon en `1709-1712`

En la primera emisión el buffer contiene las muestras desde 0, pero `valid_start = overlap_samples` también para el primer chunk, así que `[0, overlap_samples)` se declara solape izquierdo y `_process_block` lo descarta.

`stream_fil` lo hace bien: `start_with_overlap = max(0, valid_start - overlap_samples)` (`filterbank_handler.py:502`), lo que da `overlap_left = 0` en el primer chunk.

**Impacto:** pérdida silenciosa del primer segundo aproximado de cada archivo FITS, más con DM alto o frecuencias bajas.

**Solución:** `valid_start = 0` cuando `emitted - out_buf.shape[0] == 0`.

---

### [P1-07] El MJD baricéntrico usa coordenadas de FRB 121102 y Effelsberg cableadas

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Datos / correctness científica

**Ubicación:** `src/core/mjd_utils.py:47-55,199-212`

```python
ra: str = "05:31:58.70",     # FRB 121102
dec: str = "33:08:52.5",
freq_mhz: float = 1400.0,
location: str = "Effelsberg",
...
if ra is None:  ra  = getattr(config, 'SOURCE_RA',  "05:31:58.70")
```

`SOURCE_RA`, `SOURCE_DEC` y `REF_FREQ_MHZ` **no se asignan en ningún punto del código ni de los YAML**, y ni siquiera están en `config._KNOWN_CONFIG_KEYS` (`config.py:232-250`), de modo que `inject_config` los rechazaría. Los defaults se usan siempre. `location` nunca se pasa desde los llamadores (`detection_engine.py:283`, `high_freq_pipeline.py:816`).

Las coordenadas sí están disponibles en las cabeceras: `src_raj`/`src_dej` se parsean en `filterbank_handler.py:80-86` y se descartan; `RA`/`DEC`/`STT_CRD*` de PSRFITS ni se leen.

**Impacto:** la corrección baricéntrica se calcula para la línea de visión de FRB 121102 desde Effelsberg, sea cual sea la observación. El error de tiempo de llegada puede alcanzar ±499 s, la proyección completa de la órbita terrestre. Además, `mjd_bary_*_inf` corrige la dispersión a 1400 MHz aunque la banda real sea de 8 GHz o más, donde sobra prácticamente toda la corrección.

**Solución:** leer RA, DEC y telescopio de la cabecera y propagarlos; usar la frecuencia máxima real de la banda; y ante ausencia de datos, devolver `None` en las columnas baricéntricas en vez de aplicar un default.

---

### [P1-08] El MJD baricéntrico degrada a topocéntrico sin marcarlo

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Datos

**Ubicación:** `src/core/mjd_utils.py:85-87` y `:122-135`

```python
if not ASTROPY_AVAILABLE:
    logger.debug("astropy not available, returning topocentric MJD only")
    return topo_mjd, topo_mjd, topo_mjd, topo_mjd
...
except Exception as e:
    logger.warning(f"Error calculating barycentric MJD: {e}, returning topocentric MJD")
    return topo_mjd, topo_mjd, topo_mjd, topo_mjd
```

Esos cuatro valores van directos a `mjd_bary_utc`, `mjd_bary_tdb`, `mjd_bary_utc_inf` y `mjd_bary_tdb_inf` (`candidate_manager.py:211-214`) con 12 decimales, indistinguibles de un cálculo real.

**Escenario:** `jplephem` no instalado, o `EarthLocation.of_site("Effelsberg")` falla por no haber red para descargar el registro de sitios de astropy. Nótese que esa llamada ocurre **por candidato**, dentro del bucle. El CSV queda con corrección baricéntrica igual a cero, cuando la real llega a ±500 s.

**Agravante de seguridad y reproducibilidad:** `compute_bary=True` está cableado, no es configurable. `EarthLocation.of_site` y `solar_system_ephemeris.set("de432s")` disparan descargas desde Internet (registro de sitios, efeméride de unos 10 MB, tablas IERS por `iers.conf.auto_download`), en un pipeline presentado como offline. El mismo dataset produce columnas MJD distintas según haya red o no.

**Solución:** devolver `None` en los cuatro valores cuando la corrección no se pueda calcular —`to_row` ya serializa `None` como celda vacía— y añadir una columna `bary_status` con valores `ok` / `no_astropy` / `no_ephem` / `error`. Exponer `compute_bary` en `config.yaml`. Fijar `iers.conf.auto_download = False` salvo que el operador provea las tablas.

---

### [P1-09] `TSTART_MJD_CORR` del archivo anterior contamina el siguiente

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Datos

**Ubicación:** `src/input/fits_handler.py:326-334` frente a `src/core/mjd_utils.py:181-187`

```python
# fits_handler.py:326-334
if tstart_mjd is not None:
    try:
        config.TSTART_MJD = tstart_mjd
        config.TSTART_MJD_CORR = tstart_mjd + (nsuboffs * tsubint) / 86400.0
    except Exception:
        config.TSTART_MJD = tstart_mjd          # deja TSTART_MJD_CORR obsoleto
# si tstart_mjd es None, NINGUNA de las dos se resetea
```

Y el consumidor prefiere precisamente la variable obsoleta:

```python
# mjd_utils.py:182-185
tstart_mjd = getattr(config, 'TSTART_MJD_CORR', None)
if tstart_mjd is None:
    tstart_mjd = getattr(config, 'TSTART_MJD', None)
```

`filterbank_handler.py:238-241` sí hace lo correcto: pone ambas a `None` y emite warning.

**Escenario:** se procesan `scanA.fits` (con `STT_IMJD`) y `scanB.fits` (sin él). Todos los candidatos de `scanB` reciben la fecha de observación de `scanA`, con 12 decimales y sin warning.

**Solución:** asignar explícitamente `None` a ambas en el `else` y emitir warning. A medio plazo, propagar `ObservationMetadata` por parámetro en vez de leer globales.

---

### [P1-10] `is_burst` contradice la decisión de guardado cuando la Fase 3a está deshabilitada

**Confianza:** Alta · **Estado:** Confirmado (verificado manualmente) · **Categoría:** Correctness

**Ubicación:** `src/core/high_freq_pipeline.py:576-579` y `:692`

```python
else:
    logger.debug("Phase 3a: DISABLED - Skipping Intensity classification ...")
    class_prob_intensity = 1.0
    is_burst_intensity = True
```

Centinelas **dentro del rango válido**, no `None`. La tabla de decisión hace lo correcto y usa Linear cuando solo Linear está disponible (`:660-663`), pero el veredicto que se persiste no:

```python
# :692
is_burst = is_burst_intensity     # siempre True en esa configuracion
# :852  -> a CSV;  :861-864 -> a contadores
```

**Impacto:** con `enable_intensity_classification: false` y `enable_linear_classification: true`, todo candidato guardado sale `is_burst=True` aunque Linear lo haya rechazado, y el contador de BURST se infla. Además `class_prob_intensity = 1.0` se escribe como si fuera una probabilidad medida y entra en `morphology_prob = max(I, L)` (`:812`), falseando `rank_score`.

**Alcance:** el `config.yaml` por defecto trae `intensity: true, linear: false`, combinación en la que la lógica es correcta. El defecto exige la configuración inversa, que está documentada como soportada en `config.yaml` y validada en `user_config.py:122-126`. Es P0 para quien ejecute esa configuración.

**Solución:** usar `None` para "fase no ejecutada" y extraer una función pura `decide_burst(p_i, p_l, strict) -> (should_save, is_burst, reason)` que sea la única fuente del veredicto. Revisar los consumidores que asumen `float`.

**Test:** tabla paramétrica de los 12 casos (2 flags × `SAVE_ONLY_BURST`), incluyendo la regresión `intensity=false, p_l=0.1` → `is_burst=False` en el CSV.

---

### [P1-11] Desalineación entre intensidad y polarización en la rama SPEC-HF-002

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Correctness científica

**Ubicación:** `src/core/high_freq_pipeline.py:1430-1433` frente a `:1490-1491`

```python
block_ds = block_ds[overlap_left_ds: block_ds.shape[0] - overlap_right_ds]
valid_start_ds, valid_end_ds = 0, n_valid          # se pierde el offset izquierdo
...
block_raw_ds = block_raw_ds[valid_start_ds:valid_end_ds]   # = [0:n_valid]
```

`block_ds` empieza en `overlap_left_ds`; `block_raw_ds` empieza en 0.

**Impacto:** las formas de onda Linear y Circular quedan desfasadas `overlap_left_ds` muestras respecto a la intensidad. La Fase 2 compara `snr_profile_linear[peak_idx]` con `snr_profile_intensity[peak_idx]` en instantes distintos, de modo que rechaza bursts reales y valida ruido. `linear_fraction` (`:779-789`) también resulta falso.

**Solución:** recortar `block_raw_ds` con los mismos límites que `block_ds`.

---

### [P1-12] `snr_pre_dedisp` se calcula recorriendo el eje DM como si fuera el eje temporal

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Correctness

**Ubicación:** `src/core/detection_engine.py:160-167`

```python
candidate_region = band_img[y1:y2, x1:x2]   # band_img es (DM, tiempo)
snr_profile, _, _ = compute_snr_profile(candidate_region)
```

`compute_snr_profile` documenta y exige `(tiempo, frecuencia)` (`snr_utils.py:22,45`): integra sobre `axis=1` y filtra con boxcars sobre `axis=0`. Aquí `axis=0` es el eje DM, así que el resultado es un filtrado adaptado a lo largo de DM, no un SNR.

**Impacto:** la columna `snr_pre_dedisp` del CSV no es un SNR. Además alimenta `physical_consistency_score` (`:277`) y es el valor de reserva de `snr_val` cuando `finalize_patch` devuelve 0 (`:180-181`), contaminando `snr_patch_dedispersed` y `rank_score`.

Nota: esta línea comparte con P0-1 el mismo error de marco de coordenadas, ya que recorta `band_img` con coordenadas del marco 512.

**Solución:** calcular el SNR pre-dedispersión sobre el waterfall (eje tiempo × frecuencia), no sobre la imagen DM-tiempo.

---

### [P1-13] El pipeline no es idempotente: reejecutar duplica filas

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Datos

**Ubicación:** `src/output/candidate_manager.py:60-84,128`

`ensure_csv_header` retorna temprano si el archivo existe (`:77`) y `_ensure_open` abre en modo `"a"` (`:128`). No hay truncado, ni clave de deduplicación, ni comparación con filas previas.

**Escenario:** el operador corre el pipeline, ajusta `--class-prob` y vuelve a correr. El CSV contiene las detecciones de ambas corridas mezcladas y sin distinción; los PNG sí se sobrescriben. CSV y plots dejan de corresponderse.

**Solución:** definir política explícita —rotar el CSV anterior a `*.candidates.<timestamp>.csv` cuando no hay checkpoint activo, o deduplicar por la clave natural `(file, chunk_id, slice_id, band_id, t_sample)`— y añadir banderas `--overwrite` / `--resume`.

---

### [P1-14] Colisión de salidas entre archivos de entrada homónimos

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Datos

**Ubicación:** `src/core/pipeline.py:611-613,1104`; `src/core/checkpoint.py:19-20`; `src/core/data_flow_manager.py:538-545`

Toda la identidad de salida se deriva de `fits_path.stem`. Dos archivos con el mismo nombre base en directorios distintos comparten CSV (en modo append, o sea, mezcla), checkpoint (el resume del segundo salta chunks del primero), directorio de plots (PNG sobrescritos) y clave del diccionario `summary` (la segunda entrada pisa la primera).

**Escenario:** `DATA/2024-01-15/scan01.fil` y `DATA/2024-01-16/scan01.fil`.

**Solución:** derivar la identidad de una clave estable y única —ruta relativa a `DATA_DIR` saneada, o `stem` más un hash corto de la ruta absoluta— y aplicarla de forma consistente en CSV, checkpoint, directorios y clave de resumen. Es prerrequisito de cualquier ejecución concurrente futura.

---

### [P1-15] Las rutas de error reportan 0 candidatos habiendo escrito candidatos

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Datos

**Ubicación:** `src/core/pipeline.py:860-904`

Los cinco manejadores (`MemoryError`, `FileNotFoundError`, `PermissionError`, `ValueError`, `Exception`) devuelven literalmente `{"n_candidates": 0, "n_bursts": 0, ...}` y ninguno llama a `flush_all()`.

**Impacto:** contradicción total entre resumen y disco. El operador concluye "no se detectó nada" y descarta un CSV que sí contiene ciencia. Ver también P0-2.

**Solución:** `try/finally` con `flush_all()`, y reportar los conteos acumulados en `file_stats`.

---

### [P1-16] El checkpoint no valida configuración ni identidad del input

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Datos

**Ubicación:** `src/core/checkpoint.py:23-72`

El payload completo es `file_stem`, `last_completed_chunk` y `total_chunks`. No hay versión de esquema, ni hash de configuración, ni identidad del archivo de entrada. El parámetro `extra` existe y ningún llamador lo usa.

**Escenario:** corrida 1 con `--dm-max 512` procesa los chunks 0 a 40 y se corta. El operador relanza con `--dm-max 1024`. Se reanuda desde el 41: el CSV final contiene 41 chunks buscados hasta DM 512 y el resto hasta DM 1024, sin ninguna marca. El JSON de métricas registra solo `dm_max: 1024`.

**Solución:** añadir `schema_version`, `config_hash` y `input_identity` (tamaño, mtime, `TSTART_MJD`); si algo no coincide al cargar, descartar el checkpoint con warning y empezar de cero.

---

### [P1-17] El pipeline HF no tiene checkpoint

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Datos

`grep -n "checkpoint" src/core/high_freq_pipeline.py` no devuelve nada. Un archivo de 5 TB procesado por la ruta HF durante 30 horas que falla en la hora 29 se reprocesa desde el chunk 0; y como `ensure_csv_header` conserva el CSV existente (P1-13), todos los candidatos de las 29 horas se duplican.

Ni el README ni el CHANGELOG mencionan la limitación; el CHANGELOG anuncia "Smart chunking system for handling large files" sin distinguir modos.

**Solución:** se resuelve al extraer el driver común (P1-24). Como parche independiente, replicar las cuatro líneas de checkpoint en el bucle HF.

---

### [P1-18] Fuga de archivos temporales memmap multi-GB

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Recursos

**Ubicación:** `src/core/data_flow_manager.py:33-49`

```python
fd, path = tempfile.mkstemp(suffix=".mmap", prefix="drafts_dm_cube_")
os.close(fd)
arr = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
arr._mmap_path = path  # type: ignore[attr-defined]
```

`grep -rn "_mmap_path" src/` devuelve **solo esa asignación**. Ningún módulo de producción lo lee ni borra el archivo. No hay `unlink`, ni `atexit`, ni context manager. Se dispara siempre que el cubo alcanza `DM_CUBE_MEMMAP_THRESHOLD_GB` (4,0 GB por defecto, `config.py:174`), en dos sitios (`:276` y `:391`). Además `trim_valid_window` hace `.copy()` del recorte, soltando la referencia sin cerrar ni borrar.

**Escenario:** archivo largo con DM alto, 200 chunks por 5 GB cada uno, es decir 1 TB de basura en el directorio temporal. En Windows suele estar en `C:`, de modo que el disco del sistema se llena a mitad de corrida y el `OSError` resultante aparece como un error de chunk genérico. En un nodo de cluster compartido, afecta a todos los usuarios.

**Solución:** `try/finally` que cierre el memmap y haga `os.unlink`, o `weakref.finalize(arr, os.unlink, path)`. En Linux, hacer `unlink` inmediatamente tras crear el memmap también funciona: el descriptor mantiene el inodo vivo.

---

### [P1-19] Cero reintentos en todo el I/O

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Fiabilidad

`grep -rn "retry|backoff|max_attempts" src/ main.py` devuelve cero coincidencias.

El pipeline lee archivos de terabytes, típicamente desde almacenamiento de red. Un `OSError` transitorio —timeout de NFS o SMB, reconexión— en el chunk 300 de 1000 hace que ese chunk se pierda para siempre: se cuenta como fallido, se marca como completado (P0-4) y nunca se reintenta. Lo mismo para la escritura del CSV (un antivirus que abre el archivo momentáneamente en Windows) y para las descargas de astropy.

**Solución:** decorador de reintento con backoff exponencial y jitter (3 intentos, 1 a 8 s), acotado a excepciones transitorias, en lectura de chunk, `flush()` del CSV y `save_checkpoint`. Las tres son idempotentes: en el caso del `flush`, el buffer no se limpia hasta que `writerows` retorna, así que el reintento no duplica ni pierde.

---

### [P1-20] `config` es estado global mutable sin reset entre archivos

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Arquitectura

**Ubicación:** `src/config/config.py` y 49 sitios de mutación en producción

| Fase | Mutador | Qué escribe |
|---|---|---|
| Arranque (1×) | `pipeline.py:955,967` (`inject_config`) | 20 claves de usuario, perfil de hardware |
| Por archivo | `fits_handler.py:272-334,472-505` (34 escrituras) | `TIME_RESO`, `FREQ_RESO`, `FILE_LENG`, `FREQ`, `DATA_NEEDS_REVERSAL`, `NBITS`, `NPOL`, `POL_TYPE`, `TSTART_MJD`… |
| Por archivo | `filterbank_handler.py:225-241` (8) | ídem |
| Por archivo | `input/utils.py:87-111` (4) | `DOWN_FREQ_RATE`, `DOWN_TIME_RATE` — pisa lo que el usuario puso en `config.yaml` |
| Por chunk | `slice_len_calculator.py:69` | `SLICE_LEN` |
| Por chunk | `fits_handler.py:934-939,1435-1440` | `TSTART_MJD`, `TSTART_MJD_CORR` |

No existe reset entre archivos. Cualquier ruta que no asigne —porque la cabecera no trae el campo o porque un `except` se traga el error— hereda el valor del archivo anterior. P1-09 es la instancia confirmada y más dañina; `NPOL`/`POL_TYPE` obsoletos son el siguiente riesgo, ya que afectan a la extracción de polarización en HF.

**Evidencia del coste ya pagado:** `tests/conftest.py:53-58` tiene una fixture `autouse` que snapshotea y restaura la configuración entre tests, con el comentario *"Restore config globals after each test to prevent order-dependent failures"*. Sin esa red, la suite falla según el orden de ejecución.

**Impacto adicional:** impide procesar dos archivos en paralelo en el mismo proceso, que es la vía natural de escalar este pipeline.

**Solución gradual:** `contracts.py` ya define la pieza correcta (`ObservationMetadata.from_config()`) y no se usa. El camino es construirla una vez por archivo y pasarla hacia abajo por firma, empezando por `data_flow_manager` (que ya tiene entradas y salidas explícitas), con los tests de paridad como red. Como medida inmediata y barata, añadir un `reset_file_scoped_config()` al inicio de cada archivo.

**No eliminar `conftest.py::_isolate_config` hasta que esto esté resuelto.**

---

### [P1-21] La imagen Docker ejecuta un stack que CI nunca prueba

**Confianza:** Alta · **Estado:** Confirmado (verificado manualmente) · **Categoría:** Seguridad / reproducibilidad

**Ubicación:** `Dockerfile:11,64,68-88` (CPU) y `:119,178,181-201` (GPU)

| Componente | Dockerfile | Declarado en el proyecto |
|---|---|---|
| Python | `python:3.10-slim` | `requires-python = ">=3.11"`; `.python-version` = 3.12; CI = 3.11, 3.12 |
| torch | `2.1.0` | `>=2.11,<3`; lock = 2.11.0 |
| numpy | `1.24.3` | `>=2.4,<3`; lock = 2.4.6 |
| numba | `0.58.0` | `>=0.65,<0.66` |
| 15 paquetes más | **sin versión** | pinned en el lock con hashes |

**Riesgo de seguridad asociado:** el código se protege con `torch.load(..., weights_only=True)` (`pipeline.py:150,162`), pero esa garantía solo es sólida desde PyTorch 2.6 (el parseo de `weights_only=True` era evadible en versiones anteriores). La imagen fija 2.1.0, de modo que la única mitigación de deserialización del proyecto queda anulada dentro del contenedor, que es el método de ejecución documentado en el README.

**Matiz que reduce la severidad:** se buscaron APIs eliminadas en NumPy 2 (`np.NaN`, `np.Inf`, `np.float_`, `np.in1d`, `np.trapezoid`) y no hay ninguna coincidencia, así que numpy 1.24 frente a 2.4 no rompe de forma dura. Es divergencia de entorno, no fallo inmediato.

**Solución:** base `python:3.12-slim` fijada por digest e instalación con `pip install --require-hashes -r requirements.lock.txt`.

---

### [P1-22] Builds Docker no reproducibles

**Confianza:** Alta · **Estado:** Confirmado (verificado manualmente) · **Categoría:** Seguridad de cadena de suministro

`Dockerfile:64` y `:178` hacen `COPY requirements.txt .` y **nunca lo instalan**. El `requirements.lock.txt` con 1273 hashes no participa en la construcción de la imagen. Dos builds en fechas distintas producen imágenes distintas, sin hashes, resolviendo versiones en vivo desde PyPI.

Además `opencv-python` en los manifiestos frente a `opencv-python-headless` en Docker, y `fitsio` solo en Docker. Este último **no es un bug**: es un import opcional con fallback a astropy (`fits_handler.py:13-16`), documentado en `requirements.txt:9`. Sí implica que Docker y la instalación local leen FITS por rutas de código distintas, lo que es un riesgo de reproducibilidad menor pero real.

Misma corrección que P1-21.

---

### [P1-23] `force_plots: true` domina el coste de ejecución

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Performance

**Ubicación:** `config.yaml:300` y `src/core/detection_engine.py:546-596`

`config.yaml` fija `debug.force_plots: true`. En `detection_engine.py:553` eso hace `should_generate_plots = slice_has_candidates or force_plots`, es decir, siempre verdadero. Cada slice ejecuta `dedisperse_block()` y `save_all_plots()`, que produce 4 PNG a `dpi=300` con `bbox_inches="tight"`; el compuesto se rasteriza tres veces por un `fig.canvas.draw()` explícito en `plot_composite.py:439`.

**Escenario cuantificado** (archivo de 5,5 TiB, 800 canales, tsamp 0,256 ms, `time_rate: 4`, `frequency_rate: 2`, slice de 1000 ms): aproximadamente 1,94 millones de slices × 4 PNG = **7,75 millones de imágenes**. A entre 0,4 y 1,5 s por figura de 14×12 pulgadas a 300 dpi, eso son entre 860 y 3200 horas de CPU solo en matplotlib, más decenas de TB de PNG. Domina por uno o dos órdenes de magnitud sobre la dedispersión (unas 12 h) y sobre la inferencia.

**Solución:** `force_plots: false` en producción; mover `postprocess_img` y `dedisperse_block` detrás del gate; bajar a `dpi=120` y quitar `bbox_inches="tight"` en figuras de diagnóstico; fijar `matplotlib.use("Agg")`, que no aparece en ningún punto del repositorio.

**Trade-off:** se pierde la inspección visual de slices sin candidato, que es justamente la que no aporta información.

---

### [P1-24] Duplicación estructural del orquestador de archivo entre LF y HF

**Confianza:** Alta · **Estado:** Confirmado · **Categoría:** Arquitectura

**Ubicación:** `src/core/pipeline.py:494-904` (411 líneas) frente a `src/core/high_freq_pipeline.py:1194-1625` (432 líneas)

Normalizando indentación, comentarios y líneas en blanco: **172 de 326 líneas son byte-idénticas**, alrededor del 50 %. Bloques con diff vacío:

| Responsabilidad | LF | HF |
|---|---|---|
| Presupuesto de memoria (50 líneas; solo difiere el prefijo de log) | `:517-566` | `:1237-1281` |
| Resumen + `Summary/` + `ensure_csv_header` | `:596-614` | `:1299-1317` |
| Cálculo de `overlap_raw` | `:621-652` | `:1329-1358` |
| Medición de latencia de llegada de chunk | `:733-747` | `:1376-1391` |
| ETA por chunk | `:775-799` | `:1540-1566` |
| Export de métricas de validación | `:818-824` | `:1584-1590` |

**Divergencias que la duplicación ya produjo** —ninguna intencionada—:

1. `CandidateWriter.flush_all()` solo en LF → **P0-2**.
2. Checkpoint/resume solo en LF → **P1-17**.
3. Límite `MAX_CHUNK_SAMPLES` aplicado solo en LF (`:569-593` frente a `:1283-1297`).
4. `streaming_func` recibido e ignorado en HF (`:1199`, usado solo en el log de `:1364`; la lectura real usa `stream_fits_multi_pol` fijo en `:1371`) → P2.

**La divergencia real es una sola:** cómo se detectan candidatos en un slice. Todo lo demás —streaming, presupuesto de memoria, cubo DM, solape, contabilidad, persistencia— es idéntico porque el problema físico es el mismo.

**Solución:** extraer un driver funcional común (`run_file_chunks`) parametrizado por un callback de detección por chunk. **Sin herencia ni jerarquía de clases**: el repositorio no tiene una sola clase con comportamiento fuera de `CandidateWriter` y los trackers.

**Precondición obligatoria:** un test que fije el CSV de candidatos byte a byte sobre un archivo de referencia, en ambos modos. SPEC-CAND-001 ya lo exige conceptualmente; hoy `tests/test_candidate_finalizer.py` solo cubre `finalize_patch`. **Sin esa red, este refactor no es seguro.**

**Lo que NO debe unificarse:** `detect_and_classify_candidates_in_band` (CenterNet sobre imagen DM-tiempo) y `snr_detect_and_classify_candidates_in_band` (matched filter sobre IQUV) resuelven problemas científicos genuinamente distintos. Fusionarlas con banderas sería peor que la duplicación actual.

---

## 8. P2 — Impacto real, condicionado

| ID | Hallazgo | Ubicación | Efecto |
|---|---|---|---|
| P2-01 | Bucle infinito al parsear una cabecera `.fil` truncada | `filterbank_handler.py:92-94` | `except ... : continue` dentro de `while True` sin avanzar el puntero. EOF en cabecera truncada gira para siempre: un core al 100 % y, con `DEBUG`, log que llena el disco. PoC ejecutado por el auditor de seguridad con un archivo de 30 bytes |
| P2-02 | `_read_string` con longitud negativa absorbe el archivo entero | `filterbank_handler.py:37-40` | `f.read(length)` sin validar; `f.read(-1)` lee hasta EOF. OOM en nodo compartido, sorteando todo el presupuesto de memoria. PoC ejecutado |
| P2-03 | `calculate_optimal_chunk_size` lanza `NameError` garantizado | `slice_len_calculator.py:502` | Usa `overlap_decimated`, definida solo en otra función (`:142`). Es la ruta de *fallback* invocada desde `:643` cuando el presupuesto adaptativo falla, y la llamada está dentro del `except`, así que nada la captura: sube hasta `pipeline.py:940` y el archivo se salta entero |
| P2-04 | El kernel CUDA de Numba congela `TIME_RESO` como constante de compilación | `dedispersion.py:74-105` | Numba trata los atributos de módulo como constantes en tiempo de compilación. Procesando varios archivos con resoluciones distintas, del segundo en adelante el kernel GPU usa la del primero. La versión CPU sí los pasa como argumentos (`:136-139`), lo que confirma la intención |
| P2-05 | `dedisperse_block` desplaza `start` en silencio y no lo comunica | `dedispersion.py:626-628` | No devuelve el `start` ajustado (a diferencia de `dedisperse_patch`). Los últimos slices de cada chunk reciben la cascada dedispersada de un intervalo anterior, rotulada con el tiempo del slice actual |
| P2-06 | `USE_MULTI_BAND` etiqueta como "Low Band"/"High Band" planos que no son sub-bandas | `config.py:281-285`, `dedispersion.py:100-105` | El plano 1 es la serie del canal central cizallada; el plano 2 es su residuo. Ninguno es una dedispersión de sub-banda, pero ambos se suman a los conteos de candidatos, triplicándolos |
| P2-07 | PSRFITS troceados: se cuenta dos veces el desplazamiento | `fits_handler.py:1035-1039` frente a `:935` | `_expected_start_sample` es absoluto desde el inicio de la observación mientras `emitted` cuenta desde 0 en este archivo; y `TSTART_MJD_CORR` vuelve a sumar `nsuboffs·tsubint`. MJD desplazado en minutos u horas para el segundo archivo de una observación partida |
| P2-08 | `ensure_csv_header` reescribe el CSV de forma no atómica y desalinea columnas | `candidate_manager.py:60-84` | `open("w")` trunca antes de escribir, sin copia; el relleno asume que las columnas nuevas van al final, pero varias (`snr_waterfall_linear`, `dm_status`) están en medio del header, de modo que los datos antiguos quedan bajo etiquetas equivocadas |
| P2-09 | CSV y JSON sin `encoding` explícito | `candidate_manager.py:65,72,79,128` | En Windows se usa cp1252; en Linux UTF-8. Un nombre de archivo con tilde produce mojibake o `UnicodeDecodeError` al leer el CSV en el cluster. `execution_summary.py:98` y `validation_metrics.py:403` sí lo declaran |
| P2-10 | Los contratos casi no validan en runtime | `contracts.py:23-92` | Solo `ChunkPlan` tiene `__post_init__`. `from_config` fabrica valores tóxicos en silencio (`time_reso=0.0` si la cabecera no se extrajo), y no valida `dm_min < dm_max` ni la monotonía de `DMGrid.values`, que es supuesto implícito de `dm_for_row` |
| P2-11 | Timestamps naive en hora local, con resolución de 1 s | `execution_summary.py:69,93`, `validation_metrics.py:31,397` | Al retroceder el reloj por horario de verano, dos corridas producen nombres idénticos y la segunda sobrescribe la primera. En contenedor con TZ distinta, los timestamps no son comparables |
| P2-12 | `inject_config` ignora claves desconocidas pese a prometer `ValueError` | `config.py:253-272` | El docstring dice que lanza; el código emite `logger.warning` y continúa, además antes de que `setup_logging` se haya ejecutado, así que el aviso puede no aparecer. Un typo en una clave corre con el default. `CLASS_PROB_LINEAR`, `SNR_THRESH_LINEAR`, `DM_GRID_MODE`, `PREWHITEN_BEFORE_DM` y `TRIAL_CORRECTION` **no están** en `_KNOWN_CONFIG_KEYS`, luego no son inyectables |
| P2-13 | `validation_status: COMPLETE` ignora los chunks fallidos | `validation_metrics.py:393,215,357` | Solo mira el contador de OOM. Además `no_edge_losses` y `continuity_with_previous` están cableados a `True`, y `validate_continuity` no valida nada |
| P2-14 | `record_data_characteristics` usa atributos inexistentes en su fallback | `validation_metrics.py:79-81` | `config.FREQ_CENTRAL` y `config.BANDWIDTH` no existen. Si el camino principal lanza, el `except` provoca un `AttributeError` no capturado; como se llama en `pipeline.py:515`, fuera del `try` grande, el archivo se marca ERROR sin haberse procesado |
| P2-15 | Métrica de memoria falsa en el JSON de validación | `validation_metrics.py:63,66` | Escribe `max_vram_fraction: 0.7` fijo y calcula `usable_ram_gb` con `0.25` literal en vez de `config.MAX_RAM_FRACTION` (valor real 0,10), pese a leer el valor real en la línea inmediatamente anterior. Inutilizable para diagnosticar un OOM |
| P2-16 | `fastmath=True` en el kernel científico | `dedispersion.py:135`, `data_downsampler.py:21` | Autoriza reasociación y contracción FMA: el resultado no es bit a bit idéntico al fallback NumPy ni necesariamente estable entre CPU con distinto ancho SIMD. Los kernels `prange` en sí son correctos: cada iteración escribe solo su porción |
| P2-17 | Caché de Numba compartida entre procesos | `dedispersion.py`, `data_downsampler.py` (`cache=True`) | Si el usuario paraleliza lanzando varios procesos —la vía natural aquí—, dos pueden escribir el índice de caché a la vez |
| P2-18 | `OMP_NUM_THREADS` y `MKL_NUM_THREADS` se fijan después de importar NumPy y torch | `hardware_profile.py:199-205`, llamado desde `pipeline.py:968` | OpenMP y MKL leen esas variables al inicializarse. El propio código reconoce el problema para Numba y llama `numba.set_num_threads`, pero no hay equivalente para OMP/MKL ni `torch.set_num_threads`. Sobresuscripción de CPU |
| P2-19 | `CandidateWriter` sin lock ni bloqueo de archivo | `candidate_manager.py:103-149` | Hoy el pipeline es monohilo, así que no hay carrera intra-proceso. El riesgo es inter-proceso, agravado por la colisión de rutas de P1-14 |
| P2-20 | Handle del CSV sin cerrar en los caminos de error | `candidate_manager.py:126-149` | Sin context manager; `close()` solo se alcanza vía `flush_all()`. En Windows el archivo queda bloqueado hasta que el intérprete termina |
| P2-21 | 26 manejadores terminan en `pass`; unos 12 ocultan corrupción | `fits_handler.py:280,321,325,499,506`; `slice_len_calculator.py:403,585,627` | Los de `fits_handler` dejan las globales del archivo anterior (motor de P1-09) y pueden dejar el orden de frecuencias sin corregir. Los de `slice_len_calculator` caen a defaults que cambian la segmentación temporal |
| P2-22 | `except ImportError` muerto alrededor del logger | `pipeline.py:316,451`; `detection_engine.py:61-63` | `get_global_logger()` nunca lanza `ImportError` (`logging_config.py:461-466` solo hace inicialización perezosa). Handler muerto: un fallo real de logging propaga sin filtro |
| P2-23 | La decisión LF/HF cae a LF ante cualquier excepción | `pipeline.py:680-682` | `except Exception: use_hf = False`. Una observación de alta frecuencia que debía usar la ruta HF se procesa con CenterNet sobre un bow-tie colapsado, con sensibilidad radicalmente distinta. No queda registro de qué pipeline se usó por archivo |
| P2-24 | HF ignora `streaming_func` y lee siempre con `stream_fits_multi_pol` | `high_freq_pipeline.py:1199,1364,1371` | Un `.fil` que dispare el criterio bow-tie va a un lector FITS. Falla ruidosamente (`fits_handler.py:741` hace `raise RuntimeError`), no en silencio, por lo que el estado final es FAILED y no un CSV vacío con SUCCESS |
| P2-25 | El pipeline HF exige la librería `your` | `fits_handler.py:20-23,741-744` | `stream_fits_multi_pol` lanza `RuntimeError` si `your_psrfits` no está disponible. Sin `your`, todo HF es inutilizable, mientras LF degrada a astropy sin problema. Asimetría de robustez no documentada |
| P2-26 | `patch_file` colisiona entre bandas y entre candidatos | `detection_engine.py:86-90` | El sufijo del nombre es `band_img.shape[0]`, que es la altura DM, no el índice de banda. Con multi-banda las tres escriben el mismo nombre. Varias filas del CSV apuntan al mismo PNG, que corresponde a la última banda dibujada |
| P2-27 | Contradicción de `prewhiten_before_dm` entre archivos de configuración | `config.yaml:66` (`false`) frente a `advanced-config/visualization.yaml:64` (`true`) | Gana `config.yaml`, porque el otro archivo ni se lee. El prewhitening altera la física del cubo DM, según el propio comentario del YAML. Un operador que edite el archivo "avanzado" creerá que cambió el comportamiento científico |
| P2-28 | `config.yaml` se autocontradice en el presupuesto de memoria | `config.yaml:219,225,241,249,261,266,269` | Fija `max_ram_fraction: 0.10` mientras los comentarios marcan `[CURRENT]` sobre 0.15 y la tabla de `dm_chunking_threshold_gb` está calculada para 0.15. Con 16 GB y el valor real: RAM utilizable ≈ 0,98 GB frente a un umbral de 2,0 GB, el doble |
| P2-29 | Tres de los cuatro YAML avanzados se cargan y nunca se leen | `user_config.py:32-45,150` | `visualization.yaml`, `models.yaml` y `logging.yaml` suman 579 líneas y ~170 claves inertes. De `performance.yaml`, 1 de 23 claves llega a producción |
| P2-30 | `CPU_THREADS` se carga y nunca se conecta | `user_config.py:170`, `pipeline.py:968` | `apply_thread_settings(hw)` se llama sin el segundo argumento, que `hardware_profile.py:186-190` documenta como override de usuario. Una opción que existe en el YAML y no hace nada |
| P2-31 | El logging no es configurable | `pipeline.py:959` | `setup_logging(level="INFO", use_colors=True)` literal. `config.py:222-224` define `LOG_LEVEL`, `LOG_COLORS` y `LOG_FILE` con cero consumidores. No se puede subir a DEBUG en producción sin editar código |
| P2-32 | Logs sin rotación, dentro del árbol de código | `logging_config.py:128-132` | `logging.FileHandler` (no rotativo) escribiendo en `src/logging/log/`. En Docker, dentro de la capa escribible del contenedor. `advanced-config/logging.yaml:29-30` declara `max_file_size_mb` y `backup_count` que nadie aplica |
| P2-33 | Los logs se pierden con el modo de uso documentado | `docker-compose.yml:49,110` | El volumen `./logs` está comentado y el modo documentado es `docker compose run --rm`. El log completo de la corrida desaparece con el contenedor |
| P2-34 | El job de mutación de CI no es ejecutable | `.github/workflows/ci.yml:48-51` | `cosmic-ray exec mutation/cr-*.toml` sin `cosmic-ray init` previo y sin pasar la base de sesión. Fallaría siempre; además está tras `workflow_dispatch`, así que nunca corre en PR |
| P2-35 | CI rompe el modo hash-checking justo después de activarlo | `.github/workflows/ci.yml:27-28` | `pip install -r requirements.lock.txt` instala el entorno verificado; la línea siguiente, `pip install -e .[dev]`, resuelve `pytest` y transitivas contra PyPI en vivo, sin hashes |
| P2-36 | CI sin bloque `permissions:` y con acciones por tag mutable | `.github/workflows/ci.yml:17,20,40,41` | El `GITHUB_TOKEN` hereda el default del repositorio, que en configuraciones legadas es read-write sobre `contents`. `actions/checkout@v4` y `setup-python@v5` sin pin de SHA. *Positivo verificado:* no hay `pull_request_target` ni interpolación de datos de evento dentro de bloques `run:` |
| P2-37 | Pesos `.pth` sin verificación de integridad, con una opción de config que miente | `system_validator.py:78-90`, `advanced-config/models.yaml:129` | Solo se comprueba `Path.exists()`. `models.yaml` declara `check_weights_integrity: true` y ningún módulo consume esa clave. No existe ningún SHA256 de referencia |
| P2-38 | `config.yaml` montado `:rw` en el contenedor | `docker-compose.yml:36,97` | Cero ocurrencias de `yaml.dump` en el repositorio: nada escribe ese archivo. Código comprometido dentro del contenedor podría reescribir la configuración del host |

### Sobre P2-01 y P2-02

Son la única superficie donde datos de terceros causan daño directo, y ambos tienen prueba de concepto ejecutada. La corrección es de dos líneas:

```python
# filterbank_handler.py:92-94  ->  break (o raise), y un tope de iteraciones en el while
# filterbank_handler.py:37-40  ->  validar 0 <= length <= 256 antes del read
```

---

## 9. P3 — Deuda técnica y bajo riesgo

| ID | Hallazgo | Ubicación |
|---|---|---|
| P3-01 | Eje temporal construido con `linspace` de paso `dt·N/(N−1)` en vez de `dt` | `detection_engine.py:130-132,256-258`; `high_freq_pipeline.py:453-456` |
| P3-02 | `_dm_from_image_at_time` asume rejilla uniforme mientras el cubo usa `calculate_dm_values` | `high_freq_pipeline.py:89` frente a `pipeline_parameters.py:93-109` |
| P3-03 | `n_trials` ignora `DM_GRID_MODE`, afectando a `post_trials_sigma` y `rank_score` | `detection_engine.py:275`; `high_freq_pipeline.py:803` |
| P3-04 | Filterbank de 16 bits leído como `int16` con signo; `nbits < 8` cae a `uint8` y provoca `ZeroDivisionError` | `filterbank_handler.py:447-452,470,124` |
| P3-05 | `d_dm_time_g` muta el array del llamador al hacer prewhitening, condicionado al dtype | `dedispersion.py:272-285` |
| P3-06 | `prep_patch` propaga NaN al clasificador sin guarda; `prob = nan` da `is_burst=False` y `rank_score` NaN | `model_interface.py:57-65` |
| P3-07 | En LF, el tiempo del parche mostrado es siempre el del primer candidato | `detection_engine.py:187-219` |
| P3-08 | MJD serializado con `.12f`, más allá de la precisión de `float64` (los últimos 1-2 dígitos son ruido) | `candidate_manager.py:210-214` |
| P3-09 | Defaults de código divergentes del YAML cuando la clave se omite: `CLASS_PROB_LINEAR` 0.6 frente a 0.3; `SNR_THRESH_LINEAR` 5.0 frente a 3.0; `ENABLE_LINEAR_CLASSIFICATION` `True` frente a `false` | `user_config.py:94,96,119` |
| P3-10 | `eval()` sobre argumento de línea de comandos en el script de entrenamiento | `training/binary_classification/binary_train.py:43` |
| P3-11 | `torch.load` sin `weights_only=True` en los scripts de entrenamiento | `binary_train.py:64`; `centernet_train.py:72,159` |
| P3-12 | Instalación de dependencias desde PyPI en tiempo de ejecución dentro de un `except ImportError` | `analyze_alma_phases_validation.py:759-761` y 2 scripts más |
| P3-13 | Superficie innecesaria en las imágenes finales: `gcc`, `g++`, `gfortran`, `make`, `wget`, `curl`, `git` heredados en la etapa runtime | `Dockerfile:23-51,93` |
| P3-14 | GPU declarada dos veces y por el mecanismo legado (`runtime: nvidia` + `deploy.resources.devices`) | `docker-compose.yml:92,131-134` |
| P3-15 | `reservations` de CPU y memoria son inertes fuera de Swarm | `docker-compose.yml:68-69,129-130` |
| P3-16 | Sin `HEALTHCHECK` ni `ENTRYPOINT`; `restart: "no"` para corridas de días | `Dockerfile:114,228`; `docker-compose.yml:75,140` |
| P3-17 | `advanced-config/` no se copia a la imagen Docker | `Dockerfile:105-108,218-221` |
| P3-18 | Sin caché de pip en CI: se descarga torch y 15 wheels de NVIDIA en cada job y cada versión de Python | `.github/workflows/ci.yml:19-22` |
| P3-19 | `root_logger.handlers.clear()` elimina handlers instalados por terceros o por el host | `logging_config.py:141` |
| P3-20 | La validación de arranque no bloquea: incluso si faltan los `.pth`, la corrida arranca y muere después | `system_validator.py`, `pipeline.py:978-980` |
| P3-21 | 120 MB de pesos versionados sin LFS; `.git` pesa 156 MB | `src/models/*.pth`, `.gitignore:69-71` |
| P3-22 | 2,9 MB de `.sqlite` de cosmic-ray versionados, regenerables, sin regla en `.gitignore` | `mutation/*.sqlite` |
| P3-23 | `Workflow-HF.png` pesa 7,7 MB; capitalización inconsistente con `WorkFlow-LF.png` | raíz |
| P3-24 | `.gitignore` ignora `docs/` por completo: documentación nueva se perdería en silencio | `.gitignore` |
| P3-25 | `.gitignore` ignora `*.txt` globalmente con solo dos excepciones | `.gitignore:51-53` |
| P3-26 | `[tool.mutmut]` configurado; la infraestructura real es cosmic-ray y mutmut no está instalado en ningún sitio | `pyproject.toml:55-63` |
| P3-27 | Deriva de versión: `pyproject` 0.1.0 frente a README y CHANGELOG 1.0.0 | raíz |
| P3-28 | Deriva de versión de Python en cinco lugares: `.python-version` 3.12, `pyproject` ">=3.11", README "3.10+", CHANGELOG "3.8+", CI [3.11, 3.12] | raíz |
| P3-29 | `CHANGELOG.md` obsoleto desde 2025-11-14, con al menos 10 commits sustantivos posteriores | raíz |
| P3-30 | `.claude/settings.local.json` versionado con reglas de permiso terminadas en comodín | `.claude/settings.local.json:8,11,12` |
| P3-31 | `.mailmap` escrito y funcional pero sin versionar; contiene un correo corporativo previo cuya publicación es una decisión de privacidad a tomar conscientemente | raíz, `.mailmap:16` |

---

## 10. Problemas sistémicos

Los 97 hallazgos se agrupan bajo cinco causas raíz. Corregir la causa previene la reaparición; corregir solo los síntomas no.

### S-1 — No existe una definición de "completado" atada a salida durable

```
Problema central:
El progreso se marca por avance del bucle, no por persistencia confirmada.

Manifestaciones:
├── P0-2  el buffer del CSV no se vacia en HF ni en las rutas de error
├── P0-3  el checkpoint usa una convencion de indices y el consumidor otra
├── P0-4  el checkpoint se guarda tras un chunk fallido
├── P1-15 las rutas de error reportan 0 candidatos habiendo escrito candidatos
├── P1-16 el checkpoint no valida configuracion ni identidad del input
├── P1-19 cero reintentos: un fallo transitorio se convierte en perdida definitiva
└── P2-13 validation_status COMPLETE ignora los chunks fallidos
```

**Corrección de raíz:** definir una barrera de durabilidad por chunk —vaciar el CSV con `fsync`, luego escribir el checkpoint, en ese orden y dentro del mismo `try`— y hacer que el estado reportado se derive de lo que hay en disco, no de contadores en memoria.

### S-2 — El orquestador de archivo está duplicado y las dos copias divergen

```
Problema central:
Al introducirse el modo HF se copio el orquestador completo (411 vs 432 lineas,
50 % identicas) en vez de parametrizar el unico eje que realmente varia:
como se detectan candidatos en un slice.

Manifestaciones:
├── P0-2  flush_all solo en LF
├── P1-17 checkpoint solo en LF
├── P2-24 streaming_func recibido e ignorado en HF
├── P2-25 HF exige `your`; LF degrada a astropy
├──       MAX_CHUNK_SAMPLES aplicado solo en LF
└──       el ciclo pipeline <-> high_freq_pipeline, parcheado con imports tardios
```

**Corrección de raíz:** P1-24, con la precondición del test de CSV byte-idéntico.

### S-3 — No hay una invariante explícita de marco de coordenadas

```
Problema central:
Las magnitudes viajan sin declarar en que espacio viven: marco de imagen del
CNN (512x512) frente a marco del cubo; muestras crudas frente a decimadas;
indices absolutos del archivo frente a relativos al bloque.

Manifestaciones:
├── P0-1  cajas del CNN interpretadas en el marco del cubo  [el mas grave]
├── P1-01 trim_valid_window ignora el solape derecho
├── P1-04 start_sample relativo al bloque en vez de a la region valida
├── P1-05 la ruta de emergencia salta 2x solape
├── P1-06 la primera ventana de solape nunca se busca
├── P1-11 block_ds y block_raw_ds recortados con origenes distintos
├── P1-12 compute_snr_profile recorriendo el eje DM
├── P2-05 dedisperse_block desplaza start y no lo comunica
└── P2-07 PSRFITS troceados: el offset se cuenta dos veces
```

**Corrección de raíz:** tipar o nombrar las magnitudes por su marco (`t_sample_abs_raw`, `t_sample_rel_ds`, `py_img512`) y añadir aserciones de continuidad entre chunks (`end_sample[n] == start_sample[n+1]`), que hoy no existen en ningún punto.

### S-4 — La configuración no tiene una fuente de verdad única

```
Problema central:
Un valor puede provenir de config.yaml, de un YAML avanzado que nadie lee,
de un default de user_config, de un getattr con default local, o de un
literal hardcodeado. Y todos pueden discrepar.

Manifestaciones:
├── P1-20 config global mutable, mutado desde 6 modulos en runtime
├── P2-12 inject_config ignora claves y 5 opciones no son inyectables
├── P2-27 prewhiten_before_dm: false en un archivo, true en otro
├── P2-28 config.yaml se autocontradice en el presupuesto de memoria
├── P2-29 579 lineas de YAML cargadas y nunca leidas
├── P2-30 CPU_THREADS cargado y nunca conectado
├── P2-31 el logging no es configurable pese a tener 3 constantes y un YAML
├── P3-09 defaults de codigo divergentes del YAML
└──       161 getattr(config, K, default) con 57 claves y defaults contradictorios
```

**Corrección de raíz:** declarar `config.yaml` como fuente única; sustituir `getattr(config, K, default)` por `config.K` cuando `K` esté declarada —si falta, debe fallar, no adivinar—; y borrar o cablear los YAML avanzados. Un test que recorra `config.yaml` y verifique que cada `getattr` usa el mismo default que el YAML cierra la clase entera.

### S-5 — Los tests verifican el caso degenerado o la ausencia de excepción

```
Problema central:
Varios P0/P1 sobreviven porque el test que deberia cubrirlos prueba su propia
reimplementacion, fija el parametro justo en el valor donde el bug no aparece,
o solo comprueba que no se lanza una excepcion.

Manifestaciones:
├── P1-03 test_cube_windowed_parity fija el umbral para que haya UN solo trozo
├── P2-04 test_dedispersion_parity inspecciona el AST del kernel CUDA, no lo ejecuta
├── P0-1  test_contracts llama extract_candidate_dm con img_height=512,
│         exactamente lo que el pipeline NO hace
├──       test_downsampler no importa downsample_data: valida su propia referencia
├──       test_dm_chunking_threshold recolecta 0 tests
├──       test_large_file_processing: 5 de 6 tests son aritmetica propia
└──       no hay test de trim_valid_window, ni de continuidad entre chunks,
          ni de que el CSV contenga las filas contabilizadas en el resumen
```

**Corrección de raíz:** la regla es que un test debe importar y ejecutar el símbolo de producción que dice proteger. El commit más reciente del repositorio (`65419c9`, *"make parity tests actually compare against production"*) indica que el equipo ya identificó este patrón; queda aplicarlo al resto.

---

## 11. Seguridad

**Modelo de amenaza aplicado:** pipeline científico offline. Los atacantes plausibles son quien provee los archivos FITS/filterbank, quien provee los pesos `.pth`, un contribuyente al repositorio o un vecino en un cluster compartido. No hay superficie HTTP, ni autenticación, ni multi-tenencia lógica. Los hallazgos se ajustan a ese modelo.

### Hallazgos por prioridad

| ID | Sev. | Hallazgo | Superficie |
|---|---|---|---|
| P1-21 | P1 | Docker instala torch 2.1.0, anulando la única mitigación de deserialización (`weights_only=True` solo es sólido desde 2.6) | Pesos de terceros en el contenedor |
| P1-22 | P1 | El Dockerfile ignora el lockfile con hashes | Cadena de suministro de la imagen |
| P2-01 | P2 | Bucle infinito con cabecera `.fil` truncada (PoC ejecutado) | Datos de terceros |
| P2-02 | P2 | `f.read(length)` sin validar absorbe el archivo entero (PoC ejecutado) | Datos de terceros |
| P2-37 | P2 | Pesos sin checksum, con una opción de configuración que dice verificarlos y no existe | Sustitución de modelo |
| P1-08 | P2 | Descargas remotas no declaradas (registro de sitios de astropy, efeméride de ~10 MB, tablas IERS) por cada candidato, en un pipeline vendido como offline | Salida a Internet |
| P1-18 | P2 | Memmaps de 4 GB o más que nunca se borran de `/tmp` | Disco compartido |
| P2-36 | P2 | CI sin `permissions:` y con acciones por tag mutable | `GITHUB_TOKEN` |
| P3-30 | P3 | `.claude/settings.local.json` versionado con auto-aprobaciones de shell terminadas en comodín | Agentes en clones del repositorio |
| P3-10 | P3 | `eval()` sobre `sys.argv[3]` en el script de entrenamiento | CLI de entrenamiento |
| P3-11 | P3 | `torch.load` sin `weights_only` en los tres puntos de reanudación de entrenamiento | Checkpoints compartidos |
| P3-12 | P3 | `pip install` en tiempo de ejecución dentro de un `except ImportError` | Entorno Python |
| P2-38 | P3 | `config.yaml` montado `:rw` sin que nada lo escriba | Configuración del host |
| P3-13 | P3 | Compiladores y clientes HTTP en la imagen de producción | Post-explotación |
| P3-31 | P3 | `.mailmap` expone un correo corporativo previo si se commitea | Privacidad |

### Riesgos descartados por modelo de amenaza

Todo lo siguiente se buscó explícitamente y **no** se reporta como vulnerabilidad:

| Riesgo | Por qué no aplica |
|---|---|
| Path traversal en rutas de salida | Las rutas se construyen con `fits_path.stem`/`.name`, y el origen es `Path.rglob()` sobre disco local, no una cadena de red. Verificado además que **ningún** valor de cabecera FITS/FIL (`SRC_NAME`, `OBJECT`, `rawdatafile`, `source_name`) se usa para construir rutas; solo se registra en logs |
| `yaml.load` inseguro | Las dos cargas YAML usan `yaml.safe_load` (`user_config.py:27,44`) |
| Command injection | Las tres llamadas a `subprocess` usan lista de argumentos con `sys.executable` y sin `shell=True`. Cero `os.system`, `os.popen` o `shell=True` en todo el repositorio |
| SSRF clásico | No hay `requests`, `urllib`, `http.client` ni `socket` en `src/`. El único tráfico saliente es el de astropy y no acepta URL controlada por el usuario |
| `pickle`, `joblib`, `np.load(allow_pickle=True)` | Ninguna aparición. `np.load` se usa dos veces con el default `allow_pickle=False` sobre `.npy` del dataset local de entrenamiento |
| Zip-slip | Cero `zipfile`, `tarfile`, `shutil.unpack_archive`, `extractall`. El pipeline no descomprime nada |
| Secretos hardcodeados | Barrido de `api_key|secret|token|password|credential|AKIA|ghp_|xox|BEGIN` sobre todos los tipos de archivo: cero resultados. Ningún `.env`, `.pem` o `id_rsa` añadido jamás al historial. CI no referencia ningún `secrets.*` |
| Rutas absolutas de usuario en producción | Solo dos coincidencias, ambas el nombre del autor en un `LABEL` y en los créditos del README. Las de `src/scripts/` y `docker-compose.yml` sí existen pero son problema operativo, no de seguridad |
| Inyección en expresiones de GitHub Actions | No hay `pull_request_target` ni interpolación de datos de evento dentro de bloques `run:`. El único `${{ }}` es `matrix.python-version`, valor cerrado |
| `eval`/`exec` dinámico en el pipeline | Los ~15 `.eval()` son `torch.nn.Module.eval()`. El único `eval()` real está fuera del pipeline de producción |
| Desbordamiento por dimensiones de cabecera `.fil` | `np.memmap` valida la longitud contra el tamaño real del archivo y lanza `ValueError`. El DoS real de esa superficie es P2-01 y P2-02 |
| Permisos de archivos temporales | `tempfile.mkstemp` crea con modo 0600. La escritura atómica del checkpoint (temp + `replace`) está bien hecha |
| Exposición de red del contenedor | `network_mode: bridge` sin `ports:` publicados. No hay nada escuchando |

### Positivos verificados

Usuario no-root creado y activo en ambas imágenes (`Dockerfile:54,111,169,225`) y en compose (`user: "1000:1000"`); `.dockerignore` excluye `.git/`, `*.pth`, `Data/`, `Results/` y logs; sin `ARG`/`ENV` con secretos; volúmenes de modelos y datos en `:ro`; `inject_config` filtra contra una allowlist.

---

## 12. Datos e integridad

### Esquema de salida

**Positivo:** ambos pipelines serializan por `Candidate.to_row()` contra un único `CANDIDATE_HEADER` de 39 columnas, y el orden coincide. **El esquema es consistente entre escritores.** Los problemas están en el *cuándo* y el *cómo* se escribe, no en el *qué*.

**Riesgo latente:** `CANDIDATE_HEADER` (`candidate_manager.py:17-57`) y `to_row()` (`:196-264`, una lista literal de 9 elementos seguida de unos 30 `append` repartidos en 68 líneas) son dos listas paralelas mantenidas a mano. `grep CANDIDATE_HEADER tests/` no devuelve nada: **ningún test valida que sus longitudes coincidan**. Ningún lector importa el header; los nombres de columna aparecen como literales en 8 scripts, que ya se defienden con `if 'col' in df.columns: ... else: print("Advertencia")` — evidencia de que el desajuste ya ocurrió.

La corrección más barata del repositorio es un test de una línea:

```python
assert len(CANDIDATE_HEADER) == len(Candidate(...).to_row())
```

### Secuencias de fallo parcial identificadas

**SEQ-1 — Candidatos en el CSV, resumen en cero, buffer perdido** (P0-2 + P1-15)

```
[chunk 1..79] -> append_candidate() -> buffer(50) -> CSV: 250 filas en disco
                                                     buffer: 37 filas en RAM
[chunk 80]    -> OSError leyendo el archivo
                    |
                    +-> except (pipeline.py:860-904) -> return n_candidates: 0   MIENTE
                    +-> flush_all() (linea 813) NUNCA se ejecuta   -> 37 filas perdidas
                    +-> handle del CSV NUNCA se cierra              -> archivo bloqueado

ESTADO FINAL: CSV=250 filas | resumen="0 candidatos" | 37 candidatos evaporados
```

**SEQ-2 — El checkpoint garantiza que lo perdido no se recupere** (P0-4 + P0-3): descrita en la ficha de P0-4.

**SEQ-3 — CSV escrito antes que los plots**

```
detect_and_classify_candidates_in_band()
   1. append_candidate(csv, cand.to_row())     detection_engine.py:333   <- A
   2. dedisperse_block(...)                    detection_engine.py:564   <- B
   3. save_all_plots(...)                      detection_engine.py:569   <- C

Si B o C fallan -> except del chunk (pipeline.py:769)
ESTADO FINAL: fila en el CSV con patch_file apuntando a un PNG que no existe.
              El chunk cuenta como fallido, pero el candidato ya esta publicado.
```

Agravante (P2-26): aunque C tenga éxito, el nombre del PNG colisiona entre bandas y entre candidatos del mismo slice.

**SEQ-4 — Reorganización destructiva de plots al reprocesar**

```
_process_block(), pipeline.py:454-483
   1. mkdir  Composite/<stem>/ChunksWithFRBs/
   2. glob   Composite/<stem>/chunk042/*.png
   3. if destino existe: shutil.rmtree(destination_dir)   <- BORRADO IRREVERSIBLE
   4.                    shutil.move(chunk_dir, destination_dir)

Si el proceso muere entre 3 y 4 -> los plots de ese chunk desaparecen
                                   y el CSV conserva sus filas
Al reprocesar (P1-13)           -> el paso 3 borra los plots de la corrida anterior
                                   mientras el CSV ACUMULA las filas nuevas
```

**SEQ-5 — Contaminación de MJD entre archivos** (P1-09): descrita en su ficha.

**SEQ-6 — El disco se llena en silencio** (P1-18)

```
por cada chunk con cubo DM >= 4 GB:
   _allocate_dm_cube_buffer()  -> mkstemp(...)        <- +5 GB en el directorio temporal
   trim_valid_window()         -> dm_time = [...].copy()  <- se suelta la referencia
   del dm_time_full            -> el ARCHIVO sigue en disco  <- nadie hace unlink

chunk 001: +5 GB   chunk 002: +10 GB   ...   chunk 200: +1 TB
                                                  |
                                                  +-> OSError: No space left
                                                      capturado por el except del chunk
                                                      -> "PARTIAL", causa real invisible
```

### Sin base de datos

El sistema no usa base de datos relacional. No aplican N+1, índices, joins, transacciones SQL ni locking de base de datos. La persistencia es CSV, PNG, JSON y checkpoint JSON sobre sistema de archivos.

---

## 13. Concurrencia

**Superficie real, verificada por grep exhaustivo** (`threading`, `multiprocessing`, `concurrent.futures`, `asyncio`, `prange`, `num_workers`, `Pool`, `Lock`, `joblib`):

El pipeline de producción es **estrictamente monohilo a nivel de Python**. No hay `threading`, ni `multiprocessing`, ni `asyncio`, ni `concurrent.futures` en ningún módulo de `src/` fuera de `src/scripts/`. El paralelismo se limita a Numba `prange` (2 kernels), hilos de OpenMP/MKL bajo NumPy, hilos internos de PyTorch, y `DataLoader(num_workers=0)` en entrenamiento.

**Consecuencia:** no hay riesgo de fork combinado con CUDA, ni deadlock por locks anidados, porque no hay locks ni forks. Tampoco hay `async def` en el repositorio, así que la clase entera de problemas de async con I/O bloqueante no aplica.

**Análisis de carrera en los kernels `prange`:** ambos son seguros. En el downsampler cada iteración escribe solo `out[t, f]`; en la dedispersión cada iteración escribe solo `out[:, i, :]` y sus acumuladores se asignan dentro del cuerpo del bucle, por lo que son privados por iteración. El número de hilos no altera el resultado.

**Problemas reales de esta categoría:**

- **P2-16** — `fastmath=True` autoriza reasociación y contracción FMA: la ruta Numba no es bit a bit idéntica al fallback NumPy ni necesariamente estable entre CPU con distinto ancho SIMD. Es la ruta científica crítica; el coste de quitarlo es menor que el riesgo de reproducibilidad.
- **P2-17** — `cache=True` escribe el índice de caché en el `__pycache__` del paquete. Si el usuario paraleliza lanzando varios procesos —la vía natural, dado que no hay paralelismo interno—, dos pueden corromper la caché.
- **P2-18** — Sobresuscripción de CPU por fijar las variables de entorno de OpenMP y MKL después de importar NumPy y torch.
- **P1-20** — El estado global mutable impide procesar dos archivos en paralelo en el mismo proceso.
- **P1-14 y P2-19** — La colisión de rutas por `stem` y la ausencia de bloqueo de archivo hacen que dos procesos apuntando al mismo CSV se entrelacen (POSIX) o fallen con `PermissionError` (Windows).
- **P3-19 y matplotlib** — Se usa la API global `pyplot`, que no es thread-safe, y no se fuerza el backend `Agg`. Hoy no hay fuga porque cada figura se cierra, pero cualquier intento de paralelizar el plotting con hilos romperá.

**Recomendación:** si se quiere escalar, la vía correcta en este diseño es **procesos independientes por archivo**, y eso exige resolver antes P1-14 (rutas únicas) y P1-20 (estado global).

---

## 14. Performance

Escenario de referencia derivado de `config.yaml` y de las constantes del código (datos tipo Lovell: 800 canales, 8 bit, tsamp 0,256 ms; archivo de 5,5 TiB):

| Magnitud | Valor |
|---|---|
| Muestras crudas | 7,56e9 (unos 22 días de observación) |
| `dt_ds` | 1,024 ms |
| Canales tras decimar | 400 |
| `slice_len` | 977 muestras |
| `height_dm` | ~221 trials (161 con `legacy_uniform`) |
| Chunk efectivo | 496.316 muestras crudas (~127 s) |
| Número de chunks | ~15.240 |
| Slices totales | **~1,94 millones** |
| Cubo DM por chunk | 0,33 GB |

### Hotspots

| ID | Ubicación | Complejidad | Impacto | Acción |
|---|---|---|---|---|
| PERF-01 | `detection_engine.py:546-596` + `config.yaml:300` | O(N_slices) renders | **Crítico bajo escala** | `force_plots: false` (P1-23) |
| PERF-02 | `dedispersion.py:435-472` | O(H/D · C) lanzamientos | **Crítico** | Vectorizar con `gather`, índices `int32`, no materializar `idx`. Los tensores transitorios suman ~1,85 GB por iteración; con `max_chunk_samples` mayor, OOM de VRAM seguro. Además `bytes_per_dm` (`:417`) sobreestima ~6× el uso real, así que la heurística no protege: manda el suelo rígido de 16 |
| PERF-03 | `dedispersion.py:256,481`; `data_flow_manager.py:276,319,469` | 5× el tamaño del cubo | **Crítico** | Un `np.zeros` descartado en las rutas torch-GPU y CPU; `.astype(np.float32)` que copia siempre aunque ya sea float32; copia de ventana única innecesaria. ~1,3 GB de memcpy y 0,66 GB de pico extra por chunk |
| PERF-04 | `pipeline.py:423-428`; `model_interface.py:38,78,81` | O(N_slices) con sincronización | **Crítico** | `gc.collect()` por slice (≈11 h solo en GC a escala TB); `empty_cache()` cada 10 slices; `.item()` por candidato; inferencia con batch 1. No hay AMP, `channels_last`, `torch.compile` ni `cudnn.benchmark` en el repositorio, y `advanced-config/performance.yaml` declara `enable_mixed_precision` y `batch_size` que nadie lee |
| PERF-05 | `filterbank_handler.py:508-518` | 3 pasadas completas | **Crítico** | `.copy()` + `ascontiguousarray` + `astype(np.float32)`; el `astype` es redundante porque el kernel de downsampling ya castea elemento a elemento. 2,4 GB de pico donde basta 0,40 GB |
| PERF-06 | `fits_handler.py:1018-1023,1218` | 2 copias por chunk | Relevante | Emitir vista o escribir sobre buffer circular |
| PERF-07 | `dedispersion.py:176-179` | O(H·C·W) | Relevante | `count_series[...] += 1` dentro del bucle interno; array de diferencias O(C+W) por DM |
| PERF-08 | `hardware_profile.py:74` | — | Relevante | `min(self.cpu_cores_physical, 8)` limita artificialmente en servidores |
| PERF-09 | `plot_composite.py:336-374,1255-1277` y 2 módulos más | 4× copia y normalización | Relevante | El mismo bloque dedispersado se copia y se recalcula `compute_snr_profile` al menos 4 veces |
| PERF-10 | todo `src/visualization/` | — | Relevante | `matplotlib.use('Agg')` no existe en el repositorio |
| PERF-11 | `plot_individual_components.py:412-417` | +2 figuras por slice HF | Relevante | El `else:` contiene solo un `logger.info`; las llamadas están desindentadas |
| PERF-12 | `visualization_unified.py:87-96`; `detection_engine.py:76` | O(N_slices) | Relevante | El colormap devuelve `float64`, duplicando la transferencia a GPU; `postprocess_img` se ejecuta aunque no haya plots |
| PERF-13 | `data_downsampler.py:76-97` | 4 copias + bucle Python | Relevante (condicional) | Solo si `temporal_mode != sum` |
| PERF-14 | `dedispersion.py:553,638` | O(C) Python por candidato | Relevante | Vectorizar con `take_along_axis` |
| PERF-15 | `high_freq_pipeline.py:1462-1485` | 4 polarizaciones en RAM | Relevante (solo HF) | Reducir por polarización en streaming |
| PERF-16 | `snr_utils.py:95-102` | O(n·suma de anchos) | Local | `np.convolve` directo con 15 anchos hasta 300; suma acumulada da O(n) por ancho |
| PERF-17 | `high_freq_pipeline.py:43-48` | O(n) en Python | Local | Bucle sobre el perfil SNR, vectorizable |
| PERF-18 | `validation_metrics.py:51,219,366-379` | O(N_chunks) sin cota | Local | La lista de chunks nunca se poda; 5 pasadas en el export |
| PERF-19 | `centernet_utils.py:50-58,109` | Por forward | Local | `meshgrid` reconstruido en CPU y copiado a GPU en cada forward; `.cpu().unique()` sobre un tensor de una sola clase |
| PERF-20 | `slice_len_calculator.py:181-182` | — | Insignificante | `detect_hardware()` por archivo |

### Lo que ya está bien

- **Streaming real:** el archivo no se carga completo. `stream_fil` usa `np.memmap`, `stream_fits` emite por subints.
- **Modelos cargados una sola vez** por proceso (`pipeline.py:1032-1033`, fuera de los bucles), con `.eval()` una vez y `torch.no_grad()` en ambos forwards.
- **Escritura de candidatos con buffer**, sin `to_csv` completo por candidato ni comportamiento cuadrático de I/O. (El defecto de durabilidad de P0-2 es ortogonal al diseño de rendimiento, que es correcto.)
- **Checkpoint por chunk con escritura atómica**, de coste despreciable.

### Priorización

Si solo se toca una cosa: **`force_plots: false`**. Elimina entre el 80 y el 95 % del tiempo de pared a escala de terabytes y es un cambio de una línea en `config.yaml`. Después, por orden de retorno: PERF-04 (GC y batching), PERF-03 y PERF-05 (copias), PERF-02 (solo si se usa la ruta torch-GPU), PERF-07 y PERF-08.

**Advertencia de secuencia:** la optimización va después de la corrección. Optimizar el pipeline antes de arreglar P0-1 solo produce resultados incorrectos más rápido.

---

## 15. Testing

### Ejecución real

Con el entorno del repositorio tal cual, la suite **no corre**: el Python del PATH es 3.14.7 y faltan `psutil`, `torch`, `astropy`, `scipy`, `cv2`, `matplotlib`, `seaborn`.

```
E   ModuleNotFoundError: No module named 'psutil'   (src/preprocessing/slice_len_calculator.py:11)
!!!!!!!!!!!!!!!!!!! Interrupted: 5 errors during collection !!!!!!!!!!!!!!!!!!!
```

Creando un entorno virtual fuera del repositorio con las dependencias mínimas:

```
136 passed, 2 warnings in 10.68s
```

Verde, rápido y sin inestabilidad. Pero **la cobertura total es del 18 %**: 8.900 sentencias, 7.338 sin ejecutar.

Fallo adicional reproducido: con `-s` en consola Windows cp1252, la suite entera muere en la fase de recolección por un `UnicodeEncodeError` en `tests/test_dm_chunking_threshold.py:66`.

### Qué está realmente probado

| Área crítica | ¿Probada? | Evidencia | Riesgo de regresión |
|---|---|---|---|
| Dedispersión CPU | **Sí, bien** | `test_dedispersion_parity.py:152-170` llama al kernel real | Bajo |
| Cubo DM ventaneado frente a directo | **Sí** (caso degenerado) | `test_cube_windowed_parity.py:69-85`, pero con un solo trozo DM | Bajo para 1 trozo, **alto** para varios (P1-03) |
| Mapeo caja CNN a DM | **Sí, bien** | `test_mut_fast.py:366-444` — pero con `img_height=512`, que es lo que el pipeline **no** hace (P0-1) | **Crítico** |
| Constante K_DM única | **Sí** | `test_spec_constants.py:36-60`, con enforcement por escaneo de `src/` | Bajo |
| `config.yaml` | Sí | `test_config_yaml.py` | Bajo |
| Dedispersión GPU | **No** (solo AST) | `test_dedispersion_parity.py:281-355` inspecciona el árbol sintáctico | Alto |
| SNR | **Débil** | 2 aserciones sueltas; 53 % de cobertura; `compute_presto_matched_snr` y `compute_detection_significance` sin tocar | Alto |
| Downsampling | **No** | `test_downsampler.py` solo importa `unittest` y `numpy`: valida su propia referencia contra sí misma | Alto |
| Corte de slices | **No** | `plan_slices_for_chunk` no lo importa ningún test; 26 % de cobertura | Alto |
| Chunking y solape | **No** | 5 de 6 tests recalculan las fórmulas en línea | Alto |
| Lectura FITS | **No** | 3 % de cobertura sobre 917 sentencias | **Crítico** |
| Lectura filterbank | **No** | 8 % de cobertura | **Crítico** |
| Escritura CSV | **No** | `ensure_csv_header`, `append_candidate`, `CandidateWriter`, `to_row` sin test | **Crítico** |
| Pipeline HF | **No** | 6 % de cobertura sobre 730 sentencias | **Crítico** |
| Pipeline LF | **Casi no** | 13 % | **Crítico** |
| Checkpoint / reanudación | **No** | **0 % de cobertura**, módulo entero sin ejecutar | **Crítico** |
| MJD baricéntrico | **No** | 23 % | Alto |

### Defectos de la propia suite

- **`test_downsampler.py`** — el docstring dice que cualquier optimización debe coincidir con esta referencia, pero el módulo nunca importa `downsample_data`. Tautológico al 100 %: una reescritura del kernel pasaría verde.
- **`test_dm_chunking_threshold.py`** — recolecta **0 tests**. Es un script de `print` a nivel de módulo que reimplementa `calculate_dm_chunk_height` en línea con el comentario *"Fórmula REAL del código"* sin importarla nunca. Además tumba la suite entera bajo `-s` en Windows.
- **`test_large_file_processing.py`** — 5 de 6 tests son aritmética propia. El de streaming lo admite en el código: `overlap_samples = expected_overlap_samples  # In real pipeline this comes from calculator`, y luego verifica continuidad sobre su propia álgebra, de modo que `diff == 0` es cierto por construcción. `test_dm_chunking_activation` envuelve todas sus aserciones en un `if`: si la condición fuera falsa, pasa con cero aserciones.
- **Fuga de estado entre tests** — `test_large_file_processing.py:100-104` escribe `SLICE_LEN_MIN`, `MAX_CHUNK_SAMPLES` y `MAX_RAM_FRACTION`; su `finally` solo restaura 4 claves y el fixture `_isolate_config` no las incluye. Sonda ejecutada al final de la sesión: las tres quedan modificadas. Hoy no rompe nada, pero es dependencia de orden latente.
- **8 métodos duplicados en `test_mut_fast.py`** — la segunda definición sombrea la primera, así que pytest solo ejecuta la última. Los cuerpos son idénticos, de modo que no se pierde cobertura, pero **editar la primera copia no tiene ningún efecto**.
- **`test_hf_phases.py`** — reimplementa `_dm_smear_samples` localmente; el check SPEC-HF-002 real vive en línea dentro de `high_freq_pipeline.py:1417-1428` y no es invocable. `test_boundary_exactly_one_sample` solo afirma que 0.99 < 1.0 y 1.01 >= 1.0.
- **`test_candidate_finalizer.py`** — los 6 tests verifican longitud de tupla, rango y finitud; ningún valor esperado.

### Mutation testing

| Módulo | Mutantes | Completados | Supervivientes |
|---|---|---|---|
| `dm_candidate_extractor` | 173 | 173 | 2 |
| `utils` | 205 | 205 | 5 |
| `pipeline_parameters` | 530 | **372** | 35 |
| `science_metrics` | 479 | **372** | 34 |

Dos de las cuatro corridas están **incompletas**: 158 y 107 mutantes nunca se ejecutaron, así que la tasa de kill publicada es sobre población parcial y 76 supervivientes es una cota inferior.

De esos 76, **15 son equivalentes demostrables**: operan sobre anotaciones PEP-604 que `from __future__ import annotations` convierte en cadena. Los huecos reales señalan:

- `science_metrics.py:26` — `dm_step_for_smearing` solo se afirma por signo y monotonía, nunca por valor exacto.
- `science_metrics.py:34-39` (6 supervivientes) — `estimate_dm_uncertainty` se prueba con la rejilla uniforme `[0,1,2,3,4]`, donde varias fórmulas erróneas dan el mismo número. Hace falta una rejilla no uniforme.
- `science_metrics.py:45-50` — solo se afirma una desigualdad sobre `post_trials_sigma`, sin valor esperado.
- `pipeline_parameters.py:58-59` — la frontera exacta `ratio == collapse_ratio` de la decisión HF/LF no se prueba.

### Tests faltantes, priorizados

**P0 — bloqueantes**

1. `test_centernet_box_frame` — una caja en coordenadas 512 sobre un cubo con dimensiones distintas debe producir el DM y el tiempo correctos. Cierra P0-1. Coste bajo, es una función pura.
2. `test_hf_candidates_reach_csv` — N candidatos por la ruta HF con N no múltiplo de 50; el CSV debe contener N filas. Cierra P0-2.
3. `test_resume_equals_full_run` — CSV de una corrida completa idéntico al de una corrida interrumpida y reanudada. Cierra P0-3 y P0-4.
4. `test_csv_header_and_row_roundtrip` — longitud del header igual a la de la fila, y relectura con pandas de los valores escritos.
5. `test_checkpoint_save_load_clear` — ciclo completo y reanudación desde el chunk N; checkpoint corrupto no revienta el pipeline. Módulo hoy al 0 %.

**P1 — alto valor**

6. `test_downsample_data_matches_reference` — arregla el test tautológico importando producción. Coste trivial.
7. `test_trim_valid_window_respects_right_overlap` — cierra P1-01.
8. `test_overlap_covers_max_dispersion_delay` — test de propiedad: para cualquier combinación de DM_max, banda y resolución, el solape calculado por producción debe ser mayor o igual al retardo dispersivo máximo. Sustituye la aritmética tautológica actual.
9. `test_plan_slices_covers_chunk_without_gaps` — test de propiedad sobre cobertura sin huecos ni solapes no declarados.
10. `test_chunk_continuity` — `end_sample[n] == start_sample[n+1]` para toda la secuencia, en las tres rutas de lectura. Cierra P1-04, P1-05 y P1-06 de una vez.
11. `test_filterbank_header_roundtrip` — orden de frecuencias, `nchans`, `nifs`, `tstart`, `nbits` desde un `.fil` sintético.
12. `test_fits_polarization_selection` — `_select_polarization` para AABB y Stokes con `npol` 1, 2 y 4. Función pura, no requiere archivo.
13. `test_e2e_injected_frb_recovered` — extremo a extremo: `inject_synthetic_frb` (que ya existe en `snr_utils.py:305` y nadie usa) hasta el CSV.

**P2 — cierra huecos de mutación:** rejilla no uniforme para `estimate_dm_uncertainty`; valor exacto de `post_trials_sigma` y de `dm_step_for_smearing`; frontera exacta de la decisión HF/LF; contrato de la rama `coarse_to_fine`, que hoy cae a `smear_limited` como placeholder sin ningún test que lo fije.

**P3 — higiene de CI:** `ruff check --select F821,F811,F601` (cero ruido, solo errores reales); gate de cobertura con la línea base actual para impedir regresión; `windows-latest` en la matriz (habría detectado el fallo Unicode); completar las dos corridas de mutación truncadas antes de citar tasas de kill.

### Fragilidad: buena noticia

Búsqueda ejecutada: **no hay** `sleep`, red, `datetime.now`, rutas absolutas del desarrollador ni dependencia de GPU en `tests/`. Todo el RNG usa semilla explícita. El único acoplamiento de orden es la fuga de configuración descrita arriba. La suite corre en 10,7 segundos.

---

## 16. Observabilidad

### Lo que existe

- ETA y throughput por chunk (`pipeline.py:781-800`).
- Avisos de I/O lento (`:733-745`).
- Perfil de hardware al arranque (`hardware_profile.py:100-180`).
- Métricas de validación en JSON (`validation_metrics.py:402`).
- Checkpoint reanudable.
- Logging con colores y niveles, y un logger global bien encapsulado.

### Lo que falta o está roto

| Problema | Ubicación |
|---|---|
| El nivel de log no es configurable sin editar código (P2-31) | `pipeline.py:959` |
| Sin rotación de logs, y dentro del árbol de código (P2-32) | `logging_config.py:128-132` |
| Los logs se pierden con `docker compose run --rm` (P2-33) | `docker-compose.yml:49,110` |
| Métricas de memoria falsas en el JSON (P2-15) | `validation_metrics.py:63,66` |
| `validation_status` ignora los chunks fallidos (P2-13) | `validation_metrics.py:393` |
| Sin telemetría de GPU durante la corrida; solo lecturas puntuales de VRAM | `slice_len_calculator.py:392-397` |
| Sin heartbeat legible por otro proceso; el checkpoint solo guarda el último chunk | `checkpoint.py` |
| `tqdm` es dependencia declarada pero solo se usa en `src/training/` | — |
| No queda registro de qué pipeline (LF o HF) se usó por archivo (P2-23) | — |
| Logging de diagnóstico a nivel INFO dentro del bucle por candidato, y un bloque de 39 líneas de "CONFIGURATION SUMMARY" emitido por banda y por slice | `high_freq_pipeline.py:79-84,141-179` |
| `logger.warning` usado para flujo normal, lo que erosiona la señal de warning | `high_freq_pipeline.py:270,767,776-777` |
| f-strings evaluadas siempre en el camino caliente, aunque el handler las descarte | varios |

### Lo mínimo que haría falta para operar con confianza

1. Registrar en el CSV o en el JSON por archivo: pipeline usado, `bary_status`, número de chunks fallidos y sus índices.
2. Hacer que `validation_status` refleje el estado real.
3. Rotación de logs y destino bajo `results_dir`.
4. Nivel de log configurable.

---

## 17. Dependencias

**Estado general: lo más limpio del repositorio.** `requirements.txt` (19), `requirements.lock.txt` (87 con hashes) y `pyproject.toml` (19) son 100 % consistentes entre sí: cero paquetes faltantes, cero conflictos de versión, todos los pins dentro de los rangos declarados. El lockfile documenta su comando de regeneración en la cabecera.

**El problema no está en los manifiestos, sino en que el Dockerfile los ignora** (P1-21, P1-22).

| Dependencia | Tipo real | Uso | Acción |
|---|---|---|---|
| `numpy`, `astropy`, `psutil`, `pyyaml`, `numba`, `torch`, `torchvision`, `matplotlib` | Runtime | Todo el core | Mantener |
| `opencv-python` (cv2) | Runtime marginal | **Un solo import**: `visualization_unified.py:11`. `centernet_utils.py:2` lo importa sin usar | Evaluar si es sustituible; paquete pesado por un uso |
| `seaborn` | Runtime marginal | `visualization_unified.py:14`; los otros 2 imports no la usan | Verificar el uso real |
| `jplephem` | **Runtime indirecto** | Cero imports directos; la requiere astropy en `mjd_utils.py:98`. El propio código lo documenta | **Mantener.** No eliminar pese a los cero imports |
| `pandas` | Solo scripts | 15 imports, todos en `src/scripts/` | Mover a extra `analysis` |
| `scipy` | Scripts y entrenamiento | Un uso real en `binary_data.py:5`; los otros dos imports no la usan | Mover a extras |
| `scikit-learn`, `scikit-image`, `timm`, `tqdm` | Solo entrenamiento | `src/training/` | Mover a extra `training` |
| `blimpy` | Solo scripts | Un import en `MJD.py:10`, con try/except | Mover a extra `analysis` |
| `your` | Runtime opcional | `fits_handler.py:21`; **obligatorio de facto para el pipeline HF** (P2-25) | Mantener, y documentar que HF la exige |
| `fitsio` | Opcional, no declarada | 3 imports, los tres en try/except con fallback a astropy | Correcto: documentada en `requirements.txt:9` |
| **`sigpyproc`** | **Usada y NO declarada** | `time_series_with_polarization_dos.py:10,12,13,15,16` | Declarar o eliminar el script |
| `rich` | Transitiva, importada sin usar | Mismo script | Se resuelve borrando el script |
| `pytest`, `pytest-cov` | Dev | Correcto | — |
| `cosmic-ray` | Testing, sin pin | `ci.yml:44` | Pinear |
| `mutmut` | **Configurada y no instalada** | `pyproject.toml:55` | Eliminar el bloque |

**Nota:** no se recomienda actualizar ninguna dependencia solo por existir una versión nueva. La única actualización con justificación de seguridad es la de torch dentro del Dockerfile, y su motivo es alinearla con el lock, no la novedad.

---

## 18. Configuración

### Trazabilidad: `config.yaml`

**Positivo: las 36 claves hoja de `config.yaml` se cargan y se consumen.** Ninguna está muerta. La validación es desigual: 14 tienen test en `test_config_yaml.py`, 8 solo coerción de tipo, y 7 ninguna (`max_dm_smearing_ms`, `widths_ms`, `trial_correction`, `dm_policy`, `polarization.mode`, y la sección `performance` completa).

### Configuración muerta

| Archivo | Claves hoja | Estado |
|---|---|---|
| `advanced-config/performance.yaml` | 23 | **1 llega a producción** (`memory.overhead_factor`). `gpu.enable_mixed_precision`, `io.enable_async_io` y `parallel.cpu_threads` se cargan y nadie las consume; las 19 restantes ni se cargan |
| `advanced-config/logging.yaml` | ~70 | **100 % muertas** |
| `advanced-config/models.yaml` | ~54 | **100 % muertas** |
| `advanced-config/visualization.yaml` | ~45 | **100 % muertas**, y una de ellas contradice el valor real (P2-27) |

`user_config.py:32-45` mete los tres últimos en el diccionario bajo `logging_advanced`, `models_advanced` y `visualization_advanced`, y **ningún punto del código vuelve a leer esas claves**: el único consumo posterior es `performance_advanced` en `:150`.

### Constantes con cero consumidores

`config.py:181` `DM_RANGE_ADAPTIVE` · `:190-193` los cuatro `DM_PLOT_*` · `:201` `SNR_OFF_REGIONS` · `:204` `SNR_COLORMAP` · `:213` `SHADE_INVALID_TAIL` · `:222-226` `LOG_LEVEL`, `LOG_COLORS`, `LOG_FILE`, `GPU_VERBOSE`, `SHOW_PROGRESS` · `user_config.py:164,167,170` `ENABLE_MIXED_PRECISION`, `ENABLE_ASYNC_IO`, `CPU_THREADS`.

### Constantes consumidas pero no configurables

`DM_CUBE_MEMMAP_THRESHOLD_GB` (4.0), `MAX_SAMPLES_LIMIT` (10.000.000), `USE_PLANNED_CHUNKING` (True), `MAX_CHUNK_BYTES` (None), `SLICE_LEN` (512), los tres mínimos de `system_validator.py:18-20`, y `flush_interval` (50) en `candidate_manager.py:105`. Las tres últimas tienen impacto operativo directo y merecerían estar en el YAML.

### Hardcode que debería estar en YAML, y viceversa

Rutas y nombre de modelo (`config.py:123-127`) frente a `models.yaml:49-57`; selección de device (`:130-138`) frente a `models.yaml:67-71`; `get_band_configs()` (`:281-285`) frente a `visualization.yaml:106-121`; `dpi=300` repetido en 6 módulos de plotting —y `dpi=150` en uno, inconsistencia probablemente no intencional— frente a `visualization.yaml:131`; `min_chunk_samples = slice_len * 10` (`slice_len_calculator.py:442`) frente a `performance.yaml:50`.

### Patrón frágil

`config.py:186,187,205,206,212,214,215,216` usan `X = globals().get("X", default)` **después** de haber importado `X` en `:32-70`. El default nunca aplica: es un no-op que parece una decisión de diseño.

`config.py:32-70` y `:73-111` repiten la misma lista de 37 imports en un `try/except ImportError`: dos listas que hay que mantener sincronizadas a mano.

Hay **cuatro listas de claves mantenidas a mano**: `main.py:190-211` (20), `config.py:232-250` (40), `config.py:32-70` (35 imports) y los exports de `user_config.py` (~35).

### Rutas absolutas del desarrollador

| Ubicación | Contenido |
|---|---|
| `docker-compose.yml:40,101` | `D:/Your/Data/Path` — **impide arrancar tras un clone** |
| `src/scripts/combine_csv_summary.py:9,96` | `D:/Seba - Dev/TESIS/...` — y se **imprime al usuario** en el mensaje de uso |
| ~35 rutas en `src/scripts/` | `ResultsThesis/`, `Results-polarization*`; varias evaluadas a nivel de módulo, así que fallan al importar en cualquier otro checkout. `ResultsThesis/` está en `.gitignore`, de modo que nunca existe tras un clone |

**Verificado limpio:** `src/` fuera de `scripts/` y `tests/`, el CI, el Dockerfile, `config.yaml` y `advanced-config/` no contienen ninguna ruta absoluta de desarrollador.

---

## 19. Repository Hygiene

### Estado general

| Aspecto | Valoración |
|---|---|
| Marcadores TODO/FIXME/HACK | **Excelente.** Cero marcas reales. Las 11 coincidencias son falsos positivos (patrones de nombre de archivo) o documentación legítima (`dm_grid_mode: legacy_uniform`) |
| Wildcard imports | **Excelente.** Cero |
| `__pycache__` versionado | **Excelente.** Cero |
| Código comentado | **Excelente.** Solo 18 líneas en todo el repositorio |
| Consistencia de dependencias | **Excelente** entre los tres manifiestos |
| Imports sin usar | **Malo.** 116 bindings |
| Código muerto | **Malo.** ~58 símbolos, ~1.100 líneas |
| Configuración muerta | **Malo.** ~170 claves inertes |
| Artefactos versionados | **Regular.** 2,9 MB de `.sqlite` regenerables; 120 MB de pesos sin LFS |
| Documentación | **Regular.** `SPECS-physics.md` excelente; CHANGELOG obsoleto; README con dos errores |

### Residuo de una herramienta automática

Hay aproximadamente **2.400 líneas que son solo espacios en blanco** donde antes había comentarios, residuo de los commits de traducción `efd58d5` y `1480856`. Concentradas en `fits_handler.py` (228), `generate_alma_validation_report.py` (205), `plot_composite.py` (143) y `slice_len_calculator.py` (133). Son ruido cosmético; un pase de formateador las normaliza sin riesgo.

Nota: una de esas líneas en blanco está justo encima de `valid_end_ds = block_ds.shape[0]` en `trim_valid_window`, donde presumiblemente había una explicación. El borrado automático de comentarios eliminó la única pista de por qué el solape derecho no se resta (P1-01).

---

## 20. Dead Code Report

Evidencia: grep del símbolo sobre todo el corpus (`.py`, `.yaml`, `.toml`, `.md`, CI, Docker), **incluyendo literales de cadena** para detectar acceso dinámico. "1 hit" significa que solo aparece su propia definición. Se consideraron `getattr`, `importlib`, decoradores, registries y las fachadas `__getattr__` antes de marcar nada.

| Símbolo | Archivo:línea | Evidencia | Confianza | Acción |
|---|---|---|---|---|
| `set_gpu_verbose` | `logging/gpu_logging.py:21` | 3 hits: def + `__init__` + `__all__`. Cero llamadas | Confirmado | Eliminar |
| `gpu_context` | `logging/gpu_logging.py:26` | 3 hits | Confirmado | Eliminar |
| `log_gpu_operation` | `logging/gpu_logging.py:52` | 3 hits | Confirmado | Eliminar |
| `log_gpu_memory_operation` | `logging/gpu_logging.py:79` | 3 hits | Confirmado | Eliminar |
| `filter_cuda_messages` | `logging/gpu_logging.py:107` | 3 hits | Confirmado | Eliminar |
| `log_chunk_processing_start` | `logging/chunking_logging.py:73` | 3 hits | Confirmado | Eliminar |
| `log_chunk_processing_end` | `logging/chunking_logging.py:90` | 3 hits | Confirmado | Eliminar |
| `log_file_processing_summary` | `logging/chunking_logging.py:106` | 3 hits | Confirmado | Eliminar |
| `log_memory_optimization` | `logging/chunking_logging.py:123` | 3 hits | Confirmado | Eliminar |
| `log_slice_configuration` | `logging/chunking_logging.py:138` | 3 hits | Confirmado | Eliminar |
| `get_logger` | `logging/logging_config.py:453` | 3 hits; producción usa `get_global_logger` (16 hits) | Confirmado | Eliminar |
| `log_stream_fits_load_strategy` | `logging/data_loader_logging.py` | **Importado 2×** (`data_loader.py:32`, `fits_handler.py:29`), llamado 0× | Confirmado | Eliminar def y los 2 imports |
| `chunk_processing`, `slice_processing`, `debug_file_info` | `logging/logging_config.py:268,278,301` | 1 hit cada uno | Confirmado | Eliminar |
| `DM_RANGE_ADAPTIVE` | `config/config.py:181` | 1 hit | Confirmado | Eliminar |
| `DM_PLOT_MARGIN_FACTOR`, `DM_PLOT_MIN_RANGE`, `DM_PLOT_MAX_RANGE`, `DM_PLOT_DEFAULT_RANGE` | `config/config.py:190-193` | 1 hit cada uno | Confirmado | Eliminar |
| `SNR_OFF_REGIONS`, `SNR_COLORMAP` | `config/config.py:201,204` | 1 hit | Confirmado | Eliminar |
| `SHADE_INVALID_TAIL` | `config/config.py:213` | 1 hit; `visualization.yaml:68` define la clave y nadie la lee | Confirmado | Eliminar constante y clave |
| `LOG_LEVEL`, `LOG_COLORS`, `LOG_FILE`, `SHOW_PROGRESS` | `config/config.py:222-226` | 1 hit cada uno | Confirmado | Eliminar (o cablear, P2-31) |
| `ENABLE_MIXED_PRECISION` | `config/user_config.py:164` | 1 hit; lee `performance.yaml:90` | Confirmado | Eliminar constante y clave |
| `ENABLE_ASYNC_IO` | `config/user_config.py:167` | 1 hit; lee `performance.yaml:106` | Confirmado | Eliminar constante y clave |
| `CPU_THREADS` | `config/user_config.py:170` | 2 hits: def + mención en docstring de `hardware_profile.py:190` | Confirmado | **Cablear**, no eliminar: es una opción de usuario que no hace nada (P2-30) |
| `SUPPORTED_FORMATS` | `input/file_detector.py:13` | 1 hit; `SUPPORTED_EXTENSIONS` sí se usa | Confirmado | Eliminar |
| `estimate_sigma_iqr` | `analysis/snr_utils.py:108` | 1 hit | Confirmado | Eliminar |
| `compute_presto_matched_snr` | `analysis/snr_utils.py:248` | 2 hits: def + su propio mensaje de error | Confirmado | Eliminar |
| `inject_synthetic_frb` | `analysis/snr_utils.py:305` | 1 hit | Confirmado | **Mover a `tests/` como helper** — es exactamente lo que hace falta para el test E2E |
| `compute_detection_significance` | `analysis/snr_utils.py:354` | 1 hit | Confirmado | Eliminar |
| `PeakCandidateBox` | `core/high_freq_pipeline.py:31` | 1 hit; sin registry ni anotación | Confirmado | Eliminar |
| `polarization_fractions` | `input/polarization_utils.py:28` | 1 hit | Confirmado | Eliminar |
| `calculate_dispersion_bandwidth_delay` | `preprocessing/dedispersion.py:49` | 1 hit | Confirmado | Eliminar |
| `get_parameters_function`, `extract_parameters_for_target`, `validate_extracted_parameters` | `input/parameter_extractor.py:87,96,114` | 1 hit cada uno | Confirmado | Eliminar |
| `get_dynamic_dm_range_for_multiple_candidates` | `visualization/visualization_ranges.py:389` | 1 hit (el singular sí se usa 3×) | Confirmado | Eliminar |
| `calculate_undispersed_burst_time` | `visualization/plot_waterfall_dispersed.py:25` | 1 hit | Confirmado | Eliminar |
| `generate_individual_plots_from_composite_params` | `visualization/plot_individual_components.py:475` | 1 hit | Confirmado | Eliminar |
| `compute_simple_timeseries`, `normalize_simple`, `compute_off_pulse_stats` | `visualization/plot_polarization_timeseries.py:19,47,71` | 1 hit cada uno | Confirmado | Eliminar |
| `get_band_frequency_range`, `get_band_name_with_freq_range`, `_calculate_dynamic_dm_range` | `visualization/visualization_unified.py:215,234,36` | Definidas y nunca llamadas, ni dentro del propio archivo | Confirmado | Eliminar |
| `usable_ram_bytes`, `usable_vram_bytes` | `core/hardware_profile.py:76,80` | 1 hit cada uno | Confirmado | Eliminar |
| `record_buffer_event`, `record_buffer_limit` | `output/validation_metrics.py:254,283` | 1 hit cada uno | Confirmado | Eliminar |
| `reset` | `output/phase_metrics.py:172` | 1 hit | Confirmado | Eliminar |
| `_safe_int`, `_auto_config_downsampling`, `_print_debug_frequencies`, `_save_file_debug_info` | `input/data_loader.py:72-75` | 1 hit cada uno; el propio comentario dice "Legacy aliases" | Confirmado | Eliminar |
| `_config_injected` | `config/config.py:28,272` | Escrito, nunca leído | Confirmado | Eliminar |
| `_hardware_profile` | inyectado en `pipeline.py:967` | Nunca leído | Confirmado | Verificar y eliminar |
| `calculate_optimal_chunk_size` | `preprocessing/slice_len_calculator.py:353-543` | 190 líneas; solo se alcanza como fallback de excepción, y **lanza `NameError`** | Confirmado | Eliminar o reparar (P2-03) |

### Solo usado por tests — no vivo para producción

| Símbolo | Evidencia | Acción |
|---|---|---|
| `CandidateRecord` | Solo `tests/test_contracts.py` | Adoptar en `candidate_manager` o eliminar |
| `effective_time_reso` | Solo `tests/test_contracts.py:37` | Ídem |
| `src/domain/physics.py` completo | Solo 4 archivos de tests; producción importa directo de `analysis/science_metrics` | Decisión arquitectónica: adoptar o eliminar |
| `_allocate_dm_cube_buffer` | Solo `tests/test_cube_windowed_parity.py` | Helper interno legítimo; mantener |

### Probablemente muerto: fachadas nunca ejercitadas

`grep "from src.core import|from src.input import|from ..core import|from ..input import"` devuelve **cero resultados**: todo el código importa submódulos directamente.

| Elemento | Líneas | Acción |
|---|---|---|
| `__getattr__` + `__all__` (17 nombres) en `src/core/__init__.py` | 51 | Eliminar o adoptar |
| `__getattr__` + `__all__` (18 nombres) en `src/input/__init__.py` | 47 | Eliminar o adoptar |
| `src/visualization/__init__.py` | 24 | Eliminar. **Además tiene un defecto:** `plot_patches` está comentado en `:10` pero sigue listado en `__all__:20`, de modo que `from src.visualization import *` lanzaría `AttributeError` |

Contraste: `src/logging/__init__.py` sí tiene 5 consumidores reales y debe conservarse.

---

## 21. Dead Files Report

| Archivo | Líneas | Razón | Referencias | Última modificación | Confianza | Acción |
|---|---|---|---|---|---|---|
| `src/output/summary_manager.py` | 217 | Nunca importado; sus 3 funciones son privadas y tienen 1 hit | **0** | 2025-11-05 | Confirmado | Eliminar. Reemplazado por `execution_summary.py` |
| `src/logging/gpu_logging.py` | 142 | 5 de 5 funciones muertas | Solo el re-export | — | Confirmado | Eliminar módulo y entradas de `__init__.py` |
| `src/visualization/plot_patches.py` | 281 | Deshabilitado explícitamente en 2 sitios desde hace ~10 meses | Imports comentados en `plot_individual_components.py:20` y `__init__.py:10` | 2025-11-24 | Confirmado | Decidir: reactivar o eliminar. El estado "comentado en tres sitios" es el peor |
| `src/input/data_loader.py` | 80 | Shim de re-export puro; los 24 imports sin usar son re-exports sin `__all__` | 1 (`absolute_segment_plots.py:15`) | 2026-04-11 | Probable | Redirigir el consumidor y eliminar |
| `src/scripts/read_header.py` | 18 | Wrapper con import roto (`from MJD import ...` sin paquete); **verificado que falla** fuera de `cwd=src/scripts` | — | 2025-11-05 | Confirmado | Eliminar |
| `src/scripts/build_summary_matches.py` | 106 | Lista hardcodeada de 4 CSV que no existen; apunta a `scripts/bursts_mjd.csv`, ruta inexistente | — | 2025-09-07 (el más antiguo) | Confirmado | Eliminar |
| `src/scripts/analizar_canonicos_extras.py` | 230 | `sys.path.insert(0,'src/scripts')` relativo al cwd; **sin guard `__main__`, ejecuta al importarse**; falla con `FileNotFoundError` | — | 2025-12-12 | Probable | Reescribir o eliminar |
| `src/scripts/time_series_with_polarization_dos.py` | 481 | Sin `__main__`, sin docstring, 13 imports sin usar, import duplicado en las líneas 10 y 12, depende de `sigpyproc` no declarada. El sufijo "_dos" delata una copia | — | 2025-11-23 | Probable | Rescatar lo útil o eliminar |
| `src/tests/analisis_chunk_reduccion.py` | — | Script one-shot; **falla al ejecutarse** por `UnicodeEncodeError` en consola cp1252 | Solo `src/tests/README.md` | 2026-06-11 | Posible | Mover a `tools/` y arreglar el encoding |
| `src/tests/simulate_5_5tb_processing.py` | 469 | Simulación offline legítima; contiene además un `F821` (`dt_max`) | Solo `src/tests/README.md` | 2025-12-01 | Posible | Mover fuera de `src/tests/` |

### `src/scripts/` — clasificación de los 22 archivos

Ninguno está referenciado en README, CHANGELOG, SPECS, Dockerfile, CI ni `config.yaml`. No existe `src/scripts/README.md`.

| Clasificación | Archivos | Acción |
|---|---|---|
| **Herramienta de tesis legítima — conservar y documentar** | `analyze_case_fast_frex.py`, `analyze_case_b0355.py`, `analyze_case_frb121102.py`, `analyze_case_alma_psr1745.py`, `analyze_alma_validation.py`, `analyze_alma_phases_validation.py`, `generate_alma_validation_report.py`, `catalog_data_files.py`, `compare_validated_with_detections.py`, `combine_csv_summary.py` | Conservar. Documentar en un `README` del directorio |
| **Herramienta de diagnóstico legítima** | `MJD.py`, `fits_header_analyzer.py`, `absolute_segment_plots.py`, `test_slicing_alignment.py` | Conservar. **Renombrar** el último a `check_slicing_alignment.py`: el prefijo `test_` es engañoso y rompería si alguien ejecutara `pytest src/` |
| **One-shot ya consumido** | `generate_latex_tables.py`, `generate_complete_analysis.py`, `generate_detailed_canonical_table.py`, `transform_table_with_filenames.py` | Conservar solo si la tesis se regenera; si no, archivar |
| **Eliminar** | `read_header.py`, `build_summary_matches.py`, `analizar_canonicos_extras.py`, `time_series_with_polarization_dos.py` | Ver tabla anterior |

**No confundir investigación legítima con código muerto:** los diez primeros son el aparato de validación científica de la tesis. Están vivos aunque ningún código los importe.

---

## 22. Legacy Components

| Componente | Estado | Evidencia |
|---|---|---|
| `src/domain/` | **Migración incompleta.** SPEC-DM-001 afirma que todos los módulos del pipeline importan de aquí; ninguno lo hace | Solo tests |
| `contracts.py` (`ObservationMetadata`, `PipelineConfigSnapshot`, `ChunkPlan`) | **Migración incompleta.** Se construyen por chunk y se usan casi solo para logging; el camino caliente sigue leyendo el global | `pipeline.py:205-229` |
| `src/input/data_loader.py` | **Compatibilidad.** Shim de re-export con un único consumidor en `src/scripts/` | — |
| `src/output/summary_manager.py` | **Completamente obsoleto.** Sustituido por `execution_summary.py` | 0 referencias |
| `src/visualization/plot_patches.py` | **Completamente obsoleto o en pausa.** Deshabilitado hace 10 meses | Imports comentados |
| `calculate_optimal_chunk_size` | **Completamente obsoleto.** Duplica `calculate_memory_safe_chunk_size` y está roto | `slice_len_calculator.py:353` |
| `dm_grid_mode: legacy_uniform` | **Activo y deliberado.** Modo de reproducibilidad histórica | `config.yaml:79`, SPEC-DM-004 |
| `# Look in Summary/*/Validation/... (legacy structure)` en 3 scripts | **Activo por compatibilidad** con salidas antiguas | — |
| `[tool.mutmut]` | **Completamente obsoleto.** La herramienta real es cosmic-ray | `pyproject.toml:55` |

---

## 23. Duplicate Code

| ID | Ubicaciones | Tipo | Divergencia | Recomendación |
|---|---|---|---|---|
| D-01 | `pipeline.py:494-904` ↔ `high_freq_pipeline.py:1194-1625` | **Exacta, ~50 %** (172 de 326 líneas) | 4 divergencias no intencionadas, una de ellas P0 | P1-24. Es la duplicación más costosa del repositorio |
| D-02 | `fits_handler.py:807-1398` ↔ `:1402-1847` | Semántica (590 frente a 445 líneas) | Solo el backend de I/O (`your` frente a astropy). Calibración, bits, polarización, MJD y chunking son la misma lógica | Extraer `decode_row` y el ensamblador de chunks; dos lectores con el mismo contrato. **Riesgo alto**: es el archivo más frágil |
| D-03 | `fits_handler.py:925-945` ↔ `:1426-1446` | Estructural | Solo la fuente del header | Extraer `read_tstart(hdr, primary)` |
| D-04 | `detection_engine.py:569-593` ↔ `high_freq_pipeline.py:1152-1176` | Exacta parcial | **21 de 25 argumentos idénticos** | Parameter Object |
| D-05 | `detection_engine.py:111-142` ↔ `high_freq_pipeline.py:435-460` | Exacta (24 líneas; `:124-132` ↔ `:447-455` byte a byte) | Solo renombres de variable | Extraer `waterfall_snr_profile()` |
| D-06 | `pipeline.py:88-93`, `detection_engine.py:605-614`, `high_freq_pipeline.py:1189-1191` y `:1597-1604` | Semántica ×4 | Ninguna: misma regla `SAVE_ONLY_BURST` | Usar `DetectionStats.effective_counts`, que ya existe |
| D-07 | `pipeline.py:860-904` | **Exacta ×5** | Solo el literal `status` | `_error_result(status, e, t_start)` |
| D-08 | `slice_len_calculator.py:262-300` ↔ `:486-524` | Estructural | **La copia perdió `overlap_decimated` → `NameError`** | P2-03 |
| D-09 | 7 sitios en `src/visualization/` | **Exacta** | Ninguna. `plot_composite.py:1253` lo admite en un comentario | Extraer `normalize_block()`. Si cambia un percentil, los paneles dejan de ser comparables entre sí |
| D-10 | `plot_composite.py:202-225` y 5 módulos más | **Exacta (MD5 idéntico ×5)** | Ninguna | `_common.py`; y consumir `config.get_band_configs()`, que ya existe |
| D-11 | `plot_composite.py:151-200`, `plot_dm_time.py:24-73`, `visualization_unified.py:36-84` | **Exacta** | Una palabra del docstring | Colapsar en `visualization_ranges.py` |
| D-12 | 9 sitios con `nanpercentile(x, 1/99)`; 6 con `savefig(dpi=300, bbox_inches="tight")` | Exacta | `plot_polarization_timeseries.py:454` usa **dpi=150** | Extraer `draw_waterfall()` y `save_figure()` |
| D-13 | `analyze_alma_phases_validation.py:124-240` ↔ `compare_validated_with_detections.py:242-365` | **Exacta (97 de 100 líneas)** | **Una línea**: la lista de columnas excluidas | Extraer con parámetro `exclude_cols` |
| D-14 | `analyze_case_*.py` — `find_validation_json`, `load_validation_metrics`, bloque argparse | **Exacta (MD5 idéntico en los 4)** | Solo la descripción y un default | `_case_common.py`. **No fusionar los scripts**: solo comparten el 19 %; el 81 % restante es ciencia distinta por caso |
| D-15 | `analyze_alma_validation.py:240` frente a `compare_validated_with_detections.py:59` | **Semántica peligrosa** | Mismo nombre `normalize_filename`, normalizaciones **inversas** (`_`→`-` frente a lo contrario). Ambas producen claves comparadas con `==` | Renombrar o unificar. Esto es correctitud, no estética |
| D-16 | `mjd_utils.py:44` frente a `MJD.py:216,303` | Semántica | `MJD.py` reimplementa el cálculo completo, con `K_DM = 4.148808e3` **local** en vez de importar el canónico | Importar de `domain.physics` |
| D-17 | `mjd_utils.py:49,50,54` ↔ `:200,201,204` | **Exacta, en el mismo archivo** | RA, DEC y 1400.0 MHz escritos dos veces | Constantes de módulo (y ver P1-07) |
| D-18 | `data_flow_manager.py:530-546` frente a 11 sitios más | Configuración | Los fallbacks **omiten `<stem>/chunk<NNN>`** → dos layouts de salida según el camino de ejecución | Parameter Object `OutputPaths` |
| D-19 | `system_validator.py:145` (`/ 4.15`) frente a `science_metrics.py:9` | **Semántica: constante física distinta** | Escapa al regex de `test_spec_constants.py:21` porque es `4.15` sin `e3` | Importar `K_DM_MS` y **ampliar el regex del test** |
| D-20 | `time_series_with_polarization_dos.py:41` (`4.1488*10**3`) y `:102` (`1/2.41e-4`) | Semántica | **Dos valores de K_DM distintos en el mismo archivo** (4148.8 frente a 4149.38) | Se resuelve eliminando el script |
| D-21 | `config.py:32-70` ↔ `:73-111` | **Exacta** | La misma lista de 37 imports, dos veces | Un `importlib` de tres líneas |

**Cifras:** aproximadamente 900 líneas duplicadas o muertas en `src/visualization/` (19 % del subsistema), 205 entre los dos pipelines, 285 entre los `analyze_case_*` y 190 entre los scripts de validación ALMA.

**Nota sobre D-19 y D-20:** son las únicas duplicaciones con impacto científico directo. El test `test_spec_constants.py` ya existe para prohibirlas y se le escapan por el patrón del regex y por la exclusión explícita de `src/scripts/`.

---

## 24. Inefficient Code

Cubierto en detalle en la sección 14. Resumen de los patrones, no de las instancias:

| Patrón | Instancias principales |
|---|---|
| Copias de arrays innecesarias | `dedispersion.py:256,481`; `filterbank_handler.py:508-518`; `fits_handler.py:1018-1023` |
| `astype` redundante sobre un array que ya tiene el dtype correcto | `dedispersion.py:481`; `filterbank_handler.py:516` |
| Bucles Python sobre ejes grandes, vectorizables | `dedispersion.py:456-459,553,638`; `high_freq_pipeline.py:43-48`; `data_downsampler.py:89` |
| Recálculo de lo mismo varias veces por slice | `compute_snr_profile` al menos 4 veces en la cadena de plotting |
| Sincronizaciones GPU dentro del bucle | `model_interface.py:81` (`.item()` por candidato) |
| Inferencia con batch 1 | `model_interface.py:38,78` |
| GC forzado por slice | `pipeline.py:428` |
| `empty_cache()` periódico que devuelve segmentos al driver | `pipeline.py:133`; `dedispersion.py:484` |
| Trabajo ejecutado antes del gate que decide si hace falta | `detection_engine.py:76` (`postprocess_img` antes del gate de plots) |
| Estructuras que crecen sin cota | `validation_metrics.py:51` |
| Detección de hardware repetida por archivo | `slice_len_calculator.py:181-182` |
| Bucle O(H·C·W) donde basta O(C+W) | `dedispersion.py:176-179` |
| Convolución directa donde basta suma acumulada | `snr_utils.py:95-102` |

---

## 25. Overengineering

| Hallazgo | Evidencia | Por qué sobra |
|---|---|---|
| **Contratos construidos en el camino caliente y usados solo para logging** | `pipeline.py:205-229`. `ObservationMetadata.from_config` convierte el array `FREQ` completo a una tupla de floats de Python **en cada chunk** (`contracts.py:55`) para acabar en un `logger.debug`. `pipeline_snap` no se usa en `_process_block` | Coste sin retorno, e ilusión de desacoplamiento: quien lea esas líneas concluirá que el camino caliente ya está desacoplado del global, y no lo está |
| **Capa `src/domain/` passthrough** | 15 líneas totales; cero consumidores de producción | Un paquete llamado `domain` que sugiere una capa de dominio inexistente, más un test que la vigila |
| **Inyección de dependencias aparente** | `detection_engine.py:45,415` recibe `config` como parámetro; el único llamador pasa el módulo global (`pipeline.py:404`). Su gemelo HF lo importa directamente | 14 parámetros posicionales, uno de ellos inútil, y dos estilos de acoplamiento distintos para la misma dependencia |
| **Fachadas `__getattr__` perezosas sin consumidores** | `core/__init__.py:23-50`, `input/__init__.py:24-46`: 98 líneas | Indirección que hay que mantener sincronizada; una función renombrada deja el `__getattr__` mintiendo en silencio |
| **Capa de reenvío pura** | `visualization_unified.py:238-304`: 67 líneas que reenvían 33 argumentos sin una sola línea de lógica | No es una abstracción, es un alias con coste de mantenimiento |
| **`globals().get()` como falso default** | `config.py:186,187,205,206,212,214,215,216` | El import de `:32-70` ya puso el nombre en el espacio de nombres; el default es inalcanzable |
| **Lista de imports duplicada literalmente** | `config.py:32-70` y `:73-111` | Dos listas de 37 nombres a sincronizar a mano |
| **Configuración fantasma** | 579 líneas de YAML cargadas y nunca leídas | Peor que no tener configuración: el usuario cree que configuró algo |
| **Parámetro muerto propagado por 11 archivos** | `off_regions`, documentado en `snr_utils.py:29-30` como *"kept for compatibility but is unused"*, viaja por 53 referencias | Un parámetro que nunca cambia nada sigue obligando a decidir qué pasarle en cada llamada |
| **Parámetro recibido e ignorado** | `high_freq_pipeline.py:1199` (`streaming_func`) | Firma que miente, y además rompe filterbanks (P2-24) |
| **Bandera booleana que codifica un modo** | `_skip_standard_panels` gobierna 15 ramas en `plot_composite.py:794-1067` | Es `if modo_HF` escrito 15 veces en negativo |
| **Rama de abstracción sin llamadores** | `visualization_ranges.py:132-197,389-457`: 134 líneas, un 30 % del archivo | Nunca se invocó |
| **Dos toggles de verbosidad GPU, ninguno operativo** | `config.py:225` y `gpu_logging.py:19` con `set_gpu_verbose()` exportada y nunca llamada | — |
| **`if/else` degenerado** | `data_flow_manager.py:163-180`: ambas ramas llaman a `_build_dm_time_cube_chunked`, solo cambia el umbral | Colapsa a una línea |

**Complejidad accidental total estimada:** alrededor de 1.100 líneas de código y configuración que existen sin aportar comportamiento.

**Importante:** la complejidad **esencial** del proyecto es alta y está justificada. La estrategia de tres niveles del cubo DM (directo → troceado en DM → troceado temporal → memmap), los fallbacks en cadena y el presupuesto adaptativo de memoria responden a restricciones físicas reales. No confundirlos con overengineering.

---

## 26. Technical Debt

| Categoría | Volumen | Coste de servicio |
|---|---|---|
| Código muerto | ~1.100 líneas, ~58 símbolos, 3 módulos | Bajo: eliminar con la suite en verde |
| Configuración muerta | ~170 claves, 579 líneas de YAML | Bajo, pero exige decidir si hay roadmap |
| Duplicación LF/HF | ~205 líneas, con 4 divergencias ya materializadas | **Alto**: requiere test de CSV byte-idéntico previo |
| Duplicación en visualización | ~900 líneas (19 % del subsistema) | Medio: requiere test de regresión visual |
| Estado global mutable | 49 sitios de mutación, 31 módulos lectores | **Muy alto**: incremental, con tests de paridad |
| Imports sin usar | 116 bindings | Trivial: `ruff --fix F401` |
| Líneas en blanco residuales | ~2.400 | Trivial: pase de formateador |
| Ciclos de imports | 2 ciclos, 12 imports tardíos que los ocultan | Medio: mover 2 archivos |
| Cobertura de tests | 18 %; módulos críticos al 0-8 % | **Alto**, pero es la inversión de mayor retorno |
| Deriva documental | 6 afirmaciones falsas entre README, CHANGELOG y SPECS | Trivial |
| Artefactos versionados | 2,9 MB regenerables, 120 MB de pesos | Bajo para los `.sqlite`; alto y **no recomendado** para los pesos |

### Métricas de código

| Métrica | Valor |
|---|---|
| Líneas Python en `src/` | 29.042 (99 archivos) |
| Funciones con complejidad ciclomática > 12 | 40 |
| Violaciones de "más de 5 argumentos" | 59 |
| Violaciones de "más de 50 sentencias" | 52 |
| Errores de clase pyflakes | 316 (134 imports muertos, 34 variables muertas, **2 `F821` undefined-name**) |
| Parámetros sin anotar | 324 |
| `except Exception` amplios | 136 (26 seguidos de `pass`) |
| Gate de lint en CI | **ninguno** |

**Las funciones más complejas:**

| Función | Ubicación | CC | Sentencias | Parámetros |
|---|---|---|---|---|
| `create_composite_plot` | `plot_composite.py:228` | **102** | — | 31 |
| `stream_fits` | `fits_handler.py:751` | **88** | 482 | 3 |
| `snr_detect_and_classify_candidates_in_band` | `high_freq_pipeline.py:101` | **78** | 426 | 23 |
| `get_obparams` | `fits_handler.py:224` | **70** | 280 | 1 |
| `_process_file_chunked` | `pipeline.py:494` | 33 | 171 | 5 |
| `_process_file_chunked_high_freq` | `high_freq_pipeline.py:1194` | 32 | 208 | 5 |
| `detect_and_classify_candidates_in_band` | `detection_engine.py:32` | 29 | 170 | **22** |

Estas cifras son señales, no veredictos. La complejidad de `stream_fits` refleja en parte la complejidad real del formato PSRFITS; la de `create_composite_plot` no refleja nada esencial.

---

## 27. Refactoring Opportunities

| ID | Área | Problema | Smell | Principio | Patrón | Beneficio | Riesgo |
|---|---|---|---|---|---|---|---|
| REF-01 | `core/` LF+HF | Orquestador duplicado al 50 %, con 4 divergencias ya materializadas | Duplicación exacta, Shotgun Surgery | DRY, OCP | Template Method **funcional** (driver + callback) | HF gana checkpoint, flush y límite de chunk sin escribir código nuevo | **Alto** |
| REF-02 | `core/pipeline.py:658-698` | El dispatch LF/HF ocurre tras 190 líneas de setup, que HF repite | Feature Envy, bandera de control de flujo | SRP | Ninguno necesario: subir el `if` a `run_pipeline` | Elimina trabajo duplicado; corrige que un `.fil` llegue a un lector FITS | Medio |
| REF-03 | `input/fits_handler.py:751-1851` | `stream_fits` contiene dos implementaciones completas del mismo lector dentro de un `try` de 1.046 líneas | Long Method, duplicación semántica | SRP, OCP | Adapter + Template Method | El chunking —lo que más riesgo de off-by-one tiene— queda en una función pura y testeable | **Alto** |
| REF-04 | `visualization/plot_composite.py:228-1095` | 870 líneas, CC 102, 31 parámetros, 15 usos de una bandera que codifica un modo | God Function, Long Parameter List, Data Clumps | SRP, CQS | Parameter Object + composición de paneles | CC 102 → ~15; paneles testeables por separado | **Alto** |
| REF-05 | `core/high_freq_pipeline.py:101-984` | Función de 884 líneas con 40 locales; usa `'x' in locals()` para comprobar su propio scope | God Function | SRP | Ninguno: extraer 4 funciones puras | Las fases se vuelven testeables aisladas | **Alto** |
| REF-06 | `config/` | Umbrales y flags con más de una fuente de verdad | Múltiple source of truth | SSOT | Ninguno necesario | Elimina irreproducibilidad por default oculto | **Bajo** |
| REF-07 | `advanced-config/` | 579 líneas cargadas y nunca leídas | Configuración fantasma | SSOT | Ninguno: borrar o cablear | Deja de mentir al usuario | **Bajo** |
| REF-08 | `output/candidate_manager.py` | `CANDIDATE_HEADER` y `to_row()` son dos listas paralelas a mano, sin ningún test que las compare | Primitive Obsession, contrato implícito | SSOT | Dataclass + serializador derivado | Impide la desalineación silenciosa de columnas | **Bajo** |
| REF-09 | `core/` + `preprocessing/` | Ciclos C1 y C2 ocultos tras 12 imports tardíos | Dependencia circular | Acyclic Dependencies | Ninguno: mover 2 archivos | El grafo queda acíclico sin mover lógica científica | Medio |
| REF-10 | `core/contracts.py` + consumidores | Los contratos existen, se construyen y no se usan | Temporary Fields, estado global | DIP | Parameter Object (ya escrito) | Testabilidad sin monkeypatch de módulo | **Alto** |
| REF-11 | `visualization/` (7 sitios) | La receta de normalización copiada 7 veces | Duplicación exacta | DRY | Extract Function | Si cambia un percentil, los paneles siguen siendo comparables | **Bajo** |
| REF-12 | `analysis/snr_utils.py` + 11 archivos | `off_regions` documentado como no usado, propagado por 53 referencias | Parámetro muerto | YAGNI | Ninguno | Un parámetro menos en firmas de 22 | **Bajo** |
| REF-13 | `visualization_unified.py` | 140 de 305 líneas son reenvío puro o copias sin llamadores | Wrapper passthrough | YAGNI | Ninguno | Un salto menos en cada traza | **Bajo** |
| REF-14 | `scripts/` validación ALMA | `find_matches` copiada salvo una línea; arrastra 2 funciones más | Duplicación exacta | DRY | Extract + parámetro | −97 líneas | **Bajo** |
| REF-15 | `scripts/` | Dos `normalize_filename` **incompatibles** con el mismo nombre | Colisión semántica | SSOT | Renombrar | Evita cruce silencioso de claves | **Bajo** |
| REF-16 | `pipeline.py:860-904` | 5 bloques `except` con el mismo dict de retorno | Duplicación exacta | DRY | Extract Function | Un solo sitio para el esquema de error | **Bajo** |
| REF-17 | Rutas de salida | 11 sitios reconstruyen rutas; los fallbacks omiten un nivel | Duplicación de configuración | SSOT | Parameter Object `OutputPaths` | Layout determinista | Medio |
| REF-18 | `scripts/analyze_case_*.py` | ~285 líneas compartidas (19 %) | Duplicación exacta | DRY | `_case_common.py` | −285 líneas. **No fusionar los scripts** | **Bajo** |
| REF-19 | Global | ~1.100 líneas de código muerto | Dead Code | — | Ninguno | Reduce la superficie de todos los demás refactors | **Bajo** |
| REF-20 | `src/logging/` | Sombrea el módulo estándar | — | — | Ninguno: renombrar | Elimina una clase entera de fallos | **Bajo** |

### Fichas de los refactors estructurales

#### REF-01 — Driver de archivo unificado

**Problema actual:** `pipeline.py:494-904` y `high_freq_pipeline.py:1194-1625`, 172 de 326 líneas byte-idénticas. La divergencia real es una sola: cómo se detectan candidatos en un slice.

**Arquitectura actual:**

```
_process_file_chunked (411 lineas)          _process_file_chunked_high_freq (432)
  +- presupuesto de memoria  --------------- presupuesto de memoria      ESPEJADO
  +- decision de chunking    --------------- decision de chunking        ESPEJADO
  +- Summary/ + ensure_csv_header ---------- idem                        ESPEJADO
  +- calculo de overlap      --------------- calculo de overlap          ESPEJADO
  +- bucle: _process_block                   bucle: slices HF            DIFERENTE
  |    + checkpoint                          |   (sin checkpoint)
  |    + limite MAX_CHUNK_SAMPLES            |   (sin limite)
  +- export de metricas      --------------- export de metricas          ESPEJADO
  +- flush_all()                             (sin flush)                 DIVERGENTE
  +- finalize_file_status + dict ----------- idem                        ESPEJADO
```

**Arquitectura propuesta:**

```
run_file_chunks(path, save_dir, chunk_samples, stream_iter, process_chunk, *,
                base_status, use_checkpoint=True) -> dict
  +- presupuesto, chunking, dirs, overlap        (una vez)
  +- bucle con checkpoint, errores, ETA, flush   (una vez)
  |    +- process_chunk(payload)   <- unica parte variable
  +- finalize_file_status + dict                 (una vez)

pipeline.py           -> run_file_chunks(..., process_chunk=_process_block)
high_freq_pipeline.py -> run_file_chunks(..., process_chunk=_hf_chunk)
```

**Sin herencia ni jerarquía de clases.** El repositorio no tiene una sola clase con comportamiento fuera de `CandidateWriter` y los trackers de métricas; introducir `BasePipeline` obligaría a convertir 40 funciones libres en métodos y a mover el estado local a `self`.

**Tests requeridos:** test de CSV byte-idéntico sobre un archivo de referencia, en ambos modos, **antes de empezar**.

**Pasos incrementales**, cada uno verde antes del siguiente:

1. Extraer `_resolve_chunk_size()` y sustituir **las dos** copias, que ya son idénticas. Sin cambio de comportamiento.
2. Extraer `_resolve_overlap()`.
3. Extraer `_log_chunk_eta()`.
4. Extraer `_build_result()`.
5. Colapsar las 4 copias de la regla `SAVE_ONLY_BURST` en `DetectionStats.effective_counts`, que ya existe.
6. Solo entonces, unificar el bucle con el callback.

**Trade-off:** el driver debe absorber dos formas de stream (tupla de 2 frente a tupla de 4). La homogeneización del contrato de `metadata` es prerrequisito: `stream_fil` emite 13 claves, `stream_fits` emite 17 o 20 según cuál de sus 5 puntos de `yield` dispare.

#### REF-09 — Romper los ciclos moviendo dos archivos

**Problema:** `core` importa `visualization` y `visualization` importa `core`; `preprocessing` y `output` cierran el ciclo con imports tardíos.

**Causa raíz:** dos módulos de cálculo puro viven en el paquete de orquestación.

```
ANTES                                  DESPUES
core/                                  domain/
  pipeline.py                            physics.py          (ya existe)
  high_freq_pipeline.py                  pipeline_parameters.py   <- movido
  pipeline_parameters.py  <- puro        mjd_utils.py             <- movido
  mjd_utils.py            <- puro      core/
  ...                                    pipeline.py
                                         high_freq_pipeline.py
                                         ...
visualization -> core   (ciclo)        visualization -> domain  (unidireccional)
output -> core          (ciclo)        output -> domain         (unidireccional)
preprocessing -> core   (ciclo)        preprocessing -> domain  (unidireccional)
```

Esto además da contenido real a `src/domain/`, que hoy es un passthrough de 15 líneas, y de paso resuelve la contradicción de SPEC-DM-001.

**Trade-off:** movimiento de archivos con actualización de unos 20 imports. Riesgo bajo, pero ensucia el `git blame` y no aporta valor funcional inmediato. **Priorizar después de los P0 y de REF-01.**

**Lo que NO hay que hacer:** introducir puertos, adaptadores, casos de uso o interfaces. El sistema tiene un solo adaptador real —archivos astronómicos en disco— y un solo consumidor. La ceremonia no se pagaría.

---

## 28. Design Pattern Map

Patrones realmente presentes, identificados por estructura y no por nombre.

| Patrón | Ubicación | Uso real | Evaluación |
|---|---|---|---|
| **Factory Method** | `input/streaming_orchestrator.py:15-29` | `get_streaming_function()` valida y devuelve `stream_fits` o `stream_fil` | **Parcialmente aplicado.** Correcto en sí, pero el pipeline HF descarta su resultado (P2-24) |
| **Chain of Responsibility** | `preprocessing/dedispersion.py:302-378` | Torch-GPU → Numba-CUDA → Numba-CPU → NumPy, cada eslabón con `try/except` y degradación | **Correctamente aplicado** como cadena informal. Formalizarla destruiría la semántica de fallback |
| **Proxy** | `core/data_flow_manager.py:33-50` | Devuelve `np.memmap` o `np.zeros` según el tamaño | **Correctamente aplicado.** NumPy ya da la sustituibilidad |
| **Facade** | `logging/__init__.py:11-45` | Re-export de 20 símbolos, con 5 consumidores reales | **Correctamente aplicado** |
| **Facade (perezosa)** | `core/__init__.py:23-50`, `input/__init__.py:24-46` | `__getattr__` PEP 562 | **Innecesaria.** Cero importadores del paquete |
| **Facade (reenvío puro)** | `visualization_unified.py:238-304` | Reenvía 27 kwargs sin lógica | **Innecesaria** |
| **Adapter** | `plot_individual_components.py:475-496` | Desempaqueta un dict a 28 kwargs | Correctamente aplicado |
| **Adapter (peligroso)** | `detection/model_interface.py:71-77` | Si no hay modelo o torch, `classify_patch` devuelve un sigmoide sobre SNR con la misma firma | **Incorrectamente aplicado.** Falsifica probabilidades de clasificación sin señalarlo al llamador |
| **Adapter (invertido)** | `domain/physics.py:1-6` | Re-export de `K_DM_MS` | **Innecesario / invertido.** La capa "dominio" depende de `analysis`, no al revés, y nadie la usa |
| **Registry / Multiton** | `output/candidate_manager.py:103-124` | `_instances: dict[Path, CandidateWriter]` con `get()` y `flush_all()` | **Parcialmente aplicado.** El registro funciona; el ciclo de vida está roto (P0-2) |
| **Registry (tabla)** | `config/config.py:275-285` | `get_band_configs()` | Correctamente aplicado, **pero reimplementado como `if/elif band_idx` en 6 módulos de visualización** |
| **Singleton** | `logging/logging_config.py` | Logger global | Correctamente aplicado |
| **Singleton** | `visualization_ranges.py:327` frente a `:433` | Instancia de módulo… y una segunda creada en `:433` | **Parcialmente aplicado** (inconsistente, aunque la clase no tiene estado) |
| **DTO / Value Object** | `core/hardware_profile.py:20-98` | `HardwareProfile` frozen, 16 campos y propiedades derivadas | **Correctamente aplicado. El mejor ejemplo del repositorio** |
| **DTO** | `core/contracts.py:23-184` | 5 dataclasses frozen | **Parcialmente aplicado.** Bien diseñados, construidos, y casi no usados |
| **Template Method** | — | El esqueleto existe **duplicado** en 3 sitios: LF/HF a nivel de archivo, LF/HF a nivel de slice, y 4 plotters con el mismo esqueleto de 12 pasos | **Ausente pero útil**, en forma funcional (REF-01, REF-04) |
| **Strategy** | — | Lo que hay es dispatch por cadena en funciones puras de 2 a 4 ramas | **Ausente y NO necesario** |
| **Builder, Abstract Factory, Decorator, Command, Observer, State, Mediator, Composite, Bridge, Prototype** | — | — | **Ausentes y NO necesarios** (ver sección de patrones a evitar) |

---

## 29. Architecture Pattern Map

| Patrón | Evidencia | Evaluación |
|---|---|---|
| **Pipes-and-Filters** | Generadores de streaming → downsample → cubo DM → slices → detección | **Parcialmente aplicado.** La topología es correcta, pero los filtros no son componibles: leen y escriben el `config` global |
| **Batch sequential** | `run_pipeline` → target → archivo → chunk → slice → banda | **Correctamente aplicado** |
| **Store-and-forward / Checkpointing** | `core/checkpoint.py` con escritura atómica | **Parcialmente aplicado.** Bien implementado, pero el HF no lo usa y tiene los defectos P0-3, P0-4 y P1-16 |
| **DTO** | `contracts.py` | **Parcialmente aplicado** |
| **Layered** | `domain → analysis` (invertido); ciclo `pipeline ↔ high_freq_pipeline` | **Violado**, aunque el sistema tampoco pretende ser por capas |
| **Ambient Context / Service Locator implícito** | `config` como módulo mutable | **Presente e incorrecto** (P1-20) |
| **Repository** | — | **Ausente y NO necesario.** No hay segunda fuente de persistencia, ni consultas, ni transacciones |
| **Unit of Work, API Gateway, Circuit Breaker, Service Discovery, Load Balancing, SSO, Access Tokens** | — | **Ausentes y NO aplicables.** No hay servicios, ni red, ni autenticación |
| **Event-Driven, CQRS, Microservices, Hexagonal, Clean** | — | **Ausentes y NO deben introducirse** |

### Estilo arquitectónico real

**Monolito batch, un proceso, pipes-and-filters imperativo.** No es Layered, ni Hexagonal, ni Clean, ni Event-Driven, pese a que la presencia de `contracts.py` y `domain/` sugiera una intención en esa dirección.

Y **no debería serlo**: la restricción dominante es memoria por chunk sobre archivos de terabytes, no throughput entre servicios. La arquitectura actual —leer, procesar, escribir, liberar— responde directamente a esa restricción y está explícitamente presupuestada.

### Patrones que NO deben introducirse aquí

| Patrón | Dónde se estaría tentado | Coste y por qué no |
|---|---|---|
| **State** | Las fases 1, 2, 3a y 3b del HF | No hay máquina de estados: es una secuencia fija activada por flags de configuración, sin transiciones dirigidas por eventos ni estado que sobreviva entre candidatos. Un `PhaseState` con `next()` convertiría 4 `if` en 4 clases más un contexto (~200 líneas), y el defecto real (P1-10) **seguiría ahí**, porque su causa es el centinela, no el control de flujo |
| **Strategy con clases** para los backends de dedispersión | `dedispersion.py:302-378` | No es una selección, es una cadena de degradación: cada eslabón cae al siguiente *cuando falla en runtime*. Una estrategia elegida por adelantado no puede expresar "intenta y si revienta, sigue". Coste: 4 clases y una factory, más reintroducir el `try/except` en el cliente |
| **Abstract Factory / jerarquía `Reader`** | `input/` | Dos formatos, dos funciones. Si llega un tercero, una tabla de datos lo resuelve en una línea sin herencia. Coste de la versión OO: ~150 líneas y un nivel de indirección en el camino caliente de I/O |
| **Repository completo** para candidatos | `output/candidate_manager.py` | Sin segunda fuente de persistencia ni consultas, `ICandidateRepository` más implementación más inyección por 5 niveles son ~120 líneas para envolver `append_candidate`. El problema real —que header y fila se desalineen— se arregla con **un test de una línea** |
| **Observer / event bus** para métricas | `collector` y `metrics_tracker` | Hay exactamente dos observadores y cero necesidad de suscripción dinámica. El problema real es que son `Optional = None` con comprobación en 12 sitios: se arregla con un objeto nulo de 10 líneas |
| **Command** para el CLI | `main.py:216-297` | `argparse` ya es la capa de comandos. Lo que sobra es la repetición del patrón `if args.x is not None: ...`, que una tabla de 20 filas elimina sin patrón |
| **Herencia `BasePipeline` → `LFPipeline` / `HFPipeline`** | REF-01 | Es la lectura "de libro" del Template Method y la trampa. Obligaría a convertir 40 funciones libres en métodos y a resolver el ciclo que hoy se parchea con imports tardíos. La versión funcional da la misma reutilización sin tocar el estilo del repositorio |
| **`Protocol` / `ABC` para modelos** | `detection/model_interface.py` | `detect` y `classify_patch` no comparten contrato ni intención; un `Protocol` común sería una mentira de tipos |
| **Contenedor de inyección de dependencias** | El `config` global | Obligaría a enhebrar un objeto por una pila de 6 niveles. El beneficio se obtiene igual sustituyendo `getattr(config, K, default)` por `config.K` |
| **Microservicios, colas, workers distribuidos** | El pipeline completo | Añadiría serialización de arrays gigantes sin resolver la restricción real, que es memoria por chunk |

---

## 30. Dependency Graph Problems

```
Ciclo a nivel de paquete (Tarjan):
  { core, input, output, preprocessing, visualization }

Ciclos a nivel de modulo:
  core.pipeline            <-> core.high_freq_pipeline
  core.pipeline_parameters <-> preprocessing.slice_len_calculator
```

**Imports tardíos internos:** 48 en total, de los cuales unos 12 son inequívocamente rompe-ciclos. El resto son carga perezosa legítima de torch o matplotlib, y deben conservarse.

**Fan-in excesivo:** `config.py` con 31. Es el único problemático, porque es mutable. `science_metrics.py` y `snr_utils.py` con 12 cada uno son magnetos sanos: puros e inmutables, exactamente lo que debe tener fan-in alto.

**Fan-out excesivo:** `pipeline.py` con 23 y `high_freq_pipeline.py` con 18. Concentran toda la orquestación sin capa intermedia. Es consecuencia del estilo, no un defecto en sí; se reduce con REF-01.

**Fuga de API privada:** `high_freq_pipeline` importa `_optimize_memory` y `finalize_file_status`, que son símbolos privados de `pipeline`. Cualquier refactor de `pipeline.py` rompe HF en runtime, en el chunk N, no en tiempo de import.

**Imports frágiles en scripts:** `read_header.py:2` (`from MJD import ...` sin paquete, verificado que falla), `analizar_canonicos_extras.py:10` (`sys.path.insert` relativo al cwd), `fits_header_analyzer.py:82` (`from config import config`, que además dispara el sombreado de `logging`).

---

## 31. Cleanup Manifest

### Safe to Remove — evidencia concluyente

| Elemento | Tipo | Ubicación | Evidencia | Acción |
|---|---|---|---|---|
| `summary_manager.py` | Módulo (217 L) | `src/output/` | `grep "summary_manager"` → 0 hits en todo el repositorio | `git rm` |
| `gpu_logging.py` | Módulo (142 L) | `src/logging/` | 5 de 5 funciones con 3 hits, 0 llamadas | `git rm` + limpiar `__init__.py` |
| 5 funciones de `chunking_logging` | Funciones | `:73,90,106,123,138` | 3 hits cada una | Eliminar |
| `get_logger` | Función | `logging_config.py:453` | Producción usa `get_global_logger` | Eliminar |
| `log_stream_fits_load_strategy` + 2 imports | Función | 3 sitios | Importada 2×, llamada 0× | Eliminar |
| 15 constantes de configuración | Constantes | `config.py:181-226`, `user_config.py:164,167` | 1 hit cada una | Eliminar, junto con sus claves YAML |
| 4 funciones de `snr_utils` | Funciones | `:108,248,305,354` | 1 hit | Eliminar 3; **mover `inject_synthetic_frb` a `tests/`** |
| `PeakCandidateBox` | Dataclass | `high_freq_pipeline.py:31` | 1 hit, contexto verificado | Eliminar |
| 3 funciones de `parameter_extractor` | Funciones | `:87,96,114` | 1 hit | Eliminar |
| 7 funciones de visualización | Funciones | 4 módulos | 1 hit cada una | Eliminar |
| 9 métodos y helpers varios | — | ver sección 20 | 1 hit cada uno | Eliminar |
| 4 alias "Legacy" | Alias | `data_loader.py:72-75` | El propio comentario lo dice | Eliminar |
| `[tool.mutmut]` | Config (7 L) | `pyproject.toml:55-62` | CI usa cosmic-ray; mutmut no instalado | Eliminar bloque |
| `read_header.py` | Script (18 L) | `src/scripts/` | Import roto verificado | `git rm` |
| `build_summary_matches.py` | Script (106 L) | `src/scripts/` | 4 rutas inexistentes | `git rm` |
| `mutation/*.sqlite` | Artefactos (2,9 MB) | `mutation/` | Regenerables con `cosmic-ray exec` | `git rm --cached` + `.gitignore` |
| ~116 imports sin usar | Imports | ver sección 19 | Verificado por AST | `ruff --fix F401`, **excepto** los 24 de `data_loader.py` |
| 18 líneas de código comentado | Comentarios | 5 sitios | — | Eliminar 3, convertir 2 en docstring |
| ~2.400 líneas de solo espacios | — | 4 archivos principalmente | Residuo de los commits de traducción | Pase de formateador |

### Requires Verification — evidencia fuerte, decisión del autor

| Elemento | Ubicación | Pregunta a responder |
|---|---|---|
| `plot_patches.py` (281 L) | `src/visualization/` | ¿Se pretende reactivar? Lleva 10 meses comentado en 3 sitios |
| `advanced-config/visualization.yaml`, `models.yaml`, `logging.yaml` (579 L) | — | ¿Roadmap o abandono? Hoy prometen configurabilidad inexistente |
| Secciones `garbage_collection` y `profiling` de `performance.yaml` | — | Nunca se leen |
| `data_loader.py` (80 L) | `src/input/` | Redirigir su único consumidor y eliminar |
| Fachadas `__getattr__` (98 L) | `core/`, `input/` | Adoptar o eliminar. **Verificar antes si hay notebooks fuera del repositorio que las usen** |
| `src/domain/` | — | Adoptar en producción (dándole contenido real, REF-09) o eliminar y corregir SPEC-DM-001 |
| `CandidateRecord`, `effective_time_reso` | `contracts.py` | Adoptar o eliminar |
| `CPU_THREADS` | `user_config.py:170` | **Probablemente un bug**, no código muerto: es una opción de usuario que no hace nada. Cablearla |
| `calculate_optimal_chunk_size` (190 L) | `slice_len_calculator.py` | Reparar el `NameError` o eliminar la función y su rama de fallback |
| `time_series_with_polarization_dos.py` (481 L) | `src/scripts/` | ¿Queda algo rescatable? |
| `analizar_canonicos_extras.py` (230 L) | `src/scripts/` | Reescribir o eliminar |
| Los 4 `generate_*.py` | `src/scripts/` | ¿Se regenerará la tesis? |

### Keep — parece extraño pero es necesario

| Elemento | Razón |
|---|---|
| `jplephem` | Dependencia runtime real vía astropy pese a cero imports directos |
| `src/input/utils.py` | Nombre genérico, contenido cohesivo, todo usado, módulo crítico en mutation testing |
| `SPECS-physics.md` | Actualizado y **ejecutado por tests** |
| `src/models/*.pth` | Requeridos por el pipeline; migrar a LFS sería mucho ruido y poco beneficio |
| Cadena `visualization_unified` → `plot_composite` → componentes | Arquitectura por capas correcta, no duplicación |
| `dm_grid_mode: legacy_uniform` | Nombre intencional para reproducibilidad histórica |
| `mutation/cr-*.toml`, `mutation/survivors.py` | Usados por CI |
| `conftest.py::_isolate_config` | Parche sobre P1-20. **No eliminar hasta que P1-20 esté resuelto**: hoy es lo único que hace la suite determinista |
| Los fallbacks graceful en cadena | Permiten correr en CI sin GPU y en HPC sin display; los tests dependen de ellos |
| `trim_valid_window` con `.copy()` explícito | El comentario explica que una vista mantendría vivo el cubo completo. Es correcto y contraintuitivo: convertirlo en vista reintroduce el OOM que previene |
| La estrategia de tres niveles del cubo DM | Cada rama corresponde a un régimen físico real y está cubierta por tests de paridad |
| La separación de los dos motores de detección por slice | Resuelven problemas científicos distintos. REF-01 unifica la orquestación, explícitamente **no** estos dos |

---

## 32. Proposed Repository Structure

Solo se propone mover lo que resuelve un problema **verificado**. Todo lo demás se deja explícitamente quieto.

```
CURRENT TREE                          PROBLEMAS                       TARGET TREE
-------------------------------------------------------------------------------
src/logging/                 sombrea el modulo estandar      src/log_utils/
                             (rotura reproducida)

src/tests/                   no son tests; el README raiz    tools/analysis/
  analisis_chunk_reduccion     manda ejecutarlos con pytest
  simulate_5_5tb_processing    -> 0 tests recolectados

src/scripts/  (22 archivos)  cajon de sastre: validacion,    tools/validation/
                             diagnostico y one-shots         tools/diagnostics/
                             mezclados; 0 documentados;      tools/thesis/
                             3 con imports rotos             (+ tools/README.md)

src/core/pipeline_parameters.py   calculo puro en el          src/domain/
src/core/mjd_utils.py             paquete de orquestacion;      physics.py
                                  cierra 2 ciclos               pipeline_parameters.py
                                                                mjd_utils.py

src/domain/  (passthrough)   15 lineas, 0 consumidores       (absorbido arriba)
```

### Justificación de cada movimiento

| Movimiento | Problema que resuelve | Riesgo | Coste |
|---|---|---|---|
| `src/logging/` → `src/log_utils/` | La única rotura reproducible del repositorio. `import pandas` falla con `src/` en el path | **Bajo** | ~30 min: `git mv` y actualizar 6 `from ..logging import` y 5 `from .logging_config import`. Los tests no tocan `src.logging` |
| Vaciar `src/tests/` | Ambigüedad de raíz entre dos directorios de tests, y un README que manda ejecutar el equivocado | **Bajo** | ~15 min, más corregir `README.md:228` |
| Segmentar `src/scripts/` | Cajón de sastre sin documentar | **Bajo** | ~1 h. **Precaución:** `absolute_segment_plots.py` usa imports relativos; al salir de `src/` hay que convertirlos a absolutos |
| Mover `pipeline_parameters.py` y `mjd_utils.py` a `domain/` | Cierra los dos ciclos de imports y da contenido real a `src/domain/` | Medio | ~2 h, con la suite verde antes y después |

### Lo que NO debe moverse

**`src/models/` con Git LFS.** Migrar 120 MB reescribe historia o añade un commit gigante, rompe los clones existentes, obliga a todos los colaboradores a instalar LFS, y el Dockerfile y el README documentan las rutas actuales en cuatro sitios. **Mucho ruido, poco beneficio** en un repositorio de tesis. Si el tamaño molesta, la vía barata es no volver a commitear pesos nuevos.

**`src/input/utils.py`.** Nombre genérico pero contenido cohesivo y todo usado.

**La estructura `src/core|input|output|preprocessing|analysis|detection`.** La física del pipeline se mapea bien a esas carpetas.

**La cadena de `src/visualization/`.** Es arquitectura por capas correcta.

---

## 33. Target Architecture

```
CURRENT ARCHITECTURE
  monolito batch, 2 orquestadores duplicados al 50 %,
  config global mutable como bus de datos,
  SCC de 5 paquetes, contratos escritos y no adoptados
            |
            v
IDENTIFIED PROBLEMS
  4 P0 (1 de correccion cientifica, 3 de durabilidad)
  24 P1, 38 P2, 31 P3
            |
            v
ROOT CAUSES
  S-1  "completado" no atado a salida durable
  S-2  orquestador duplicado
  S-3  sin invariante de marco de coordenadas
  S-4  configuracion sin fuente de verdad unica
  S-5  tests que verifican el caso degenerado
            |
            v
REFACTORINGS
  REF-01  driver de archivo unificado (funcional, no herencia)
  REF-09  mover 2 modulos puros a domain/ -> grafo aciclico
  REF-10  adoptar los contratos que ya existen -> reducir el global
  REF-06/07  una sola fuente de verdad de configuracion
  REF-19  eliminar ~1.100 lineas muertas
            |
            v
TARGET ARCHITECTURE
  monolito batch (mismo estilo, deliberadamente)
  1 driver de archivo + 2 estrategias de deteccion
  ObservationMetadata propagado por firma en el camino caliente
  config.yaml como unica fuente de verdad, de solo lectura tras el arranque
  grafo de paquetes aciclico: domain <- {core, input, output, preprocessing, visualization}
  barrera de durabilidad explicita por chunk
```

**El estilo arquitectónico no cambia.** No se propone Clean, ni Hexagonal, ni Event-Driven, ni microservicios. Se propone que el monolito batch que ya existe cumpla sus propias invariantes.

### Antes y después del camino caliente

```
ANTES
  _process_block(det, cls, block, metadata: dict, path, dir, chunk_idx, csv, collector)
    +- lee config.TIME_RESO, config.FREQ, config.DM_min, config.SLICE_LEN...  (global)
    +- construye ObservationMetadata, PipelineConfigSnapshot, ChunkPlan
    +- ...y los usa solo para logger.debug
    +- sigue leyendo config.* en el resto de la funcion

DESPUES
  _process_block(det, cls, block, plan: ChunkPlan, meta: ObservationMetadata,
                 snap: PipelineConfigSnapshot, grid: DMGrid, paths: OutputPaths)
    +- sin lecturas del global en el camino caliente
    +- ChunkPlan.__post_init__ valida invariantes que hoy nadie ejerce
    +- dos archivos pueden procesarse en procesos distintos sin contaminacion
```

Este cambio es de superficie grande y debe hacerse **módulo a módulo**, empezando por `data_flow_manager` (que ya tiene entradas y salidas explícitas), con los tests de paridad como red, y **después** de REF-01, que ya unifica ese código.

---

## 34. Incremental Migration Plan

No se propone "reorganizar todo `src/`". Cada paso es verificable de forma independiente y deja el repositorio en verde.

### Ejemplo detallado: eliminar el módulo muerto `gpu_logging.py`

```
Paso 1  Confirmar con la suite completa en verde y el entorno completo instalado.
Paso 2  grep del simbolo sobre TODO el corpus, incluidos strings y YAML.
Paso 3  Eliminar las 5 entradas de src/logging/__init__.py (imports y __all__).
Paso 4  Ejecutar la suite. Verde.
Paso 5  git rm src/logging/gpu_logging.py
Paso 6  Ejecutar la suite y un arranque real del pipeline sobre un archivo pequeno.
```

### Ejemplo detallado: unificar el driver de archivo (REF-01)

```
Paso 1  Escribir el test de CSV byte-identico sobre un archivo de referencia,
        en modo LF y en modo HF. Debe pasar con el codigo actual.
        SIN ESTE PASO, NO AVANZAR.
Paso 2  Extraer _resolve_chunk_size() y sustituir LAS DOS copias, que ya son
        identicas. Ejecutar el test: debe seguir byte-identico.
Paso 3  Igual con _resolve_overlap().
Paso 4  Igual con _log_chunk_eta().
Paso 5  Igual con _build_result().
Paso 6  Colapsar las 4 copias de la regla SAVE_ONLY_BURST en
        DetectionStats.effective_counts, que ya existe.
Paso 7  Homogeneizar el contrato de `metadata` entre stream_fil y stream_fits
        (13 claves frente a 17 o 20 segun el punto de yield).
Paso 8  Solo entonces: introducir run_file_chunks con callback y migrar LF.
Paso 9  Migrar HF. El checkpoint, el limite de chunk y el flush aparecen solos.
Paso 10 Ejecutar el test de CSV byte-identico y una corrida real completa.
```

### Ejemplo detallado: renombrar `src/logging/`

```
Paso 1  Suite en verde.
Paso 2  git mv src/logging src/log_utils
Paso 3  Actualizar los 6 `from ..logging import` y los 5 `from .logging_config import`.
Paso 4  Suite en verde.
Paso 5  Verificar la reproduccion del fallo original: con src/ en sys.path,
        `import pandas` ahora debe funcionar.
Paso 6  Arranque real del pipeline sobre un archivo pequeno.
```

---

## 35. Quick Wins

Cambios de una o pocas líneas, riesgo bajo, impacto inmediato.

| # | Cambio | Ubicación | Impacto |
|---|---|---|---|
| 1 | `img_height=512, img_width=512` | `detection_engine.py:147-148` | **Corrige P0-1.** El cambio de mayor impacto científico de toda la lista |
| 2 | `if chunk_idx <= resume_after:` | `pipeline.py:718` | **Corrige P0-3** |
| 3 | Mover `save_checkpoint` dentro del `try` | `pipeline.py:802` | **Corrige P0-4** |
| 4 | `atexit.register(CandidateWriter.flush_all)` + `finally` en ambos orquestadores | `candidate_manager.py`, `pipeline.py`, `high_freq_pipeline.py` | **Corrige P0-2** |
| 5 | `valid_end_ds = max(valid_start_ds, block_ds.shape[0] - overlap_right_ds)` | `data_flow_manager.py:463` | **Corrige P1-01** (candidatos duplicados) |
| 6 | `break` en vez de `continue`, y validar la longitud antes del `read` | `filterbank_handler.py:92,39` | **Corrige P2-01 y P2-02**, los dos únicos vectores con PoC |
| 7 | `force_plots: false` | `config.yaml:300` | Elimina el 80-95 % del tiempo de pared a escala TB |
| 8 | `matplotlib.use("Agg")` al arranque | `main.py` o `pipeline.py` | Evita intentos de abrir ventanas y fija el backend |
| 9 | `encoding="utf-8"` en las 3 aperturas del CSV | `candidate_manager.py:65,72,79,128` | **Corrige P2-09** |
| 10 | `assert len(CANDIDATE_HEADER) == len(Candidate(...).to_row())` | nuevo test | Cierra el riesgo de desalineación de columnas |
| 11 | `config.yaml:36,97` a `:ro` | `docker-compose.yml` | **Corrige P2-38** |
| 12 | `${DRAFTS_DATA_DIR:?}` en vez de `D:/Your/Data/Path` | `docker-compose.yml:40,101` | El proyecto arranca tras un clone |
| 13 | `permissions: {contents: read}` | `.github/workflows/ci.yml` | **Corrige P2-36** |
| 14 | `ruff check --select F821,F811,F601` en CI | `.github/workflows/ci.yml` | Cero ruido, solo errores reales. Habría detectado P2-03 |
| 15 | Corregir `pytest src/tests -q` → `pytest -q` y el badge de Python | `README.md:228,4` | El error más visible para un usuario nuevo |
| 16 | `mutation/*.sqlite` a `.gitignore` + `git rm --cached` | — | Libera 2,9 MB |
| 17 | Commitear `.mailmap` | — | Ya funciona localmente; beneficia al resto |
| 18 | Eliminar el bloque `[tool.mutmut]` | `pyproject.toml:55-62` | Configuración muerta que induce a error |
| 19 | `apply_thread_settings(hw, user_threads=config.CPU_THREADS)` | `pipeline.py:968` | **Corrige P2-30**: una opción de usuario que no hacía nada |
| 20 | Alinear `max_ram_fraction` con la tabla de comentarios, o recalcular la tabla | `config.yaml:225` | **Corrige P2-28** |

Los seis primeros cierran los cuatro P0 y los tres vectores de datos de terceros. Entre todos suman menos de 20 líneas de cambio.

---

## 36. Production Readiness Checklist

```
[ ] Correctness            BLOQUEANTE  -- P0-1 invalida los DM del pipeline LF
[ ] Security               parcial     -- sin secretos ni inyeccion; Docker divergente (P1-21),
                                          2 vectores DoS con PoC (P2-01, P2-02)
[ ] Data integrity         BLOQUEANTE  -- P0-2, P0-3, P0-4; no idempotente (P1-13);
                                          colision por stem (P1-14)
[ ] Error handling         parcial     -- 136 except amplios, 26 con pass, 0 reintentos (P1-19)
[x] Concurrency            OK          -- monohilo por diseno; kernels prange verificados seguros.
                                          No escalar por procesos sin resolver P1-14 y P1-20
[ ] Performance            parcial     -- force_plots domina el coste (P1-23); riesgo OOM de VRAM
                                          si se sube max_chunk_samples (PERF-02)
[ ] Testing                BLOQUEANTE  -- 18 % de cobertura; checkpoint al 0 %; lectores al 3-8 %;
                                          varios tests tautologicos
[ ] Observability          parcial     -- hay ETA y metricas; sin rotacion de logs, nivel no
                                          configurable, metricas de memoria falsas
[ ] Configuration          parcial     -- config.yaml integro; 579 lineas de YAML inertes;
                                          config.yaml se autocontradice
[x] Secrets                OK          -- ninguno en el repositorio ni en el historial
[ ] Deployment             BLOQUEANTE  -- no arranca tras un clone (docker-compose.yml:40)
[ ] Rollback               NO EXISTE   -- sin versionado ni retencion de resultados
[ ] Backup                 NO EXISTE   -- CSV en modo append, JSON sobrescrito
[x] Dependency health      OK          -- 3 manifiestos consistentes, lock con 1273 hashes.
                                          (El Dockerfile los ignora: eso es P1-21, no un problema
                                          de los manifiestos)
[ ] Repository hygiene     parcial     -- ~1.100 lineas muertas, ~170 claves inertes,
                                          artefactos versionados
```

### Blockers, en orden

1. **P0-1** — corrección científica del pipeline LF.
2. **P0-2, P0-3, P0-4** — durabilidad de los candidatos.
3. **Cobertura de tests** en los caminos que acaban de corregirse: sin ellos, los cuatro P0 pueden reaparecer.
4. **`docker-compose.yml:40,101`** — el proyecto no arranca tras un clone.
5. **P1-21** — el contenedor ejecuta un stack que CI nunca prueba.
6. **P2-32, P2-33** — los logs se pierden y no rotan, en corridas de días.

---

## 37. Prioritized Remediation Plan

### Fase 0 — P0 (bloqueante, antes de cualquier otra cosa)

| Prioridad | ID | Problema | Impacto | Esfuerzo | Riesgo del cambio | Acción |
|---|---|---|---|---|---|---|
| 1 | P0-1 | Marco de coordenadas de las cajas | Todos los DM del LF | 1 línea | Bajo | `img_height=512, img_width=512` |
| 2 | P0-2 | Buffer del CSV sin vaciar | Hasta 49 candidatos por archivo, o todos | ~10 líneas | Muy bajo | `finally` + `atexit` |
| 3 | P0-3 | Off-by-one al reanudar | 1 chunk por reanudación | 1 línea | Bajo | `<= resume_after` |
| 4 | P0-4 | Checkpoint tras chunk fallido | Impide la recuperación | 1 línea movida | Bajo | Mover dentro del `try` |
| 5 | — | Tests de regresión de los cuatro anteriores | — | ~1 día | Ninguno | Ver sección 15, lista P0 |

**Después de la Fase 0 es obligatorio reprocesar cualquier catálogo producido con el pipeline LF.**

### Fase 1 — Correctness (P1 de resultado incorrecto)

| Prioridad | ID | Problema | Esfuerzo | Riesgo |
|---|---|---|---|---|
| 6 | P1-01 | `trim_valid_window` ignora el solape derecho → candidatos duplicados | 1 línea | Bajo |
| 7 | P1-12 | `snr_pre_dedisp` sobre el eje DM | Pocas líneas | Bajo |
| 8 | P1-02 | Tres criterios de inversión de frecuencia | Medio | **Medio-alto**: exige verificar el comportamiento de `your` |
| 9 | P1-03 | Rejilla DM corrupta al trocear | Medio | Medio |
| 10 | P1-04, P1-05, P1-06 | Los tres defectos de `start_sample` y continuidad entre chunks | Medio | Medio |
| 11 | P1-11 | Desalineación de polarización en la rama SPEC-HF-002 | Pocas líneas | Bajo |
| 12 | P1-10 | Centinelas de `is_burst` | Medio | Bajo |
| 13 | P1-07, P1-08 | MJD baricéntrico: coordenadas cableadas y degradación silenciosa | Medio | Bajo |
| 14 | P1-09 | Contaminación de `TSTART_MJD_CORR` | Pocas líneas | Bajo |

### Fase 2 — Fiabilidad

| Prioridad | ID | Problema | Esfuerzo |
|---|---|---|---|
| 15 | P1-18 | Fuga de memmaps multi-GB | Bajo |
| 16 | P1-15 | Rutas de error que reportan 0 candidatos | Bajo |
| 17 | P1-13, P1-14 | Idempotencia y colisión de rutas de salida | Medio |
| 18 | P1-16, P1-17 | Checkpoint validado, y checkpoint en HF | Medio |
| 19 | P1-19 | Reintentos con backoff en las 3 operaciones de I/O | Medio |
| 20 | P2-01, P2-02 | Los dos vectores DoS de la cabecera `.fil` | 2 líneas |
| 21 | — | Tests de los caminos críticos hoy al 0-8 % | **Alto, y el de mayor retorno** |

### Fase 3 — Infraestructura y limpieza

| Prioridad | ID | Problema | Esfuerzo |
|---|---|---|---|
| 22 | — | `docker-compose.yml`: ruta de desarrollador y `:rw` | Bajo |
| 23 | P1-21, P1-22 | Dockerfile sobre el lockfile, Python 3.12, base por digest; job de build en CI | Medio |
| 24 | P2-32, P2-33 | Rotación de logs y persistencia fuera del contenedor | Bajo |
| 25 | P2-34, P2-35, P2-36 | CI: mutación ejecutable, hash-checking, permisos, lint | Bajo |
| 26 | REF-19, REF-20 | Eliminar ~1.100 líneas muertas; renombrar `src/logging/` | Bajo |
| 27 | P2-29, REF-07 | Decidir sobre `advanced-config/` | Bajo |
| 28 | P2-28 | Coherencia interna de `config.yaml` | Bajo |

### Fase 4 — Arquitectura

| Prioridad | ID | Problema | Esfuerzo | Precondición |
|---|---|---|---|---|
| 29 | REF-01 | Driver de archivo unificado (resuelve P1-17 y previene futuras divergencias) | Alto | **Test de CSV byte-idéntico** |
| 30 | REF-02 | Subir el dispatch LF/HF | Medio | REF-01 parcial |
| 31 | REF-09 | Mover 2 módulos puros a `domain/`, romper los ciclos | Medio | Suite verde |
| 32 | REF-05 | Descomponer la función de 884 líneas del HF | Alto | REF-01 |
| 33 | REF-10 | Adoptar los contratos, reducir el global | **Muy alto** | Incremental, con tests de paridad |
| 34 | REF-03 | Separar los dos lectores de `stream_fits` | Alto | Golden test de bytes |

### Fase 5 — Performance

Después de la corrección, nunca antes.

| Prioridad | ID | Esfuerzo |
|---|---|---|
| 35 | P1-23 (`force_plots`) | 1 línea |
| 36 | PERF-04 (GC por chunk, batching, `cudnn.benchmark`) | Medio |
| 37 | PERF-03, PERF-05 (copias de arrays) | Medio |
| 38 | PERF-02 (dedispersión torch vectorizada) | Alto |

### Fase 6 — Mantenibilidad

| Prioridad | ID | Esfuerzo |
|---|---|---|
| 39 | REF-11, REF-13, REF-16 (duplicación en visualización y errores) | Bajo |
| 40 | REF-04 (descomponer `create_composite_plot`) | Alto, con golden-image test previo |
| 41 | REF-14, REF-15, REF-18 (scripts) | Bajo |
| 42 | Reorganizar `src/scripts/` y `src/tests/` | Bajo |
| 43 | Actualizar CHANGELOG, README y las 2 afirmaciones falsas de SPECS | Trivial |

### Criterio de priorización aplicado

```
impacto x probabilidad x alcance
--------------------------------
      costo de correccion
```

Por eso P0-1 encabeza la lista pese a no ser una caída: impacto máximo (invalida resultados), probabilidad 1 (ocurre siempre), alcance total (todos los candidatos LF) y coste de corrección de una línea. Y por eso REF-10, que es arquitectónicamente el cambio más valioso, queda en la Fase 4: su coste de corrección es muy alto y su impacto es diferido.

---

## Nota final de método

**Verificación cruzada realizada:** los cuatro P0 y una muestra de los P1 se reprodujeron o se verificaron línea a línea de forma independiente, después de recibirlos de los subagentes. En particular se verificaron: la cadena completa del marco de coordenadas de P0-1 (cuatro archivos), la semántica de índices de P0-3 (tres archivos), la indentación exacta de P0-4, el mecanismo de buffer de P0-2, el sombreado de `logging` (reproducido con un intérprete real), y la firma de `trim_valid_window` de P1-01.

**Falsos positivos descartados explícitamente:** se recibieron y se descartaron, entre otros, los siguientes hallazgos de los subagentes: el "default divergente" `SNR_THRESH = 3.0` en `model_interface.py:75` (inalcanzable, porque `config.SNR_THRESH` siempre existe); `fitsio` como dependencia no declarada (es un import opcional deliberado con fallback documentado); numpy 1.24 frente a 2.4 como incompatibilidad dura (se verificó que el código no usa ninguna API eliminada en NumPy 2); la duplicación en la cadena de `src/visualization/` (es arquitectura por capas correcta); path traversal (el origen de las rutas es el sistema de archivos local, y se verificó que ningún valor de cabecera se usa para construir rutas); y la severidad P0 inicialmente atribuida a P1-10 y a P2-24.

**Lo que esta auditoría NO cubrió:** no se ejecutó el pipeline sobre datos reales, no se construyeron las imágenes Docker, no se ejecutó el job de mutación, y no se verificó empíricamente el comportamiento de `your.get_data` respecto al orden de canales, que es la incógnita central de P1-02.
