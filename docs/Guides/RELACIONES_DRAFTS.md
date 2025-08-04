# 🔗 Relaciones y Dependencias de la Carpeta `drafts/`

## 📊 **Diagrama de Arquitectura del Pipeline**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🚀 ENTRADA PRINCIPAL                              │
│                              main.py                                        │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        📋 PIPELINE PRINCIPAL                                │
│                           pipeline.py                                       │
│  • Carga modelos de detección y clasificación                              │
│  • Orquesta el procesamiento de archivos                                   │
│  • Maneja chunks y memoria                                                 │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ⚙️ CONFIGURACIÓN Y UTILIDADES                            │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   config.py │  │auto_slice_  │  │slice_len_   │  │summary_     │        │
│  │             │  │len.py       │  │utils.py     │  │utils.py     │        │
│  │• Parámetros │  │• Cálculo    │  │• Gestión    │  │• Reportes   │        │
│  │• Switches   │  │  automático │  │  dinámica   │  │• Logs       │        │
│  │• Modelos    │  │  de SLICE_  │  │  de SLICE_  │  │• Resúmenes  │        │
│  │• Rutas      │  │  LEN        │  │  LEN        │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        📁 ENTRADA/SALIDA (I/O)                              │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │     io.py   │  │filterbank_  │  │   io_utils  │  │candidate_   │        │
│  │             │  │io.py        │  │.py          │  │utils.py     │        │
│  │• Archivos   │  │• Archivos   │  │• Carga y    │  │• Gestión    │        │
│  │  FITS       │  │  FIL        │  │  preproces. │  │  de CSV     │        │
│  │• Metadatos  │  │• Metadatos  │  │• Datos      │  │• Headers    │        │
│  │• Parámetros │  │• Streaming  │  │  unificados │  │• Append     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🔄 PREPROCESAMIENTO                                      │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │preprocessing│  │dedispersion │  │dynamic_dm_  │  │astro_       │        │
│  │.py          │  │.py          │  │range.py     │  │conversions  │        │
│  │• Downsample │  │• Dedispersión│  │• Rangos DM  │  │.py          │        │
│  │• Normaliz.  │  │• GPU/CPU    │  │  dinámicos  │  │• Conversión │        │
│  │• Filtros    │  │• Patches    │  │• Zoom auto  │  │  pixel→DM   │        │
│  │             │  │• Bloques    │  │• Visualiz.  │  │• Tiempo     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🎯 DETECCIÓN Y CLASIFICACIÓN                             │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │pipeline_    │  │   utils.py  │  │  metrics.py │  │candidate.py │        │
│  │utils.py     │  │             │  │             │  │             │        │
│  │• Procesa    │  │• Detección  │  │• Cálculo    │  │• Estructura │        │
│  │  bandas     │  │  CenterNet  │  │  SNR        │  │  de datos   │        │
│  │• Procesa    │  │• Clasific.  │  │• Métricas   │  │• Candidatos │        │
│  │  slices     │  │  ResNet     │  │• Estadísticas│  │• CSV rows   │        │
│  │• Gestión    │  │• Patches    │  │             │  │             │        │
│  │  candidatos │  │             │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    📊 ANÁLISIS Y SNR                                        │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   snr_utils │  │consistency_ │  │             │  │             │        │
│  │   .py       │  │fixes.py     │  │             │  │             │        │
│  │• Cálculo    │  │• Gestor de  │  │             │  │             │        │
│  │  SNR        │  │  consistencia│  │             │  │             │        │
│  │• Perfiles   │  │• Unificación│  │             │  │             │        │
│  │• Picos      │  │  DM/SNR     │  │             │  │             │        │
│  │• Regiones   │  │• Reportes   │  │             │  │             │        │
│  │• Inyección  │  │• Debugging  │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🎨 VISUALIZACIÓN                                         │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │visualization│  │image_utils  │  │plot_manager │  │             │        │
│  │.py          │  │.py          │  │.py          │  │             │        │
│  │• Composite  │  │• Preproces. │  │• Orquesta   │  │             │        │
│  │  plots      │  │  imágenes   │  │  todos los  │  │             │        │
│  │• Waterfalls │  │• Postproces.│  │  plots      │  │             │        │
│  │• Patches    │  │• DM dinámico│  │• Gestión    │  │             │        │
│  │• SNR plots  │  │• Detección  │  │  de memoria │  │             │        │
│  │• Rangos     │  │  plots      │  │• Optimiz.   │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        📤 SALIDA DE RESULTADOS                              │
│                                                                             │
│  • Archivos CSV con candidatos detectados                                  │
│  • Imágenes de detección (composite, patches, waterfalls)                  │
│  • Reportes de consistencia y debugging                                    │
│  • Logs de procesamiento                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 **Flujo de Datos Detallado**

### **1. Inicialización (`pipeline.py`)**

```python
# Carga modelos
det_model = _load_detection_model()      # CenterNet
cls_model = _load_class_model()          # ResNet18

# Configuración
config.FREQ, config.TIME_RESO, etc.     # Desde archivo FITS/FIL
```

### **2. Procesamiento de Archivos**

```python
# Para cada archivo:
for fits_path in data_files:
    # Carga metadatos
    get_obparams(fits_path)              # io.py / filterbank_io.py

    # Carga datos
    data = load_and_preprocess_data()    # io_utils.py

    # Procesa en chunks
    _process_file_chunked()              # pipeline.py
```

### **3. Procesamiento de Chunks**

```python
# Para cada chunk:
for chunk_idx, (block, metadata) in enumerate(stream):
    # Dedispersión
    dm_time = d_dm_time_g(block)         # dedispersion.py

    # Procesa slices
    for j in range(time_slices):
        process_slice()                  # pipeline_utils.py
```

### **4. Procesamiento de Slices**

```python
# Para cada slice:
for band_idx, band_config in band_configs:
    # Procesa banda
    process_band()                       # pipeline_utils.py

    # Detección
    top_conf, top_boxes = detect()       # utils.py

    # Para cada candidato:
    for conf, box in zip(top_conf, top_boxes):
        # Conversión pixel→DM
        dm_val = pixel_to_physical()     # astro_conversions.py

        # Dedispersión del patch
        patch = dedisperse_patch()       # dedispersion.py

        # Clasificación
        class_prob = classify_patch()    # utils.py

        # Cálculo SNR
        snr_val = compute_snr()          # metrics.py / snr_utils.py

        # Guarda candidato
        append_candidate()               # candidate_utils.py
```

### **5. Visualización**

```python
# Para cada slice con candidatos:
save_all_plots()                        # plot_manager.py
├── save_slice_summary()                # visualization.py
│   ├── save_detection_plot()           # image_utils.py
│   ├── plot_waterfall_block()          # image_utils.py
│   └── save_patch_plot()               # visualization.py
└── save_plot()                         # visualization.py
```

## 📋 **Dependencias por Archivo**

### **Archivos Principales (Sin Dependencias Externas)**

- `config.py` - Configuración global
- `candidate.py` - Estructura de datos
- `__init__.py` - Inicialización del módulo

### **Archivos de I/O**

- `io.py` → `config.py`
- `filterbank_io.py` → `config.py`
- `io_utils.py` → `io.py`, `filterbank_io.py`
- `candidate_utils.py` → `candidate.py`

### **Archivos de Procesamiento**

- `preprocessing.py` → `config.py`
- `dedispersion.py` → `config.py`
- `astro_conversions.py` → `config.py`
- `dynamic_dm_range.py` → `config.py`

### **Archivos de Detección**

- `utils.py` → `config.py`
- `metrics.py` → `config.py`
- `pipeline_utils.py` → Múltiples módulos
- `snr_utils.py` → `config.py`

### **Archivos de Visualización**

- `image_utils.py` → `config.py`, `astro_conversions.py`, `snr_utils.py`
- `visualization.py` → Múltiples módulos
- `plot_manager.py` → `visualization.py`, `image_utils.py`

### **Archivos de Gestión**

- `auto_slice_len.py` → `config.py`
- `slice_len_utils.py` → `config.py`, `auto_slice_len.py`
- `summary_utils.py` → `config.py`
- `consistency_fixes.py` → `config.py`, `astro_conversions.py`, `snr_utils.py`

### **Archivo Principal**

- `pipeline.py` → Todos los módulos anteriores

## 🔗 **Relaciones Críticas**

### **1. Flujo de Datos Principal**

```
pipeline.py → pipeline_utils.py → utils.py → candidate_utils.py
     ↓              ↓              ↓              ↓
visualization.py → image_utils.py → plot_manager.py → Resultados
```

### **2. Gestión de Configuración**

```
config.py ← Todos los módulos
     ↓
auto_slice_len.py → slice_len_utils.py
```

### **3. Procesamiento de SNR**

```
snr_utils.py ← metrics.py
     ↓
consistency_fixes.py → pipeline_utils.py
```

### **4. Visualización**

```
visualization.py ← image_utils.py
     ↓
plot_manager.py → Todos los plots
```

## 🎯 **Puntos de Integración Clave**

### **1. Para el Gestor de Consistencia (`consistency_fixes.py`)**

- **Entrada**: `pipeline_utils.py` (línea ~30 en `process_band()`)
- **Salida**: `visualization.py` (línea ~243 en `save_slice_summary()`)
- **Beneficio**: Unificación de DM y SNR en todo el pipeline

### **2. Para Optimizaciones de Memoria**

- **Entrada**: `pipeline.py` (línea ~44 en `_optimize_memory()`)
- **Salida**: Todos los módulos de visualización
- **Beneficio**: Gestión eficiente de memoria en archivos grandes

### **3. Para SLICE_LEN Dinámico**

- **Entrada**: `auto_slice_len.py` → `slice_len_utils.py`
- **Salida**: `pipeline_utils.py` (línea ~30 en `get_pipeline_parameters()`)
- **Beneficio**: Optimización automática según características del archivo

## 📊 **Estadísticas del Pipeline**

- **Total de archivos**: 22 módulos
- **Archivos principales**: 4 (pipeline, config, utils, visualization)
- **Archivos de I/O**: 4 (io, filterbank_io, io_utils, candidate_utils)
- **Archivos de procesamiento**: 4 (preprocessing, dedispersion, astro_conversions, dynamic_dm_range)
- **Archivos de detección**: 4 (utils, metrics, pipeline_utils, snr_utils)
- **Archivos de visualización**: 3 (visualization, image_utils, plot_manager)
- **Archivos de gestión**: 5 (auto_slice_len, slice_len_utils, summary_utils, consistency_fixes, candidate)
- **Archivos de configuración**: 2 (config, **init**)

## 🚀 **Conclusión**

El pipeline `drafts/` está **bien estructurado** con una **separación clara de responsabilidades**:

- ✅ **Modularidad**: Cada módulo tiene una función específica
- ✅ **Escalabilidad**: Fácil agregar nuevas funcionalidades
- ✅ **Mantenibilidad**: Código organizado y documentado
- ✅ **Flexibilidad**: Configuración centralizada y dinámica

La integración del **gestor de consistencia** (`consistency_fixes.py`) resolverá las discrepancias identificadas y mejorará la **confiabilidad** del pipeline.
