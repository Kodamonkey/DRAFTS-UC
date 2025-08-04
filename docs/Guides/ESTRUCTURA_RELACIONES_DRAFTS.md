# 🗂️ Estructura de Carpetas Basada en Relaciones Reales

## 📊 **Estructura que Refleja el Flujo de Datos**

Basándome en el análisis de relaciones de `RELACIONES_DRAFTS.md`, propongo esta estructura que **refleja exactamente el flujo de datos**:

```
drafts/
├── __init__.py
├── config.py                    # ⚙️ Centro de configuración (todos dependen de él)
├── pipeline.py                  # 🚀 Pipeline principal (orquestador)
│
├── input/                       # 📁 ENTRADA/SALIDA (I/O)
│   ├── __init__.py
│   ├── io.py                    # Archivos FITS
│   ├── filterbank_io.py         # Archivos FIL
│   ├── io_utils.py              # Carga unificada de datos
│   ├── candidate.py             # Estructura de candidatos
│   └── candidate_utils.py       # Gestión de CSV
│
├── preprocessing/               # 🔄 PREPROCESAMIENTO
│   ├── __init__.py
│   ├── preprocessing.py         # Downsampling y normalización
│   ├── dedispersion.py          # Dedispersión GPU/CPU
│   ├── astro_conversions.py     # Conversiones pixel→DM
│   └── dynamic_dm_range.py      # Rangos DM dinámicos
│
├── detection/                   # 🎯 DETECCIÓN Y CLASIFICACIÓN
│   ├── __init__.py
│   ├── pipeline_utils.py        # Lógica principal de procesamiento
│   ├── utils.py                 # Detección CenterNet + Clasificación ResNet
│   └── metrics.py               # Cálculo de SNR básico
│
├── analysis/                    # 📊 ANÁLISIS Y SNR
│   ├── __init__.py
│   ├── snr_utils.py             # Análisis avanzado de SNR
│   └── consistency_fixes.py     # Gestor de consistencia DM/SNR
│
├── visualization/               # 🎨 VISUALIZACIÓN
│   ├── __init__.py
│   ├── visualization.py         # Plots principales (composite, waterfalls)
│   ├── image_utils.py           # Utilidades de imagen y detección plots
│   └── plot_manager.py          # Gestión de plots y memoria
│
└── system/                      # ⚙️ UTILIDADES DEL SISTEMA
    ├── __init__.py
    ├── auto_slice_len.py        # Cálculo automático de SLICE_LEN
    ├── slice_len_utils.py       # Gestión dinámica de SLICE_LEN
    └── summary_utils.py         # Reportes y logs
```

## 🔄 **Flujo de Datos Reflejado en la Estructura**

### **1. 🚀 Entrada Principal**

```
main.py → pipeline.py
```

### **2. 📁 Carga de Datos**

```
pipeline.py → input/
├── io.py (FITS)
├── filterbank_io.py (FIL)
├── io_utils.py (carga unificada)
└── candidate_utils.py (CSV)
```

### **3. 🔄 Preprocesamiento**

```
input/ → preprocessing/
├── preprocessing.py (downsampling)
├── dedispersion.py (dedispersión)
├── astro_conversions.py (pixel→DM)
└── dynamic_dm_range.py (rangos DM)
```

### **4. 🎯 Detección**

```
preprocessing/ → detection/
├── pipeline_utils.py (procesa bandas/slices)
├── utils.py (CenterNet + ResNet)
└── metrics.py (SNR básico)
```

### **5. 📊 Análisis**

```
detection/ → analysis/
├── snr_utils.py (SNR avanzado)
└── consistency_fixes.py (unificación DM/SNR)
```

### **6. 🎨 Visualización**

```
analysis/ → visualization/
├── visualization.py (composite plots)
├── image_utils.py (detection plots)
└── plot_manager.py (gestión plots)
```

### **7. ⚙️ Sistema**

```
system/ (utilidades globales)
├── auto_slice_len.py
├── slice_len_utils.py
└── summary_utils.py
```

## 📋 **Mapeo de Archivos por Responsabilidad**

### **📁 `input/` - Entrada/Salida**

- **Responsabilidad**: Carga de datos y gestión de candidatos
- **Flujo**: `pipeline.py` → `input/` → `preprocessing/`
- **Archivos**:
  - `io.py` → `config.py`
  - `filterbank_io.py` → `config.py`
  - `io_utils.py` → `io.py`, `filterbank_io.py`
  - `candidate.py` → (estructura de datos)
  - `candidate_utils.py` → `candidate.py`

### **🔄 `preprocessing/` - Preprocesamiento**

- **Responsabilidad**: Procesamiento de datos y conversiones
- **Flujo**: `input/` → `preprocessing/` → `detection/`
- **Archivos**:
  - `preprocessing.py` → `config.py`
  - `dedispersion.py` → `config.py`
  - `astro_conversions.py` → `config.py`
  - `dynamic_dm_range.py` → `config.py`

### **🎯 `detection/` - Detección y Clasificación**

- **Responsabilidad**: ML y procesamiento principal
- **Flujo**: `preprocessing/` → `detection/` → `analysis/`
- **Archivos**:
  - `pipeline_utils.py` → Múltiples módulos
  - `utils.py` → `config.py`
  - `metrics.py` → `config.py`

### **📊 `analysis/` - Análisis y SNR**

- **Responsabilidad**: Análisis avanzado y consistencia
- **Flujo**: `detection/` → `analysis/` → `visualization/`
- **Archivos**:
  - `snr_utils.py` → `config.py`
  - `consistency_fixes.py` → `config.py`, `astro_conversions.py`, `snr_utils.py`

### **🎨 `visualization/` - Visualización**

- **Responsabilidad**: Generación de plots
- **Flujo**: `analysis/` → `visualization/` → Resultados
- **Archivos**:
  - `visualization.py` → Múltiples módulos
  - `image_utils.py` → `config.py`, `astro_conversions.py`, `snr_utils.py`
  - `plot_manager.py` → `visualization.py`, `image_utils.py`

### **⚙️ `system/` - Utilidades del Sistema**

- **Responsabilidad**: Utilidades globales y configuración
- **Flujo**: Usado por todos los módulos
- **Archivos**:
  - `auto_slice_len.py` → `config.py`
  - `slice_len_utils.py` → `config.py`, `auto_slice_len.py`
  - `summary_utils.py` → `config.py`

## 🔗 **Relaciones Críticas Preservadas**

### **1. Flujo de Datos Principal**

```
pipeline.py → detection/pipeline_utils.py → detection/utils.py → input/candidate_utils.py
     ↓              ↓              ↓              ↓
visualization/visualization.py → visualization/image_utils.py → visualization/plot_manager.py
```

### **2. Gestión de Configuración**

```
config.py ← Todos los módulos
     ↓
system/auto_slice_len.py → system/slice_len_utils.py
```

### **3. Procesamiento de SNR**

```
detection/metrics.py → analysis/snr_utils.py
     ↓
analysis/consistency_fixes.py → detection/pipeline_utils.py
```

### **4. Visualización**

```
analysis/ → visualization/visualization.py → visualization/image_utils.py
     ↓
visualization/plot_manager.py → Todos los plots
```

## 🎯 **Ventajas de esta Estructura**

### **1. 📊 Refleja el Flujo Real**

- Cada carpeta representa una etapa del pipeline
- Las dependencias están claramente organizadas
- El flujo de datos es intuitivo

### **2. 🔍 Navegación Lógica**

- `input/` → `preprocessing/` → `detection/` → `analysis/` → `visualization/`
- Fácil seguir el flujo de datos
- Cada carpeta tiene una responsabilidad clara

### **3. 🚀 Escalabilidad**

- Fácil agregar nuevos módulos en la etapa correcta
- Separación clara entre etapas del pipeline
- Estructura preparada para expansiones

### **4. 🛠️ Mantenibilidad**

- Código organizado por funcionalidad
- Dependencias claras y organizadas
- Fácil debugging por etapa

## 📋 **Plan de Migración**

### **Paso 1: Crear Estructura**

```bash
mkdir -p drafts/input
mkdir -p drafts/preprocessing
mkdir -p drafts/detection
mkdir -p drafts/analysis
mkdir -p drafts/visualization
mkdir -p drafts/system
```

### **Paso 2: Mover Archivos**

```bash
# Input
mv drafts/io.py drafts/input/
mv drafts/filterbank_io.py drafts/input/
mv drafts/io_utils.py drafts/input/
mv drafts/candidate.py drafts/input/
mv drafts/candidate_utils.py drafts/input/

# Preprocessing
mv drafts/preprocessing.py drafts/preprocessing/
mv drafts/dedispersion.py drafts/preprocessing/
mv drafts/astro_conversions.py drafts/preprocessing/
mv drafts/dynamic_dm_range.py drafts/preprocessing/

# Detection
mv drafts/pipeline_utils.py drafts/detection/
mv drafts/utils.py drafts/detection/
mv drafts/metrics.py drafts/detection/

# Analysis
mv drafts/snr_utils.py drafts/analysis/
mv drafts/consistency_fixes.py drafts/analysis/

# Visualization
mv drafts/visualization.py drafts/visualization/
mv drafts/image_utils.py drafts/visualization/
mv drafts/plot_manager.py drafts/visualization/

# System
mv drafts/auto_slice_len.py drafts/system/
mv drafts/slice_len_utils.py drafts/system/
mv drafts/summary_utils.py drafts/system/
```

### **Paso 3: Actualizar Imports**

```python
# Antes
from . import config
from .pipeline_utils import process_band
from .snr_utils import compute_snr_profile

# Después
from .. import config
from ..detection.pipeline_utils import process_band
from ..analysis.snr_utils import compute_snr_profile
```

## 🎯 **Conclusión**

Esta estructura **refleja exactamente** las relaciones identificadas en tu análisis:

- ✅ **Flujo de datos real**: `input` → `preprocessing` → `detection` → `analysis` → `visualization`
- ✅ **Dependencias claras**: Cada carpeta tiene responsabilidades específicas
- ✅ **Escalabilidad**: Fácil agregar módulos en la etapa correcta
- ✅ **Mantenibilidad**: Código organizado por funcionalidad

¿Quieres que proceda a crear el script de migración automática para implementar esta estructura basada en relaciones?
