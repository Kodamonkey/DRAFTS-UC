# 🗂️ Propuesta de Estructura de Carpetas para `drafts/`

## 📊 **Estructura Actual vs Propuesta**

### **Estructura Actual (Plana)**

```
drafts/
├── __init__.py
├── config.py
├── pipeline.py
├── pipeline_utils.py
├── io.py
├── filterbank_io.py
├── io_utils.py
├── candidate.py
├── candidate_utils.py
├── preprocessing.py
├── dedispersion.py
├── astro_conversions.py
├── dynamic_dm_range.py
├── utils.py
├── metrics.py
├── snr_utils.py
├── consistency_fixes.py
├── auto_slice_len.py
├── slice_len_utils.py
├── summary_utils.py
├── visualization.py
├── image_utils.py
└── plot_manager.py
```

### **Estructura Propuesta (Organizada)**

```
drafts/
├── __init__.py
├── config.py                    # Configuración central
├── pipeline.py                  # Pipeline principal
│
├── core/                        # 🧠 Núcleo del sistema
│   ├── __init__.py
│   ├── pipeline_utils.py        # Lógica principal de procesamiento
│   ├── utils.py                 # Utilidades de detección/clasificación
│   └── consistency_fixes.py     # Gestor de consistencia DM/SNR
│
├── io/                          # 📁 Entrada/Salida
│   ├── __init__.py
│   ├── io.py                    # Archivos FITS
│   ├── filterbank_io.py         # Archivos FIL
│   ├── io_utils.py              # Carga unificada de datos
│   ├── candidate.py             # Estructura de candidatos
│   └── candidate_utils.py       # Gestión de CSV
│
├── preprocessing/               # 🔄 Preprocesamiento
│   ├── __init__.py
│   ├── preprocessing.py         # Downsampling y normalización
│   ├── dedispersion.py          # Dedispersión GPU/CPU
│   ├── astro_conversions.py     # Conversiones pixel→DM
│   └── dynamic_dm_range.py      # Rangos DM dinámicos
│
├── detection/                   # 🎯 Detección y Análisis
│   ├── __init__.py
│   ├── metrics.py               # Cálculo de SNR básico
│   └── snr_utils.py             # Análisis avanzado de SNR
│
├── visualization/               # 🎨 Visualización
│   ├── __init__.py
│   ├── visualization.py         # Plots principales
│   ├── image_utils.py           # Utilidades de imagen
│   └── plot_manager.py          # Gestión de plots
│
└── utils/                       # ⚙️ Utilidades del Sistema
    ├── __init__.py
    ├── auto_slice_len.py        # Cálculo automático de SLICE_LEN
    ├── slice_len_utils.py       # Gestión dinámica de SLICE_LEN
    └── summary_utils.py         # Reportes y logs
```

## 🎯 **Beneficios de la Organización**

### **1. 📁 Separación Clara de Responsabilidades**

- **`core/`**: Lógica principal del pipeline
- **`io/`**: Todo lo relacionado con entrada/salida
- **`preprocessing/`**: Procesamiento de datos
- **`detection/`**: Análisis y detección
- **`visualization/`**: Generación de plots
- **`utils/`**: Utilidades del sistema

### **2. 🔍 Facilidad de Navegación**

- Encontrar archivos específicos es más rápido
- Nuevos desarrolladores entienden mejor la estructura
- Menos confusión sobre qué hace cada módulo

### **3. 🚀 Escalabilidad**

- Fácil agregar nuevos módulos en la carpeta correcta
- Mejor organización para futuras expansiones
- Separación clara entre funcionalidades

### **4. 🛠️ Mantenibilidad**

- Código más organizado y profesional
- Más fácil hacer debugging
- Mejor control de versiones

## 📋 **Plan de Migración**

### **Paso 1: Crear Estructura de Carpetas**

```bash
mkdir -p drafts/core
mkdir -p drafts/io
mkdir -p drafts/preprocessing
mkdir -p drafts/detection
mkdir -p drafts/visualization
mkdir -p drafts/utils
```

### **Paso 2: Mover Archivos**

```bash
# Core
mv drafts/pipeline_utils.py drafts/core/
mv drafts/utils.py drafts/core/
mv drafts/consistency_fixes.py drafts/core/

# IO
mv drafts/io.py drafts/io/
mv drafts/filterbank_io.py drafts/io/
mv drafts/io_utils.py drafts/io/
mv drafts/candidate.py drafts/io/
mv drafts/candidate_utils.py drafts/io/

# Preprocessing
mv drafts/preprocessing.py drafts/preprocessing/
mv drafts/dedispersion.py drafts/preprocessing/
mv drafts/astro_conversions.py drafts/preprocessing/
mv drafts/dynamic_dm_range.py drafts/preprocessing/

# Detection
mv drafts/metrics.py drafts/detection/
mv drafts/snr_utils.py drafts/detection/

# Visualization
mv drafts/visualization.py drafts/visualization/
mv drafts/image_utils.py drafts/visualization/
mv drafts/plot_manager.py drafts/visualization/

# Utils
mv drafts/auto_slice_len.py drafts/utils/
mv drafts/slice_len_utils.py drafts/utils/
mv drafts/summary_utils.py drafts/utils/
```

### **Paso 3: Actualizar Imports**

Necesitarás actualizar todos los imports en los archivos. Por ejemplo:

**Antes:**

```python
from . import config
from .pipeline_utils import process_band
from .snr_utils import compute_snr_profile
```

**Después:**

```python
from .. import config
from ..core.pipeline_utils import process_band
from ..detection.snr_utils import compute_snr_profile
```

## 🔧 **Script de Migración Automática**

Te puedo crear un script que:

1. ✅ Cree las carpetas automáticamente
2. ✅ Mueva los archivos
3. ✅ Actualice todos los imports
4. ✅ Cree los `__init__.py` necesarios
5. ✅ Verifique que todo funcione

## 📊 **Comparación de Estructuras**

| Aspecto                  | Estructura Actual          | Estructura Propuesta |
| ------------------------ | -------------------------- | -------------------- |
| **Archivos por carpeta** | 22 en una carpeta          | 3-5 por carpeta      |
| **Navegación**           | Difícil encontrar archivos | Fácil y lógica       |
| **Escalabilidad**        | Limitada                   | Excelente            |
| **Mantenibilidad**       | Media                      | Alta                 |
| **Profesionalismo**      | Básico                     | Profesional          |

## 🎯 **Recomendación Final**

**SÍ, definitivamente deberías reorganizar** porque:

1. ✅ **Tu pipeline es complejo** (22 módulos) y necesita organización
2. ✅ **Estás en desarrollo activo** - mejor hacerlo ahora que después
3. ✅ **Facilitará futuras mejoras** y mantenimiento
4. ✅ **Mejorará la legibilidad** del código
5. ✅ **Es una práctica estándar** en proyectos profesionales

¿Quieres que proceda a crear el script de migración automática para reorganizar tu estructura de carpetas?
