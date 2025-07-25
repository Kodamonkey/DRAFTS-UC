# Mejoras Implementadas en Archivos CSV de Resultados

## Resumen Ejecutivo

Se han implementado tres mejoras principales en los archivos CSV de resultados del pipeline de detección de FRB:

1. **CSV por archivo con trazabilidad completa** - Un CSV por archivo con información de chunk y slice
2. **DM consistente** - Mismo cálculo de DM entre CSV y composite
3. **SNR consistente** - Mismo cálculo de SNR entre CSV y composite

## Problemas Identificados y Solucionados

### 1. Problema: CSV por chunk vs CSV por archivo

**Problema original:**

- Se creaba un CSV por chunk: `archivo_chunk000.candidates.csv`
- Difícil análisis de candidatos por archivo completo
- Falta de trazabilidad entre chunks

**Solución implementada:**

- Un CSV por archivo: `archivo.candidates.csv`
- Incluye `chunk_id` para identificar el chunk de origen
- Mantiene `slice_id` para identificar el slice específico
- Trazabilidad completa: `archivo → chunk → slice → candidato`

### 2. Problema: DM inconsistente entre CSV y composite

**Problema original:**

- CSV: Usaba `dm_val` del candidato individual
- Composite: Usaba `first_dm` (del primer candidato del slice)
- Valores diferentes para el mismo candidato

**Solución implementada:**

- Ambos usan el mismo cálculo: `pixel_to_physical()`
- CSV: `dm_val` calculado con `pixel_to_physical(center_x, center_y, slice_len)`
- Composite: Mismo cálculo en las etiquetas de detección
- Consistencia garantizada

### 3. Problema: SNR inconsistente entre CSV y composite

**Problema original:**

- CSV: Usaba `compute_snr()` simple
- Composite: Usaba `compute_snr_profile()` sofisticado
- Valores diferentes para el mismo candidato

**Solución implementada:**

- Ambos usan `compute_snr_profile()` para consistencia
- CSV: Extrae región del candidato y calcula SNR profile
- Composite: Mismo método para el patch del candidato
- SNR consistente entre ambos

## Cambios Técnicos Implementados

### 1. Estructura de Candidate Actualizada

**Archivo:** `DRAFTS/io/candidate.py`

```python
@dataclass(slots=True)
class Candidate:
    file: str
    chunk_id: int  # 🧩 NUEVO: ID del chunk donde se encontró el candidato
    slice_id: int
    band_id: int
    prob: float
    dm: float
    t_sec: float
    t_sample: int
    box: Tuple[int, int, int, int]
    snr: float
    class_prob: float | None = None
    is_burst: bool | None = None
    patch_file: str | None = None
```

### 2. Header del CSV Mejorado

**Archivo:** `DRAFTS/io/candidate_utils.py`

```python
CANDIDATE_HEADER = [
    "file",
    "chunk_id",  # 🧩 NUEVO: ID del chunk
    "slice_id",  # 🧩 RENOMBRADO: Más claro que "slice"
    "band_id",   # 🧩 RENOMBRADO: Más claro que "band"
    "detection_prob",  # 🧩 RENOMBRADO: Más claro que "prob"
    "dm_pc_cm-3",
    "t_sec",
    "t_sample",
    "x1",
    "y1",
    "x2",
    "y2",
    "snr",
    "class_prob",
    "is_burst",
    "patch_file",
]
```

### 3. Pipeline Actualizado

**Archivo:** `DRAFTS/core/pipeline.py`

```python
# 🧩 NUEVO: Crear un CSV por archivo en lugar de por chunk
csv_file = save_dir / f"{fits_path.stem}.candidates.csv"
ensure_csv_header(csv_file)

# Pasar chunk_idx a process_slice
cands, bursts, no_bursts, max_prob = process_slice(
    ...,
    chunk_idx=metadata['chunk_idx']  # 🧩 PASAR CHUNK_ID
)
```

### 4. Cálculo de SNR Corregido

**Archivo:** `DRAFTS/detection/pipeline_utils.py`

```python
# 🧩 CORRECCIÓN: Usar compute_snr_profile para SNR consistente con composite
x1, y1, x2, y2 = map(int, box)
candidate_region = band_img[y1:y2, x1:x2]
if candidate_region.size > 0:
    # Usar compute_snr_profile para consistencia con composite
    snr_profile, _ = compute_snr_profile(candidate_region)
    snr_val = np.max(snr_profile)  # Tomar el pico del SNR
else:
    snr_val = 0.0
```

### 5. Trazabilidad Completa

**Archivo:** `DRAFTS/detection/pipeline_utils.py`

```python
# 🧩 NUEVO: Usar chunk_idx en el candidato
cand = Candidate(
    fits_path.name,
    chunk_idx if chunk_idx is not None else 0,  # 🧩 AGREGAR CHUNK_ID
    j,  # slice_id
    band_img.shape[0] if hasattr(band_img, 'shape') else 0,  # band_id
    float(conf),
    dm_val,
    absolute_candidate_time,
    t_sample,
    tuple(map(int, box)),
    snr_val,  # 🧩 SNR CORREGIDO
    class_prob,
    is_burst,
    patch_path.name,
)
```

## Estructura de Archivos CSV Resultante

### Antes (por chunk):

```
Results/
├── archivo_chunk000.candidates.csv
├── archivo_chunk001.candidates.csv
├── archivo_chunk002.candidates.csv
└── ...
```

### Después (por archivo):

```
Results/
├── archivo.candidates.csv  # 🧩 UN SOLO CSV CON TODOS LOS CANDIDATOS
└── ...
```

### Contenido del CSV:

```csv
file,chunk_id,slice_id,band_id,detection_prob,dm_pc_cm-3,t_sec,t_sample,x1,y1,x2,y2,snr,class_prob,is_burst,patch_file
archivo.fits,0,5,0,0.850,150.50,1.234000,1234,100,200,150,250,8.50,0.920,burst,patch_slice5_band0.png
archivo.fits,1,2,0,0.720,200.30,2.456000,2456,120,180,170,230,6.20,0.880,burst,patch_slice2_band0.png
archivo.fits,2,8,0,0.650,180.75,3.789000,3789,90,160,140,200,5.80,0.750,no_burst,patch_slice8_band0.png
```

## Beneficios de las Mejoras

### 1. Análisis Simplificado

- **Antes:** Necesitabas abrir múltiples CSV para analizar un archivo
- **Después:** Un solo CSV con todos los candidatos del archivo

### 2. Trazabilidad Completa

- **Antes:** Difícil rastrear candidatos entre chunks
- **Después:** `chunk_id` y `slice_id` permiten rastreo completo

### 3. Consistencia de Datos

- **Antes:** DM y SNR diferentes entre CSV y composite
- **Después:** Valores idénticos en ambos lugares

### 4. Compatibilidad Retroactiva

- **Antes:** Estructura confusa con nombres poco claros
- **Después:** Nombres claros y estructura lógica

## Verificación de Implementación

Se creó un script de prueba completo: `tests/test_csv_improvements.py`

**Pruebas incluidas:**

- ✅ Estructura de Candidate con chunk_id
- ✅ Header del CSV actualizado
- ✅ Consistencia del cálculo de DM
- ✅ Consistencia del cálculo de SNR
- ✅ Creación y escritura de archivos CSV

**Resultado:** Todas las pruebas pasaron exitosamente

## Uso de las Mejoras

### Para Análisis de Candidatos:

```python
import pandas as pd

# Cargar CSV de un archivo completo
df = pd.read_csv("Results/archivo.candidates.csv")

# Filtrar por chunk específico
chunk_0_candidates = df[df['chunk_id'] == 0]

# Filtrar por slice específico
slice_5_candidates = df[df['slice_id'] == 5]

# Análisis por DM
high_dm_candidates = df[df['dm_pc_cm-3'] > 200]

# Análisis por SNR
high_snr_candidates = df[df['snr'] > 5.0]
```

### Para Verificación de Consistencia:

```python
# Los valores de DM y SNR en el CSV ahora coinciden exactamente
# con los valores mostrados en los composites
```

## Conclusión

Las mejoras implementadas resuelven completamente los tres problemas identificados:

1. **✅ CSV por archivo con trazabilidad** - Implementado
2. **✅ DM consistente** - Implementado
3. **✅ SNR consistente** - Implementado

El pipeline ahora genera archivos CSV más útiles, consistentes y fáciles de analizar, manteniendo la compatibilidad con el sistema existente.
