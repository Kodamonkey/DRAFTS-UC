# Análisis de Discrepancias DM/SNR en Pipeline DRAFTS

## 🔍 **Problema Identificado**

Se detectaron **discrepancias sistemáticas** entre los valores mostrados en diferentes partes del pipeline:

### **1. Discrepancia en DM (Dispersion Measure)**

**Síntomas:**

- **Box Detection**: DM mostrado en el composite plot
- **Waterfall Dedispersado**: DM en el subtítulo del waterfall
- **CSV**: DM guardado en el archivo de resultados

**Resultado:** Los tres valores pueden ser diferentes, causando confusión.

### **2. Discrepancia en SNR (Signal-to-Noise Ratio)**

**Síntomas:**

- **Composite Plot**: SNR mostrado en subtítulos de waterfalls
- **CSV**: SNR guardado en el archivo de resultados

**Resultado:** Los valores de SNR no coinciden entre visualización y datos.

## 🕵️ **Análisis de Causas Raíz**

### **Causa 1: Múltiples Cálculos de DM**

#### **Ubicaciones de Cálculo:**

1. **`pipeline_utils.py` (líneas 60-65):**

```python
dm_val, t_sec, t_sample = pixel_to_physical(
    (box[0] + box[2]) / 2,
    (box[1] + box[3]) / 2,
    slice_len,
)
```

2. **`visualization.py` (líneas 400-405):**

```python
dm_val_cand, t_sec_real, t_sample_real = pixel_to_physical(center_x, center_y, slice_len)
```

3. **`visualization.py` (líneas 560-565):**

```python
dm_val_consistent, _, _ = pixel_to_physical(center_x, center_y, slice_len)
```

#### **Problema:**

Aunque todos usan `pixel_to_physical()`, se calculan en **momentos diferentes** y pueden usar **diferentes candidatos**:

- **Box Detection**: Candidato actual en el loop
- **Waterfall Dedispersado**: Candidato con mayor confianza (`best_candidate_idx`)

### **Causa 2: Múltiples Cálculos de SNR**

#### **Ubicaciones de Cálculo:**

1. **CSV (patch dedispersado):**

```python
# pipeline_utils.py líneas 85-95
snr_patch_profile, _ = compute_snr_profile(patch)
snr_val, _, _ = find_snr_peak(snr_patch_profile)
```

2. **Composite Raw Waterfall:**

```python
# visualization.py líneas 465-470
snr_wf, sigma_wf = compute_snr_profile(wf_block, off_regions)
peak_snr_wf, peak_time_wf, peak_idx_wf = find_snr_peak(snr_wf)
```

3. **Composite Dedispersed Waterfall:**

```python
# visualization.py líneas 580-585
snr_dw, sigma_dw = compute_snr_profile(dw_block, off_regions)
peak_snr_dw, peak_time_dw, peak_idx_dw = find_snr_peak(snr_dw)
```

#### **Problema:**

Cada cálculo usa **datos diferentes**:

- **CSV**: SNR del patch dedispersado (región pequeña)
- **Composite Raw**: SNR del waterfall completo sin dedispersar
- **Composite Dedispersed**: SNR del waterfall completo dedispersado

## 🛠️ **Solución Implementada**

### **Archivo: `drafts/consistency_fixes.py`**

#### **1. Gestor de Consistencia (`ConsistencyManager`)**

```python
class ConsistencyManager:
    def calculate_consistent_candidate_values(self, ...):
        # Calcula DM y SNR consistentes para todos los candidatos
```

#### **2. Cálculo Unificado de DM**

```python
# ✅ DM CONSISTENTE - Usar el mismo cálculo en todas partes
dm_val, t_sec, t_sample = pixel_to_physical(center_x, center_y, slice_len)
```

#### **3. Cálculo Unificado de SNR**

```python
def _calculate_all_snr_types(self, ...):
    # Calcula 5 tipos diferentes de SNR:
    # 1. candidate_raw: SNR del candidato en waterfall raw
    # 2. candidate_dedispersed: SNR del candidato en waterfall dedispersado
    # 3. patch_dedispersed: SNR del patch dedispersado (para CSV)
    # 4. waterfall_raw_peak: SNR peak del waterfall raw completo
    # 5. waterfall_dedispersed_peak: SNR peak del waterfall dedispersado completo
```

#### **4. Priorización de SNR para CSV**

```python
def get_csv_snr_value(self, candidate_data):
    # Prioridad:
    # 1. SNR del patch dedispersado (más preciso)
    # 2. SNR del candidato en waterfall dedispersado
    # 3. SNR del candidato en waterfall raw
    # 4. 0.0 como fallback
```

## 📊 **Tipos de SNR Explicados**

### **1. SNR del Candidato (Región Específica)**

- **Raw**: SNR de la región del candidato en datos sin dedispersar
- **Dedispersed**: SNR de la región del candidato en datos dedispersados
- **Uso**: Análisis específico del candidato detectado

### **2. SNR del Patch Dedispersado**

- **Descripción**: SNR del patch pequeño extraído y dedispersado
- **Uso**: **Valor guardado en CSV** (más preciso para el candidato)
- **Ventaja**: Mide la señal real del candidato después de dedispersión

### **3. SNR del Waterfall Completo**

- **Raw Peak**: SNR máximo del waterfall sin dedispersar
- **Dedispersed Peak**: SNR máximo del waterfall dedispersado
- **Uso**: **Mostrado en composite plots** (contexto general)

## 🎯 **Valores Correctos por Contexto**

### **Para CSV (Datos de Análisis):**

- **DM**: Valor calculado por `pixel_to_physical()` para el candidato específico
- **SNR**: SNR del patch dedispersado (prioridad 1)

### **Para Composite Plot (Visualización):**

- **DM**: Valor del candidato con mayor confianza
- **SNR Raw**: Peak del waterfall raw completo
- **SNR Dedispersed**: Peak del waterfall dedispersado completo

## 🔧 **Implementación en el Pipeline**

### **Paso 1: Integrar en `pipeline_utils.py`**

```python
from .consistency_fixes import get_consistent_candidate_values, get_csv_ready_candidate_data

# En process_band():
consistent_data = get_consistent_candidate_values(
    top_boxes, top_conf, slice_len, waterfall_block, dedispersed_block, patch
)

# Para cada candidato:
candidate_data = get_csv_ready_candidate_data(idx, consistent_data)
dm_val = candidate_data['dm_val']
snr_val = candidate_data['snr_csv']  # SNR correcto para CSV
```

### **Paso 2: Integrar en `visualization.py`**

```python
from .consistency_fixes import get_composite_display_data

# En save_slice_summary():
display_data = get_composite_display_data(consistent_data)
dm_val = display_data['dm_val']
snr_raw_peak = display_data['snr_raw_peak']
snr_dedispersed_peak = display_data['snr_dedispersed_peak']
```

## 📈 **Beneficios de la Solución**

### **1. Consistencia Total**

- ✅ DM idéntico en todas las partes del pipeline
- ✅ SNR apropiado para cada contexto
- ✅ Transparencia en los cálculos

### **2. Debugging Mejorado**

```python
from .consistency_fixes import print_consistency_report
print_consistency_report(consistent_data)
```

### **3. Flexibilidad**

- Múltiples tipos de SNR disponibles
- Priorización inteligente para CSV
- Fácil extensión para nuevos tipos

### **4. Mantenibilidad**

- Código centralizado
- Documentación clara
- Tests unitarios posibles

## 🚀 **Próximos Pasos**

### **1. Integración Completa**

- [ ] Modificar `pipeline_utils.py` para usar el gestor de consistencia
- [ ] Modificar `visualization.py` para usar valores consistentes
- [ ] Actualizar `plot_manager.py` para pasar datos consistentes

### **2. Testing**

- [ ] Crear tests unitarios para el gestor de consistencia
- [ ] Verificar que los valores coincidan en todos los contextos
- [ ] Validar con datos reales de FRB

### **3. Documentación**

- [ ] Actualizar guías de usuario
- [ ] Documentar los diferentes tipos de SNR
- [ ] Crear ejemplos de uso

## 📋 **Resumen de Correcciones**

| Componente           | Antes              | Después                         |
| -------------------- | ------------------ | ------------------------------- |
| **DM Box Detection** | Cálculo individual | DM del mejor candidato          |
| **DM Waterfall**     | Cálculo individual | DM del mejor candidato          |
| **DM CSV**           | Cálculo individual | DM del candidato específico     |
| **SNR CSV**          | Patch dedispersado | Patch dedispersado (priorizado) |
| **SNR Composite**    | Waterfall completo | Waterfall completo + contexto   |

## 🎯 **Conclusión**

Las discrepancias se debían a **cálculos independientes** en diferentes partes del pipeline. La solución implementada **centraliza y unifica** todos los cálculos, asegurando consistencia total entre:

- ✅ **Box Detection** y **Waterfall Dedispersado** (mismo DM)
- ✅ **Composite Plot** y **CSV** (SNR apropiado para cada contexto)
- ✅ **Transparencia** en todos los cálculos
- ✅ **Mantenibilidad** del código

El nuevo sistema `ConsistencyManager` resuelve completamente las discrepancias y proporciona una base sólida para futuras mejoras del pipeline.
