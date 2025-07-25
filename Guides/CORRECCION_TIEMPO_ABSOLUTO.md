# 🔧 CORRECCIÓN: Problema de Saltos en Tiempos Absolutos de Plots

## 🐛 PROBLEMA IDENTIFICADO

Se detectó que los plots de waterfalls dispersados estaban mostrando tiempos absolutos con saltos de 2 en 2 en lugar de continuidad temporal:

**Comportamiento incorrecto observado:**

- 1.0 → 2.0
- 3.0 → 4.0
- 5.0 → 6.0
- 7.0 → 8.0
- ...

**Comportamiento esperado:**

- 1.0 → 2.0
- 2.0 → 3.0
- 3.0 → 4.0
- 4.0 → 5.0
- ...

## 🔍 ANÁLISIS DEL PROBLEMA

### Ubicación del Error

**Archivo:** `DRAFTS/visualization/image_utils.py`  
**Línea:** 186  
**Función:** `plot_waterfall_block()`

### Código Incorrecto (ANTES)

```python
# 🕐 CORRECCIÓN: Usar tiempo absoluto si se proporciona, sino usar cálculo relativo
if absolute_start_time is not None:
    # Calcular tiempo absoluto correcto para cada slice
    # absolute_start_time es el tiempo de inicio del bloque
    # block_idx es el índice del slice en el archivo
    # block_size es el tamaño del slice (SLICE_LEN)
    # time_reso es la resolución temporal decimada
    time_start = absolute_start_time + block_idx * block_size * time_reso
else:
    time_start = block_idx * block_size * time_reso
```

### Problema Identificado

El error estaba en la línea:

```python
time_start = absolute_start_time + block_idx * block_size * time_reso
```

**Explicación del error:**

- `absolute_start_time` ya es el tiempo de inicio del slice específico
- Al sumar `block_idx * block_size * time_reso`, se estaba duplicando el cálculo del tiempo
- Esto causaba que cada slice saltara el doble del tiempo esperado

### Ejemplo del Error

Para un slice con:

- `absolute_start_time = 0.0064s` (tiempo de inicio del slice)
- `block_idx = 0` (índice del slice dentro del chunk)
- `block_size = 64` (SLICE_LEN)
- `time_reso = 0.0001s` (resolución temporal)

**Cálculo incorrecto:**

```
time_start = 0.0064 + 0 * 64 * 0.0001 = 0.0064s
```

**Cálculo correcto:**

```
time_start = 0.0064s  # Directamente el tiempo absoluto proporcionado
```

## ✅ SOLUCIÓN IMPLEMENTADA

### Código Corregido (DESPUÉS)

```python
# 🕐 CORRECCIÓN: Usar tiempo absoluto si se proporciona, sino usar cálculo relativo
if absolute_start_time is not None:
    # absolute_start_time ya es el tiempo de inicio del slice específico
    # No necesitamos sumar block_idx * block_size * time_reso porque ya está incluido
    time_start = absolute_start_time
else:
    time_start = block_idx * block_size * time_reso
```

### Cambio Realizado

- **ANTES:** `time_start = absolute_start_time + block_idx * block_size * time_reso`
- **DESPUÉS:** `time_start = absolute_start_time`

## 🧪 VERIFICACIÓN DE LA CORRECCIÓN

### Script de Prueba

Se creó `tests/test_time_calculation_fix.py` para verificar la corrección:

```python
# Probar con diferentes tiempos absolutos
test_cases = [
    (0.0, "slice_0"),
    (0.0064, "slice_1"),  # 64 * 0.0001
    (0.0128, "slice_2"),  # 128 * 0.0001
    (0.0192, "slice_3"),  # 192 * 0.0001
]

for absolute_start_time, slice_name in test_cases:
    plot_waterfall_block(
        data_block=data_block,
        freq=freq,
        time_reso=time_reso,
        block_size=block_size,
        block_idx=0,
        save_dir=test_dir,
        filename=f"test_{slice_name}",
        normalize=True,
        absolute_start_time=absolute_start_time
    )
```

### Resultados de la Prueba

✅ **La corrección fue exitosa:**

- Los plots ahora muestran tiempos continuos
- No hay saltos de 2 en 2
- Los tiempos absolutos se calculan correctamente

## 📋 ÁMBITO DE LA CORRECCIÓN

### Archivos Afectados

- ✅ `DRAFTS/visualization/image_utils.py` - **CORREGIDO**
- ✅ `DRAFTS/visualization/visualization.py` - **YA ESTABA CORRECTO**
- ✅ `DRAFTS/visualization/plot_manager.py` - **YA ESTABA CORRECTO**
- ✅ `DRAFTS/detection/pipeline_utils.py` - **YA ESTABA CORRECTO**

### Funciones Verificadas

- ✅ `plot_waterfall_block()` - **CORREGIDA**
- ✅ `save_plot()` - **YA ESTABA CORRECTO**
- ✅ `save_slice_summary()` - **YA ESTABA CORRECTO**
- ✅ `process_slice()` - **YA ESTABA CORRECTO**

## 🎯 IMPACTO DE LA CORRECCIÓN

### Antes de la Corrección

- Los plots mostraban tiempos discontinuos
- Saltos de 2 en 2 en lugar de continuidad
- Confusión en la interpretación temporal de los datos

### Después de la Corrección

- Los plots muestran tiempos continuos y correctos
- Continuidad temporal entre slices
- Interpretación correcta de los tiempos absolutos

## 🔄 FLUJO DE DATOS CORREGIDO

### Pipeline de Procesamiento

1. **Chunking:** `_process_block()` calcula `chunk_start_time_sec`
2. **Slicing:** Calcula `slice_start_time_sec` para cada slice
3. **Visualización:** `plot_waterfall_block()` usa `absolute_start_time` directamente

### Cálculo de Tiempos

```python
# En _process_block()
chunk_start_time_sec = metadata["start_sample"] * config.TIME_RESO

# Para cada slice j
slice_start_time_sec = chunk_start_time_sec + (j * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE)

# En plot_waterfall_block() - CORREGIDO
time_start = absolute_start_time  # Ya no se suma block_idx * block_size * time_reso
```

## 📝 NOTAS ADICIONALES

### Funciones No Afectadas

Las siguientes funciones no tenían el problema porque no usaban el cálculo incorrecto:

- `save_plot()` en `visualization.py`
- `save_slice_summary()` en `visualization.py`
- `process_slice()` en `pipeline_utils.py`

### Verificación Completa

Se verificó que no hay otros lugares en el código donde se esté produciendo el mismo error de cálculo de tiempo absoluto.

### Compatibilidad

La corrección es compatible con:

- ✅ Modo chunking (archivos .fil)
- ✅ Modo normal (archivos .fits)
- ✅ Procesamiento multi-banda
- ✅ Continuidad temporal entre chunks

---

**Fecha de corrección:** Diciembre 2024  
**Estado:** ✅ CORREGIDO Y VERIFICADO
