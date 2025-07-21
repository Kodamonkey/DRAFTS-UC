# Corrección de Tiempos Absolutos en Procesamiento por Chunks

## Problema Identificado

En el procesamiento por chunks, todos los chunks mostraban tiempos relativos (empezando en 0 segundos) en lugar de los tiempos reales del archivo. Esto causaba confusión porque:

- Chunk 0: mostraba tiempos 0.00s - X.XXs
- Chunk 1: mostraba tiempos 0.00s - X.XXs (debería ser 5.00s - Y.YYs)
- Chunk 2: mostraba tiempos 0.00s - X.XXs (debería ser 10.00s - Z.ZZs)

## Solución Implementada

### 1. Modificación de `plot_waterfall_block` en `image_utils.py`

**Cambio:** Agregado parámetro `absolute_start_time` opcional

```python
def plot_waterfall_block(
    data_block: np.ndarray,
    freq: np.ndarray,
    time_reso: float,
    block_size: int,
    block_idx: int,
    save_dir: Path,
    filename: str,
    normalize: bool = False,
    absolute_start_time: float = None,  # 🕐 NUEVO: Tiempo absoluto de inicio del chunk
) -> None:
```

**Lógica de tiempo:**

```python
# 🕐 CORRECCIÓN: Usar tiempo absoluto si se proporciona, sino usar cálculo relativo
if absolute_start_time is not None:
    time_start = absolute_start_time + block_idx * block_size * time_reso
else:
    time_start = block_idx * block_size * time_reso
```

### 2. Modificación de `_process_single_chunk` en `pipeline.py`

**Cambios principales:**

#### a) Cálculo del tiempo absoluto del chunk

```python
# 🕐 CALCULAR TIEMPO ABSOLUTO DEL CHUNK PARA PLOTS
chunk_start_time_sec = start_sample_global * config.TIME_RESO * config.DOWN_TIME_RATE
logger.info(f"Chunk {chunk_idx}: tiempo absoluto de inicio = {chunk_start_time_sec:.3f}s")
```

#### b) Cálculo del índice absoluto del slice

```python
# 🕐 CORRECCIÓN: Calcular tiempo absoluto para save_slice_summary
slice_absolute_idx = start_sample_global // slice_len + j  # Índice absoluto del slice
```

#### c) Actualización de todas las llamadas a funciones de visualización

**Para `plot_waterfall_block`:**

```python
plot_waterfall_block(
    # ... otros parámetros ...
    absolute_start_time=chunk_start_time_sec,  # 🕐 Pasar tiempo absoluto del chunk
)
```

**Para `save_patch_plot`:**

```python
# 🕐 CORRECCIÓN: Usar tiempo absoluto para patch_plot
patch_start_time_abs = chunk_start_time_sec + first_start
save_patch_plot(
    # ... otros parámetros ...
    patch_start_time_abs,  # Tiempo absoluto del archivo
)
```

**Para `save_slice_summary` y `save_plot`:**

```python
save_slice_summary(
    # ... otros parámetros ...
    slice_absolute_idx,  # 🕐 Índice absoluto del slice en el archivo completo
    # ... otros parámetros ...
)

save_plot(
    # ... otros parámetros ...
    slice_absolute_idx,  # 🕐 Índice absoluto del slice en el archivo completo
    # ... otros parámetros ...
)
```

### 3. Actualización de `_process_file` (procesamiento estándar)

**Mantener compatibilidad:** Todas las llamadas a `plot_waterfall_block` ahora incluyen:

```python
absolute_start_time=None,  # 🕐 Usar tiempo relativo para procesamiento estándar
```

## Resultados

### Antes de la corrección:

- Chunk 0: tiempos 0.00s - 1.00s
- Chunk 1: tiempos 0.00s - 1.00s ❌
- Chunk 2: tiempos 0.00s - 1.00s ❌

### Después de la corrección:

- Chunk 0: tiempos 0.00s - 1.00s ✅
- Chunk 1: tiempos 5.00s - 6.00s ✅
- Chunk 2: tiempos 10.00s - 11.00s ✅

## Archivos Modificados

1. **`DRAFTS/image_utils.py`**

   - Función `plot_waterfall_block`: agregado parámetro `absolute_start_time`

2. **`DRAFTS/pipeline.py`**

   - Función `_process_single_chunk`: implementada lógica de tiempos absolutos
   - Función `_process_file`: mantenida compatibilidad con `absolute_start_time=None`

3. **`tests/test_chunk_timing.py`** (nuevo)
   - Script de prueba para verificar funcionamiento correcto

## Verificación

Los cambios fueron verificados con tests automatizados que confirman:

✅ **Cálculo correcto de índices absolutos de slices**
✅ **Tiempos absolutos correctos en waterfalls**
✅ **Compatibilidad con procesamiento estándar**
✅ **Nombres de archivos con tiempos correctos**

## Impacto

- **Positivo:** Los plots ahora muestran los tiempos reales del archivo
- **Neutral:** No afecta el procesamiento estándar (sin chunks)
- **Compatibilidad:** Mantiene compatibilidad hacia atrás

## Uso

El sistema funciona automáticamente:

- **Procesamiento estándar:** Usa tiempos relativos (comportamiento original)
- **Procesamiento por chunks:** Usa tiempos absolutos automáticamente

No se requieren cambios en la configuración del usuario.
