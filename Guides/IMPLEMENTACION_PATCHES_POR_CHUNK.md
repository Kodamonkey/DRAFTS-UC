# Implementación de Estructura de Carpetas por Chunk para Patches

## Resumen

Se ha implementado una nueva funcionalidad que organiza los plots de patches candidates en carpetas correspondientes al chunk en que fueron generados, siguiendo la misma estructura que ya existía para `Detections/` y `Composite/`.

## Problema Original

Los plots de patches candidates se guardaban en una estructura fija:

```
Patches/
└── archivo_fits/
    ├── patch_slice0_band256.png
    ├── patch_slice1_band256.png
    └── ...
```

Esto no permitía organizar los patches por chunk cuando se procesaban archivos grandes en bloques.

## Solución Implementada

### 1. Estructura de Carpetas por Chunk

Ahora los patches se organizan en la siguiente estructura:

```
Patches/
└── archivo_fits_chunk000/
    ├── patch_slice0_band256.png
    ├── patch_slice1_band256.png
    └── ...
└── archivo_fits_chunk001/
    ├── patch_slice0_band256.png
    ├── patch_slice1_band256.png
    └── ...
```

### 2. Cambios en el Código

#### A. Pipeline Principal (`DRAFTS/core/pipeline.py`)

**Líneas 188-192**: Creación de carpeta de patches por chunk

```python
# === CHUNKED FOLDER STRUCTURE FOR PLOTS ===
chunk_folder_name = f"{fits_path.stem}_chunk{chunk_idx:03d}"
composite_dir = save_dir / "Composite" / chunk_folder_name
composite_dir.mkdir(parents=True, exist_ok=True)
detections_dir = save_dir / "Detections" / chunk_folder_name
detections_dir.mkdir(parents=True, exist_ok=True)
patches_dir = save_dir / "Patches" / chunk_folder_name  # 🆕 NUEVO
patches_dir.mkdir(parents=True, exist_ok=True)          # 🆕 NUEVO
```

**Líneas 204-210**: Paso del parámetro `patches_dir` a `process_slice`

```python
cands, bursts, no_bursts, max_prob = process_slice(
    j, dm_time, block, slice_len, det_model, cls_model, fits_path, save_dir,
    freq_down, csv_file, config.TIME_RESO * config.DOWN_TIME_RATE, band_configs,
    snr_list, waterfall_dispersion_dir, waterfall_dedispersion_dir, config,
    absolute_start_time=slice_start_time_sec,
    composite_dir=composite_dir,
    detections_dir=detections_dir,
    patches_dir=patches_dir  # 🆕 NUEVO PARÁMETRO
)
```

#### B. Pipeline Utils (`DRAFTS/detection/pipeline_utils.py`)

**Líneas 29-47**: Modificación de `process_band` para aceptar `patches_dir`

```python
def process_band(
    det_model,
    cls_model,
    band_img,
    slice_len,
    j,
    fits_path,
    save_dir,
    data,
    freq_down,
    csv_file,
    time_reso_ds,
    snr_list,
    config,
    absolute_start_time=None,
    patches_dir=None  # 🆕 NUEVO PARÁMETRO
):
```

**Líneas 60-65**: Lógica condicional para `patch_path`

```python
if patches_dir is not None:
    patch_path = patches_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
else:
    patch_dir = save_dir / "Patches" / fits_path.stem
    patch_path = patch_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
```

**Líneas 136-157**: Modificación de `process_slice` para aceptar `patches_dir`

```python
def process_slice(
    j,
    dm_time,
    block,
    slice_len,
    det_model,
    cls_model,
    fits_path,
    save_dir,
    freq_down,
    csv_file,
    time_reso_ds,
    band_configs,
    snr_list,
    waterfall_dispersion_dir,
    waterfall_dedispersion_dir,
    config,
    absolute_start_time=None,
    composite_dir=None,
    detections_dir=None,
    patches_dir=None,  # 🆕 NUEVO PARÁMETRO
):
```

**Líneas 197-201**: Lógica condicional para `patch_path` en `process_slice`

```python
if patches_dir is not None:
    patch_path = patches_dir / f"{fits_stem}_slice{j:03d}.png"
else:
    patch_path = save_dir / "Patches" / f"{fits_stem}_slice{j:03d}.png"
```

**Líneas 225-235**: Paso del parámetro `patches_dir` a `process_band`

```python
band_result = process_band(
    det_model,
    cls_model,
    band_img,
    slice_len,
    j,
    fits_path,
    save_dir,
    block,
    freq_down,
    csv_file,
    time_reso_ds,
    snr_list,
    config,
    absolute_start_time=absolute_start_time,
    patches_dir=patches_dir,  # 🆕 NUEVO PARÁMETRO
)
```

## Compatibilidad hacia Atrás

La implementación mantiene compatibilidad hacia atrás:

- **Con `patches_dir=None`**: Usa la estructura antigua `Patches/archivo_fits/`
- **Con `patches_dir` especificado**: Usa la nueva estructura `Patches/archivo_fits_chunkXXX/`

## Verificación

Se creó un script de prueba (`tests/test_patches_chunked_structure.py`) que verifica:

1. ✅ **Estructura por chunk**: Los patches se guardan en carpetas por chunk cuando se especifica `patches_dir`
2. ✅ **Compatibilidad hacia atrás**: El código funciona correctamente sin `patches_dir` (modo antiguo)

### Resultados de las Pruebas

```
🧪 INICIANDO PRUEBAS DE ESTRUCTURA DE PATCHES POR CHUNK

=== Probando estructura de carpetas por chunk para patches ===
✅ patch_path correcto: test_patches_chunked\Patches\test_file_chunk000\patch_slice5_band256.png

=== Probando compatibilidad hacia atrás (sin patches_dir) ===
✅ patch_path correcto (modo antiguo): test_patches_backward\Patches\test_file\patch_slice5_band256.png

📊 RESUMEN DE PRUEBAS:
   🧩 Estructura por chunk: ✅ PASÓ
   🔄 Compatibilidad hacia atrás: ✅ PASÓ

🎉 ¡TODAS LAS PRUEBAS PASARON!
```

## Beneficios

1. **Organización mejorada**: Los patches se organizan por chunk, facilitando el análisis de archivos grandes
2. **Consistencia**: Misma estructura que `Detections/` y `Composite/`
3. **Escalabilidad**: Permite procesar archivos muy grandes sin mezclar patches de diferentes chunks
4. **Compatibilidad**: No rompe el código existente

## Uso

La funcionalidad se activa automáticamente cuando se usa el modo chunking:

```python
# En el pipeline principal, cuando chunk_samples > 0
run_pipeline(chunk_samples=1000000)  # Activa modo chunking
```

Los patches se guardarán automáticamente en la estructura:

```
Results/ObjectDetection/resnet50/
├── Patches/
│   ├── archivo_fits_chunk000/
│   │   ├── patch_slice0_band256.png
│   │   └── patch_slice1_band256.png
│   └── archivo_fits_chunk001/
│       ├── patch_slice0_band256.png
│       └── patch_slice1_band256.png
├── Composite/
│   ├── archivo_fits_chunk000/
│   └── archivo_fits_chunk001/
└── Detections/
    ├── archivo_fits_chunk000/
    └── archivo_fits_chunk001/
```

## Archivos Modificados

1. `DRAFTS/core/pipeline.py` - Creación de carpetas y paso de parámetros
2. `DRAFTS/detection/pipeline_utils.py` - Lógica de rutas condicionales
3. `tests/test_patches_chunked_structure.py` - Script de verificación (nuevo)

## Conclusión

La implementación es exitosa y mantiene la compatibilidad hacia atrás. Los patches candidates ahora se organizan correctamente por chunk, siguiendo la misma estructura que los otros tipos de plots en el pipeline.
