# Guía de Limpieza de RFI en DRAFTS

## 🎯 Resumen

El sistema de limpieza de RFI (Radio Frequency Interference) de DRAFTS implementa múltiples técnicas avanzadas para mejorar la detección de FRBs eliminando interferencias terrestres y espaciales.

## 📋 Técnicas Implementadas

### 1. **Enmascarado de Canales de Frecuencia**

- **Propósito**: Detecta y elimina canales de frecuencia persistentemente contaminados
- **Métodos**: MAD (Median Absolute Deviation), Desviación Estándar, Curtosis
- **Configuración**: `RFI_FREQ_SIGMA_THRESH = 5.0`

### 2. **Enmascarado Temporal**

- **Propósito**: Detecta y elimina muestras temporales con RFI de banda ancha
- **Métodos**: MAD, Desviación Estándar, Análisis de Outliers
- **Configuración**: `RFI_TIME_SIGMA_THRESH = 5.0`

### 3. **Filtro Zero-DM**

- **Propósito**: Elimina señales no dispersas (RFI terrestre)
- **Principio**: Resta el perfil temporal promedio, preservando señales dispersas
- **Configuración**: `RFI_ZERO_DM_SIGMA_THRESH = 4.0`

### 4. **Filtrado de Impulsos**

- **Propósito**: Elimina RFI impulsivo de corta duración
- **Método**: Filtro mediano 2D para detectar y suprimir impulsos
- **Configuración**: `RFI_IMPULSE_SIGMA_THRESH = 6.0`

### 5. **Análisis de Polarización**

- **Propósito**: Usa características de polarización para identificar RFI
- **Principio**: RFI terrestre suele tener polarización diferente a señales astrofísicas
- **Configuración**: `RFI_POLARIZATION_THRESH = 0.8`

## ⚙️ Configuración

### Archivo `config.py`

```python
# Configuración de Mitigación de RFI
RFI_FREQ_SIGMA_THRESH = 5.0      # Umbral sigma para enmascarado de canales
RFI_TIME_SIGMA_THRESH = 5.0      # Umbral sigma para enmascarado temporal
RFI_ZERO_DM_SIGMA_THRESH = 4.0   # Umbral sigma para filtro Zero-DM
RFI_IMPULSE_SIGMA_THRESH = 6.0   # Umbral sigma para filtrado de impulsos
RFI_POLARIZATION_THRESH = 0.8    # Umbral para filtrado de polarización
RFI_ENABLE_ALL_FILTERS = True    # Habilita todos los filtros de RFI
RFI_INTERPOLATE_MASKED = True    # Interpola valores enmascarados
RFI_SAVE_DIAGNOSTICS = True      # Guarda gráficos de diagnóstico
RFI_CHANNEL_DETECTION_METHOD = "mad"  # Método para detectar canales malos
RFI_TIME_DETECTION_METHOD = "mad"     # Método para detectar muestras temporales malas
```

### Parámetros Recomendados por Escenario

#### 🏠 **Observatorio Urbano (Alto RFI)**

```python
RFI_FREQ_SIGMA_THRESH = 3.0      # Más agresivo
RFI_TIME_SIGMA_THRESH = 3.0      # Más agresivo
RFI_ZERO_DM_SIGMA_THRESH = 3.0   # Más agresivo
RFI_IMPULSE_SIGMA_THRESH = 4.0   # Más agresivo
```

#### 🏔️ **Observatorio Remoto (Bajo RFI)**

```python
RFI_FREQ_SIGMA_THRESH = 6.0      # Menos agresivo
RFI_TIME_SIGMA_THRESH = 6.0      # Menos agresivo
RFI_ZERO_DM_SIGMA_THRESH = 5.0   # Menos agresivo
RFI_IMPULSE_SIGMA_THRESH = 8.0   # Menos agresivo
```

## 🚀 Uso Básico

### Integración Automática en Pipeline

```python
# En tu script principal
from DRAFTS.pipeline import apply_rfi_cleaning

# Cargar datos
waterfall = load_your_data()  # (tiempo, frecuencia/DM)
stokes_v = load_polarization_data()  # Opcional

# Aplicar limpieza automática
cleaned_waterfall, rfi_stats = apply_rfi_cleaning(
    waterfall,
    stokes_v=stokes_v,
    output_dir=Path("./results")
)

# Continuar con detección de FRBs
# ...
```

### Uso Manual Avanzado

```python
from DRAFTS.rfi_mitigation import RFIMitigator

# Configurar mitigador
rfi_mitigator = RFIMitigator(
    freq_sigma_thresh=5.0,
    time_sigma_thresh=5.0,
    zero_dm_sigma_thresh=4.0,
    impulse_sigma_thresh=6.0,
    polarization_thresh=0.8
)

# Aplicar limpieza paso a paso
freq_mask = rfi_mitigator.detect_bad_channels(waterfall, method="mad")
time_mask = rfi_mitigator.detect_bad_time_samples(waterfall, method="mad")
waterfall_zero_dm = rfi_mitigator.zero_dm_filter(waterfall)
waterfall_impulse = rfi_mitigator.impulse_filter(waterfall_zero_dm)

# Aplicar máscaras finales
cleaned_waterfall = rfi_mitigator.apply_masks(
    waterfall_impulse, freq_mask, time_mask, interpolate=True
)
```

## 📊 Monitoreo y Diagnóstico

### Estadísticas Automáticas

Después de la limpieza, `rfi_stats` contiene:

```python
{
    'bad_channels': 15,                    # Número de canales flagged
    'channel_fraction_flagged': 0.12,     # Fracción de canales flagged
    'bad_time_samples': 50,               # Muestras temporales flagged
    'time_fraction_flagged': 0.05,        # Fracción temporal flagged
    'zero_dm_flagged': 25,                # Muestras Zero-DM flagged
    'impulses_flagged': 120,              # Impulsos flagged
    'total_flagged_fraction': 0.08        # Fracción total flagged
}
```

### Gráficos de Diagnóstico

Si `RFI_SAVE_DIAGNOSTICS = True`, se generan automáticamente:

1. **Comparación Before/After**: Waterfalls original vs. limpio
2. **RFI Removido**: Visualización del RFI detectado
3. **Perfiles Temporales**: Comparación de perfiles promedio
4. **Espectros**: Comparación de espectros promedio
5. **Estadísticas**: Tabla con métricas de limpieza

## 🧪 Testing y Validación

### Ejecutar Tests

```bash
# Test básico de RFI
python test_rfi_integration.py

# Tests unitarios completos
python -m pytest tests/test_rfi_mitigation.py -v
```

### Crear Datos de Prueba

```python
from test_rfi_integration import create_test_data_with_rfi

# Crea datos sintéticos con FRB y RFI
waterfall, stokes_v = create_test_data_with_rfi(
    n_time=1024,
    n_freq=256,
    frb_strength=8.0,
    rfi_fraction=0.15
)
```

## 🔧 Optimización y Ajuste

### Métricas de Rendimiento

1. **Mejora en SNR**: ¿Aumentó el SNR del FRB después de limpieza?
2. **Preservación de Señal**: ¿Se mantiene la forma del pulso?
3. **Eficiencia de RFI**: ¿Se removió RFI sin afectar señales reales?
4. **Tiempo de Procesamiento**: ¿Es aceptable para procesamiento en tiempo real?

### Ajuste de Parámetros

```python
# Para ajustar umbrales, analiza las estadísticas
def tune_rfi_parameters(waterfall_samples):
    results = {}

    for thresh in [3.0, 4.0, 5.0, 6.0, 7.0]:
        rfi_mitigator = RFIMitigator(freq_sigma_thresh=thresh)
        cleaned, stats = rfi_mitigator.clean_waterfall(waterfall_samples)

        results[thresh] = {
            'flagged_fraction': stats['total_flagged_fraction'],
            'snr_improvement': calculate_snr_improvement(waterfall_samples, cleaned)
        }

    return results
```

## 🔬 Casos de Uso Específicos

### 1. **Procesamiento de Datos FAST**

```python
# Configuración optimizada para FAST
RFI_FREQ_SIGMA_THRESH = 4.0
RFI_ZERO_DM_SIGMA_THRESH = 3.5
RFI_ENABLE_ALL_FILTERS = True
```

### 2. **Procesamiento de Datos GBT**

```python
# Configuración para Green Bank Telescope
RFI_FREQ_SIGMA_THRESH = 5.0
RFI_TIME_SIGMA_THRESH = 4.0
RFI_POLARIZATION_THRESH = 0.7
```

### 3. **Búsqueda en Tiempo Real**

```python
# Configuración para procesamiento rápido
RFI_ENABLE_ALL_FILTERS = True
RFI_INTERPOLATE_MASKED = False  # Más rápido
RFI_SAVE_DIAGNOSTICS = False   # Más rápido
```

## 🎯 Mejores Prácticas

### 1. **Orden de Aplicación**

El orden recomendado de filtros es:

1. Detección de canales/muestras malas
2. Filtro Zero-DM
3. Filtrado de impulsos
4. Filtrado de polarización
5. Aplicación de máscaras

### 2. **Validación**

- Siempre verifica que las señales conocidas se preserven
- Usa datos con FRBs inyectados para validar
- Monitorea estadísticas de RFI regularmente

### 3. **Ajuste Adaptativo**

```python
# Ajusta parámetros basado en contenido de RFI
def adaptive_rfi_cleaning(waterfall):
    # Estima nivel de RFI inicial
    rfi_level = estimate_rfi_level(waterfall)

    if rfi_level > 0.2:  # Alto RFI
        thresh_factor = 0.7
    elif rfi_level < 0.05:  # Bajo RFI
        thresh_factor = 1.3
    else:
        thresh_factor = 1.0

    # Ajusta parámetros
    rfi_mitigator = RFIMitigator(
        freq_sigma_thresh=5.0 * thresh_factor,
        time_sigma_thresh=5.0 * thresh_factor,
        # ...
    )
```

## ⚠️ Limitaciones y Consideraciones

### Limitaciones

- **Señales Débiles**: Filtros agresivos pueden eliminar FRBs débiles
- **RFI Intermitente**: Difícil de detectar si no es persistente
- **Polarización Compleja**: Requiere datos de polarización completos
- **Tiempo de Procesamiento**: Puede ser lento para datos grandes

### Consideraciones

- Siempre valida con datos conocidos
- Ajusta parámetros según tu entorno específico
- Monitorea estadísticas de falsos positivos/negativos
- Considera procesamiento en paralelo para datos grandes

## 🔄 Integración con Pipeline Existente

La limpieza de RFI se integra automáticamente en el pipeline DRAFTS:

```python
# En pipeline.py, se aplica automáticamente si está habilitado
if config.RFI_ENABLE_ALL_FILTERS:
    cleaned_waterfall, rfi_stats = apply_rfi_cleaning(
        waterfall,
        stokes_v=stokes_v,
        output_dir=results_dir
    )
else:
    cleaned_waterfall = waterfall
    rfi_stats = {}
```

El sistema mantiene **compatibilidad completa** con el pipeline existente, permitiendo habilitar/deshabilitar la limpieza según necesidad.
