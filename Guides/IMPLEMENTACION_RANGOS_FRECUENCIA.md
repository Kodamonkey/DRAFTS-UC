# Implementación: Visualización de Rangos de Frecuencia por Banda

## 📋 Resumen

Se ha implementado la funcionalidad para mostrar en los gráficos del sistema DRAFTS el rango específico de frecuencias que está usando cada banda (Full/Low/High). Esta mejora proporciona mayor claridad sobre qué parte del espectro está siendo procesada en cada análisis.

## 🎯 Problema Resuelto

**Antes:** Los gráficos mostraban solo el nombre genérico de la banda (ej: "Full Band", "Low Band", "High Band") sin especificar qué frecuencias estaban siendo procesadas.

**Después:** Los gráficos ahora muestran el rango específico de frecuencias para cada banda (ej: "Full Band (1200-1500 MHz)", "Low Band (1200-1350 MHz)", "High Band (1350-1500 MHz)").

## 🔧 Implementación Técnica

### 1. Nuevas Funciones en `visualization.py`

```python
def get_band_frequency_range(band_idx: int) -> Tuple[float, float]:
    """Calcula el rango de frecuencias (min, max) para una banda específica."""

def get_band_name_with_freq_range(band_idx: int, band_name: str) -> str:
    """Retorna el nombre de la banda con información del rango de frecuencias."""
```

### 2. Lógica de División de Bandas

- **Full Band (índice 0)**: Rango completo de frecuencias disponibles
- **Low Band (índice 1)**: Desde frecuencia mínima hasta canal medio
- **High Band (índice 2)**: Desde canal medio hasta frecuencia máxima

### 3. Integración en el Pipeline

Se actualizaron las siguientes funciones para incluir el parámetro `band_idx`:

- `save_plot()` - Gráficos de detección
- `save_patch_plot()` - Gráficos de patches de candidatos
- `save_slice_summary()` - Gráficos compuestos
- `save_detection_plot()` - Gráficos de detección base

## 📊 Tipos de Gráficos Afectados

### 1. Gráficos de Detección (`/Detections/`)

- **Antes:** `"FRB20180301_0001 - Full Band"`
- **Después:** `"FRB20180301_0001 - Full Band (1200-1500 MHz)"`

### 2. Gráficos Compuestos (`/Composite/`)

- **Antes:** `"Composite Summary: FRB20180301_0001 - Full Band - Slice 6"`
- **Después:** `"Composite Summary: FRB20180301_0001 - Full Band (1200-1500 MHz) - Slice 6"`

### 3. Gráficos de Patches (`/Patches/`)

- **Antes:** `"Candidate Patch - Full Band"`
- **Después:** `"Candidate Patch - Full Band (1200-1500 MHz)"`

## 🧪 Validación

### Tests Implementados

```bash
# Test de las funciones básicas
python test_frequency_display.py

# Demostración visual
python demo_frequency_display.py
```

### Casos de Prueba

✅ **Configuraciones típicas:** FAST (1050-1450 MHz), Arecibo (1200-1500 MHz)  
✅ **Casos extremos:** Pocas frecuencias, número impar de canales  
✅ **Diferentes factores de reducción:** DOWN_FREQ_RATE = 1, 2, 4, 8  
✅ **Todas las bandas:** Full Band, Low Band, High Band

## 🎨 Ejemplos Visuales

### Para un archivo típico (1200-1500 MHz, 400 canales):

```
📡 División en bandas:
   • Full Band (1200-1500 MHz)
     - Descripción: Toda la banda de observación
     - Ancho de banda: 300.0 MHz
     - Canales: 0 a 200 (de 200 total)

   • Low Band (1200-1350 MHz)
     - Descripción: Mitad inferior del espectro
     - Ancho de banda: 150.0 MHz
     - Canales: 0 a 100 (de 200 total)

   • High Band (1350-1500 MHz)
     - Descripción: Mitad superior del espectro
     - Ancho de banda: 150.0 MHz
     - Canales: 100 a 200 (de 200 total)
```

### Títulos de Gráficos Actualizados:

```
🖼️  Títulos en gráficos de detección:
   • "FRB20180301_0001 - Full Band (1200-1500 MHz)
     Slice 6/20 | Time Resolution: 32.8 μs | DM Range: 50–180 (auto) pc cm⁻³"

   • "FRB20180301_0001 - Low Band (1200-1350 MHz)
     Slice 6/20 | Time Resolution: 32.8 μs | DM Range: 50–180 (auto) pc cm⁻³"

   • "FRB20180301_0001 - High Band (1350-1500 MHz)
     Slice 6/20 | Time Resolution: 32.8 μs | DM Range: 50–180 (auto) pc cm⁻³"
```

## 📁 Archivos Modificados

### 1. `DRAFTS/visualization.py`

- ✅ Nuevas funciones `get_band_frequency_range()` y `get_band_name_with_freq_range()`
- ✅ Parámetro `band_idx` agregado a funciones existentes
- ✅ Integración en títulos de gráficos

### 2. `DRAFTS/image_utils.py`

- ✅ Parámetro `band_idx` agregado a `save_detection_plot()`
- ✅ Uso de rangos específicos de banda en títulos

### 3. `DRAFTS/pipeline.py`

- ✅ Pasar `band_idx` a todas las llamadas de funciones de visualización
- ✅ Integración en ambos pipelines (normal y chunked)

## 🔍 Compatibilidad

### ✅ Mantiene compatibilidad con:

- Configuraciones existentes de telescopios
- Archivos de datos actuales (.fits, .fil)
- Factores de reducción variables
- Modo single-band y multi-band

### ✅ Funciona con:

- Cualquier rango de frecuencias
- Cualquier número de canales
- Cualquier factor de downsampling
- Archivos de diferentes tamaños

## 🚀 Beneficios para el Usuario

1. **Claridad Visual:** Inmediatamente visible qué frecuencias están siendo procesadas
2. **Análisis Científico:** Facilita la interpretación de resultados por banda
3. **Debugging:** Ayuda a identificar problemas específicos de frecuencia
4. **Documentación:** Los gráficos son autodocumentados con información de frecuencias
5. **Flexibilidad:** Funciona con cualquier configuración de telescopio

## 🔧 Uso

La funcionalidad se activa automáticamente cuando se ejecuta el pipeline:

```python
# Configurar multi-banda en config.py
USE_MULTI_BAND: bool = True

# Ejecutar pipeline - los gráficos mostrarán automáticamente los rangos
python main.py  # o d-center-main.py
```

## 📚 Notas Técnicas

### Cálculo de Rangos

- Los rangos se calculan dinámicamente basándose en `config.FREQ`
- Se respeta el factor de reducción `DOWN_FREQ_RATE`
- La división siempre es 50/50 en el canal medio

### Rendimiento

- ✅ Impacto mínimo en rendimiento (solo cálculos de metadatos)
- ✅ No afecta la velocidad de procesamiento de datos
- ✅ Cálculos realizados solo cuando se generan gráficos

## 🎯 Próximos Pasos (Opcional)

1. **Bandas Personalizables:** Permitir al usuario definir divisiones custom
2. **Información Adicional:** Mostrar resolución espectral por banda
3. **Colores por Banda:** Esquemas de colores únicos para cada banda
4. **Waterfall Annotations:** Marcar rangos en gráficos de waterfall

---

**Implementado por:** GitHub Copilot  
**Fecha:** Julio 2025  
**Versión:** DRAFTS-FE v1.0+  
**Status:** ✅ Completado y Validado
