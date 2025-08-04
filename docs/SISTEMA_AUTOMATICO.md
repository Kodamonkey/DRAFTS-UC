# Sistema Automático de Parámetros - DRAFTS Pipeline

## 🎯 Resumen

El sistema automático de parámetros elimina la necesidad de configurar manualmente `chunk_samples` y otros parámetros técnicos. Ahora solo necesitas configurar **`SLICE_DURATION_MS`** en `user_config.py` y el sistema calcula automáticamente todos los demás parámetros optimizados.

## 🚀 Uso Básico

### Configuración Mínima

En `user_config.py`, solo necesitas configurar:

```python
# Duración de cada slice temporal (milisegundos)
SLICE_DURATION_MS: float = 64.0
```

### Ejecución

```bash
# Modo automático (recomendado)
python main.py

# O explícitamente
python main.py --chunk-samples 0
```

## 🔧 Cómo Funciona

### 1. Cálculo de SLICE_LEN

El sistema calcula automáticamente `SLICE_LEN` basado en:

```python
# Fórmula: SLICE_LEN = round(SLICE_DURATION_MS / (TIME_RESO × DOWN_TIME_RATE × 1000))
slice_len = round(64.0 / (0.001 * 1 * 1000))  # Ejemplo: 64 muestras
```

### 2. Cálculo de chunk_samples

El sistema optimiza `chunk_samples` considerando:

- **Memoria disponible**: Máximo 25% de RAM disponible
- **Eficiencia**: 200-300 slices por chunk
- **Duración**: 30-60 segundos por chunk
- **Fragmentación**: Múltiplo exacto de `slice_len`

### 3. Validación Automática

Todos los parámetros se validan automáticamente:

- Límites mínimos y máximos
- Consistencia entre parámetros
- Compatibilidad con el archivo de datos

## 📊 Configuraciones Típicas

| Caso de Uso     | SLICE_DURATION_MS | Descripción                               |
| --------------- | ----------------- | ----------------------------------------- |
| FRB rápidos     | 32.0 ms           | Slices cortos para pulsos muy rápidos     |
| FRB general     | 64.0 ms           | Balance entre sensibilidad y velocidad    |
| Pulsos largos   | 128.0 ms          | Slices más largos para pulsos extendidos  |
| Señales débiles | 256.0 ms          | Mayor integración temporal para mejor SNR |

## 💾 Optimización de Memoria

El sistema considera automáticamente:

- **Memoria RAM disponible** del sistema
- **Tamaño del archivo** de datos
- **Resolución temporal** y frecuencial
- **Factor de decimado** aplicado
- **Número de canales** de frecuencia

### Estrategias de Optimización

1. **Basado en slices**: ~250 slices por chunk
2. **Basado en duración**: 45 segundos por chunk
3. **Basado en memoria**: 25% de RAM disponible
4. **Límites prácticos**: 50-1000 slices por chunk

## 📝 Logs Informativos

El sistema genera logs detallados:

```
✅ Parámetros calculados automáticamente:
   • Slice: 512 muestras (64.0 ms)
   • Chunk: 128,000 muestras (45.2s)
   • Slices por chunk: 250
   • Total estimado: 78 chunks, 19,531 slices

🔧 Archivo de alta resolución temporal
   TIME_RESO: 0.0001s
   FREQ_RESO: 1024 canales
   FILE_LENG: 10,000,000 muestras
   DOWN_TIME_RATE: 1
   DOWN_FREQ_RATE: 1
```

## 🗂️ Optimización de Carpetas

### Problema Resuelto

**ANTES**: Se creaban carpetas vacías para Composite, Detections y Patches incluso cuando no había candidatos en un chunk.

**DESPUÉS**: Las carpetas solo se crean cuando realmente se van a generar plots.

### Tipos de Carpetas Optimizadas

| Carpeta                   | Comportamiento          | Condición de Creación         |
| ------------------------- | ----------------------- | ----------------------------- |
| `Composite/`              | Solo si hay candidatos  | `slice_has_candidates = True` |
| `Detections/`             | Solo si hay detecciones | `len(top_conf) > 0`           |
| `Patches/`                | Solo si hay patches     | `first_patch is not None`     |
| `waterfall_dispersion/`   | Solo si hay datos       | `waterfall_block.size > 0`    |
| `waterfall_dedispersion/` | Solo si hay candidatos  | `slice_has_candidates = True` |

### Beneficios

- **🗂️ Mejor organización**: Solo carpetas con contenido
- **💾 Ahorro de espacio**: No hay carpetas vacías
- **⚡ Mejor rendimiento**: Menos operaciones de I/O
- **🔍 Fácil navegación**: Encontrar resultados más rápido
- **🧹 Limpieza automática**: No hay que limpiar carpetas vacías

### Estructura Optimizada

```
Results/ObjectDetection/
├── Composite/           # Solo si hay candidatos
│   └── [archivo]/
│       └── chunk[XXX]/   # Solo si hay plots
├── Detections/          # Solo si hay detecciones
│   └── [archivo]/
│       └── chunk[XXX]/   # Solo si hay plots
├── Patches/             # Solo si hay patches
│   └── [archivo]/
│       └── chunk[XXX]/   # Solo si hay plots
├── waterfall_dispersion/    # Solo si hay datos
│   └── [archivo]/
│       └── chunk[XXX]/      # Solo si hay plots
└── waterfall_dedispersion/  # Solo si hay candidatos
    └── [archivo]/
        └── chunk[XXX]/      # Solo si hay plots
```

## 🔄 Compatibilidad

### Modo Automático (Recomendado)

```python
run_pipeline(chunk_samples=0)  # Cálculo automático
```

### Modo Manual (Legacy)

```python
run_pipeline(chunk_samples=2_097_152)  # Valor fijo
```

## 🧪 Pruebas

### Script de Prueba

```bash
python tests/test_automatic_parameters.py
```

### Ejemplo de Uso

```bash
python tests/ejemplo_uso_automatico.py
```

## ⚙️ Configuración Avanzada

### Límites del Sistema

En `config.py` puedes ajustar los límites:

```python
# Configuraciones de slice avanzadas (sistema)
SLICE_LEN_MIN: int = 32                     # Límite inferior
SLICE_LEN_MAX: int = 2048                   # Límite superior
```

### Parámetros de Memoria

```python
# Porcentaje de memoria a usar por chunk
max_memory_per_chunk_gb = available_memory_gb * 0.25

# Mínimo y máximo de slices por chunk
min_chunk_samples = slice_len * 50   # Mínimo 50 slices
max_chunk_samples = slice_len * 1000 # Máximo 1000 slices
```

## 🎉 Ventajas

### Para el Usuario

- ✅ **Configuración simple**: Solo `SLICE_DURATION_MS`
- ✅ **Optimización automática**: Memoria y rendimiento
- ✅ **Validación automática**: Sin errores de configuración
- ✅ **Logs informativos**: Proceso transparente

### Para el Sistema

- ✅ **Eficiencia de memoria**: Uso óptimo de RAM
- ✅ **Rendimiento optimizado**: Chunks balanceados
- ✅ **Compatibilidad**: Funciona con cualquier archivo
- ✅ **Escalabilidad**: Se adapta al hardware disponible

## 🚨 Solución de Problemas

### Error: "Parámetros calculados inválidos"

- Verificar que `TIME_RESO` > 0
- Verificar que `FILE_LENG` > 0
- Ajustar `SLICE_DURATION_MS` si es muy extremo

### Error: "No se pudo obtener información de memoria"

- Instalar `psutil`: `pip install psutil`
- El sistema usará valores por defecto

### Chunk muy grande/pequeño

- Ajustar `SLICE_DURATION_MS`
- Verificar configuración de memoria del sistema
- Revisar logs para entender la decisión del sistema

## 📚 Referencias

- [user_config.py](../drafts/user_config.py) - Configuración del usuario
- [slice_len_calculator.py](../drafts/preprocessing/slice_len_calculator.py) - Lógica de cálculo
- [pipeline.py](../drafts/pipeline.py) - Integración en el pipeline
- [main.py](../main.py) - Punto de entrada

---

**¡El sistema automático hace que tu pipeline sea mucho más fácil de usar y optimizado!** 🎯
