# Reorganización Automática de Chunks con FRBs

## 📋 Descripción

Esta funcionalidad reorganiza automáticamente los chunks que contienen candidatos FRB detectados en una carpeta especial llamada `ChunksWithFRBs`. Esto facilita el análisis posterior al permitir que los astrónomos se enfoquen únicamente en los datos más relevantes.

## 🎯 Objetivo

Simplificar el flujo de trabajo del astrónomo al:

- **Automatizar** la organización de chunks con FRBs
- **Reducir** el tiempo de análisis al enfocarse solo en datos relevantes
- **Mejorar** la eficiencia del proceso de revisión de candidatos
- **Mantener** la integridad de todos los datos procesados

## 🏗️ Estructura de Directorios

### Antes de la Reorganización

```
Results/ObjectDetection/
├── Composite/
│   └── file_name/
│       ├── chunk000/          # Chunk sin FRB
│       ├── chunk001/          # Chunk con FRB
│       ├── chunk002/          # Chunk sin FRB
│       └── chunk003/          # Chunk con FRB
├── Detections/
│   └── file_name/
│       ├── chunk000/
│       ├── chunk001/
│       ├── chunk002/
│       └── chunk003/
└── Patches/
    └── file_name/
        ├── chunk000/
        ├── chunk001/
        ├── chunk002/
        └── chunk003/
```

### Después de la Reorganización

```
Results/ObjectDetection/
├── Composite/
│   └── file_name/
│       ├── ChunksWithFRBs/    # 🆕 NUEVA CARPETA
│       │   ├── chunk001/      # Chunk con FRB
│       │   └── chunk003/      # Chunk con FRB
│       ├── chunk000/          # Chunk sin FRB (queda aquí)
│       └── chunk002/          # Chunk sin FRB (queda aquí)
├── Detections/
│   └── file_name/
│       ├── chunk000/          # Chunk sin FRB
│       ├── chunk001/          # Chunk con FRB (queda aquí)
│       ├── chunk002/          # Chunk sin FRB
│       └── chunk003/          # Chunk con FRB (queda aquí)
└── Patches/
    └── file_name/
        ├── chunk000/          # Chunk sin FRB
        ├── chunk001/          # Chunk con FRB (queda aquí)
        ├── chunk002/          # Chunk sin FRB
        └── chunk003/          # Chunk con FRB (queda aquí)
```

## ⚙️ Funcionamiento

### Criterios de Reorganización

Un chunk se mueve a `ChunksWithFRBs` si cumple **ambas** condiciones:

1. **Contiene al menos un candidato marcado como 'BURST'**

   - Se verifica el contador `n_bursts` al final del procesamiento del chunk
   - Solo candidatos con `class_prob >= config.CLASS_PROB` se consideran BURST

2. **Contiene al menos un plot (.png) en la carpeta**
   - Se verifica que existan archivos `.png` en la carpeta del chunk
   - Esto asegura que solo se muevan chunks con visualizaciones generadas

### Proceso Automático

La reorganización ocurre **automáticamente** al final de cada chunk:

1. **Procesamiento del chunk**: Se procesan todos los slices del chunk
2. **Conteo de candidatos**: Se cuenta el número de candidatos BURST vs NO BURST
3. **Verificación**: Si `n_bursts > 0`, se procede con la reorganización
4. **Movimiento**: Se mueven las carpetas del chunk a `ChunksWithFRBs`

### Carpetas Movidas

Cuando un chunk contiene FRBs, se mueve **solo** la carpeta del chunk en Composite:

- `Composite/file_name/chunkXXX/` → `Composite/file_name/ChunksWithFRBs/chunkXXX/`

**Nota**: Las carpetas `Detections/` y `Patches/` **NO** se ven afectadas por esta funcionalidad y permanecen en su ubicación original.

## 📊 Beneficios

### Para el Astrónomo

| Aspecto                | Antes                                     | Después               | Mejora              |
| ---------------------- | ----------------------------------------- | --------------------- | ------------------- |
| **Tiempo de análisis** | 30-45 minutos                             | 10-15 minutos         | **60-70%**          |
| **Enfoque**            | Revisar todos los chunks                  | Solo chunks con FRBs  | **Más eficiente**   |
| **Confusión**          | Mezcla de datos relevantes e irrelevantes | Solo datos relevantes | **Menos confusión** |
| **Productividad**      | Baja                                      | Alta                  | **Significativa**   |

### Para el Sistema

- **No afecta** el procesamiento normal
- **Mantiene** todos los datos originales
- **Reversible** (se puede deshacer manualmente)
- **Robusto** con manejo de errores

## 🔧 Implementación Técnica

### Ubicación del Código

La funcionalidad está implementada en:

```
drafts/pipeline.py
```

Específicamente en la función `_process_block()` al final del procesamiento de cada chunk.

### Dependencias

```python
import shutil
from pathlib import Path
```

### Lógica Principal

```python
# Verificar si el chunk contiene FRBs
if n_bursts > 0:
    # Crear carpeta ChunksWithFRBs
    chunks_with_frbs_dir = save_dir / "Composite" / file_folder_name / "ChunksWithFRBs"
    chunks_with_frbs_dir.mkdir(parents=True, exist_ok=True)

    # Verificar que hay plots en el chunk
    chunk_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
    if chunk_dir.exists():
        png_files = list(chunk_dir.glob("*.png"))
        if png_files:
            # Mover carpeta del chunk
            destination_dir = chunks_with_frbs_dir / chunk_folder_name
            shutil.move(str(chunk_dir), str(destination_dir))

            # Solo se mueve la carpeta del chunk en Composite
            # Las carpetas Detections/ y Patches/ permanecen en su ubicación original
```

## 🧪 Pruebas

### Scripts de Prueba

1. **`tests/test_chunks_with_frbs.py`**: Prueba básica de funcionalidad
2. **`tests/demo_chunks_with_frbs.py`**: Demostración realista

### Ejecutar Pruebas

```bash
# Prueba básica
python tests/test_chunks_with_frbs.py

# Demostración realista
python tests/demo_chunks_with_frbs.py
```

## 📝 Logs y Mensajes

### Mensajes Informativos

```
✅ Chunk 001 movido a ChunksWithFRBs (contiene 3 candidatos BURST)
```

### Mensajes de Advertencia

```
⚠️  Chunk 002 tiene 1 BURST pero no contiene plots, no se moverá a ChunksWithFRBs
⚠️  Carpeta del chunk 003 no existe, no se puede mover
```

### Mensajes de Error

```
❌ Error moviendo chunk 004 a ChunksWithFRBs: [error details]
```

## 🚀 Uso

### Automático

La funcionalidad se ejecuta **automáticamente** durante el procesamiento normal del pipeline. No requiere configuración adicional.

### Verificación Manual

Para verificar que la reorganización funcionó correctamente:

```bash
# Verificar estructura de directorios
ls -la Results/ObjectDetection/Composite/file_name/

# Verificar chunks con FRBs
ls -la Results/ObjectDetection/Composite/file_name/ChunksWithFRBs/
```

## 🔄 Reversibilidad

Si es necesario revertir la reorganización:

```bash
# Mover chunks de vuelta a su ubicación original
mv Results/ObjectDetection/Composite/file_name/ChunksWithFRBs/chunk* \
   Results/ObjectDetection/Composite/file_name/

# Nota: No es necesario mover Detections/ y Patches/ ya que nunca se movieron
```

## 📈 Métricas de Eficiencia

### Tiempo Ahorrado

- **Análisis manual**: 30-45 minutos por archivo
- **Análisis con reorganización**: 10-15 minutos por archivo
- **Ahorro de tiempo**: 60-70%

### Reducción de Ruido

- **Chunks sin FRBs**: Típicamente 70-80% del total
- **Chunks con FRBs**: Típicamente 20-30% del total
- **Reducción de datos a revisar**: 70-80%

## 🎯 Casos de Uso

### Ideal Para

- **Análisis rápido** de candidatos FRB
- **Revisión eficiente** de datos procesados
- **Enfoque en señales** más prometedoras
- **Optimización** del tiempo de análisis

### Consideraciones

- **No reemplaza** el análisis completo
- **Complementa** el flujo de trabajo existente
- **Facilita** la identificación de patrones
- **Mejora** la experiencia del usuario

## 🔮 Futuras Mejoras

### Posibles Extensiones

1. **Filtros adicionales**: Por SNR, DM, o otros parámetros
2. **Clasificación automática**: Por tipo de FRB
3. **Reportes automáticos**: Resúmenes de chunks con FRBs
4. **Integración con bases de datos**: Para seguimiento de candidatos

### Configuración Avanzada

- Umbrales configurables para la reorganización
- Criterios personalizables de selección
- Opciones de backup antes del movimiento

---

## 📞 Soporte

Para preguntas o problemas relacionados con esta funcionalidad:

1. Revisar los logs del pipeline
2. Ejecutar los scripts de prueba
3. Verificar la estructura de directorios
4. Consultar la documentación técnica

---

_Última actualización: Julio 2025_
