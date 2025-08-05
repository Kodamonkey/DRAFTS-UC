# Configuración de Visualización Obligatoria

## Descripción

Este sistema permite controlar la creación de plots específicos (Composite, Detections, Waterfall De-dispersed) independientemente de si se detectan candidatos FRB o no. Esto es útil para:

- **Análisis sistemático**: Generar visualizaciones de todos los slices para análisis posterior
- **Debugging**: Verificar que el procesamiento funciona correctamente en slices sin candidatos
- **Validación**: Comparar slices con y sin candidatos para entender mejor el ruido de fondo
- **Documentación**: Crear un registro visual completo del procesamiento

## Flags de Configuración

Las siguientes flags se pueden configurar en `drafts/user_config.py`:

### `ALWAYS_CREATE_COMPOSITE`

- **Tipo**: `bool`
- **Valor por defecto**: `False`
- **Descripción**: Si `True`, crea el Composite Plot en cada slice procesado, independientemente de si hay candidatos detectados
- **Uso**: Útil para análisis sistemático y debugging

### `ALWAYS_CREATE_DETECTIONS`

- **Tipo**: `bool`
- **Valor por defecto**: `False`
- **Descripción**: Si `True`, crea el Detection Plot en cada slice procesado, independientemente de si hay candidatos detectados
- **Uso**: Útil para verificar el funcionamiento del modelo de detección

### `ALWAYS_CREATE_WATERFALL_DEDISP`

- **Tipo**: `bool`
- **Valor por defecto**: `False`
- **Descripción**: Si `True`, crea el Waterfall De-dispersed en cada slice procesado, independientemente de si hay candidatos detectados
- **Uso**: Útil para análisis de dispersión y verificación de datos

## Comportamiento

### Modo Normal (Todas las flags en `False`)

- Solo se crean plots cuando se detectan candidatos FRB
- Comportamiento original del sistema
- Ahorra espacio en disco y tiempo de procesamiento

### Modo Obligatorio (Alguna flag en `True`)

- Se crean los plots correspondientes en cada slice procesado
- Si no hay candidatos, se usan valores por defecto:
  - `DM = 0.0` para waterfalls de-dispersed
  - Listas vacías para detecciones
  - Sin información de candidatos en composite

## Ejemplos de Configuración

### Ejemplo 1: Solo Composite obligatorio

```python
ALWAYS_CREATE_COMPOSITE: bool = True
ALWAYS_CREATE_DETECTIONS: bool = False
ALWAYS_CREATE_WATERFALL_DEDISP: bool = False
```

### Ejemplo 2: Todos los plots obligatorios

```python
ALWAYS_CREATE_COMPOSITE: bool = True
ALWAYS_CREATE_DETECTIONS: bool = True
ALWAYS_CREATE_WATERFALL_DEDISP: bool = True
```

### Ejemplo 3: Solo para debugging de detección

```python
ALWAYS_CREATE_COMPOSITE: bool = False
ALWAYS_CREATE_DETECTIONS: bool = True
ALWAYS_CREATE_WATERFALL_DEDISP: bool = False
```

## Consideraciones de Rendimiento

### Impacto en Tiempo de Procesamiento

- **Modo Normal**: Solo procesa slices con candidatos
- **Modo Obligatorio**: Procesa todos los slices
- **Impacto estimado**: 2-5x más tiempo dependiendo de la cantidad de slices sin candidatos

### Impacto en Almacenamiento

- **Modo Normal**: Solo guarda plots de slices con candidatos
- **Modo Obligatorio**: Guarda plots de todos los slices
- **Impacto estimado**: 3-10x más espacio en disco

### Recomendaciones

1. **Para procesamiento en lote**: Usar modo normal
2. **Para análisis detallado**: Usar modo obligatorio
3. **Para debugging**: Usar solo las flags necesarias
4. **Para validación**: Usar todas las flags

## Estructura de Archivos Generados

### Con Candidatos Detectados

```
Results/
├── Composite/
│   └── archivo_slice000.png (con información de candidatos)
├── Detections/
│   └── archivo_slice000.png (con bounding boxes)
└── waterfall_dedispersion/
    └── archivo_dm123.45_fullband.png
```

### Sin Candidatos (Modo Obligatorio)

```
Results/
├── Composite/
│   └── archivo_slice000.png (sin información de candidatos)
├── Detections/
│   └── archivo_slice000.png (sin bounding boxes)
└── waterfall_dedispersion/
    └── archivo_dm0.00_fullband.png
```

## Mensajes de Log

### Modo Normal

```
Slice 000: Sin candidatos detectados
```

### Modo Obligatorio

```
Creando plots obligatorios para slice 000 (sin candidatos)
```

## Troubleshooting

### Problema: No se crean plots obligatorios

**Solución**: Verificar que las flags estén correctamente configuradas en `user_config.py`

### Problema: Error al crear plots sin candidatos

**Solución**: Verificar que las funciones de visualización manejen correctamente valores `None` y listas vacías

### Problema: Consumo excesivo de memoria

**Solución**: Procesar archivos más pequeños o usar menos flags obligatorias

## Notas Técnicas

- Las flags se evalúan en tiempo de ejecución
- El sistema mantiene compatibilidad hacia atrás
- Los plots sin candidatos usan valores por defecto seguros
- La lógica de detección no se ve afectada por estas flags
