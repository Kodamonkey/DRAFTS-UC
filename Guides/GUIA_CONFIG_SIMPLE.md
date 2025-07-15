# Guía de Configuración Simplificada para Astrónomos

## Resumen

El sistema de configuración se ha simplificado drásticamente. **Solo necesitas modificar un archivo pequeño** con los parámetros esenciales.

## Archivos de Configuración

### 🟢 `config_simple.py` - **El único archivo que debes modificar**

```python
# Solo estos 5 parámetros:
DATA_DIR = Path("./Data")           # Carpeta con tus archivos .fits/.fil
RESULTS_DIR = Path("./Results")     # Carpeta para resultados
FRB_TARGETS = ["B0355+54"]          # Lista de archivos a procesar
DM_min = 0                          # DM mínimo de búsqueda
DM_max = 1024                       # DM máximo de búsqueda
DET_PROB = 0.1                      # Sensibilidad (0.05=alta, 0.2=baja)
```

### 🔄 `config_auto.py` - **Configuración automática** (no tocar)

Este archivo toma tu configuración simple y la expande automáticamente a todos los parámetros técnicos necesarios.

### 📚 `config.py` - **Configuración completa** (para referencia)

El archivo original completo, ahora solo para consulta o configuración avanzada.

## Cómo Usar

### 1. Configuración Básica

```python
# Edita solo config_simple.py:
DATA_DIR = Path("./MisDatos")
FRB_TARGETS = ["FRB20180301", "B0355+54"]
DM_min = 50
DM_max = 2000
DET_PROB = 0.1
```

### 2. Ejecutar el Pipeline

```python
# En tu script principal:
from DRAFTS.config_auto import *  # Carga configuración completa automáticamente
# ... resto del pipeline
```

### 3. Ver tu Configuración

```python
from DRAFTS.config_auto import print_user_config
print_user_config()  # Muestra solo tus parámetros
```

## Parámetros Explicados

### `DM_min` y `DM_max`

- Define el **rango completo de búsqueda** de Dispersion Measure
- El sistema buscará candidatos en todo este rango
- Los plots se centrarán automáticamente en los candidatos encontrados

### `DET_PROB` (Sensibilidad)

- `0.05`: Muy sensible (detecta más candidatos, más falsos positivos)
- `0.10`: Balanceado (recomendado para la mayoría de casos)
- `0.20`: Conservador (menos candidatos, menos falsos positivos)

### `FRB_TARGETS`

- Lista de nombres de archivos (sin extensión)
- El sistema buscará automáticamente `.fits` o `.fil`
- Ejemplo: `["FRB20180301", "B0355+54"]`

## Visualización Automática

### ✅ Lo que se hace automáticamente:

- **Zoom en candidatos**: Los plots se centran en los DM detectados
- **Rango dinámico**: Se ajusta automáticamente ±30% del DM óptimo
- **Múltiples bandas**: Full/Low/High band automáticas
- **Slice temporal**: Se calcula automáticamente desde metadatos

### ❌ Lo que NO necesitas configurar:

- Parámetros de visualización DM dinámicos
- Configuración de slices temporales
- Parámetros de chunking de memoria
- Configuración de modelos
- Configuración de RFI

## Casos de Uso Típicos

### Búsqueda Exploratoria

```python
DM_min = 0
DM_max = 3000
DET_PROB = 0.1  # Balanceado
```

### Alta Precisión

```python
DM_min = 100
DM_max = 1000
DET_PROB = 0.05  # Muy sensible
```

### Procesamiento Rápido

```python
DM_min = 200
DM_max = 800
DET_PROB = 0.15  # Menos sensible
```

## Migración desde Configuración Anterior

Si tenías el `config.py` anterior:

1. Copia tus valores de `DM_min`, `DM_max`, `DET_PROB` a `config_simple.py`
2. Copia las rutas `DATA_DIR`, `RESULTS_DIR`, `FRB_TARGETS`
3. Ignora todos los demás parámetros (ahora son automáticos)
4. Cambia `from DRAFTS.config import *` por `from DRAFTS.config_auto import *`

## Preguntas Frecuentes

**P: ¿Puedo seguir usando configuración avanzada?**
R: Sí, puedes modificar `config.py` directamente para control total.

**P: ¿Los plots siguen siendo dinámicos?**
R: Sí, se centran automáticamente en candidatos detectados.

**P: ¿Cómo cambio parámetros de RFI o chunking?**
R: Para configuración avanzada, edita `config.py` o `config_auto.py`.

**P: ¿El rango DM afecta la visualización?**
R: No, el rango DM es para búsqueda. Los plots se ajustan automáticamente a los candidatos encontrados.
