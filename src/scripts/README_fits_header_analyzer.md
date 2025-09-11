# Analizador de Headers FITS - DRAFTS Pipeline

## Descripción

El script `fits_header_analyzer.py` es una herramienta robusta para analizar archivos FITS/PSRFITS utilizados en el pipeline de detección de FRBs (Fast Radio Bursts). Proporciona información detallada sobre los headers, estructura del archivo, metadatos astronómicos y características técnicas.

## Características

- ✅ **Análisis completo de headers**: Primarios y de extensiones
- ✅ **Información astronómica detallada**: Coordenadas, frecuencias, polarización
- ✅ **Análisis de estructura**: Número de extensiones, tamaños, tipos de datos
- ✅ **Validaciones de integridad**: Detección de errores y warnings
- ✅ **Sistema de logging integrado**: Compatible con el pipeline DRAFTS
- ✅ **Manejo robusto de errores**: Continúa análisis aunque haya problemas
- ✅ **Múltiples formatos soportados**: FITS estándar y PSRFITS

## Uso

### Análisis de archivos específicos

```bash
# Analizar un archivo específico
python src/scripts/fits_header_analyzer.py Data/raw/B0355+54_FB_20220918.fits

# Analizar múltiples archivos
python src/scripts/fits_header_analyzer.py archivo1.fits archivo2.fits
```

### Análisis por lotes

```bash
# Analizar todos los archivos en DATA_DIR
python src/scripts/fits_header_analyzer.py --all

# Analizar archivos en FRB_TARGETS
python src/scripts/fits_header_analyzer.py --targets

# Analizar archivos en un directorio específico
python src/scripts/fits_header_analyzer.py --dir /ruta/al/directorio
```

### Opciones avanzadas

```bash
# Modo silencioso (solo reportes importantes)
python src/scripts/fits_header_analyzer.py --quiet archivo.fits

# Guardar reporte en archivo JSON
python src/scripts/fits_header_analyzer.py --output reporte.json archivo.fits

# Mostrar ayuda completa
python src/scripts/fits_header_analyzer.py --help
```

## Información proporcionada

### Estructura del archivo

- Tamaño total del archivo
- Número de extensiones
- Tipo de cada extensión (PrimaryHDU, BinTableHDU, etc.)
- Tamaño de datos por extensión

### Información astronómica

- Coordenadas (RA, DEC) en grados y formato HMS/DMS
- Coordenadas galácticas (longitud, latitud)
- Telescopio e instrumento utilizados
- Modo de observación
- Fecha y hora de observación
- Frecuencia central y ancho de banda

### Información técnica

- Número de canales de frecuencia
- Número de polarizaciones
- Bits por muestra
- Tiempo de muestreo
- Parámetros de calibración

### Análisis de extensiones

- Estructura detallada de datos binarios
- Tipos de datos y formas de arrays
- Headers específicos por extensión
- Estadísticas básicas de datos

## Ejemplos de salida

```
================================================================================
ANÁLISIS DE HEADER FITS
================================================================================
Archivo: B0355+54_FB_20220918.fits
Ruta: Data\raw\B0355+54_FB_20220918.fits
Tamaño: 1128.99 MB

ESTRUCTURA DEL ARCHIVO:
  • Es FITS válido: True
  • Es PSRFITS: False
  • Número de extensiones: 3

INFORMACIÓN ASTRONÓMICA:
  • TELESCOP: Effelsberg
  • OBS_MODE: SEARCH
  • RA: 03:58:53.700
  • DEC: +54:13:13.800
  • OBSFREQ: 1400.0
  • OBSBW: 400.0
  • OBSNCHAN: 512
```

## Dependencias

- **astropy**: Para manejo de archivos FITS
- **numpy**: Para análisis de datos
- **pathlib**: Para manejo de rutas (incluido en Python 3.4+)

## Integración con DRAFTS

El script está diseñado para integrarse perfectamente con el pipeline DRAFTS:

- Utiliza la configuración del proyecto (`config.py`)
- Compatible con el sistema de logging del proyecto
- Maneja las rutas de datos definidas en la configuración
- Respeta las preferencias de logging del usuario

## Manejo de errores

- **Archivos no encontrados**: Reporta claramente cuáles archivos no existen
- **Headers corruptos**: Continúa análisis y reporta problemas específicos
- **Formatos no soportados**: Identifica archivos que no son FITS válidos
- **Dependencias faltantes**: Funciona con dependencias opcionales (fitsio)

## Archivo de configuración

El script respeta la configuración definida en `src/config/user_config.py`:

- `DATA_DIR`: Directorio por defecto para archivos FITS
- `FRB_TARGETS`: Lista de archivos objetivo para análisis
- Configuraciones de logging y debugging

## Notas técnicas

- El script utiliza `astropy.io.fits` para análisis robusto
- Maneja archivos grandes eficientemente usando `memmap=True`
- Proporciona warnings para headers faltantes importantes
- Calcula parámetros derivados como tiempo total de observación
- Soporta tanto FITS estándar como formato PSRFITS extendido
