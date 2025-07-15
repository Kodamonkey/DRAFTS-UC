"""
RESUMEN COMPLETO: ¿QUÉ HACE SLICE_LEN Y POR QUÉ AFECTA TUS RESULTADOS?

Este documento explica de manera clara y práctica el parámetro SLICE_LEN
y por qué cambiar este valor modifica tus resultados de detección.
"""

## 🎯 RESUMEN EJECUTIVO

**SLICE_LEN** es el parámetro más importante que controla la resolución temporal
en tu pipeline de detección de FRBs. Define cuántas muestras temporales
se procesan juntas en cada "ventana" de análisis.

**Valor actual**: 64 muestras
**Impacto**: Divide cada archivo en ventanas de 64 muestras que se analizan independientemente

---

## 🔬 QUÉ HACE EXACTAMENTE SLICE_LEN

### 1. **División Temporal del Archivo**

```python
# En pipeline.py, línea ~289
slice_len, time_slice = _slice_parameters(width_total, config.SLICE_LEN)
# Si archivo = 1024 muestras y SLICE_LEN = 64:
# → time_slice = 1024 // 64 = 16 slices
```

Tu archivo se divide en múltiples "slices" (rebanadas temporales):

- **SLICE_LEN = 32**: Archivo 1024 → 32 slices de 32 muestras cada uno
- **SLICE_LEN = 64**: Archivo 1024 → 16 slices de 64 muestras cada uno
- **SLICE_LEN = 128**: Archivo 1024 → 8 slices de 128 muestras cada uno

### 2. **Extracción de Datos por Slice**

```python
# En pipeline.py, línea ~331
slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
waterfall_block = data[j * slice_len : (j + 1) * slice_len]
```

Para cada slice j:

- Se extrae una región DM vs. tiempo de tamaño `[n_bandas, n_DMs, SLICE_LEN]`
- Se extrae un waterfall de tamaño `[SLICE_LEN, n_frecuencias]`

### 3. **Conversión a Imagen 512x512 para CNN**

```python
# En image_utils.py
img = cv2.resize(img, (512, 512))
```

Cada slice se redimensiona a 512x512 pixels para el modelo CNN:

- **SLICE_LEN muestras** → **512 pixels temporales**
- **Resolución temporal por pixel** = SLICE_LEN / 512

### 4. **Conversión Pixel → Coordenadas Físicas**

```python
# En astro_conversions.py, línea ~22
scale_time = slice_len / 512.0
sample_off = px * scale_time
t_seconds = sample_off * config.TIME_RESO * config.DOWN_TIME_RATE
```

Cuando el modelo CNN detecta algo en pixel (px, py):

- **Posición temporal** = px × (SLICE_LEN / 512) × TIME_RESO
- **DM** = py × (DM_range / 512) + DM_min

---

## 🎯 POR QUÉ SLICE_LEN AFECTA TUS RESULTADOS

### **1. Resolución Temporal**

| SLICE_LEN | Resolución/pixel      | Bueno para                  |
| --------- | --------------------- | --------------------------- |
| 32        | 0.0625 muestras/pixel | Señales muy cortas (< 10ms) |
| 64        | 0.125 muestras/pixel  | FRBs típicos (10-50ms)      |
| 128       | 0.25 muestras/pixel   | Señales largas (> 50ms)     |

### **2. Número de Oportunidades de Detección**

- **SLICE_LEN pequeño** → Más slices → Más oportunidades de detectar señales
- **SLICE_LEN grande** → Menos slices → Menos oportunidades, pero menos fragmentación

### **3. Contexto Temporal**

- **SLICE_LEN pequeño** → Ventana temporal corta → Puede fragmentar señales largas
- **SLICE_LEN grande** → Ventana temporal larga → Mejor contexto, pero menor resolución

### **4. Precisión de Localización**

Con TIME_RESO = 0.001s (1ms):

- **SLICE_LEN = 32**: Precisión = 32/512 × 1ms = **0.0625ms por pixel**
- **SLICE_LEN = 64**: Precisión = 64/512 × 1ms = **0.125ms por pixel**
- **SLICE_LEN = 128**: Precisión = 128/512 × 1ms = **0.25ms por pixel**

---

## 📊 EJEMPLO PRÁCTICO

### Archivo de 2048 muestras (2.048 segundos):

| SLICE_LEN | N_Slices | Duración/Slice | Precisión/Pixel | Tipo de Señales     |
| --------- | -------- | -------------- | --------------- | ------------------- |
| 32        | 64       | 32ms           | 0.0625ms        | Pulsos muy cortos   |
| 64        | 32       | 64ms           | 0.125ms         | **FRBs típicos** ⭐ |
| 128       | 16       | 128ms          | 0.25ms          | Señales dispersas   |
| 256       | 8        | 256ms          | 0.5ms           | Eventos muy largos  |

### Si detectas un FRB en pixel (256, 256):

**Con SLICE_LEN = 32:**

- Posición temporal = 256 × (32/512) × 1ms = **16ms** dentro del slice
- Precisión alta, ideal para pulsos cortos

**Con SLICE_LEN = 64:**

- Posición temporal = 256 × (64/512) × 1ms = **32ms** dentro del slice
- Balance óptimo para FRBs típicos

**Con SLICE_LEN = 128:**

- Posición temporal = 256 × (128/512) × 1ms = **64ms** dentro del slice
- Mejor para señales largas o muy dispersas

---

## 🎯 RECOMENDACIONES ESPECÍFICAS

### **Para Señales Cortas (< 20ms)**

```python
SLICE_LEN: int = 32
```

- ✅ Alta resolución temporal
- ✅ Mejor localización de pulsos cortos
- ❌ Más slices → más tiempo de procesamiento

### **Para FRBs Típicos (20-100ms)** ⭐ **RECOMENDADO**

```python
SLICE_LEN: int = 64  # Tu configuración actual
```

- ✅ Balance óptimo resolución/contexto
- ✅ Compatible con modelo entrenado
- ✅ Buena para mayoría de casos

### **Para Señales Largas (> 100ms)**

```python
SLICE_LEN: int = 128
```

- ✅ Mejor contexto temporal
- ✅ Menos fragmentación de señales largas
- ❌ Menor resolución temporal

---

## 🚀 CÓMO EXPERIMENTAR

### **Opción 1: Manual**

1. Editar `DRAFTS/config.py`:
   ```python
   SLICE_LEN: int = 32  # Probar valor nuevo
   ```
2. Ejecutar pipeline: `python main.py`
3. Comparar resultados en CSV

### **Opción 2: Automático**

```bash
python experiment_slice_len.py
```

Este script probará automáticamente [32, 64, 128] y comparará resultados.

---

## 🔍 MÉTRICAS PARA EVALUAR

1. **Número de candidatos detectados** (más no siempre es mejor)
2. **SNR promedio de detecciones** (mayor SNR = mejor calidad)
3. **Distribución temporal de detecciones** (¿coherente con señales esperadas?)
4. **Tiempo de procesamiento** (SLICE_LEN pequeño = más lento)

---

## 💡 CONCLUSIÓN

**SLICE_LEN = 64** es una buena configuración general, pero:

- Si buscas **pulsos muy cortos**: prueba **SLICE_LEN = 32**
- Si buscas **señales muy dispersas**: prueba **SLICE_LEN = 128**
- Si no estás seguro: **experimenta con ambos** y compara resultados

El parámetro SLICE_LEN es fundamental porque define la "lupa temporal"
con la que tu pipeline examina los datos. ¡Experimentar con diferentes
valores puede revelar señales que antes pasaban desapercibidas!
