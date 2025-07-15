# ✅ CONFIGURACIÓN REORGANIZADA CON SWITCHES DE CONTROL

## 📁 **NUEVA ESTRUCTURA DEL CONFIG.PY**

### 🎛️ **1. SWITCHES DE CONTROL** (Líneas 27-37)

```python
# --- Control de Slice Temporal ---
SLICE_LEN_AUTO: bool = False                # True = automático, False = manual
SLICE_LEN_INTELLIGENT: bool = False         # True = análisis inteligente, False = usar SLICE_LEN fijo
SLICE_LEN_OVERRIDE_MANUAL: bool = False     # True = sistema anula manual, False = respetar manual

# --- Control de Rango DM Dinámico ---
DM_DYNAMIC_RANGE_ENABLE: bool = False       # True = zoom automático, False = rango fijo
DM_RANGE_ADAPTIVE: bool = False             # True = adaptar según confianza, False = factor fijo

# --- Control de RFI ---
RFI_ENABLE_ALL_FILTERS: bool = False        # True = todos los filtros, False = solo básicos
RFI_INTERPOLATE_MASKED: bool = False        # True = interpolar valores, False = mantener enmascarados
RFI_SAVE_DIAGNOSTICS: bool = False          # True = guardar gráficos, False = no guardar
```

### 🎯 **2. CONFIGURACIÓN PRINCIPAL** (Líneas 42-59)

Variables básicas que siempre se configuran:

```python
DATA_DIR, RESULTS_DIR, FRB_TARGETS
DM_min, DM_max
DET_PROB, CLASS_PROB, SNR_THRESH
USE_MULTI_BAND, ENABLE_CHUNK_PROCESSING, MAX_SAMPLES_LIMIT
```

### 🔧 **3. CONFIGURACIÓN MANUAL** (Líneas 64-84) - **LA SECCIÓN CLAVE**

**Variables que modificas cuando switches = False:**

#### **Slice Temporal Manual:**

```python
SLICE_LEN: int = 32                         # Valor manual de slice temporal
SLICE_DURATION_SECONDS: float = 0.032      # Duración deseada por slice
SLICE_LEN_MIN: int = 0                      # Valor mínimo de SLICE_LEN
SLICE_LEN_MAX: int = 1024                   # Valor máximo de SLICE_LEN
```

#### **Rango DM Manual:**

```python
DM_RANGE_FACTOR: float = 0.3                # Factor de rango (±30%)
DM_PLOT_MARGIN_FACTOR: float = 0.25         # Margen adicional
DM_RANGE_MIN_WIDTH: float = 80.0            # Ancho mínimo del rango DM
DM_RANGE_MAX_WIDTH: float = 300.0           # Ancho máximo del rango DM
DM_PLOT_MIN_RANGE: float = 120.0            # Rango mínimo del plot
DM_PLOT_MAX_RANGE: float = 400.0            # Rango máximo del plot
DM_PLOT_DEFAULT_RANGE: float = 250.0        # Rango por defecto
```

#### **RFI Manual:**

```python
RFI_FREQ_SIGMA_THRESH = 5.0                # Umbral sigma para canales
RFI_TIME_SIGMA_THRESH = 5.0                # Umbral sigma temporal
RFI_ZERO_DM_SIGMA_THRESH = 4.0             # Umbral sigma Zero-DM
RFI_IMPULSE_SIGMA_THRESH = 6.0             # Umbral sigma impulsos
RFI_POLARIZATION_THRESH = 0.8              # Umbral polarización
RFI_CHANNEL_DETECTION_METHOD = "mad"        # Método detección canales
RFI_TIME_DETECTION_METHOD = "mad"           # Método detección temporal
```

### 🤖 **4. CONFIGURACIÓN AUTOMÁTICA** (Líneas 89+)

Variables que se configuran automáticamente:

- Metadatos del archivo (FREQ, FREQ_RESO, TIME_RESO, FILE_LENG)
- Parámetros de decimación
- Configuración de SNR y visualización
- Configuración de chunking
- Modelos y sistema

## 🎯 **CÓMO USAR LA NUEVA ORGANIZACIÓN**

### **Para MODO MANUAL** (configuración actual):

1. **Switches = False** (ya configurados)
2. **Modifica solo la sección "CONFIGURACIÓN MANUAL"** (líneas 64-84)
3. Ejemplo:
   ```python
   SLICE_LEN = 64                    # Cambiar slice temporal
   DM_RANGE_FACTOR = 0.4             # Cambiar zoom DM
   RFI_FREQ_SIGMA_THRESH = 4.0       # Cambiar umbral RFI
   ```

### **Para MODO AUTOMÁTICO:**

1. **Cambiar switches a True:**
   ```python
   SLICE_LEN_AUTO = True
   DM_DYNAMIC_RANGE_ENABLE = True
   RFI_ENABLE_ALL_FILTERS = True
   ```
2. **Las variables manuales se ignoran automáticamente**

## 📊 **BENEFICIOS DE LA REORGANIZACIÓN**

### ✅ **Para Desarrollo (Modo Manual):**

- **Sección clara** de variables que puedes modificar
- **Control total** sobre slice temporal, rango DM y RFI
- **Fácil experimentación** con diferentes valores

### ✅ **Para Producción (Modo Automático):**

- **Un solo cambio** en switches para activar todo automático
- **Optimización automática** basada en metadatos del archivo
- **Sin configuración manual** necesaria

### ✅ **Para Depuración:**

- **Switches mixtos**: algunos manual, otros automático
- **Diagnósticos**: `RFI_SAVE_DIAGNOSTICS = True`
- **Comparación fácil** entre modos

## 🚀 **CASOS DE USO RÁPIDOS**

### **Ajustar slice temporal manualmente:**

```python
# En switches: SLICE_LEN_AUTO = False
# En configuración manual:
SLICE_LEN = 64                    # Doblar resolución temporal
```

### **Ajustar zoom DM manualmente:**

```python
# En switches: DM_DYNAMIC_RANGE_ENABLE = False
# En configuración manual:
DM_RANGE_FACTOR = 0.5             # Zoom más amplio (±50%)
```

### **Filtros RFI más agresivos:**

```python
# En configuración manual:
RFI_FREQ_SIGMA_THRESH = 3.0       # Más agresivo (detecta más RFI)
RFI_TIME_SIGMA_THRESH = 3.0       # Más agresivo
```

---

## 🏆 **RESULTADO FINAL**

✅ **Organización clara** por función y modo de uso  
✅ **Sección dedicada** para variables manuales  
✅ **Switches centralizados** al principio  
✅ **Documentación integrada** con casos de uso  
✅ **Fácil migración** entre modo manual y automático

**¡Ahora es súper fácil encontrar y modificar las variables correctas!**
