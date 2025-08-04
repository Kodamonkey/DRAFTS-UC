# 🚀 SLICE_LEN DINÁMICO: Configuración Intuitiva Basada en Tiempo

## 📋 Resumen

Se ha implementado una nueva funcionalidad que permite configurar `SLICE_LEN` de manera mucho más intuitiva, especificando la **duración temporal deseada en segundos** en lugar del número de muestras.

### 🎯 Beneficios

- ✅ **Más intuitivo**: Especifica duración en segundos (ej: 0.032s = 32ms)
- ✅ **Automático**: Calcula SLICE_LEN óptimo basado en TIME_RESO del archivo
- ✅ **Adaptable**: Se ajusta automáticamente a diferentes resoluciones temporales
- ✅ **Seguro**: Incluye límites mínimos y máximos para evitar valores extremos
- ✅ **Compatible**: Mantiene funcionalidad manual existente

## ⚙️ Configuración

### **Nueva Configuración en `config.py`:**

```python
# Configuración dinámica de SLICE_LEN basada en duración temporal
SLICE_DURATION_SECONDS: float = 0.032  # Duración deseada por slice en segundos (32ms)
SLICE_LEN_AUTO: bool = True             # Calcular SLICE_LEN automáticamente
SLICE_LEN_MIN: int = 16                 # Valor mínimo permitido
SLICE_LEN_MAX: int = 512                # Valor máximo permitido

# SLICE_LEN manual (usado solo si SLICE_LEN_AUTO = False)
SLICE_LEN: int = 32
```

### **Parámetros Explicados:**

| Parámetro                | Tipo  | Descripción                                                      |
| ------------------------ | ----- | ---------------------------------------------------------------- |
| `SLICE_DURATION_SECONDS` | float | Duración deseada por slice en segundos                           |
| `SLICE_LEN_AUTO`         | bool  | Si `True`, calcula automáticamente; si `False`, usa valor manual |
| `SLICE_LEN_MIN`          | int   | Valor mínimo permitido para evitar slices muy pequeños           |
| `SLICE_LEN_MAX`          | int   | Valor máximo permitido para evitar slices muy grandes            |

## 🎯 Ejemplos de Uso

### **1. Pulsos Muy Cortos (< 20ms)**

```python
SLICE_DURATION_SECONDS: float = 0.016  # 16ms
SLICE_LEN_AUTO: bool = True
SLICE_LEN_MIN: int = 8
SLICE_LEN_MAX: int = 64
```

### **2. FRBs Típicos (20-100ms) - RECOMENDADO**

```python
SLICE_DURATION_SECONDS: float = 0.032  # 32ms
SLICE_LEN_AUTO: bool = True
SLICE_LEN_MIN: int = 16
SLICE_LEN_MAX: int = 256
```

### **3. Señales Largas (> 100ms)**

```python
SLICE_DURATION_SECONDS: float = 0.128  # 128ms
SLICE_LEN_AUTO: bool = True
SLICE_LEN_MIN: int = 32
SLICE_LEN_MAX: int = 512
```

### **4. Configuración Manual (como antes)**

```python
SLICE_LEN_AUTO: bool = False
SLICE_LEN: int = 64  # Valor fijo
```

## 🔄 Cómo Funciona

### **Proceso Automático:**

1. **Lectura del archivo**: El pipeline lee `TIME_RESO` del archivo
2. **Cálculo automático**:
   ```
   SLICE_LEN_ideal = SLICE_DURATION_SECONDS / (TIME_RESO * DOWN_TIME_RATE)
   ```
3. **Ajuste a potencia de 2**: Se ajusta al valor más cercano que sea potencia de 2
4. **Aplicación de límites**: Se asegura que esté dentro de `[SLICE_LEN_MIN, SLICE_LEN_MAX]`
5. **Uso en pipeline**: Se usa el valor calculado en todo el procesamiento

### **Ejemplo Práctico:**

```
Archivo con TIME_RESO = 0.001s (1ms por muestra)
SLICE_DURATION_SECONDS = 0.032s (32ms deseados)

Cálculo: 0.032s / 0.001s = 32 muestras
Ajuste: 32 es potencia de 2, se mantiene
Resultado: SLICE_LEN = 32
```

## 🛠️ Herramientas Incluidas

### **1. Script de Demostración**

```bash
python demo_dynamic_slice_len.py
```

Muestra cómo funciona la funcionalidad con diferentes ejemplos.

### **2. Configurador Interactivo**

```bash
python configure_slice_len.py
```

Asistente interactivo para configurar fácilmente los parámetros.

### **3. Prueba de Funcionalidad**

```bash
python test_dynamic_slice_len.py
```

Prueba que la funcionalidad está funcionando correctamente.

## 🔍 Análisis de Configuración

### **Verificar Configuración Actual:**

```python
from DRAFTS.slice_len_utils import print_slice_len_analysis
from DRAFTS import config

print_slice_len_analysis(config)
```

### **Ejemplo de Salida:**

```
🔬 === ANÁLISIS DE CONFIGURACIÓN SLICE_LEN ===

📋 CONFIGURACIÓN ACTUAL:
   • Modo automático: ✅ Habilitado
   • Duración objetivo: 0.032 s (32.0 ms)
   • SLICE_LEN manual: 32
   • TIME_RESO: 0.001000 s
   • DOWN_TIME_RATE: 1

🎯 CÁLCULO DINÁMICO:
   • SLICE_LEN calculado: 32
   • Duración real: 0.032 s (32.0 ms)
   • Diferencia vs objetivo: 0.0%
```

## 📊 Comparación: Antes vs Ahora

### **Antes (Manual):**

```python
# ❌ Poco intuitivo
SLICE_LEN: int = 32  # ¿Cuánto tiempo representa?

# ❌ Hay que calcular manualmente para cada archivo
# Si TIME_RESO = 0.001s → 32 muestras = 32ms
# Si TIME_RESO = 0.0005s → 32 muestras = 16ms
```

### **Ahora (Dinámico):**

```python
# ✅ Muy intuitivo
SLICE_DURATION_SECONDS: float = 0.032  # 32ms claramente especificados
SLICE_LEN_AUTO: bool = True

# ✅ Se adapta automáticamente
# Con TIME_RESO = 0.001s → SLICE_LEN = 32 (32ms)
# Con TIME_RESO = 0.0005s → SLICE_LEN = 64 (32ms)
```

## 🚀 Migración

### **Para Usuarios Existentes:**

1. **Mantener comportamiento actual** (no cambiar nada):

   ```python
   SLICE_LEN_AUTO: bool = False
   SLICE_LEN: int = 64  # Tu valor actual
   ```

2. **Migrar a sistema dinámico**:

   ```python
   # Calcular duración actual
   # Si SLICE_LEN = 64 y TIME_RESO = 0.001s:
   # Duración = 64 × 0.001 = 0.064s

   SLICE_DURATION_SECONDS: float = 0.064  # Equivalente a tu SLICE_LEN = 64
   SLICE_LEN_AUTO: bool = True
   ```

## 💡 Recomendaciones

### **Por Tipo de Señal:**

| Tipo de Señal   | Duración Recomendada | Configuración           |
| --------------- | -------------------- | ----------------------- |
| Pulsars rápidos | 0.016s (16ms)        | Pulsos muy cortos       |
| FRBs típicos    | 0.032s (32ms)        | **Recomendado general** |
| FRBs dispersos  | 0.064s (64ms)        | Señales medias          |
| Casos extremos  | 0.128s (128ms)       | Señales muy largas      |

### **Para Optimización:**

1. **Comienza con 0.032s** (32ms) - funciona bien en la mayoría de casos
2. **Si detectas pocas señales**: prueba 0.016s (16ms) para mayor resolución
3. **Si las señales se fragmentan**: prueba 0.064s (64ms) para más contexto
4. **Para casos especiales**: usa el configurador interactivo

## ✅ Conclusión

La nueva funcionalidad de **SLICE_LEN dinámico** hace que la configuración sea:

- 🎯 **Más intuitiva**: Especifica tiempo en segundos
- 🔧 **Más flexible**: Se adapta automáticamente a diferentes archivos
- 🛡️ **Más segura**: Incluye límites y validaciones
- 📈 **Más eficiente**: Optimiza automáticamente para tus datos

¡Disfruta de la nueva funcionalidad más user-friendly!
