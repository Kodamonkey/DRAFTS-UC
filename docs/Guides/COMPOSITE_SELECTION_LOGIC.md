# Lógica de Selección de Candidatos para Composite Plot

## 🎯 Descripción General

Se ha implementado una nueva lógica inteligente para seleccionar qué candidato se centraliza en el **Candidate Patch** del Composite Plot cuando hay múltiples candidatos en el mismo slice.

## 🔄 Comportamiento Anterior vs Nuevo

### ❌ Comportamiento Anterior

- **Siempre** se centralizaba en el **primer candidato** detectado
- **No importaba** si era BURST o NO BURST
- Podía resultar en centralizar en un candidato NO BURST cuando había un BURST disponible

### ✅ Nuevo Comportamiento

- **Prioriza candidatos BURST** sobre candidatos NO BURST
- Si hay múltiples candidatos, busca el **primer BURST** disponible
- Si **no hay BURST**, mantiene el comportamiento anterior (primer candidato)

## 📋 Lógica de Selección

```python
# 🎯 SELECCIÓN INTELIGENTE: Priorizar candidatos BURST sobre NO BURST
if best_patch is None:
    # Primer candidato siempre se guarda como fallback
    best_patch = proc_patch
    best_start = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
    best_dm = dm_val
    best_is_burst = is_burst
elif is_burst and not best_is_burst:
    # Si encontramos un BURST y el mejor actual es NO BURST, actualizar
    best_patch = proc_patch
    best_start = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
    best_dm = dm_val
    best_is_burst = is_burst
# Si ambos son BURST o ambos son NO BURST, mantener el primero (orden de detección)
```

## 🧪 Casos de Prueba

### Caso 1: Un solo candidato

- **NO BURST**: Se centraliza en el candidato
- **BURST**: Se centraliza en el candidato

### Caso 2: Múltiples candidatos con BURST

- **Primero NO BURST, segundo BURST**: Se centraliza en el **segundo (BURST)**
- **Primero BURST, segundo NO BURST**: Se centraliza en el **primero (BURST)**
- **Múltiples BURST**: Se centraliza en el **primer BURST** detectado

### Caso 3: Múltiples candidatos sin BURST

- **Todos NO BURST**: Se centraliza en el **primer candidato** (comportamiento anterior)

## 🔧 Implementación Técnica

### Archivos Modificados

- `drafts/detection_engine.py`: Función `process_band()`

### Variables Nuevas

```python
# 🎯 NUEVA LÓGICA: Variables para seleccionar el mejor candidato para el Composite
best_patch = None
best_start = None
best_dm = None
best_is_burst = False
```

### Información Adicional en Logs

```python
# Log informativo sobre la selección del candidato
if len(all_candidates) > 1:
    burst_count = sum(1 for c in all_candidates if c['is_burst'])
    global_logger.logger.info(
        f"Slice {j} - {band_name}: {len(all_candidates)} candidatos "
        f"({burst_count} BURST, {len(all_candidates) - burst_count} NO BURST). "
        f"Seleccionado: {'BURST' if best_is_burst else 'NO BURST'} (DM={final_dm:.2f})"
    )
```

## 📊 Beneficios

1. **Mejor calidad visual**: Los Composite Plots muestran candidatos más relevantes
2. **Priorización inteligente**: Los candidatos BURST tienen prioridad sobre NO BURST
3. **Compatibilidad**: Mantiene el comportamiento anterior cuando no hay BURST
4. **Transparencia**: Logs informativos muestran qué candidato se seleccionó y por qué

## 🚀 Uso

La funcionalidad se activa automáticamente. No requiere configuración adicional.

### Ejemplo de Log

```
Slice 005 - Full Band: 3 candidatos (1 BURST, 2 NO BURST).
Seleccionado: BURST (DM=150.0)
```

## 🔍 Verificación

Para verificar que la funcionalidad funciona correctamente:

1. Ejecutar el pipeline con datos que contengan múltiples candidatos
2. Revisar los logs para confirmar la selección correcta
3. Verificar que los Composite Plots muestran el candidato BURST cuando está disponible

## 📝 Notas Técnicas

- La lógica se aplica **por banda** (Full Band, Low Band, High Band)
- Se mantiene **compatibilidad total** con el código existente
- Los candidatos siguen guardándose en el CSV en el orden de detección
- Solo afecta la **visualización** del Composite Plot
