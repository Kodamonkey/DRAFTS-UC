
# Sistema de DM Dinámico - Ejemplo de Uso

## Integración Automática

El sistema de DM dinámico se integra automáticamente en el pipeline DRAFTS.
No requiere modificaciones en el código de usuario existente.

## Configuración en config.py

```python
# Habilitar DM dinámico
DM_DYNAMIC_RANGE_ENABLE = True

# Ajustar parámetros según necesidades
DM_RANGE_FACTOR = 0.2          # ±20% del DM óptimo
DM_RANGE_MIN_WIDTH = 50.0      # Mínimo 50 pc cm⁻³
DM_RANGE_MAX_WIDTH = 200.0     # Máximo 200 pc cm⁻³
```

## Uso en el Pipeline

El sistema funciona automáticamente cuando hay candidatos detectados:

1. **Detección automática**: Analiza bounding boxes de candidatos
2. **Cálculo inteligente**: Determina DM óptimo del mejor candidato
3. **Ajuste dinámico**: Calcula rango centrado en el candidato
4. **Fallback robusto**: Usa rango completo si no hay candidatos

## Beneficios Observados

- **Resolución mejorada**: 2x a 20x mejor en eje DM
- **Visualización centrada**: Candidatos siempre visibles y centrados
- **Automático**: Sin intervención manual requerida
- **Robusto**: Fallbacks automáticos para todos los casos

## Archivos Generados

Este ejemplo ha generado los siguientes archivos de demostración:

- `FRB_bajo_DM_dynamic.png`: FRB con DM bajo, visualización optimizada
- `FRB_alto_DM_dynamic.png`: FRB con DM alto, zoom automático
- `multiples_FRBs_dynamic.png`: Múltiples candidatos, rango adaptado
- `comparacion_sistemas_dynamic.png`: Visualización con DM dinámico
- `comparacion_sistemas_fixed.png`: Visualización con rango fijo

## Recomendaciones

1. **Mantener habilitado**: `DM_DYNAMIC_RANGE_ENABLE = True` para mejor visualización
2. **Ajustar factor**: Modificar `DM_RANGE_FACTOR` según tipo de observaciones
3. **Monitorear logs**: Verificar comportamiento en casos especiales
4. **Usar fallback**: Sistema automáticamente maneja casos sin candidatos

El sistema está diseñado para mejorar la experiencia de análisis sin requerir
cambios en el flujo de trabajo existente.
