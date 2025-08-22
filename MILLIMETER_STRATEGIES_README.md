# Estrategias de Detección en Régimen Milimétrico (ALMA Band 3) - DRAFTS-MB

## Descripción General

Este documento describe la implementación de dos estrategias complementarias para mejorar la detección de **BURSTS** en el régimen milimétrico (ALMA Band 3) en **DRAFTS-MB**. Estas estrategias abordan el desafío de detectar bursts cuando la dispersión es mínima y el bow-tie en tiempo-DM casi no "abre".

## Estrategias Implementadas

### E1: Expandir Rango y Paso de DM

**Objetivo**: Expandir el rango y paso de DM para "abrir" lo suficiente el bow-tie y verificar **DM\* > 0**.

**Características**:

- Grid DM expandido desde 0 hasta `dm_max` configurable
- Paso DM optimizado basado en tolerancia de emborronamiento residual
- Fórmula: Δt_smear ≈ 4.15×10⁶ × δDM × (ν_low⁻² - ν_high⁻²) ≤ smear_frac × W
- A mm, (ν⁻²) cambia poco, por lo que ΔDM puede ser grueso (barato)

**Configuración**:

```yaml
STRATEGY_DM_EXPAND:
  enabled: true
  dm_max: 2000 # DM máximo expandido
  smear_frac: 0.25 # Δt_residual ≤ 0.25 · W
  min_dm_sigmas: 3.0 # Exigir centro DM* > 0 con ≥ 3σ
```

### E2: Pescar en DM≈0 con Validación

**Objetivo**: Detectar candidatos cerca de DM≈0 con umbral laxo, seguido de validación usando un chequeo DM-aware local y consistencia por sub-bandas.

**Características**:

- Grid DM fino desde 0 hasta `dm_fish_max` (típicamente 50 pc cm⁻³)
- Umbral CenterNet más laxo para elevar recall
- Validación posterior con micro-rejillas locales
- Verificación de consistencia multi-banda

**Configuración**:

```yaml
STRATEGY_FISH_NEAR_ZERO:
  enabled: true
  dm_fish_max: 50 # Barrido muy bajo solo para "pescar"
  fish_thresh: 0.3 # Umbral CenterNet más laxo
  refine:
    dm_local_max: 300 # Micro-rejilla de validación local
    ddm_local: 1 # Paso DM para micro-rejilla
    min_delta_snr: 2.0 # SNR(DM*) - SNR(0) mínima
    min_dm_star: 5 # DM* mínimo aceptable
    subband_consistency_pc: 20 # Tolerancia relativa entre sub-bandas (%)
```

## Arquitectura del Sistema

### 1. Planificador de DM (`dm_planner.py`)

Construye dos grids de DM complementarios según las estrategias E1 y E2:

```python
from drafts.preprocessing.dm_planner import build_dm_grids

# Construir grids
grid_expand, grid_fish, meta_expand, meta_fish = build_dm_grids(obparams)

# Validar grids
validate_dm_grids(grid_expand, grid_fish, meta_expand, meta_fish)
```

### 2. Dedispersión Modificada (`dedispersion.py`)

Las funciones de dedispersión ahora aceptan listas específicas de DMs y devuelven metadatos:

```python
from drafts.preprocessing.dedispersion import d_dm_time_g

# Dedispersión con grid específico
dm_time_cube, metadata = d_dm_time_g(
    data, height, width,
    dm_values=grid_expand  # o grid_fish
)

# Convertir índices a valores físicos
from drafts.preprocessing.dedispersion import dm_index_to_physical
dm_physical = dm_index_to_physical(dm_idx, metadata)
```

### 3. Validador DM-Aware (`validators/dm_validator.py`)

Implementa validación local con micro-rejillas y análisis de consistencia:

```python
from drafts.validators.dm_validator import DMValidator

validator = DMValidator()
result = validator.validate_candidate(
    candidate, data, freq_values, time_resolution, subbands
)

if result.passed:
    print(f"DM* = {result.dm_star:.1f} ± {result.dm_star_err:.1f}")
    print(f"ΔSNR = {result.delta_snr:.2f}")
    print(f"Acuerdo sub-bandas = {result.subband_agreement:.1f}%")
```

### 4. Candidatos Extendidos (`candidate_manager.py`)

La clase `Candidate` ahora incluye campos para validación DM-aware:

```python
from drafts.output.candidate_manager import Candidate

candidate = Candidate(
    # ... campos básicos ...
    dm_star=28.3,                    # DM* óptimo encontrado
    dm_star_err=1.2,                 # Error en DM*
    snr_dm0=5.1,                     # SNR a DM=0
    snr_dmstar=8.2,                  # SNR a DM*
    delta_snr=3.1,                   # ΔSNR = SNR(DM*) - SNR(0)
    subband_agreement=85.5,          # Acuerdo entre sub-bandas (%)
    validation_passed=True,           # True si pasa validación
    validation_reason=None,           # Razón del fallo si validation_passed=False
    strategy="E2_fish"                # E1_expand, E2_fish, o None
)

# Calcular score de prioridad
priority_score = candidate.calculate_priority_score()

# Obtener resumen de validación
validation_summary = candidate.get_validation_summary()
```

## Flujo de Procesamiento

### Flujo Completo

```text
load_fits → rfi_mask → get_obparams →
(grid_expand, grid_fish) = build_dm_grids(...) →
(DMt_expand, meta_ex) = d_dm_time_g(..., grid_expand) →
(DMt_fish, meta_fi) = d_dm_time_g(..., grid_fish)

# E1: Detección normal con verificación DM* > 0
boxes_ex = detect(DMt_expand, thresh=det.thresh)
cand_ex = keep_if_dm_center_significantly_above_zero(boxes_ex, meta_ex, cfg)

# E2: Detección laxa + validación
boxes_fi = detect(DMt_fish, thresh=fish_thresh)
cand_fi = dm_validator.run(boxes_fi, windowed_data, subbands, cfg)

# Combinar y priorizar candidatos
candidates = merge(cand_ex + cand_fi)
candidates.sort(key=lambda c: c.calculate_priority_score(), reverse=True)

# Clasificación solo para candidatos validados
for c in candidates:
    if c.validation_passed:
        wf_DMstar = dedisperse_and_crop(data, DM=c.dm_star, t=c.t0, win=...)
        label = resnet18.classify(wf_DMstar)
        save_outputs(c, label, artefacts)
```

### Detalles de Validación E2

1. **Micro-rejilla local**: `DM ∈ [0, dm_local_max]` con `ddm_local`
2. **Re-dedispersar** solo la ventana del candidato
3. **Curva SNR vs DM**: Hallar DM\* y exigir:
   - `DM* ≥ min_dm_star`
   - `SNR(DM*) − SNR(0) ≥ min_delta_snr`
4. **Consistencia por sub-bandas**: Repetir 1-3 y exigir DM\* compatible
5. **Consistencia por chunks**: Re-ejecutar bloque vecino si es necesario

## Configuración

### Parámetros Principales

```python
# En user_config.py
STRATEGY_DM_EXPAND = {
    'enabled': True,                # Habilitar estrategia E1
    'dm_max': 2000,                 # DM máximo expandido
    'smear_frac': 0.25,             # Tolerancia de emborronamiento
    'min_dm_sigmas': 3.0,           # Significancia mínima DM* > 0
}

STRATEGY_FISH_NEAR_ZERO = {
    'enabled': True,                 # Habilitar estrategia E2
    'dm_fish_max': 50,              # DM máximo para "pescar"
    'fish_thresh': 0.3,             # Umbral laxo para E2
    'refine': {
        'dm_local_max': 300,        # Micro-rejilla local
        'ddm_local': 1,             # Paso DM local
        'min_delta_snr': 2.0,       # ΔSNR mínima
        'min_dm_star': 5,           # DM* mínimo
        'subband_consistency_pc': 20, # Tolerancia sub-bandas
    }
}
```

### Ajustes por Dataset

- **ALMA Band 3**: `dm_max=2000`, `dm_fish_max=50`
- **VLA L-band**: `dm_max=1000`, `dm_fish_max=30`
- **Parkes**: `dm_max=500`, `dm_fish_max=20`

## Logging y Artefactos

### Información Persistida

Para cada candidato, se guarda:

- `DM*`, `SNR@0`, `SNR@DM*`, `ΔSNR`, `subband_agreement`
- **Thumbnails**: tiempo-DM (E1/E2), waterfall a DM=0 y DM=DM\*
- Curva **SNR(DM)** local para análisis

### Logs de Validación

```
[INFO] Validando candidato test_candidate_001 en t=1.500s
[INFO] Candidato test_candidate_001 validado: DM*=28.3, ΔSNR=3.1, subband_agreement=85.5%
```

## Ventajas del Diseño

### E1 (DM Expandido)

- Recupera **morfología** bow-tie suficiente
- Permite **prueba directa** de que DM\* > 0
- **Costo bajo** (ΔDM más grueso a mm)

### E2 (Pesca + Validación)

- **Alto recall** con CenterNet laxo
- **Evidencia de dispersión** con micro-búsqueda local
- **Consistencia multibanda** como chequeo anti-RFI
- **Sin re-entrenar** redes existentes

## Próximos Pasos

1. **Integración en Pipeline Principal**: Modificar `pipeline.py` para usar las nuevas estrategias
2. **Optimización de Rendimiento**: Paralelizar validaciones E2
3. **Data Augmentation**: Re-entrenar CenterNet con bow-ties "aplastados"
4. **Validación en Datos Reales**: Probar con observaciones ALMA Band 3

## Referencias

- Fórmula de dispersión: `Δt ≈ 4.15×10⁶ × DM × (ν_low⁻² - ν_high⁻²)`
- Bow-tie effect: Dependencia temporal de la dispersión
- SNR calculation: `(peak - mean) / std`
- FWHM error estimation: `FWHM / 2.355`

---

**Autor**: DRAFTS-MB Team  
**Fecha**: 2024  
**Versión**: 1.0
