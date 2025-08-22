"""
Planificador de DM para estrategias de detección en régimen milimétrico (ALMA Band 3)
===============================================================================

Este módulo implementa la lógica para construir dos grids de DM complementarios:

E1 (Expandir DM): Grid expandido para "abrir" el bow-tie y verificar DM* > 0
E2 (Pescar en DM≈0): Grid fino cerca de DM=0 para elevar recall con validación posterior

Autor: DRAFTS-MB Team
Fecha: 2024
"""

import numpy as np
from typing import Tuple, Dict, Any
from ..config import STRATEGY_DM_EXPAND, STRATEGY_FISH_NEAR_ZERO


def calculate_dm_step_from_smear(
    freq_low: float, 
    freq_high: float, 
    time_resolution: float,
    smear_frac: float = 0.25
) -> float:
    """
    Calcula el paso DM óptimo basado en la tolerancia de emborronamiento residual.
    
    Fórmula: Δt_smear ≈ 4.15×10⁶ × δDM × (ν_low⁻² - ν_high⁻²) ≤ smear_frac × W
    
    Args:
        freq_low: Frecuencia más baja en MHz
        freq_high: Frecuencia más alta en MHz
        time_resolution: Resolución temporal en segundos
        smear_frac: Fracción del ancho temporal permitida para emborronamiento
        
    Returns:
        float: Paso DM óptimo en pc cm⁻³
    """
    # Constante de dispersión en MHz² s pc⁻¹ cm³
    K_DM = 4.15e6
    
    # Calcular diferencia de frecuencias al cuadrado
    freq_diff = (1.0 / freq_low**2) - (1.0 / freq_high**2)
    
    # Tiempo máximo de emborronamiento permitido
    max_smear_time = smear_frac * time_resolution
    
    # Calcular paso DM que satisface la restricción
    dm_step = max_smear_time / (K_DM * freq_diff)
    
    return dm_step


def make_grid_expand(
    obparams: Dict[str, Any], 
    config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construye el grid expandido para la estrategia E1.
    
    Args:
        obparams: Parámetros de observación (freq_low, freq_high, time_resolution)
        config: Configuración de la estrategia E1
        
    Returns:
        Tuple[np.ndarray, Dict]: (grid DM, metadatos del grid)
    """
    if not config.get('enabled', False):
        return np.array([]), {}
    
    freq_low = obparams.get('freq_low', 100.0)  # MHz
    freq_high = obparams.get('freq_high', 110.0)  # MHz
    time_resolution = obparams.get('time_resolution', 0.001)  # segundos
    
    # Calcular paso DM óptimo basado en emborronamiento
    dm_step = calculate_dm_step_from_smear(
        freq_low, freq_high, time_resolution, 
        config.get('smear_frac', 0.25)
    )
    
    # Asegurar que el paso no sea demasiado pequeño
    dm_step = max(dm_step, 1.0)
    
    # Construir grid desde 0 hasta dm_max
    dm_max = config.get('dm_max', 2000)
    dm_values = np.arange(0, dm_max + dm_step, dm_step)
    
    # Metadatos del grid
    meta = {
        'strategy': 'E1_expand',
        'dm_min': 0.0,
        'dm_max': dm_max,
        'dm_step': dm_step,
        'n_dm': len(dm_values),
        'freq_low': freq_low,
        'freq_high': freq_high,
        'time_resolution': time_resolution,
        'smear_frac': config.get('smear_frac', 0.25),
        'min_dm_sigmas': config.get('min_dm_sigmas', 3.0)
    }
    
    return dm_values, meta


def make_grid_fish(
    obparams: Dict[str, Any], 
    config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construye el grid "fish" para la estrategia E2.
    
    Args:
        obparams: Parámetros de observación
        config: Configuración de la estrategia E2
        
    Returns:
        Tuple[np.ndarray, Dict]: (grid DM, metadatos del grid)
    """
    if not config.get('enabled', False):
        return np.array([]), {}
    
    # Grid fino desde 0 hasta dm_fish_max
    dm_fish_max = config.get('dm_fish_max', 50)
    dm_step_fine = 1.0  # Paso fino para "pescar" cerca de DM=0
    
    dm_values = np.arange(0, dm_fish_max + dm_step_fine, dm_step_fine)
    
    # Metadatos del grid
    meta = {
        'strategy': 'E2_fish',
        'dm_min': 0.0,
        'dm_max': dm_fish_max,
        'dm_step': dm_step_fine,
        'n_dm': len(dm_values),
        'fish_thresh': config.get('fish_thresh', 0.3),
        'refine': config.get('refine', {})
    }
    
    return dm_values, meta


def build_dm_grids(
    obparams: Dict[str, Any], 
    config_expand: Dict[str, Any] = None,
    config_fish: Dict[str, Any] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Construye ambos grids de DM según las estrategias E1 y E2.
    
    Args:
        obparams: Parámetros de observación
        config_expand: Configuración de E1 (usar STRATEGY_DM_EXPAND si es None)
        config_fish: Configuración de E2 (usar STRATEGY_FISH_NEAR_ZERO si es None)
        
    Returns:
        Tuple: (grid_expand, grid_fish, meta_expand, meta_fish)
    """
    # Usar configuraciones por defecto si no se proporcionan
    if config_expand is None:
        config_expand = STRATEGY_DM_EXPAND
    if config_fish is None:
        config_fish = STRATEGY_FISH_NEAR_ZERO
    
    # Construir grid expandido (E1)
    grid_expand, meta_expand = make_grid_expand(obparams, config_expand)
    
    # Construir grid fish (E2)
    grid_fish, meta_fish = make_grid_fish(obparams, config_fish)
    
    return grid_expand, grid_fish, meta_expand, meta_fish


def validate_dm_grids(
    grid_expand: np.ndarray, 
    grid_fish: np.ndarray,
    meta_expand: Dict[str, Any],
    meta_fish: Dict[str, Any]
) -> bool:
    """
    Valida que los grids de DM sean consistentes y válidos.
    
    Args:
        grid_expand: Grid expandido de E1
        grid_fish: Grid fish de E2
        meta_expand: Metadatos del grid expandido
        meta_fish: Metadatos del grid fish
        
    Returns:
        bool: True si los grids son válidos
    """
    # Verificar que ambos grids estén habilitados
    if len(grid_expand) == 0 and len(grid_fish) == 0:
        raise ValueError("Al menos una estrategia debe estar habilitada")
    
    # Verificar consistencia del grid expandido
    if len(grid_expand) > 0:
        if meta_expand['dm_min'] < 0:
            raise ValueError("DM mínimo no puede ser negativo")
        if meta_expand['dm_max'] <= meta_expand['dm_min']:
            raise ValueError("DM máximo debe ser mayor que DM mínimo")
        if meta_expand['dm_step'] <= 0:
            raise ValueError("Paso DM debe ser positivo")
    
    # Verificar consistencia del grid fish
    if len(grid_fish) > 0:
        if meta_fish['dm_min'] < 0:
            raise ValueError("DM mínimo fish no puede ser negativo")
        if meta_fish['dm_max'] <= meta_fish['dm_min']:
            raise ValueError("DM máximo fish debe ser mayor que DM mínimo")
        if meta_fish['dm_step'] <= 0:
            raise ValueError("Paso DM fish debe ser positivo")
    
    # Verificar que no haya solapamiento problemático
    if len(grid_expand) > 0 and len(grid_fish) > 0:
        if meta_fish['dm_max'] > meta_expand['dm_max']:
            raise ValueError("Grid fish no puede extenderse más allá del grid expandido")
    
    return True


def get_optimal_dm_range_for_candidate(
    candidate_dm: float,
    meta_expand: Dict[str, Any],
    meta_fish: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Determina el rango DM óptimo para visualizar un candidato específico.
    
    Args:
        candidate_dm: DM del candidato en pc cm⁻³
        meta_expand: Metadatos del grid expandido
        meta_fish: Metadatos del grid fish
        
    Returns:
        Tuple[float, float]: (dm_min_plot, dm_max_plot)
    """
    # Si el candidato está en el rango fish, usar ese grid
    if len(meta_fish) > 0 and candidate_dm <= meta_fish['dm_max']:
        dm_min = max(0, candidate_dm - 10)
        dm_max = min(meta_fish['dm_max'], candidate_dm + 10)
    else:
        # Usar el grid expandido
        dm_min = max(0, candidate_dm - 50)
        dm_max = min(meta_expand['dm_max'], candidate_dm + 50)
    
    return dm_min, dm_max
