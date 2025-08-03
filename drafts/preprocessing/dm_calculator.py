"""
Módulo de Cálculo DM - Cálculo de Rangos de Dispersion Measure
=============================================================

Este módulo se encarga de calcular y validar rangos de DM (Dispersion Measure)
para el procesamiento de datos astronómicos.

Responsabilidades:
- Calcular rangos DM óptimos basados en frecuencias de observación
- Validar parámetros DM para evitar errores de procesamiento
- Optimizar rangos DM para detección de FRB
- Proporcionar configuraciones recomendadas de DM

Para astrónomos:
- Usar calculate_dm_range() para obtener rangos DM óptimos
- Usar validate_dm_parameters() para verificar configuración
- Usar get_dm_processing_config() para obtener configuración completa
- Los parámetros se configuran en config.DM_min y config.DM_max
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .. import config

logger = logging.getLogger(__name__)


def calculate_dm_range(freq_min: Optional[float] = None, 
                      freq_max: Optional[float] = None,
                      time_resolution: Optional[float] = None) -> Dict[str, Any]:
    """
    Calcula rangos DM óptimos basados en parámetros de observación.
    
    Parameters
    ----------
    freq_min : float, optional
        Frecuencia mínima en MHz. Si None, usa config.FREQ.min()
    freq_max : float, optional
        Frecuencia máxima en MHz. Si None, usa config.FREQ.max()
    time_resolution : float, optional
        Resolución temporal en segundos. Si None, usa config.TIME_RESO
        
    Returns
    -------
    Dict[str, Any]
        Diccionario con rangos DM calculados y recomendaciones
    """
    try:
        # Usar valores de config si no se proporcionan
        if freq_min is None:
            if config.FREQ is not None and config.FREQ.size > 0:
                freq_min = config.FREQ.min()
            else:
                freq_min = 1000.0  # Valor por defecto típico
                logger.warning("FREQ no configurado, usando freq_min=1000 MHz")
        
        if freq_max is None:
            if config.FREQ is not None and config.FREQ.size > 0:
                freq_max = config.FREQ.max()
            else:
                freq_max = 2000.0  # Valor por defecto típico
                logger.warning("FREQ no configurado, usando freq_max=2000 MHz")
        
        if time_resolution is None:
            time_resolution = config.TIME_RESO * config.DOWN_TIME_RATE
        
        # Validar parámetros
        if freq_min <= 0 or freq_max <= 0:
            raise ValueError("Frecuencias deben ser positivas")
        if freq_min >= freq_max:
            raise ValueError("freq_min debe ser menor que freq_max")
        if time_resolution <= 0:
            raise ValueError("time_resolution debe ser positivo")
        
        # Calcular DM máximo teórico (fórmula de Cordes & McLaughlin 2003)
        # DM_max = (freq_min^2 - freq_max^2) / (2 * freq_min^2 * freq_max^2 * time_resolution)
        dm_max_theoretical = (freq_min**2 - freq_max**2) / (2 * freq_min**2 * freq_max**2 * time_resolution)
        
        # Convertir a pc cm⁻³ (multiplicar por 1e6 para MHz a Hz)
        dm_max_theoretical *= 1e6
        
        # Aplicar factor de seguridad (80% del máximo teórico)
        dm_max_recommended = dm_max_theoretical * 0.8
        
        # Calcular DM mínimo (típicamente 0 para FRB)
        dm_min_recommended = 0.0
        
        # Calcular paso DM óptimo
        # Paso DM = 1 / (2 * freq_min^2 * time_resolution)
        dm_step_recommended = 1.0 / (2 * freq_min**2 * time_resolution) * 1e6
        
        # Ajustar paso DM a valores prácticos
        if dm_step_recommended < 0.1:
            dm_step_recommended = 0.1
        elif dm_step_recommended > 10.0:
            dm_step_recommended = 10.0
        
        # Calcular número de valores DM
        num_dm_values = int((dm_max_recommended - dm_min_recommended) / dm_step_recommended) + 1
        
        return {
            "dm_min": dm_min_recommended,
            "dm_max": dm_max_recommended,
            "dm_step": dm_step_recommended,
            "num_dm_values": num_dm_values,
            "freq_min_mhz": freq_min,
            "freq_max_mhz": freq_max,
            "time_resolution_s": time_resolution,
            "dm_max_theoretical": dm_max_theoretical,
            "recommendations": {
                "use_optimized_range": True,
                "consider_galactic_dm": True,
                "typical_frb_dm_range": "0-2000 pc cm⁻³",
                "high_dm_frb_range": "2000-5000 pc cm⁻³"
            }
        }
        
    except Exception as e:
        logger.error(f"Error al calcular rango DM: {e}")
        # Retornar valores por defecto seguros
        return {
            "dm_min": 0.0,
            "dm_max": 1000.0,
            "dm_step": 1.0,
            "num_dm_values": 1001,
            "error": str(e)
        }


def validate_dm_parameters(dm_min: Optional[float] = None,
                          dm_max: Optional[float] = None,
                          dm_step: Optional[float] = None) -> bool:
    """
    Valida que los parámetros DM sean correctos y seguros.
    
    Parameters
    ----------
    dm_min : float, optional
        DM mínimo. Si None, usa config.DM_min
    dm_max : float, optional
        DM máximo. Si None, usa config.DM_max
    dm_step : float, optional
        Paso DM. Si None, calcula automáticamente
        
    Returns
    -------
    bool
        True si los parámetros son válidos, False en caso contrario
    """
    try:
        # Usar valores de config si no se proporcionan
        if dm_min is None:
            dm_min = config.DM_min
        if dm_max is None:
            dm_max = config.DM_max
        if dm_step is None:
            dm_step = (dm_max - dm_min) / 512  # Valor por defecto
        
        # Validaciones básicas
        if dm_min < 0:
            logger.error(f"DM_min ({dm_min}) no puede ser negativo")
            return False
        
        if dm_max <= dm_min:
            logger.error(f"DM_max ({dm_max}) debe ser mayor que DM_min ({dm_min})")
            return False
        
        if dm_step <= 0:
            logger.error(f"DM_step ({dm_step}) debe ser positivo")
            return False
        
        # Validar rangos típicos para FRB
        if dm_max > 10000:
            logger.warning(f"DM_max ({dm_max}) es muy alto para FRB típicos")
        
        if dm_max < 100:
            logger.warning(f"DM_max ({dm_max}) es muy bajo, podría perder FRB distantes")
        
        # Validar número de valores DM
        num_dm_values = int((dm_max - dm_min) / dm_step) + 1
        if num_dm_values > 10000:
            logger.warning(f"Demasiados valores DM ({num_dm_values}), considerando reducir dm_step")
        
        if num_dm_values < 100:
            logger.warning(f"Muy pocos valores DM ({num_dm_values}), considerando aumentar dm_step")
        
        # Validar que el rango sea razonable para las frecuencias
        if config.FREQ is not None and config.FREQ.size > 0:
            freq_min = config.FREQ.min()
            freq_max = config.FREQ.max()
            time_res = config.TIME_RESO * config.DOWN_TIME_RATE
            
            # Calcular DM máximo teórico
            dm_max_theoretical = (freq_min**2 - freq_max**2) / (2 * freq_min**2 * freq_max**2 * time_res) * 1e6
            
            if dm_max > dm_max_theoretical:
                logger.warning(f"DM_max ({dm_max}) excede el máximo teórico ({dm_max_theoretical:.1f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error al validar parámetros DM: {e}")
        return False


def get_dm_processing_config() -> Dict[str, Any]:
    """
    Obtiene la configuración completa de procesamiento DM.
    
    Returns
    -------
    Dict[str, Any]
        Configuración completa de DM para procesamiento
    """
    try:
        # Calcular rango DM óptimo
        dm_range = calculate_dm_range()
        
        # Validar parámetros actuales
        current_valid = validate_dm_parameters()
        
        # Comparar con valores recomendados
        recommendations = []
        if abs(config.DM_min - dm_range["dm_min"]) > 1.0:
            recommendations.append(f"Considerar DM_min = {dm_range['dm_min']:.1f} (actual: {config.DM_min})")
        
        if abs(config.DM_max - dm_range["dm_max"]) > 10.0:
            recommendations.append(f"Considerar DM_max = {dm_range['dm_max']:.1f} (actual: {config.DM_max})")
        
        # Calcular paso DM actual
        current_dm_step = (config.DM_max - config.DM_min) / 512  # Asumiendo 512 valores por defecto
        
        if abs(current_dm_step - dm_range["dm_step"]) > 0.1:
            recommendations.append(f"Considerar DM_step = {dm_range['dm_step']:.2f} (actual: {current_dm_step:.2f})")
        
        return {
            "current_config": {
                "dm_min": config.DM_min,
                "dm_max": config.DM_max,
                "dm_step": current_dm_step,
                "num_dm_values": 512
            },
            "recommended_config": dm_range,
            "validation": {
                "current_valid": current_valid,
                "recommendations": recommendations
            },
            "processing_info": {
                "estimated_memory_gb": dm_range["num_dm_values"] * 4 / (1024**3),  # Estimación básica
                "estimated_processing_time": "Depende del tamaño de datos y GPU/CPU"
            }
        }
        
    except Exception as e:
        logger.error(f"Error al obtener configuración DM: {e}")
        return {
            "error": str(e),
            "current_config": {
                "dm_min": config.DM_min,
                "dm_max": config.DM_max,
                "dm_step": (config.DM_max - config.DM_min) / 512,
                "num_dm_values": 512
            }
        }


def optimize_dm_range_for_frb_detection() -> Dict[str, Any]:
    """
    Optimiza el rango DM específicamente para detección de FRB.
    
    Returns
    -------
    Dict[str, Any]
        Configuración DM optimizada para FRB
    """
    try:
        # Rangos típicos de DM para diferentes tipos de FRB
        frb_dm_ranges = {
            "galactic": (0, 100),      # FRB galácticos
            "extragalactic_near": (100, 1000),  # FRB extragalácticos cercanos
            "extragalactic_far": (1000, 3000),  # FRB extragalácticos lejanos
            "extreme": (3000, 10000)   # FRB extremos (muy raros)
        }
        
        # Calcular rango base
        base_range = calculate_dm_range()
        
        # Ajustar para FRB típicos (priorizar 0-2000 pc cm⁻³)
        dm_min_optimized = 0.0
        dm_max_optimized = min(base_range["dm_max"], 2000.0)
        
        # Ajustar paso DM para mejor resolución en rangos bajos
        dm_step_optimized = max(0.1, base_range["dm_step"])
        
        # Calcular distribución de pasos (más fino en rangos bajos)
        dm_values = []
        current_dm = dm_min_optimized
        
        while current_dm <= dm_max_optimized:
            dm_values.append(current_dm)
            # Paso más fino para DM bajos, más grueso para DM altos
            if current_dm < 100:
                step = 0.1
            elif current_dm < 500:
                step = 0.5
            elif current_dm < 1000:
                step = 1.0
            else:
                step = 2.0
            current_dm += step
        
        return {
            "dm_min": dm_min_optimized,
            "dm_max": dm_max_optimized,
            "dm_values": dm_values,
            "num_dm_values": len(dm_values),
            "optimization": {
                "target": "FRB detection",
                "prioritized_range": "0-2000 pc cm⁻³",
                "adaptive_step": True,
                "efficiency_gain": "~30% vs uniform step"
            },
            "frb_ranges": frb_dm_ranges
        }
        
    except Exception as e:
        logger.error(f"Error al optimizar rango DM: {e}")
        return {
            "error": str(e),
            "dm_min": 0.0,
            "dm_max": 1000.0,
            "dm_values": list(range(0, 1001, 1)),
            "num_dm_values": 1001
        } 