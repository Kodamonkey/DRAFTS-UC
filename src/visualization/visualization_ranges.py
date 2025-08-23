"""Dynamic DM range calculator for visualization - optimizes plot ranges based on detected candidates."""

import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class DynamicDMRangeCalculator:
    """Calculadora de rangos DM dinámicos para visualización óptima."""
    
    def __init__(self):
        self.dm_ranges_cache = {}
        
    def calculate_optimal_dm_range(
        self,
        dm_optimal: float,
        dm_global_min: int = 0,
        dm_global_max: int = 1024,
        range_factor: float = 0.2,
        min_range_width: float = 50.0,
        max_range_width: float = 200.0,
        time_reso: float = 0.001,
        freq_range: Tuple[float, float] = (1200.0, 1500.0)
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calcula el rango DM óptimo centrado en el candidato detectado.
        
        Parameters
        ----------
        dm_optimal : float
            DM óptimo del candidato detectado
        dm_global_min : int
            DM mínimo global del análisis
        dm_global_max : int
            DM máximo global del análisis
        range_factor : float
            Factor de rango como fracción del DM óptimo (0.2 = ±20%)
        min_range_width : float
            Ancho mínimo del rango DM en pc cm⁻³
        max_range_width : float
            Ancho máximo del rango DM en pc cm⁻³
        time_reso : float
            Resolución temporal en segundos
        freq_range : tuple
            Rango de frecuencias (freq_min, freq_max) en MHz
            
        Returns
        -------
        tuple
            (dm_plot_min, dm_plot_max, calculation_details)
        """
        
        logger.info("Calculando rango DM dinámico para candidato con DM=%.1f", dm_optimal)
        
        # Calcular ancho de rango basado en el DM óptimo
        if dm_optimal > 0:
            base_range_width = dm_optimal * range_factor * 2  # Factor simétrico
        else:
            base_range_width = min_range_width
        
        # Aplicar límites de ancho
        range_width = np.clip(base_range_width, min_range_width, max_range_width)
        
        # Calcular rango centrado
        dm_plot_center = dm_optimal
        dm_plot_min = dm_plot_center - range_width / 2
        dm_plot_max = dm_plot_center + range_width / 2
        
        # Ajustar si se sale de los límites globales
        if dm_plot_min < dm_global_min:
            dm_plot_min = dm_global_min
            dm_plot_max = min(dm_global_min + range_width, dm_global_max)
        
        if dm_plot_max > dm_global_max:
            dm_plot_max = dm_global_max
            dm_plot_min = max(dm_global_max - range_width, dm_global_min)
        
        # Análisis de dispersión temporal para este rango
        dispersion_analysis = self._analyze_dispersion_for_range(
            dm_plot_min, dm_plot_max, time_reso, freq_range
        )
        
        calculation_details = {
            'dm_optimal': dm_optimal,
            'dm_plot_min': dm_plot_min,
            'dm_plot_max': dm_plot_max,
            'dm_plot_center': dm_plot_center,
            'range_width': range_width,
            'base_range_width': base_range_width,
            'range_factor_used': range_factor,
            'constrained_by_global_limits': (
                dm_plot_min <= dm_global_min or dm_plot_max >= dm_global_max
            ),
            'dispersion_analysis': dispersion_analysis
        }
        
        logger.info("Rango DM dinámico: %.1f - %.1f (ancho: %.1f, centrado en: %.1f)", 
                   dm_plot_min, dm_plot_max, range_width, dm_plot_center)
        
        return dm_plot_min, dm_plot_max, calculation_details
    
    def _analyze_dispersion_for_range(
        self, 
        dm_min: float, 
        dm_max: float, 
        time_reso: float, 
        freq_range: Tuple[float, float]
    ) -> Dict[str, float]:
        """Analiza características de dispersión para el rango DM específico."""
        
        freq_min, freq_max = freq_range
        K_DM = 4.148808e3  # Constante de dispersión
        
        # Dispersión temporal para el rango
        disp_delay_min = K_DM * dm_min * (1/freq_min**2 - 1/freq_max**2)
        disp_delay_max = K_DM * dm_max * (1/freq_min**2 - 1/freq_max**2)
        disp_delay_range = disp_delay_max - disp_delay_min
        
        # Número de muestras temporales afectadas
        disp_samples_range = disp_delay_range / time_reso
        
        return {
            'disp_delay_min': disp_delay_min,
            'disp_delay_max': disp_delay_max,
            'disp_delay_range': disp_delay_range,
            'disp_samples_range': disp_samples_range,
            'temporal_resolution_factor': disp_samples_range / 512  # Para plots 512x512
        }
    
    def calculate_multiple_candidates_range(
        self,
        dm_candidates: List[float],
        dm_global_min: int = 0,
        dm_global_max: int = 1024,
        coverage_factor: float = 1.2,
        **kwargs
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calcula rango DM que cubra múltiples candidatos detectados.
        
        Parameters
        ----------
        dm_candidates : list
            Lista de DMs de candidatos detectados
        coverage_factor : float
            Factor de cobertura para incluir todos los candidatos (1.2 = 20% extra)
            
        Returns
        -------
        tuple
            (dm_plot_min, dm_plot_max, calculation_details)
        """
        
        if not dm_candidates:
            # Si no hay candidatos, usar rango centrado en el medio
            dm_center = (dm_global_min + dm_global_max) / 2
            return self.calculate_optimal_dm_range(
                dm_center, dm_global_min, dm_global_max, **kwargs
            )
        
        dm_candidates = np.array(dm_candidates)
        dm_candidates_min = np.min(dm_candidates)
        dm_candidates_max = np.max(dm_candidates)
        dm_candidates_center = np.mean(dm_candidates)
        dm_candidates_range = dm_candidates_max - dm_candidates_min
        
        logger.info("Calculando rango para %d candidatos: DM %.1f - %.1f", 
                   len(dm_candidates), dm_candidates_min, dm_candidates_max)
        
        # Expandir rango para cubrir todos los candidatos
        expanded_range = dm_candidates_range * coverage_factor
        expanded_range = max(expanded_range, kwargs.get('min_range_width', 50.0))
        expanded_range = min(expanded_range, kwargs.get('max_range_width', 200.0))
        
        dm_plot_min = dm_candidates_center - expanded_range / 2
        dm_plot_max = dm_candidates_center + expanded_range / 2
        
        # Aplicar límites globales
        dm_plot_min = max(dm_plot_min, dm_global_min)
        dm_plot_max = min(dm_plot_max, dm_global_max)
        
        calculation_details = {
            'dm_candidates': dm_candidates.tolist(),
            'dm_candidates_min': dm_candidates_min,
            'dm_candidates_max': dm_candidates_max,
            'dm_candidates_center': dm_candidates_center,
            'dm_candidates_range': dm_candidates_range,
            'expanded_range': expanded_range,
            'dm_plot_min': dm_plot_min,
            'dm_plot_max': dm_plot_max,
            'coverage_factor': coverage_factor,
            'n_candidates': len(dm_candidates)
        }
        
        return dm_plot_min, dm_plot_max, calculation_details
    
    def calculate_adaptive_dm_range(
        self,
        dm_optimal: float,
        confidence: float = 0.8,
        dm_global_min: int = 0,
        dm_global_max: int = 1024,
        adaptive_factor: bool = True,
        **kwargs
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calcula rango DM adaptativo basado en confianza de detección.
        
        Parameters
        ----------
        dm_optimal : float
            DM óptimo del candidato
        confidence : float
            Confianza de la detección (0-1)
        adaptive_factor : bool
            Si True, ajusta el rango basado en la confianza
            
        Returns
        -------
        tuple
            (dm_plot_min, dm_plot_max, calculation_details)
        """
        
        # Ajustar factor de rango basado en confianza
        if adaptive_factor:
            # Alta confianza → rango más estrecho (más zoom)
            # Baja confianza → rango más amplio (más contexto)
            base_range_factor = kwargs.get('range_factor', 0.2)
            confidence_adjustment = 1.5 - confidence  # 0.7 para conf=0.8, 1.0 para conf=0.5
            adjusted_range_factor = base_range_factor * confidence_adjustment
            kwargs['range_factor'] = adjusted_range_factor
            
            logger.info("Ajuste adaptativo: confianza=%.2f, factor=%.3f", 
                       confidence, adjusted_range_factor)
        
        dm_plot_min, dm_plot_max, details = self.calculate_optimal_dm_range(
            dm_optimal, dm_global_min, dm_global_max, **kwargs
        )
        
        details.update({
            'confidence': confidence,
            'adaptive_factor_enabled': adaptive_factor,
            'confidence_adjustment': 1.5 - confidence if adaptive_factor else 1.0
        })
        
        return dm_plot_min, dm_plot_max, details
    
    def get_dm_range_for_visualization(
        self,
        visualization_type: str,
        dm_optimal: float,
        confidence: float = 0.8,
        dm_global_min: int = 0,
        dm_global_max: int = 1024,
        **kwargs
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Obtiene rango DM optimizado para tipo específico de visualización.
        
        Parameters
        ----------
        visualization_type : str
            Tipo de visualización: 'composite', 'patch', 'detailed', 'overview'
        dm_optimal : float
            DM óptimo del candidato
        confidence : float
            Confianza de la detección
            
        Returns
        -------
        tuple
            (dm_plot_min, dm_plot_max, calculation_details)
        """
        
        # Configuraciones específicas por tipo de visualización
        viz_configs = {
            'composite': {
                'range_factor': 0.15,      # Rango estrecho para composites
                'min_range_width': 30.0,
                'max_range_width': 100.0
            },
            'patch': {
                'range_factor': 0.1,       # Rango muy estrecho para patches
                'min_range_width': 20.0,
                'max_range_width': 60.0
            },
            'detailed': {
                'range_factor': 0.25,      # Rango medio para análisis detallado
                'min_range_width': 50.0,
                'max_range_width': 150.0
            },
            'overview': {
                'range_factor': 0.4,       # Rango amplio para vista general
                'min_range_width': 100.0,
                'max_range_width': 300.0
            }
        }
        
        # Usar configuración específica o por defecto
        config = viz_configs.get(visualization_type, viz_configs['detailed'])
        
        # Combinar con kwargs del usuario
        final_kwargs = {**config, **kwargs}
        
        logger.info("Calculando rango DM para visualización '%s'", visualization_type)
        
        dm_plot_min, dm_plot_max, details = self.calculate_adaptive_dm_range(
            dm_optimal=dm_optimal,
            confidence=confidence,
            dm_global_min=dm_global_min,
            dm_global_max=dm_global_max,
            **final_kwargs
        )
        
        details.update({
            'visualization_type': visualization_type,
            'viz_config_used': config
        })
        
        return dm_plot_min, dm_plot_max, details


# Instancia global del calculador
dm_range_calculator = DynamicDMRangeCalculator()

def get_dynamic_dm_range_for_candidate(
    dm_optimal: float,
    config_module,
    visualization_type: str = 'detailed',
    confidence: float = 0.8,
    **kwargs
) -> Tuple[float, float]:
    """
    Función principal para obtener rango DM dinámico en el pipeline.
    
    Parameters
    ----------
    dm_optimal : float
        DM óptimo del candidato detectado
    config_module : module
        Módulo de configuración
    visualization_type : str
        Tipo de visualización
    confidence : float
        Confianza de la detección
        
    Returns
    -------
    tuple
        (dm_plot_min, dm_plot_max)
    """
    
    # Obtener parámetros del config
    dm_global_min = getattr(config_module, 'DM_min', 0)
    dm_global_max = getattr(config_module, 'DM_max', 1024)
    time_reso = getattr(config_module, 'TIME_RESO', 0.001)
    
    # Obtener rango de frecuencias
    freq_array = getattr(config_module, 'FREQ', None)
    if freq_array is not None and len(freq_array) > 0:
        freq_min, freq_max = float(np.min(freq_array)), float(np.max(freq_array))
    else:
        freq_min, freq_max = 1200.0, 1500.0
    
    # Calcular rango dinámico
    dm_plot_min, dm_plot_max, details = dm_range_calculator.get_dm_range_for_visualization(
        visualization_type=visualization_type,
        dm_optimal=dm_optimal,
        confidence=confidence,
        dm_global_min=dm_global_min,
        dm_global_max=dm_global_max,
        time_reso=time_reso,
        freq_range=(freq_min, freq_max),
        **kwargs
    )
    
    logger.info("Rango DM dinámico para %s: %.1f - %.1f (candidato DM=%.1f)", 
               visualization_type, dm_plot_min, dm_plot_max, dm_optimal)
    
    return dm_plot_min, dm_plot_max

def get_dynamic_dm_range_for_multiple_candidates(
    dm_candidates: List[float],
    config_module,
    visualization_type: str = 'overview',
    **kwargs
) -> Tuple[float, float]:
    """
    Obtiene rango DM dinámico para múltiples candidatos.
    
    Parameters
    ----------
    dm_candidates : list
        Lista de DMs de candidatos detectados
    config_module : module
        Módulo de configuración
    visualization_type : str
        Tipo de visualización
        
    Returns
    -------
    tuple
        (dm_plot_min, dm_plot_max)
    """
    
    dm_global_min = getattr(config_module, 'DM_min', 0)
    dm_global_max = getattr(config_module, 'DM_max', 1024)
    
    dm_plot_min, dm_plot_max, details = dm_range_calculator.calculate_multiple_candidates_range(
        dm_candidates=dm_candidates,
        dm_global_min=dm_global_min,
        dm_global_max=dm_global_max,
        **kwargs
    )
    
    logger.info("Rango DM para %d candidatos: %.1f - %.1f", 
               len(dm_candidates), dm_plot_min, dm_plot_max)
    
    return dm_plot_min, dm_plot_max

if __name__ == "__main__":
    # Ejemplo de uso
    calculator = DynamicDMRangeCalculator()
    
    # Ejemplo 1: Candidato individual
    dm_opt = 245.5
    dm_min, dm_max, details = calculator.calculate_optimal_dm_range(
        dm_optimal=dm_opt,
        dm_global_min=0,
        dm_global_max=1024,
        range_factor=0.2
    )
    
    print(f"Candidato DM={dm_opt:.1f}")
    print(f"Rango dinámico: {dm_min:.1f} - {dm_max:.1f}")
    print(f"Ancho: {dm_max - dm_min:.1f} pc cm⁻³")
    
    # Ejemplo 2: Múltiples candidatos
    candidates = [120.3, 125.7, 130.1]
    dm_min_multi, dm_max_multi, details_multi = calculator.calculate_multiple_candidates_range(
        dm_candidates=candidates,
        dm_global_min=0,
        dm_global_max=1024
    )
    
    print(f"\nMúltiples candidatos: {candidates}")
    print(f"Rango dinámico: {dm_min_multi:.1f} - {dm_max_multi:.1f}")
