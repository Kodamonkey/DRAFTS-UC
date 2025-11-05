# This module calculates visualization ranges for candidates.

"""Dynamic DM range calculator for visualization - optimizes plot ranges based on detected candidates."""

import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class DynamicDMRangeCalculator:
    """Dynamic DM range calculator for optimal visualization."""
    
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
        Calculates optimal DM range centered on detected candidate.
        
        Parameters
        ----------
        dm_optimal : float
            Optimal DM of detected candidate
        dm_global_min : int
            Global minimum DM of analysis
        dm_global_max : int
            Global maximum DM of analysis
        range_factor : float
            Range factor as fraction of optimal DM (0.2 = ±20%)
        min_range_width : float
            Minimum DM range width in pc cm⁻³
        max_range_width : float
            Maximum DM range width in pc cm⁻³
        time_reso : float
            Temporal resolution in seconds
        freq_range : tuple
            Frequency range (freq_min, freq_max) in MHz
            
        Returns
        -------
        tuple
            (dm_plot_min, dm_plot_max, calculation_details)
        """
        
        logger.info("Calculating dynamic DM range for candidate with DM=%.1f", dm_optimal)
        
                                                        
        if dm_optimal > 0:
            base_range_width = dm_optimal * range_factor * 2                    
        else:
            base_range_width = min_range_width
        
                                  
        range_width = np.clip(base_range_width, min_range_width, max_range_width)
        
                                 
        dm_plot_center = dm_optimal
        dm_plot_min = dm_plot_center - range_width / 2
        dm_plot_max = dm_plot_center + range_width / 2
        
                                                    
        if dm_plot_min < dm_global_min:
            dm_plot_min = dm_global_min
            dm_plot_max = min(dm_global_min + range_width, dm_global_max)
        
        if dm_plot_max > dm_global_max:
            dm_plot_max = dm_global_max
            dm_plot_min = max(dm_global_max - range_width, dm_global_min)
        
                                                         
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
        
        logger.info("Dynamic DM range: %.1f - %.1f (width: %.1f, centered on: %.1f)", 
                   dm_plot_min, dm_plot_max, range_width, dm_plot_center)
        
        return dm_plot_min, dm_plot_max, calculation_details
    
    def _analyze_dispersion_for_range(
        self, 
        dm_min: float, 
        dm_max: float, 
        time_reso: float, 
        freq_range: Tuple[float, float]
    ) -> Dict[str, float]:
        """Analyzes dispersion characteristics for specific DM range."""
        
        freq_min, freq_max = freq_range
        K_DM = 4.148808e3                           
        
                                           
        disp_delay_min = K_DM * dm_min * (1/freq_min**2 - 1/freq_max**2)
        disp_delay_max = K_DM * dm_max * (1/freq_min**2 - 1/freq_max**2)
        disp_delay_range = disp_delay_max - disp_delay_min
        
                                                 
        disp_samples_range = disp_delay_range / time_reso
        
        return {
            'disp_delay_min': disp_delay_min,
            'disp_delay_max': disp_delay_max,
            'disp_delay_range': disp_delay_range,
            'disp_samples_range': disp_samples_range,
            'temporal_resolution_factor': disp_samples_range / 512                      
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
        Calculates DM range that covers multiple detected candidates.
        
        Parameters
        ----------
        dm_candidates : list
            List of DMs from detected candidates
        coverage_factor : float
            Coverage factor to include all candidates (1.2 = 20% extra)
            
        Returns
        -------
        tuple
            (dm_plot_min, dm_plot_max, calculation_details)
        """
        
        if not dm_candidates:
                                                                   
            dm_center = (dm_global_min + dm_global_max) / 2
            return self.calculate_optimal_dm_range(
                dm_center, dm_global_min, dm_global_max, **kwargs
            )
        
        dm_candidates = np.array(dm_candidates)
        dm_candidates_min = np.min(dm_candidates)
        dm_candidates_max = np.max(dm_candidates)
        dm_candidates_center = np.mean(dm_candidates)
        dm_candidates_range = dm_candidates_max - dm_candidates_min
        
        logger.info("Calculating range for %d candidates: DM %.1f - %.1f",
                    len(dm_candidates), dm_candidates_min, dm_candidates_max)
        
                                                         
        expanded_range = dm_candidates_range * coverage_factor
        expanded_range = max(expanded_range, kwargs.get('min_range_width', 50.0))
        expanded_range = min(expanded_range, kwargs.get('max_range_width', 200.0))
        
        dm_plot_min = dm_candidates_center - expanded_range / 2
        dm_plot_max = dm_candidates_center + expanded_range / 2
        
                                  
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
        """Calculate an adaptive DM range based on detection confidence.

        Parameters
        ----------
        dm_optimal : float
            Optimal DM of the candidate
        confidence : float
            Detection confidence (0-1)
        adaptive_factor : bool
            If True, adjust the range according to the confidence

        Returns
        -------
        tuple
            (dm_plot_min, dm_plot_max, calculation_details)
        """
        
                                                     
        if adaptive_factor:
                                                            
                                                              
            base_range_factor = kwargs.get('range_factor', 0.2)
            confidence_adjustment = 1.5 - confidence                                        
            adjusted_range_factor = base_range_factor * confidence_adjustment
            kwargs['range_factor'] = adjusted_range_factor
            
            logger.info(
                "Adaptive adjustment: confidence=%.2f, factor=%.3f",
                confidence,
                adjusted_range_factor,
            )
        
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
        """Get the optimized DM range for a specific visualization type.

        Parameters
        ----------
        visualization_type : str
            Visualization type: 'composite', 'patch', 'detailed', 'overview'
        dm_optimal : float
            Optimal DM of the candidate
        confidence : float
            Detection confidence

        Returns
        -------
        tuple
            (dm_plot_min, dm_plot_max, calculation_details)
        """
        
                                                               
        viz_configs = {
            'composite': {
                'range_factor': 0.15,                                      
                'min_range_width': 30.0,
                'max_range_width': 100.0
            },
            'patch': {
                'range_factor': 0.1,                                        
                'min_range_width': 20.0,
                'max_range_width': 60.0
            },
            'detailed': {
                'range_factor': 0.25,                                           
                'min_range_width': 50.0,
                'max_range_width': 150.0
            },
            'overview': {
                'range_factor': 0.4,                                        
                'min_range_width': 100.0,
                'max_range_width': 300.0
            }
        }
        
                                                     
        config = viz_configs.get(visualization_type, viz_configs['detailed'])
        
                                         
        final_kwargs = {**config, **kwargs}
        
        logger.info("Calculating DM range for visualization '%s'", visualization_type)
        
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


                                 
dm_range_calculator = DynamicDMRangeCalculator()

def get_dynamic_dm_range_for_candidate(
    dm_optimal: float,
    config_module,
    visualization_type: str = 'detailed',
    confidence: float = 0.8,
    **kwargs
) -> Tuple[float, float]:
    """Main function to obtain the dynamic DM range in the pipeline.

    Parameters
    ----------
    dm_optimal : float
        Optimal DM of the detected candidate
    config_module : module
        Configuration module
    visualization_type : str
        Visualization type
    confidence : float
        Detection confidence

    Returns
    -------
    tuple
        (dm_plot_min, dm_plot_max)
    """
    
                                   
    dm_global_min = getattr(config_module, 'DM_min', 0)
    dm_global_max = getattr(config_module, 'DM_max', 1024)
    time_reso = getattr(config_module, 'TIME_RESO', 0.001)
    
                                  
    freq_array = getattr(config_module, 'FREQ', None)
    if freq_array is not None and len(freq_array) > 0:
        freq_min, freq_max = float(np.min(freq_array)), float(np.max(freq_array))
    else:
        freq_min, freq_max = 1200.0, 1500.0
    
                             
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
    
    logger.info(
        "Dynamic DM range for %s: %.1f - %.1f (candidate DM=%.1f)",
        visualization_type,
        dm_plot_min,
        dm_plot_max,
        dm_optimal,
    )
    
    return dm_plot_min, dm_plot_max

def get_dynamic_dm_range_for_multiple_candidates(
    dm_candidates: List[float],
    config_module,
    visualization_type: str = 'overview',
    **kwargs
) -> Tuple[float, float]:
    """Obtain a dynamic DM range for multiple candidates.

    Parameters
    ----------
    dm_candidates : list
        List of detected candidate DMs
    config_module : module
        Configuration module
    visualization_type : str
        Visualization type

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
    
    logger.info(
        "DM range for %d candidates: %.1f - %.1f",
        len(dm_candidates),
        dm_plot_min,
        dm_plot_max,
    )
    
    return dm_plot_min, dm_plot_max

if __name__ == "__main__":
                    
    calculator = DynamicDMRangeCalculator()
    
                                     
    dm_opt = 245.5
    dm_min, dm_max, details = calculator.calculate_optimal_dm_range(
        dm_optimal=dm_opt,
        dm_global_min=0,
        dm_global_max=1024,
        range_factor=0.2
    )
    
    print(f"Candidate DM={dm_opt:.1f}")
    print(f"Dynamic range: {dm_min:.1f} - {dm_max:.1f}")
    print(f"Width: {dm_max - dm_min:.1f} pc cm⁻³")
    
                                     
    candidates = [120.3, 125.7, 130.1]
    dm_min_multi, dm_max_multi, details_multi = calculator.calculate_multiple_candidates_range(
        dm_candidates=candidates,
        dm_global_min=0,
        dm_global_max=1024
    )
    
    print(f"\nMultiple candidates: {candidates}")
    print(f"Dynamic range: {dm_min_multi:.1f} - {dm_max_multi:.1f}")
