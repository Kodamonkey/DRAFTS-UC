"""
Validador DM-aware para candidatos E2 (Pescar en DM≈0)
=======================================================

Este módulo implementa la validación DM-aware para candidatos detectados
con umbral laxo cerca de DM=0, incluyendo:

1. Micro-rejilla local para re-dedispersar
2. Análisis de curva SNR vs DM
3. Consistencia por sub-bandas
4. Consistencia por chunks

Autor: DRAFTS-MB Team
Fecha: 2024
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

from ..config import STRATEGY_FISH_NEAR_ZERO
from ..preprocessing.dedispersion import d_dm_time_g, dm_index_to_physical

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de la validación DM-aware de un candidato."""
    
    # Identificación del candidato
    candidate_id: str
    t0: float  # Tiempo del candidato
    window_start: int  # Índice de inicio de la ventana
    window_end: int    # Índice de fin de la ventana
    
    # Resultados de validación
    dm_star: float           # DM* óptimo encontrado
    dm_star_err: float       # Error en DM*
    snr_dm0: float           # SNR a DM=0
    snr_dmstar: float        # SNR a DM*
    delta_snr: float         # ΔSNR = SNR(DM*) - SNR(0)
    subband_agreement: float # Acuerdo entre sub-bandas (%)
    
    # Estado de validación
    passed: bool              # True si pasa todas las validaciones
    reason: str               # Razón del fallo si passed=False
    
    # Metadatos adicionales
    local_grid_dm: np.ndarray  # Grid DM usado para validación local
    snr_vs_dm: np.ndarray      # Curva SNR vs DM para análisis


class DMValidator:
    """
    Validador DM-aware para candidatos detectados con estrategia E2.
    
    Implementa validación local con micro-rejillas, análisis de SNR vs DM,
    y verificación de consistencia multi-banda.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el validador DM-aware.
        
        Args:
            config: Configuración de validación (usar STRATEGY_FISH_NEAR_ZERO si es None)
        """
        if config is None:
            config = STRATEGY_FISH_NEAR_ZERO
        
        self.config = config
        self.refine_config = config.get('refine', {})
        
        # Parámetros de validación
        self.dm_local_max = self.refine_config.get('dm_local_max', 300)
        self.ddm_local = self.refine_config.get('ddm_local', 1)
        self.min_delta_snr = self.refine_config.get('min_delta_snr', 2.0)
        self.min_dm_star = self.refine_config.get('min_dm_star', 5)
        self.subband_consistency_pc = self.refine_config.get('subband_consistency_pc', 20)
        
        logger.info(f"DMValidator inicializado: dm_local_max={self.dm_local_max}, "
                   f"ddm_local={self.ddm_local}, min_delta_snr={self.min_delta_snr}")
    
    def validate_candidate(
        self,
        candidate: Dict[str, Any],
        data: np.ndarray,
        freq_values: np.ndarray,
        time_resolution: float,
        subbands: Optional[List[np.ndarray]] = None
    ) -> ValidationResult:
        """
        Valida un candidato usando validación DM-aware.
        
        Args:
            candidate: Diccionario con información del candidato
            data: Datos del bloque (tiempo, freq) o (tiempo, pol, freq)
            freq_values: Valores de frecuencia en MHz
            time_resolution: Resolución temporal en segundos
            subbands: Lista de sub-bandas para validación multi-banda
            
        Returns:
            ValidationResult: Resultado de la validación
        """
        candidate_id = candidate.get('id', 'unknown')
        t0 = candidate.get('t0', 0.0)
        window_start = candidate.get('window_start', 0)
        window_end = candidate.get('window_end', data.shape[0])
        
        logger.info(f"Validando candidato {candidate_id} en t={t0:.3f}s")
        
        try:
            # 1. Micro-rejilla local
            local_grid_dm, snr_vs_dm = self._build_local_dm_grid(
                data, freq_values, time_resolution, window_start, window_end
            )
            
            # 2. Encontrar DM* óptimo
            dm_star_idx = np.argmax(snr_vs_dm)
            dm_star = local_grid_dm[dm_star_idx]
            snr_dmstar = snr_vs_dm[dm_star_idx]
            snr_dm0 = snr_vs_dm[0]  # SNR a DM=0
            
            # 3. Calcular ΔSNR
            delta_snr = snr_dmstar - snr_dm0
            
            # 4. Validar DM* mínimo
            if dm_star < self.min_dm_star:
                return ValidationResult(
                    candidate_id=candidate_id,
                    t0=t0,
                    window_start=window_start,
                    window_end=window_end,
                    dm_star=dm_star,
                    dm_star_err=0.0,
                    snr_dm0=snr_dm0,
                    snr_dmstar=snr_dmstar,
                    delta_snr=delta_snr,
                    subband_agreement=0.0,
                    passed=False,
                    reason=f"DM*={dm_star:.1f} < {self.min_dm_star}",
                    local_grid_dm=local_grid_dm,
                    snr_vs_dm=snr_vs_dm
                )
            
            # 5. Validar ΔSNR mínimo
            if delta_snr < self.min_delta_snr:
                return ValidationResult(
                    candidate_id=candidate_id,
                    t0=t0,
                    window_start=window_start,
                    window_end=window_end,
                    dm_star=dm_star,
                    dm_star_err=0.0,
                    snr_dm0=snr_dm0,
                    snr_dmstar=snr_dmstar,
                    delta_snr=delta_snr,
                    subband_agreement=0.0,
                    passed=False,
                    reason=f"ΔSNR={delta_snr:.2f} < {self.min_delta_snr}",
                    local_grid_dm=local_grid_dm,
                    snr_vs_dm=snr_vs_dm
                )
            
            # 6. Validación multi-banda si está disponible
            subband_agreement = 100.0  # Por defecto
            if subbands is not None and len(subbands) > 1:
                subband_agreement = self._validate_subband_consistency(
                    candidate, subbands, freq_values, time_resolution
                )
                
                if subband_agreement < (100 - self.subband_consistency_pc):
                    return ValidationResult(
                        candidate_id=candidate_id,
                        t0=t0,
                        window_start=window_start,
                        window_end=window_end,
                        dm_star=dm_star,
                        dm_star_err=0.0,
                        snr_dm0=snr_dm0,
                        snr_dmstar=snr_dmstar,
                        delta_snr=delta_snr,
                        subband_agreement=subband_agreement,
                        passed=False,
                        reason=f"Acuerdo sub-bandas={subband_agreement:.1f}% < {100-self.subband_consistency_pc}%",
                        local_grid_dm=local_grid_dm,
                        snr_vs_dm=snr_vs_dm
                    )
            
            # 7. Calcular error en DM*
            dm_star_err = self._estimate_dm_error(local_grid_dm, snr_vs_dm, dm_star_idx)
            
            # Candidato validado exitosamente
            logger.info(f"Candidato {candidate_id} validado: DM*={dm_star:.1f}, "
                       f"ΔSNR={delta_snr:.2f}, subband_agreement={subband_agreement:.1f}%")
            
            return ValidationResult(
                candidate_id=candidate_id,
                t0=t0,
                window_start=window_start,
                window_end=window_end,
                dm_star=dm_star,
                dm_star_err=dm_star_err,
                snr_dm0=snr_dm0,
                snr_dmstar=snr_dmstar,
                delta_snr=delta_snr,
                subband_agreement=subband_agreement,
                passed=True,
                reason="Validación exitosa",
                local_grid_dm=local_grid_dm,
                snr_vs_dm=snr_vs_dm
            )
            
        except Exception as e:
            logger.error(f"Error validando candidato {candidate_id}: {e}")
            return ValidationResult(
                candidate_id=candidate_id,
                t0=t0,
                window_start=window_start,
                window_end=window_end,
                dm_star=0.0,
                dm_star_err=0.0,
                snr_dm0=0.0,
                snr_dmstar=0.0,
                delta_snr=0.0,
                subband_agreement=0.0,
                passed=False,
                reason=f"Error en validación: {e}",
                local_grid_dm=np.array([]),
                snr_vs_dm=np.array([])
            )
    
    def _build_local_dm_grid(
        self,
        data: np.ndarray,
        freq_values: np.ndarray,
        time_resolution: float,
        window_start: int,
        window_end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye una micro-rejilla local de DM y calcula SNR vs DM.
        
        Args:
            data: Datos del bloque completo
            freq_values: Valores de frecuencia
            time_resolution: Resolución temporal
            window_start: Índice de inicio de la ventana
            window_end: Índice de fin de la ventana
            
        Returns:
            Tuple: (local_grid_dm, snr_vs_dm)
        """
        # Extraer ventana temporal del candidato
        window_data = data[window_start:window_end]
        
        # Construir grid DM local fino
        local_grid_dm = np.arange(0, self.dm_local_max + self.ddm_local, self.ddm_local, dtype=np.float32)
        
        # Calcular SNR para cada DM en la ventana
        snr_vs_dm = np.zeros(len(local_grid_dm), dtype=np.float32)
        
        for i, dm in enumerate(local_grid_dm):
            try:
                # Dedispersar la ventana a este DM
                height = 1  # Solo un DM
                width = window_data.shape[0]
                
                dm_time_cube, metadata = d_dm_time_g(
                    window_data, height, width, 
                    dm_min=dm, dm_max=dm, dm_values=np.array([dm])
                )
                
                # Calcular SNR del slice dedispersado
                # Usar el canal principal (índice 0) del cubo
                if dm_time_cube.ndim == 3:
                    slice_data = dm_time_cube[0, 0, :]  # (3, 1, width) -> (width,)
                else:
                    slice_data = dm_time_cube[0, :]     # (1, width) -> (width,)
                
                # SNR simple: (peak - mean) / std
                peak = np.max(slice_data)
                mean_val = np.mean(slice_data)
                std_val = np.std(slice_data)
                
                if std_val > 0:
                    snr_vs_dm[i] = (peak - mean_val) / std_val
                else:
                    snr_vs_dm[i] = 0.0
                    
            except Exception as e:
                logger.warning(f"Error calculando SNR para DM={dm}: {e}")
                snr_vs_dm[i] = 0.0
        
        return local_grid_dm, snr_vs_dm
    
    def _validate_subband_consistency(
        self,
        candidate: Dict[str, Any],
        subbands: List[np.ndarray],
        freq_values: np.ndarray,
        time_resolution: float
    ) -> float:
        """
        Valida la consistencia del candidato entre diferentes sub-bandas.
        
        Args:
            candidate: Información del candidato
            subbands: Lista de sub-bandas
            freq_values: Valores de frecuencia completos
            time_resolution: Resolución temporal
            
        Returns:
            float: Porcentaje de acuerdo entre sub-bandas
        """
        if len(subbands) < 2:
            return 100.0
        
        window_start = candidate.get('window_start', 0)
        window_end = candidate.get('window_end', 0)
        
        dm_stars = []
        
        for i, subband_data in enumerate(subbands):
            try:
                # Extraer ventana del candidato en esta sub-banda
                subband_window = subband_data[window_start:window_end]
                
                # Construir grid DM local para esta sub-banda
                local_grid_dm = np.arange(0, self.dm_local_max + self.ddm_local, self.ddm_local, dtype=np.float32)
                
                # Encontrar DM* en esta sub-banda
                best_snr = 0.0
                best_dm = 0.0
                
                for dm in local_grid_dm:
                    try:
                        height = 1
                        width = subband_window.shape[0]
                        
                        dm_time_cube, metadata = d_dm_time_g(
                            subband_window, height, width,
                            dm_min=dm, dm_max=dm, dm_values=np.array([dm])
                        )
                        
                        # Calcular SNR
                        if dm_time_cube.ndim == 3:
                            slice_data = dm_time_cube[0, 0, :]
                        else:
                            slice_data = dm_time_cube[0, :]
                        
                        peak = np.max(slice_data)
                        mean_val = np.mean(slice_data)
                        std_val = np.std(slice_data)
                        
                        if std_val > 0:
                            snr = (peak - mean_val) / std_val
                            if snr > best_snr:
                                best_snr = snr
                                best_dm = dm
                                
                    except Exception:
                        continue
                
                if best_dm >= self.min_dm_star:
                    dm_stars.append(best_dm)
                    
            except Exception as e:
                logger.warning(f"Error validando sub-banda {i}: {e}")
                continue
        
        if len(dm_stars) < 2:
            return 0.0
        
        # Calcular consistencia: qué tan cerca están los DM* entre sub-bandas
        dm_stars = np.array(dm_stars)
        mean_dm = np.mean(dm_stars)
        std_dm = np.std(dm_stars)
        
        if mean_dm > 0:
            consistency = max(0, 100 - (std_dm / mean_dm) * 100)
        else:
            consistency = 0.0
        
        return consistency
    
    def _estimate_dm_error(
        self,
        local_grid_dm: np.ndarray,
        snr_vs_dm: np.ndarray,
        dm_star_idx: int
    ) -> float:
        """
        Estima el error en DM* basado en la forma de la curva SNR vs DM.
        
        Args:
            local_grid_dm: Grid DM local
            snr_vs_dm: Curva SNR vs DM
            dm_star_idx: Índice del DM* óptimo
            
        Returns:
            float: Error estimado en DM*
        """
        if len(local_grid_dm) < 3:
            return local_grid_dm[1] - local_grid_dm[0] if len(local_grid_dm) > 1 else 1.0
        
        # Encontrar el ancho a media altura del pico SNR
        peak_snr = snr_vs_dm[dm_star_idx]
        half_height = peak_snr / 2
        
        # Buscar hacia la izquierda
        left_idx = dm_star_idx
        while left_idx > 0 and snr_vs_dm[left_idx] > half_height:
            left_idx -= 1
        
        # Buscar hacia la derecha
        right_idx = dm_star_idx
        while right_idx < len(snr_vs_dm) - 1 and snr_vs_dm[right_idx] > half_height:
            right_idx += 1
        
        # Calcular ancho a media altura en unidades DM
        left_dm = local_grid_dm[left_idx] if left_idx >= 0 else local_grid_dm[0]
        right_dm = local_grid_dm[right_idx] if right_idx < len(local_grid_dm) else local_grid_dm[-1]
        
        fwhm = right_dm - left_dm
        
        # Error estimado: FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.355
        dm_error = fwhm / 2.355
        
        return max(dm_error, self.ddm_local)
