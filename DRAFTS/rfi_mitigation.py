"""
Módulo de mitigación de RFI (Radio Frequency Interference) para el pipeline DRAFTS.

Este módulo implementa técnicas avanzadas de limpieza de RFI incluyendo:
- Enmascarado de canales contaminados
- Filtrado Zero-DM 
- Filtrado de impulsos
- Análisis de polarización
- Detección automática de RFI

Autor: DRAFTS Pipeline Team
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from scipy import ndimage, signal
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
from pathlib import Path


class RFIMitigator:
    """
    Clase principal para mitigación de RFI en datos de pulsar/FRB.
    
    Implementa múltiples técnicas de limpieza de RFI para mejorar
    la detección de señales dispersas.
    """
    
    def __init__(self, 
                 freq_sigma_thresh: float = 5.0,
                 time_sigma_thresh: float = 5.0,
                 zero_dm_sigma_thresh: float = 4.0,
                 impulse_sigma_thresh: float = 6.0,
                 polarization_thresh: float = 0.8):
        """
        Inicializa el mitigador de RFI.
        
        Parameters
        ----------
        freq_sigma_thresh : float
            Umbral sigma para enmascarado de canales de frecuencia
        time_sigma_thresh : float
            Umbral sigma para enmascarado temporal
        zero_dm_sigma_thresh : float
            Umbral sigma para filtro Zero-DM
        impulse_sigma_thresh : float
            Umbral sigma para filtrado de impulsos
        polarization_thresh : float
            Umbral para análisis de polarización (0-1)
        """
        self.freq_sigma_thresh = freq_sigma_thresh
        self.time_sigma_thresh = time_sigma_thresh
        self.zero_dm_sigma_thresh = zero_dm_sigma_thresh
        self.impulse_sigma_thresh = impulse_sigma_thresh
        self.polarization_thresh = polarization_thresh
        
        # Máscaras y estadísticas
        self.freq_mask = None
        self.time_mask = None
        self.rfi_stats = {}
    
    def detect_bad_channels(self, 
                          waterfall: np.ndarray,
                          method: str = "mad") -> np.ndarray:
        """
        Detecta canales de frecuencia contaminados por RFI.
        
        Parameters
        ----------
        waterfall : np.ndarray
            Datos waterfall (tiempo, frecuencia)
        method : str
            Método de detección: "mad", "std", "kurtosis"
            
        Returns
        -------
        np.ndarray
            Máscara booleana de canales buenos (True = bueno)
        """
        n_time, n_freq = waterfall.shape
        
        # Calcula estadísticas por canal de frecuencia
        freq_means = np.mean(waterfall, axis=0)
        freq_stds = np.std(waterfall, axis=0)
        
        if method == "mad":
            # Método robusto usando MAD
            freq_mads = median_abs_deviation(waterfall, axis=0)
            global_mad = np.median(freq_mads)
            mad_deviation = np.abs(freq_mads - global_mad) / global_mad
            bad_channels = mad_deviation > self.freq_sigma_thresh
            
        elif method == "std":
            # Método basado en desviación estándar
            global_std = np.median(freq_stds)
            std_deviation = np.abs(freq_stds - global_std) / global_std
            bad_channels = std_deviation > self.freq_sigma_thresh
            
        elif method == "kurtosis":
            # Método basado en curtosis (detecta no-gaussianidad)
            from scipy.stats import kurtosis
            freq_kurtosis = kurtosis(waterfall, axis=0)
            kurt_threshold = np.median(freq_kurtosis) + self.freq_sigma_thresh * np.std(freq_kurtosis)
            bad_channels = freq_kurtosis > kurt_threshold
            
        else:
            raise ValueError(f"Método desconocido: {method}")
        
        # Invierte para obtener canales buenos
        good_channels = ~bad_channels
        
        # Guarda estadísticas
        self.rfi_stats['bad_channels'] = np.sum(bad_channels)
        self.rfi_stats['good_channels'] = np.sum(good_channels)
        self.rfi_stats['channel_fraction_flagged'] = np.sum(bad_channels) / n_freq
        
        return good_channels
    
    def detect_bad_time_samples(self, 
                               waterfall: np.ndarray,
                               method: str = "mad") -> np.ndarray:
        """
        Detecta muestras temporales contaminadas por RFI.
        
        Parameters
        ----------
        waterfall : np.ndarray
            Datos waterfall (tiempo, frecuencia)
        method : str
            Método de detección: "mad", "std", "outlier"
            
        Returns
        -------
        np.ndarray
            Máscara booleana de muestras temporales buenas (True = bueno)
        """
        n_time, n_freq = waterfall.shape
        
        # Calcula estadísticas por muestra temporal
        time_profile = np.mean(waterfall, axis=1)
        
        if method == "mad":
            # Método robusto usando MAD
            time_mad = median_abs_deviation(time_profile)
            time_median = np.median(time_profile)
            outliers = np.abs(time_profile - time_median) > self.time_sigma_thresh * time_mad
            
        elif method == "std":
            # Método basado en desviación estándar
            time_mean = np.mean(time_profile)
            time_std = np.std(time_profile)
            outliers = np.abs(time_profile - time_mean) > self.time_sigma_thresh * time_std
            
        elif method == "outlier":
            # Método basado en percentiles
            q25, q75 = np.percentile(time_profile, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - self.time_sigma_thresh * iqr
            upper_bound = q75 + self.time_sigma_thresh * iqr
            outliers = (time_profile < lower_bound) | (time_profile > upper_bound)
            
        else:
            raise ValueError(f"Método desconocido: {method}")
        
        # Invierte para obtener muestras buenas
        good_samples = ~outliers
        
        # Guarda estadísticas
        self.rfi_stats['bad_time_samples'] = np.sum(outliers)
        self.rfi_stats['good_time_samples'] = np.sum(good_samples)
        self.rfi_stats['time_fraction_flagged'] = np.sum(outliers) / n_time
        
        return good_samples
    
    def zero_dm_filter(self, 
                      waterfall: np.ndarray,
                      zero_dm_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aplica filtro Zero-DM para eliminar RFI no dispersa.
        
        El filtro Zero-DM resta la señal promedio que no está dispersa,
        preservando solo señales con dispersión temporal característica de pulsares/FRBs.
        
        Parameters
        ----------
        waterfall : np.ndarray
            Datos waterfall (tiempo, frecuencia)
        zero_dm_profile : np.ndarray, optional
            Perfil Zero-DM personalizado. Si None, se calcula automáticamente.
            
        Returns
        -------
        np.ndarray
            Waterfall filtrado
        """
        if zero_dm_profile is None:
            # Calcula perfil Zero-DM como promedio en frecuencia
            zero_dm_profile = np.mean(waterfall, axis=1)
        
        # Detecta outliers en el perfil Zero-DM
        zero_dm_mad = median_abs_deviation(zero_dm_profile)
        zero_dm_median = np.median(zero_dm_profile)
        
        # Identifica muestras con RFI fuerte
        rfi_mask = np.abs(zero_dm_profile - zero_dm_median) > self.zero_dm_sigma_thresh * zero_dm_mad
        
        # Suaviza el perfil Zero-DM para evitar artefactos
        smoothed_profile = signal.medfilt(zero_dm_profile, kernel_size=5)
        
        # Aplica filtro solo donde no hay RFI detectado
        filtered_profile = zero_dm_profile.copy()
        filtered_profile[~rfi_mask] = smoothed_profile[~rfi_mask]
        
        # Resta el perfil Zero-DM filtrado
        waterfall_filtered = waterfall - filtered_profile[:, np.newaxis]
        
        # Guarda estadísticas
        self.rfi_stats['zero_dm_flagged'] = np.sum(rfi_mask)
        self.rfi_stats['zero_dm_fraction'] = np.sum(rfi_mask) / len(zero_dm_profile)
        
        return waterfall_filtered
    
    def impulse_filter(self, 
                      waterfall: np.ndarray,
                      kernel_size: int = 3) -> np.ndarray:
        """
        Aplica filtrado de impulsos para eliminar RFI impulsivo.
        
        Parameters
        ----------
        waterfall : np.ndarray
            Datos waterfall (tiempo, frecuencia)
        kernel_size : int
            Tamaño del kernel para filtro mediano
            
        Returns
        -------
        np.ndarray
            Waterfall filtrado
        """
        # Aplica filtro mediano 2D para detectar impulsos
        filtered_waterfall = ndimage.median_filter(waterfall, size=kernel_size)
        
        # Calcula residuos
        residuals = waterfall - filtered_waterfall
        
        # Detecta impulsos usando MAD
        residual_mad = median_abs_deviation(residuals.flatten())
        impulse_mask = np.abs(residuals) > self.impulse_sigma_thresh * residual_mad
        
        # Reemplaza impulsos con valores filtrados
        waterfall_cleaned = waterfall.copy()
        waterfall_cleaned[impulse_mask] = filtered_waterfall[impulse_mask]
        
        # Guarda estadísticas
        self.rfi_stats['impulses_flagged'] = np.sum(impulse_mask)
        self.rfi_stats['impulse_fraction'] = np.sum(impulse_mask) / waterfall.size
        
        return waterfall_cleaned
    
    def polarization_filter(self, 
                           stokes_i: np.ndarray,
                           stokes_v: Optional[np.ndarray] = None,
                           stokes_q: Optional[np.ndarray] = None,
                           stokes_u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aplica filtrado basado en polarización para eliminar RFI.
        
        Las señales de pulsar/FRB suelen tener características de polarización
        diferentes a las de RFI terrestres.
        
        Parameters
        ----------
        stokes_i : np.ndarray
            Intensidad total (Stokes I)
        stokes_v : np.ndarray, optional
            Polarización circular (Stokes V)
        stokes_q : np.ndarray, optional
            Polarización lineal Q (Stokes Q)
        stokes_u : np.ndarray, optional
            Polarización lineal U (Stokes U)
            
        Returns
        -------
        np.ndarray
            Waterfall filtrado basado en polarización
        """
        if stokes_v is None:
            # Si no hay información de polarización, retorna datos originales
            return stokes_i
        
        # Calcula grado de polarización
        if stokes_q is not None and stokes_u is not None:
            # Polarización total
            pol_intensity = np.sqrt(stokes_q**2 + stokes_u**2 + stokes_v**2)
        else:
            # Solo polarización circular
            pol_intensity = np.abs(stokes_v)
        
        # Calcula fracción de polarización
        # Evita división por cero
        epsilon = 1e-10
        pol_fraction = pol_intensity / (stokes_i + epsilon)
        
        # Identifica regiones con polarización anómala (típicamente RFI)
        high_pol_mask = pol_fraction > self.polarization_thresh
        
        # Aplica filtro mediano en regiones de alta polarización
        filtered_stokes_i = stokes_i.copy()
        if np.any(high_pol_mask):
            filtered_stokes_i[high_pol_mask] = ndimage.median_filter(
                stokes_i, size=3
            )[high_pol_mask]
        
        # Guarda estadísticas
        self.rfi_stats['high_pol_flagged'] = np.sum(high_pol_mask)
        self.rfi_stats['pol_fraction'] = np.sum(high_pol_mask) / stokes_i.size
        
        return filtered_stokes_i
    
    def apply_masks(self, 
                   waterfall: np.ndarray,
                   freq_mask: Optional[np.ndarray] = None,
                   time_mask: Optional[np.ndarray] = None,
                   interpolate: bool = True) -> np.ndarray:
        """
        Aplica máscaras de frecuencia y tiempo al waterfall.
        
        Parameters
        ----------
        waterfall : np.ndarray
            Datos waterfall (tiempo, frecuencia)
        freq_mask : np.ndarray, optional
            Máscara de frecuencia (True = bueno)
        time_mask : np.ndarray, optional
            Máscara de tiempo (True = bueno)
        interpolate : bool
            Si True, interpola valores enmascarados
            
        Returns
        -------
        np.ndarray
            Waterfall enmascarado
        """
        masked_waterfall = waterfall.copy()
        
        if freq_mask is not None:
            # Enmascara canales de frecuencia malos
            masked_waterfall[:, ~freq_mask] = np.nan
            
        if time_mask is not None:
            # Enmascara muestras temporales malas
            masked_waterfall[~time_mask, :] = np.nan
            
        if interpolate:
            # Interpola valores enmascarados
            masked_waterfall = self._interpolate_masked_values(masked_waterfall)
        
        return masked_waterfall
    
    def _interpolate_masked_values(self, waterfall: np.ndarray) -> np.ndarray:
        """
        Interpola valores enmascarados (NaN) en el waterfall.
        
        Parameters
        ----------
        waterfall : np.ndarray
            Waterfall con valores NaN para interpolar
            
        Returns
        -------
        np.ndarray
            Waterfall interpolado
        """
        # Usa interpolación 2D para rellenar valores NaN
        from scipy.interpolate import griddata
        
        n_time, n_freq = waterfall.shape
        
        # Crea grilla de coordenadas
        time_coords, freq_coords = np.meshgrid(
            np.arange(n_time), 
            np.arange(n_freq), 
            indexing='ij'
        )
        
        # Identifica puntos válidos y NaN
        valid_mask = ~np.isnan(waterfall)
        
        if np.sum(valid_mask) < 0.1 * waterfall.size:
            # Si hay muy pocos puntos válidos, usa valores medianos
            waterfall_filled = np.full_like(waterfall, np.nanmedian(waterfall))
        else:
            # Interpola usando puntos válidos
            valid_points = np.column_stack([
                time_coords[valid_mask],
                freq_coords[valid_mask]
            ])
            valid_values = waterfall[valid_mask]
            
            # Interpola en toda la grilla
            waterfall_filled = griddata(
                valid_points, 
                valid_values,
                (time_coords, freq_coords),
                method='nearest'
            )
        
        return waterfall_filled
    
    def clean_waterfall(self, 
                       waterfall: np.ndarray,
                       stokes_v: Optional[np.ndarray] = None,
                       stokes_q: Optional[np.ndarray] = None,
                       stokes_u: Optional[np.ndarray] = None,
                       apply_all_filters: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aplica pipeline completo de limpieza de RFI.
        
        Parameters
        ----------
        waterfall : np.ndarray
            Datos waterfall (tiempo, frecuencia)
        stokes_v : np.ndarray, optional
            Polarización circular (Stokes V)
        stokes_q : np.ndarray, optional
            Polarización lineal Q (Stokes Q)
        stokes_u : np.ndarray, optional
            Polarización lineal U (Stokes U)
        apply_all_filters : bool
            Si True, aplica todos los filtros disponibles
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Waterfall limpio y estadísticas de RFI
        """
        print("[INFO] Iniciando limpieza de RFI...")
        
        # Resetea estadísticas
        self.rfi_stats = {}
        
        # Guarda datos originales
        original_waterfall = waterfall.copy()
        cleaned_waterfall = waterfall.copy()
        
        if apply_all_filters:
            # 1. Detecta canales de frecuencia malos
            print("[INFO] Detectando canales de frecuencia contaminados...")
            freq_mask = self.detect_bad_channels(cleaned_waterfall)
            
            # 2. Detecta muestras temporales malas
            print("[INFO] Detectando muestras temporales contaminadas...")
            time_mask = self.detect_bad_time_samples(cleaned_waterfall)
            
            # 3. Aplica filtro Zero-DM
            print("[INFO] Aplicando filtro Zero-DM...")
            cleaned_waterfall = self.zero_dm_filter(cleaned_waterfall)
            
            # 4. Aplica filtrado de impulsos
            print("[INFO] Aplicando filtrado de impulsos...")
            cleaned_waterfall = self.impulse_filter(cleaned_waterfall)
            
            # 5. Aplica filtrado de polarización (si disponible)
            if stokes_v is not None:
                print("[INFO] Aplicando filtrado de polarización...")
                cleaned_waterfall = self.polarization_filter(
                    cleaned_waterfall, stokes_v, stokes_q, stokes_u
                )
            
            # 6. Aplica máscaras finales
            print("[INFO] Aplicando máscaras finales...")
            cleaned_waterfall = self.apply_masks(
                cleaned_waterfall, freq_mask, time_mask, interpolate=True
            )
        
        # Calcula estadísticas finales
        self.rfi_stats['total_flagged_fraction'] = (
            np.sum(np.isnan(cleaned_waterfall)) / cleaned_waterfall.size
        )
        
        # Asegura que no quedan NaN
        cleaned_waterfall = np.nan_to_num(cleaned_waterfall)
        
        print(f"[INFO] Limpieza completada. Fracción total flagged: {self.rfi_stats['total_flagged_fraction']:.3f}")
        
        return cleaned_waterfall, self.rfi_stats
    
    def plot_rfi_diagnostics(self, 
                           original_waterfall: np.ndarray,
                           cleaned_waterfall: np.ndarray,
                           output_path: Optional[Path] = None) -> None:
        """
        Genera gráficos de diagnóstico de limpieza de RFI.
        
        Parameters
        ----------
        original_waterfall : np.ndarray
            Datos originales
        cleaned_waterfall : np.ndarray
            Datos limpiados
        output_path : Path, optional
            Ruta para guardar los gráficos
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Waterfall original
        im1 = axes[0, 0].imshow(original_waterfall.T, aspect='auto', origin='lower')
        axes[0, 0].set_title('Waterfall Original')
        axes[0, 0].set_xlabel('Tiempo')
        axes[0, 0].set_ylabel('Frecuencia')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Waterfall limpio
        im2 = axes[0, 1].imshow(cleaned_waterfall.T, aspect='auto', origin='lower')
        axes[0, 1].set_title('Waterfall Limpio')
        axes[0, 1].set_xlabel('Tiempo')
        axes[0, 1].set_ylabel('Frecuencia')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Diferencia
        diff = original_waterfall - cleaned_waterfall
        im3 = axes[0, 2].imshow(diff.T, aspect='auto', origin='lower')
        axes[0, 2].set_title('RFI Removido')
        axes[0, 2].set_xlabel('Tiempo')
        axes[0, 2].set_ylabel('Frecuencia')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Perfiles temporales
        time_orig = np.mean(original_waterfall, axis=1)
        time_clean = np.mean(cleaned_waterfall, axis=1)
        axes[1, 0].plot(time_orig, label='Original', alpha=0.7)
        axes[1, 0].plot(time_clean, label='Limpio', alpha=0.7)
        axes[1, 0].set_title('Perfil Temporal')
        axes[1, 0].set_xlabel('Tiempo')
        axes[1, 0].set_ylabel('Intensidad')
        axes[1, 0].legend()
        
        # Espectros
        freq_orig = np.mean(original_waterfall, axis=0)
        freq_clean = np.mean(cleaned_waterfall, axis=0)
        axes[1, 1].plot(freq_orig, label='Original', alpha=0.7)
        axes[1, 1].plot(freq_clean, label='Limpio', alpha=0.7)
        axes[1, 1].set_title('Espectro Promedio')
        axes[1, 1].set_xlabel('Frecuencia')
        axes[1, 1].set_ylabel('Intensidad')
        axes[1, 1].legend()
        
        # Estadísticas de RFI
        if hasattr(self, 'rfi_stats') and self.rfi_stats:
            stats_text = '\n'.join([
                f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}"
                for key, value in self.rfi_stats.items()
            ])
            axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                           verticalalignment='top', fontsize=10)
            axes[1, 2].set_title('Estadísticas RFI')
            axes[1, 2].set_xticks([])
            axes[1, 2].set_yticks([])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Gráficos guardados en: {output_path}")
        
        plt.show()


def create_rfi_config_section() -> str:
    """
    Genera sección de configuración de RFI para config.py.
    
    Returns
    -------
    str
        Configuración de RFI formateada
    """
    return """
# Configuración de Mitigación de RFI ----------------------------------------
RFI_FREQ_SIGMA_THRESH = 5.0      # Umbral sigma para enmascarado de canales
RFI_TIME_SIGMA_THRESH = 5.0      # Umbral sigma para enmascarado temporal  
RFI_ZERO_DM_SIGMA_THRESH = 4.0   # Umbral sigma para filtro Zero-DM
RFI_IMPULSE_SIGMA_THRESH = 6.0   # Umbral sigma para filtrado de impulsos
RFI_POLARIZATION_THRESH = 0.8    # Umbral para filtrado de polarización
RFI_ENABLE_ALL_FILTERS = True    # Habilita todos los filtros de RFI
RFI_INTERPOLATE_MASKED = True    # Interpola valores enmascarados
RFI_SAVE_DIAGNOSTICS = True      # Guarda gráficos de diagnóstico
"""
