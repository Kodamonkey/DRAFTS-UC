"""
Sistema de cálculo automático de SLICE_LEN ideal basado en metadatos del archivo.

Este módulo analiza automáticamente las características del archivo (resolución temporal,
ancho de banda, dispersión esperada, etc.) y calcula el SLICE_LEN óptimo para maximizar
la resolución de detección sin perder contexto temporal.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class SliceLenOptimizer:
    """Optimizador automático de SLICE_LEN basado en metadatos del archivo."""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_file_characteristics(
        self,
        time_reso: float,
        freq_reso: float,
        file_length: int,
        freq_range: Tuple[float, float],
        dm_max: int = 1024,
        down_time_rate: int = 1,
        filename: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Analiza las características del archivo para optimizar SLICE_LEN.
        
        Parameters
        ----------
        time_reso : float
            Resolución temporal en segundos por muestra
        freq_reso : float
            Resolución en frecuencia en MHz
        file_length : int
            Longitud del archivo en muestras
        freq_range : tuple
            Rango de frecuencias (freq_min, freq_max) en MHz
        dm_max : int
            DM máximo a considerar
        down_time_rate : int
            Factor de decimación temporal
        filename : str
            Nombre del archivo para logs
            
        Returns
        -------
        dict
            Diccionario con análisis completo del archivo
        """
        
        logger.info("Analizando características del archivo: %s", filename)
        
        # Parámetros básicos
        effective_time_reso = time_reso * down_time_rate
        total_duration = file_length * time_reso
        freq_min, freq_max = freq_range
        bandwidth = freq_max - freq_min
        
        # Análisis de dispersión temporal
        dispersion_analysis = self._analyze_dispersion_characteristics(
            freq_min, freq_max, dm_max, effective_time_reso
        )
        
        # Análisis de resolución requerida
        resolution_analysis = self._analyze_resolution_requirements(
            effective_time_reso, total_duration, bandwidth
        )
        
        # Análisis de contenido del archivo
        content_analysis = self._analyze_file_content(
            filename, file_length, effective_time_reso
        )
        
        # Análisis de procesamiento
        processing_analysis = self._analyze_processing_requirements(
            file_length, effective_time_reso
        )
        
        analysis = {
            'basic_params': {
                'time_reso': time_reso,
                'effective_time_reso': effective_time_reso,
                'freq_reso': freq_reso,
                'file_length': file_length,
                'total_duration': total_duration,
                'bandwidth': bandwidth,
                'freq_range': freq_range,
                'dm_max': dm_max
            },
            'dispersion': dispersion_analysis,
            'resolution': resolution_analysis,
            'content': content_analysis,
            'processing': processing_analysis,
            'filename': filename
        }
        
        self.analysis_results[filename] = analysis
        return analysis
    
    def _analyze_dispersion_characteristics(
        self, freq_min: float, freq_max: float, dm_max: int, time_reso: float
    ) -> Dict[str, float]:
        """Analiza características de dispersión del archivo."""
        
        # Constante de dispersión (s MHz^2 pc^-1 cm^3)
        K_DM = 4.148808e3
        
        # Dispersión temporal máxima esperada
        max_disp_delay = K_DM * dm_max * (1/freq_min**2 - 1/freq_max**2)
        
        # Dispersión temporal por canal
        channel_disp = K_DM * dm_max * (2 / (freq_min * freq_max * (freq_max - freq_min)))
        
        # Número de muestras afectadas por dispersión
        disp_samples = max_disp_delay / time_reso
        
        # Análisis de coherencia temporal
        coherence_time = min(time_reso * 10, max_disp_delay / 4)  # Tiempo de coherencia estimado
        
        return {
            'max_disp_delay': max_disp_delay,
            'channel_disp': channel_disp,
            'disp_samples': disp_samples,
            'coherence_time': coherence_time,
            'disp_factor': min(disp_samples / 512, 2.0)  # Factor de influencia en SLICE_LEN
        }
    
    def _analyze_resolution_requirements(
        self, time_reso: float, total_duration: float, bandwidth: float
    ) -> Dict[str, float]:
        """Analiza requisitos de resolución temporal."""
        
        # Resolución temporal óptima basada en características físicas
        # Para FRBs típicos: 0.1 - 10 ms
        optimal_pulse_duration = np.clip(time_reso * 50, 0.001, 0.1)  # Entre 1-100ms
        
        # Resolución requerida para diferentes tipos de señales
        short_pulse_req = 0.01   # 10ms para pulsos cortos
        medium_pulse_req = 0.05  # 50ms para FRBs típicos
        long_pulse_req = 0.2     # 200ms para señales largas
        
        # Score de resolución basado en características del archivo
        if time_reso < 0.0001:  # Sub-ms resolution
            resolution_score = 'ultra_high'
            recommended_duration = short_pulse_req
        elif time_reso < 0.001:  # ~1ms resolution
            resolution_score = 'high'
            recommended_duration = medium_pulse_req
        elif time_reso < 0.01:   # ~10ms resolution
            resolution_score = 'medium'
            recommended_duration = long_pulse_req
        else:                    # >10ms resolution
            resolution_score = 'low'
            recommended_duration = long_pulse_req * 2
        
        return {
            'optimal_pulse_duration': optimal_pulse_duration,
            'resolution_score': resolution_score,
            'recommended_duration': recommended_duration,
            'time_reso_quality': 'excellent' if time_reso < 0.001 else 
                                'good' if time_reso < 0.01 else 'adequate'
        }
    
    def _analyze_file_content(
        self, filename: str, file_length: int, time_reso: float
    ) -> Dict[str, Any]:
        """Analiza el contenido y tipo de archivo."""
        
        file_type = 'unknown'
        expected_signal_duration = 0.05  # Default 50ms
        
        filename_lower = filename.lower()
        
        # Análisis basado en nombre de archivo
        if 'frb' in filename_lower:
            file_type = 'frb'
            expected_signal_duration = 0.03  # 30ms típico para FRBs
        elif 'pulsar' in filename_lower or 'psr' in filename_lower:
            file_type = 'pulsar'
            expected_signal_duration = 0.01  # 10ms típico para pulsars
        elif 'burst' in filename_lower:
            file_type = 'burst'
            expected_signal_duration = 0.02  # 20ms típico para bursts
        elif '.fil' in filename_lower:
            file_type = 'filterbank'
            expected_signal_duration = 0.025  # 25ms conservador para filterbank
        elif '.fits' in filename_lower:
            file_type = 'fits'
            expected_signal_duration = 0.04   # 40ms conservador para FITS
        
        # Análisis de duración total
        total_duration = file_length * time_reso
        
        if total_duration < 1.0:      # <1s
            duration_category = 'short'
            context_factor = 0.5
        elif total_duration < 10.0:   # 1-10s
            duration_category = 'medium'
            context_factor = 1.0
        elif total_duration < 100.0:  # 10-100s
            duration_category = 'long'
            context_factor = 1.5
        else:                         # >100s
            duration_category = 'very_long'
            context_factor = 2.0
        
        return {
            'file_type': file_type,
            'expected_signal_duration': expected_signal_duration,
            'total_duration': total_duration,
            'duration_category': duration_category,
            'context_factor': context_factor,
            'file_length': file_length
        }
    
    def _analyze_processing_requirements(
        self, file_length: int, time_reso: float
    ) -> Dict[str, Any]:
        """Analiza requisitos de procesamiento."""
        
        # Estimación de carga computacional
        total_samples = file_length
        
        if total_samples < 1000:
            processing_load = 'light'
            recommended_slices = 4
        elif total_samples < 10000:
            processing_load = 'medium'
            recommended_slices = 8
        elif total_samples < 100000:
            processing_load = 'heavy'
            recommended_slices = 16
        else:
            processing_load = 'very_heavy'
            recommended_slices = 32
        
        # Balance entre resolución y eficiencia
        efficiency_factor = np.clip(total_samples / 50000, 0.5, 2.0)
        
        return {
            'processing_load': processing_load,
            'recommended_slices': recommended_slices,
            'efficiency_factor': efficiency_factor,
            'total_samples': total_samples
        }
    
    def calculate_optimal_slice_len(self, analysis: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Calcula el SLICE_LEN óptimo basado en el análisis completo.
        
        Parameters
        ----------
        analysis : dict
            Análisis completo del archivo
            
        Returns
        -------
        tuple
            (slice_len_optimal, optimization_details)
        """
        
        basic = analysis['basic_params']
        dispersion = analysis['dispersion']
        resolution = analysis['resolution']
        content = analysis['content']
        processing = analysis['processing']
        
        # Calcular SLICE_LEN base según duración de señal esperada
        base_duration = content['expected_signal_duration']
        base_slice_len = base_duration / basic['effective_time_reso']
        
        # Ajustes basados en dispersión
        disp_factor = dispersion['disp_factor']
        disp_adjusted = base_slice_len * (1 + disp_factor * 0.5)
        
        # Ajustes basados en resolución
        resolution_factor = {
            'ultra_high': 0.7,  # Reducir para alta resolución
            'high': 0.8,
            'medium': 1.0,
            'low': 1.3          # Aumentar para baja resolución
        }.get(resolution['resolution_score'], 1.0)
        
        resolution_adjusted = disp_adjusted * resolution_factor
        
        # Ajustes basados en contenido
        context_factor = content['context_factor']
        content_adjusted = resolution_adjusted * context_factor
        
        # Ajustes basados en procesamiento
        efficiency_factor = processing['efficiency_factor']
        final_slice_len = content_adjusted * efficiency_factor
        
        # Redondear a potencia de 2 más cercana
        powers_of_2 = [2**i for i in range(3, 12)]  # 8, 16, 32, ..., 2048
        optimal_slice_len = min(powers_of_2, key=lambda x: abs(x - final_slice_len))
        
        # Aplicar límites de seguridad
        optimal_slice_len = max(8, min(optimal_slice_len, 2048))
        
        # Calcular duración real
        real_duration = optimal_slice_len * basic['effective_time_reso']
        
        # Calcular número de slices
        n_slices = basic['file_length'] // optimal_slice_len
        
        optimization_details = {
            'base_slice_len': base_slice_len,
            'disp_adjusted': disp_adjusted,
            'resolution_adjusted': resolution_adjusted,
            'content_adjusted': content_adjusted,
            'final_slice_len': final_slice_len,
            'optimal_slice_len': optimal_slice_len,
            'real_duration': real_duration,
            'n_slices': n_slices,
            'factors': {
                'disp_factor': disp_factor,
                'resolution_factor': resolution_factor,
                'context_factor': context_factor,
                'efficiency_factor': efficiency_factor
            }
        }
        
        logger.info("SLICE_LEN óptimo calculado: %d (duración: %.3f s, %d slices)", 
                   optimal_slice_len, real_duration, n_slices)
        
        return optimal_slice_len, optimization_details
    
    def get_automatic_slice_len(
        self,
        time_reso: float,
        freq_reso: float,
        file_length: int,
        freq_range: Tuple[float, float],
        dm_max: int = 1024,
        down_time_rate: int = 1,
        filename: str = "unknown"
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Obtiene el SLICE_LEN automático para un archivo específico.
        
        Esta es la función principal que debe ser llamada desde el pipeline.
        
        Returns
        -------
        tuple
            (optimal_slice_len, full_analysis)
        """
        
        # Análisis completo del archivo
        analysis = self.analyze_file_characteristics(
            time_reso=time_reso,
            freq_reso=freq_reso,
            file_length=file_length,
            freq_range=freq_range,
            dm_max=dm_max,
            down_time_rate=down_time_rate,
            filename=filename
        )
        
        # Cálculo óptimo
        optimal_slice_len, optimization_details = self.calculate_optimal_slice_len(analysis)
        
        # Añadir detalles de optimización al análisis
        analysis['optimization'] = optimization_details
        
        return optimal_slice_len, analysis
    
    def print_analysis_report(self, filename: str = None):
        """Imprime un reporte detallado del análisis."""
        
        if filename and filename in self.analysis_results:
            analysis = self.analysis_results[filename]
            self._print_single_analysis(analysis)
        else:
            # Imprimir todos los análisis
            for fname, analysis in self.analysis_results.items():
                self._print_single_analysis(analysis)
                print()
    
    def _print_single_analysis(self, analysis: Dict[str, Any]):
        """Imprime análisis de un archivo individual."""
        
        filename = analysis['filename']
        basic = analysis['basic_params']
        opt = analysis.get('optimization', {})
        
        print(f"📊 === ANÁLISIS AUTOMÁTICO: {filename} ===")
        print(f"📁 Archivo: {basic['file_length']} muestras, {basic['total_duration']:.3f} s")
        print(f"⏱️  Resolución: {basic['effective_time_reso']*1000:.3f} ms/muestra")
        print(f"📡 Frecuencia: {basic['freq_range'][0]:.1f}-{basic['freq_range'][1]:.1f} MHz")
        print(f"📊 Tipo: {analysis['content']['file_type']}")
        
        if 'optimal_slice_len' in opt:
            print(f"🎯 SLICE_LEN óptimo: {opt['optimal_slice_len']}")
            print(f"⏰ Duración real: {opt['real_duration']*1000:.1f} ms")
            print(f"🔢 Número de slices: {opt['n_slices']}")
            print(f"📈 Factores aplicados:")
            for factor, value in opt['factors'].items():
                print(f"   • {factor}: {value:.3f}")


# Instancia global del optimizador
slice_optimizer = SliceLenOptimizer()

def get_automatic_slice_len_from_file(config_module) -> int:
    """
    Obtiene SLICE_LEN automático basado en metadatos del archivo cargado.
    
    Esta función debe ser llamada después de cargar los metadatos del archivo.
    
    Parameters
    ----------
    config_module : module
        Módulo de configuración con metadatos del archivo
        
    Returns
    -------
    int
        SLICE_LEN óptimo calculado automáticamente
    """
    
    # Obtener parámetros del archivo
    time_reso = getattr(config_module, 'TIME_RESO', 0.001)
    freq_reso = getattr(config_module, 'FREQ_RESO', 1.0)
    file_length = getattr(config_module, 'FILE_LENG', 1024)
    down_time_rate = getattr(config_module, 'DOWN_TIME_RATE', 1)
    dm_max = getattr(config_module, 'DM_max', 1024)
    
    # Obtener rango de frecuencias
    freq_array = getattr(config_module, 'FREQ', None)
    if freq_array is not None and len(freq_array) > 0:
        freq_min, freq_max = float(np.min(freq_array)), float(np.max(freq_array))
    else:
        # Valores por defecto si no hay información de frecuencia
        freq_min, freq_max = 1000.0, 1500.0  # MHz
        logger.warning("No se encontró información de frecuencia, usando valores por defecto")
    
    # Obtener nombre del archivo si está disponible
    filename = "current_file"
    if hasattr(config_module, 'CURRENT_FILE'):
        filename = str(config_module.CURRENT_FILE)
    
    # Calcular SLICE_LEN óptimo
    optimal_slice_len, full_analysis = slice_optimizer.get_automatic_slice_len(
        time_reso=time_reso,
        freq_reso=freq_reso,
        file_length=file_length,
        freq_range=(freq_min, freq_max),
        dm_max=dm_max,
        down_time_rate=down_time_rate,
        filename=filename
    )
    
    logger.info("SLICE_LEN automático calculado: %d para archivo %s", optimal_slice_len, filename)
    
    return optimal_slice_len

if __name__ == "__main__":
    # Ejemplo de uso
    optimizer = SliceLenOptimizer()
    
    # Ejemplo con archivo típico
    slice_len, analysis = optimizer.get_automatic_slice_len(
        time_reso=0.001,
        freq_reso=1.0,
        file_length=2048,
        freq_range=(1200.0, 1500.0),
        dm_max=1024,
        filename="example_frb.fil"
    )
    
    optimizer.print_analysis_report()
