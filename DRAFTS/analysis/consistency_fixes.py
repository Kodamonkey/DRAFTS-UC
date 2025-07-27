"""
Correcciones para unificar cálculos de DM y SNR en todo el pipeline.

Este módulo asegura que los valores de DM y SNR sean consistentes entre:
- Box Detection en Composite Plot
- Subtítulos de Waterfalls
- Valores guardados en CSV
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from ..preprocessing.astro_conversions import pixel_to_physical
from .snr_utils import compute_snr_profile, find_snr_peak

class ConsistencyManager:
    """Gestor de consistencia para valores de DM y SNR."""
    
    def __init__(self):
        self.candidate_data = {}
        
    def calculate_consistent_candidate_values(
        self,
        top_boxes: List,
        top_conf: List,
        slice_len: int,
        waterfall_block: np.ndarray,
        dedispersed_block: np.ndarray,
        patch: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Calcula valores consistentes de DM y SNR para todos los candidatos.
        
        Parameters
        ----------
        top_boxes : List
            Lista de bounding boxes de candidatos
        top_conf : List
            Lista de confianzas de detección
        slice_len : int
            Longitud del slice temporal
        waterfall_block : np.ndarray
            Datos del waterfall sin dedispersar
        dedispersed_block : np.ndarray
            Datos del waterfall dedispersado
        patch : np.ndarray, optional
            Patch dedispersado del candidato
            
        Returns
        -------
        Dict[str, Any]
            Diccionario con valores consistentes para cada candidato
        """
        
        if not top_boxes or not top_conf:
            return {}
            
        consistent_data = {}
        
        for idx, (box, conf) in enumerate(zip(top_boxes, top_conf)):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # ✅ 1. DM CONSISTENTE - Usar el mismo cálculo en todas partes
            dm_val, t_sec, t_sample = pixel_to_physical(center_x, center_y, slice_len)
            
            # ✅ 2. SNR CONSISTENTE - Calcular múltiples tipos para transparencia
            snr_values = self._calculate_all_snr_types(
                box, waterfall_block, dedispersed_block, patch
            )
            
            consistent_data[idx] = {
                'box': box,
                'confidence': conf,
                'dm_val': dm_val,
                't_sec': t_sec,
                't_sample': t_sample,
                'center_x': center_x,
                'center_y': center_y,
                'snr_values': snr_values
            }
            
        return consistent_data
    
    def _calculate_all_snr_types(
        self,
        box: Tuple[int, int, int, int],
        waterfall_block: np.ndarray,
        dedispersed_block: np.ndarray,
        patch: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calcula todos los tipos de SNR para un candidato.
        
        Returns
        -------
        Dict[str, float]
            Diccionario con diferentes tipos de SNR
        """
        
        x1, y1, x2, y2 = map(int, box)
        snr_values = {}
        
        # 1. SNR del candidato en waterfall raw (región específica)
        if waterfall_block is not None and waterfall_block.size > 0:
            try:
                candidate_region_raw = waterfall_block[:, y1:y2]
                if candidate_region_raw.size > 0:
                    snr_profile_raw, _ = compute_snr_profile(candidate_region_raw)
                    snr_values['candidate_raw'] = np.max(snr_profile_raw)
                else:
                    snr_values['candidate_raw'] = 0.0
            except Exception as e:
                print(f"Error calculando SNR candidate_raw: {e}")
                snr_values['candidate_raw'] = 0.0
        else:
            snr_values['candidate_raw'] = 0.0
        
        # 2. SNR del candidato en waterfall dedispersado (región específica)
        if dedispersed_block is not None and dedispersed_block.size > 0:
            try:
                candidate_region_dedisp = dedispersed_block[:, y1:y2]
                if candidate_region_dedisp.size > 0:
                    snr_profile_dedisp, _ = compute_snr_profile(candidate_region_dedisp)
                    snr_values['candidate_dedispersed'] = np.max(snr_profile_dedisp)
                else:
                    snr_values['candidate_dedispersed'] = 0.0
            except Exception as e:
                print(f"Error calculando SNR candidate_dedispersed: {e}")
                snr_values['candidate_dedispersed'] = 0.0
        else:
            snr_values['candidate_dedispersed'] = 0.0
        
        # 3. SNR del patch dedispersado (para CSV)
        if patch is not None and patch.size > 0:
            try:
                snr_profile_patch, _ = compute_snr_profile(patch)
                snr_values['patch_dedispersed'], _, _ = find_snr_peak(snr_profile_patch)
            except Exception as e:
                print(f"Error calculando SNR patch_dedispersed: {e}")
                snr_values['patch_dedispersed'] = 0.0
        else:
            snr_values['patch_dedispersed'] = 0.0
        
        # 4. SNR peak del waterfall raw completo
        if waterfall_block is not None and waterfall_block.size > 0:
            try:
                snr_profile_wf, _ = compute_snr_profile(waterfall_block)
                snr_values['waterfall_raw_peak'], _, _ = find_snr_peak(snr_profile_wf)
            except Exception as e:
                print(f"Error calculando SNR waterfall_raw_peak: {e}")
                snr_values['waterfall_raw_peak'] = 0.0
        else:
            snr_values['waterfall_raw_peak'] = 0.0
        
        # 5. SNR peak del waterfall dedispersado completo
        if dedispersed_block is not None and dedispersed_block.size > 0:
            try:
                snr_profile_dw, _ = compute_snr_profile(dedispersed_block)
                snr_values['waterfall_dedispersed_peak'], _, _ = find_snr_peak(snr_profile_dw)
            except Exception as e:
                print(f"Error calculando SNR waterfall_dedispersed_peak: {e}")
                snr_values['waterfall_dedispersed_peak'] = 0.0
        else:
            snr_values['waterfall_dedispersed_peak'] = 0.0
        
        return snr_values
    
    def get_best_candidate_data(self, consistent_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Obtiene los datos del candidato con mayor confianza.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Datos del mejor candidato o None si no hay candidatos
        """
        
        if not consistent_data:
            return None
            
        # Encontrar candidato con mayor confianza
        best_idx = max(consistent_data.keys(), 
                      key=lambda idx: consistent_data[idx]['confidence'])
        
        return consistent_data[best_idx]
    
    def get_csv_snr_value(self, candidate_data: Dict[str, Any]) -> float:
        """
        Obtiene el valor de SNR que debe guardarse en CSV.
        
        Prioridad:
        1. SNR del patch dedispersado (más preciso)
        2. SNR del candidato en waterfall dedispersado
        3. SNR del candidato en waterfall raw
        4. 0.0 como fallback
        """
        
        snr_values = candidate_data.get('snr_values', {})
        
        # Prioridad 1: Patch dedispersado
        if snr_values.get('patch_dedispersed', 0.0) > 0.0:
            return snr_values['patch_dedispersed']
        
        # Prioridad 2: Candidato en waterfall dedispersado
        if snr_values.get('candidate_dedispersed', 0.0) > 0.0:
            return snr_values['candidate_dedispersed']
        
        # Prioridad 3: Candidato en waterfall raw
        if snr_values.get('candidate_raw', 0.0) > 0.0:
            return snr_values['candidate_raw']
        
        # Fallback
        return 0.0
    
    def get_composite_display_values(self, consistent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtiene valores para mostrar en el composite plot.
        
        Returns
        -------
        Dict[str, Any]
            Valores optimizados para visualización
        """
        
        if not consistent_data:
            return {
                'dm_val': 0.0,
                'snr_raw_peak': 0.0,
                'snr_dedispersed_peak': 0.0,
                'snr_candidate': 0.0
            }
        
        # Usar el mejor candidato para DM
        best_candidate = self.get_best_candidate_data(consistent_data)
        if best_candidate is None:
            return {
                'dm_val': 0.0,
                'snr_raw_peak': 0.0,
                'snr_dedispersed_peak': 0.0,
                'snr_candidate': 0.0
            }
        
        dm_val = best_candidate['dm_val']
        snr_values = best_candidate['snr_values']
        
        return {
            'dm_val': dm_val,
            'snr_raw_peak': snr_values.get('waterfall_raw_peak', 0.0),
            'snr_dedispersed_peak': snr_values.get('waterfall_dedispersed_peak', 0.0),
            'snr_candidate': snr_values.get('candidate_dedispersed', 0.0)
        }


# Instancia global del gestor de consistencia
consistency_manager = ConsistencyManager()

def get_consistent_candidate_values(
    top_boxes: List,
    top_conf: List,
    slice_len: int,
    waterfall_block: np.ndarray,
    dedispersed_block: np.ndarray,
    patch: np.ndarray = None
) -> Dict[str, Any]:
    """
    Función principal para obtener valores consistentes de candidatos.
    
    Returns
    -------
    Dict[str, Any]
        Datos consistentes de todos los candidatos
    """
    
    return consistency_manager.calculate_consistent_candidate_values(
        top_boxes, top_conf, slice_len, waterfall_block, dedispersed_block, patch
    )

def get_csv_ready_candidate_data(
    candidate_idx: int,
    consistent_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Obtiene datos del candidato listos para guardar en CSV.
    
    Parameters
    ----------
    candidate_idx : int
        Índice del candidato
    consistent_data : Dict[str, Any]
        Datos consistentes de todos los candidatos
        
    Returns
    -------
    Dict[str, Any]
        Datos del candidato con SNR correcto para CSV
    """
    
    if candidate_idx not in consistent_data:
        return {}
    
    candidate_data = consistent_data[candidate_idx]
    
    # Obtener SNR para CSV
    csv_snr = consistency_manager.get_csv_snr_value(candidate_data)
    
    return {
        'dm_val': candidate_data['dm_val'],
        't_sec': candidate_data['t_sec'],
        't_sample': candidate_data['t_sample'],
        'confidence': candidate_data['confidence'],
        'snr_csv': csv_snr,
        'snr_values': candidate_data['snr_values']
    }

def get_composite_display_data(
    consistent_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Obtiene datos para mostrar en el composite plot.
    
    Returns
    -------
    Dict[str, Any]
        Datos optimizados para visualización del composite
    """
    
    return consistency_manager.get_composite_display_values(consistent_data)

def print_consistency_report(consistent_data: Dict[str, Any]):
    """
    Imprime un reporte de consistencia para debugging.
    """
    
    print("\n" + "="*60)
    print("📊 REPORTE DE CONSISTENCIA DM/SNR")
    print("="*60)
    
    if not consistent_data:
        print("❌ No hay candidatos para analizar")
        return
    
    for idx, candidate in consistent_data.items():
        print(f"\n🎯 Candidato #{idx+1}:")
        print(f"   📍 DM: {candidate['dm_val']:.2f} pc cm⁻³")
        print(f"   ⏱️  Tiempo: {candidate['t_sec']:.6f} s")
        print(f"   🎯 Confianza: {candidate['confidence']:.3f}")
        
        snr_values = candidate['snr_values']
        print(f"   📊 SNRs:")
        print(f"      • Candidato Raw: {snr_values.get('candidate_raw', 0.0):.2f}σ")
        print(f"      • Candidato Dedispersado: {snr_values.get('candidate_dedispersed', 0.0):.2f}σ")
        print(f"      • Patch Dedispersado: {snr_values.get('patch_dedispersed', 0.0):.2f}σ")
        print(f"      • Waterfall Raw Peak: {snr_values.get('waterfall_raw_peak', 0.0):.2f}σ")
        print(f"      • Waterfall Dedispersado Peak: {snr_values.get('waterfall_dedispersed_peak', 0.0):.2f}σ")
    
    # Mostrar mejor candidato
    best_candidate = consistency_manager.get_best_candidate_data(consistent_data)
    if best_candidate:
        print(f"\n🏆 MEJOR CANDIDATO:")
        print(f"   📍 DM: {best_candidate['dm_val']:.2f} pc cm⁻³")
        print(f"   🎯 Confianza: {best_candidate['confidence']:.3f}")
        csv_snr = consistency_manager.get_csv_snr_value(best_candidate)
        print(f"   📊 SNR para CSV: {csv_snr:.2f}σ")
    
    print("="*60) 