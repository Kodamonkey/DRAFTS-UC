"""
Tests para el módulo de mitigación de RFI.

Este archivo contiene tests unitarios para verificar que la limpieza de RFI
funcione correctamente en diferentes escenarios.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from DRAFTS.rfi_mitigation import RFIMitigator
from DRAFTS.snr_utils import compute_snr_profile, find_snr_peak, inject_synthetic_frb


class TestRFIMitigator:
    """Test suite para RFIMitigator."""
    
    def setup_method(self):
        """Configuración para cada test."""
        self.rfi_mitigator = RFIMitigator(
            freq_sigma_thresh=5.0,
            time_sigma_thresh=5.0,
            zero_dm_sigma_thresh=4.0,
            impulse_sigma_thresh=6.0,
            polarization_thresh=0.8
        )
        
        # Datos de prueba básicos
        self.n_time = 512
        self.n_freq = 128
        self.clean_waterfall = np.random.normal(0, 1, (self.n_time, self.n_freq))
    
    def test_detect_bad_channels_mad(self):
        """Test detección de canales malos usando MAD."""
        # Crea waterfall con algunos canales contaminados
        contaminated_waterfall = self.clean_waterfall.copy()
        
        # Contamina algunos canales
        bad_channels = [10, 25, 50, 75]
        for ch in bad_channels:
            contaminated_waterfall[:, ch] += np.random.normal(0, 10, self.n_time)
        
        # Detecta canales malos
        good_channels = self.rfi_mitigator.detect_bad_channels(
            contaminated_waterfall, method="mad"
        )
        
        # Verifica que se detectaron canales malos
        assert np.sum(~good_channels) > 0
        assert self.rfi_mitigator.rfi_stats['bad_channels'] > 0
        assert self.rfi_mitigator.rfi_stats['channel_fraction_flagged'] > 0
        
        # Verifica que se detectaron la mayoría de los canales contaminados
        detected_bad = np.where(~good_channels)[0]
        overlap = len(set(bad_channels) & set(detected_bad))
        assert overlap >= len(bad_channels) // 2  # Al menos 50% detectados
    
    def test_detect_bad_channels_std(self):
        """Test detección de canales malos usando desviación estándar."""
        contaminated_waterfall = self.clean_waterfall.copy()
        
        # Contamina canales con ruido persistente
        bad_channels = [15, 30, 45]
        for ch in bad_channels:
            contaminated_waterfall[:, ch] += np.random.normal(5, 2, self.n_time)
        
        good_channels = self.rfi_mitigator.detect_bad_channels(
            contaminated_waterfall, method="std"
        )
        
        assert np.sum(~good_channels) > 0
        assert self.rfi_mitigator.rfi_stats['bad_channels'] > 0
    
    def test_detect_bad_time_samples(self):
        """Test detección de muestras temporales malas."""
        contaminated_waterfall = self.clean_waterfall.copy()
        
        # Contamina algunas muestras temporales
        bad_times = [100, 200, 300, 400]
        for t in bad_times:
            contaminated_waterfall[t, :] += np.random.normal(0, 8, self.n_freq)
        
        good_samples = self.rfi_mitigator.detect_bad_time_samples(
            contaminated_waterfall, method="mad"
        )
        
        # Verifica detección
        assert np.sum(~good_samples) > 0
        assert self.rfi_mitigator.rfi_stats['bad_time_samples'] > 0
        assert self.rfi_mitigator.rfi_stats['time_fraction_flagged'] > 0
        
        # Verifica que se detectaron muestras contaminadas
        detected_bad = np.where(~good_samples)[0]
        overlap = len(set(bad_times) & set(detected_bad))
        assert overlap >= len(bad_times) // 2
    
    def test_zero_dm_filter(self):
        """Test filtro Zero-DM."""
        # Crea waterfall con RFI de banda ancha
        waterfall_with_rfi = self.clean_waterfall.copy()
        
        # Añade RFI de banda ancha (correlacionado en frecuencia)
        rfi_profile = np.random.normal(0, 5, self.n_time)
        for i in range(self.n_freq):
            waterfall_with_rfi[:, i] += rfi_profile * (0.8 + 0.2 * np.random.random())
        
        # Aplica filtro Zero-DM
        filtered_waterfall = self.rfi_mitigator.zero_dm_filter(waterfall_with_rfi)
        
        # Verifica que se redujeron las correlaciones temporales
        original_time_var = np.var(np.mean(waterfall_with_rfi, axis=1))
        filtered_time_var = np.var(np.mean(filtered_waterfall, axis=1))
        
        assert filtered_time_var < original_time_var
        assert self.rfi_mitigator.rfi_stats['zero_dm_flagged'] >= 0
    
    def test_impulse_filter(self):
        """Test filtrado de impulsos."""
        waterfall_with_impulses = self.clean_waterfall.copy()
        
        # Añade impulsos aleatorios
        n_impulses = 50
        impulse_positions = np.random.randint(0, self.n_time * self.n_freq, n_impulses)
        
        for pos in impulse_positions:
            t_idx = pos // self.n_freq
            f_idx = pos % self.n_freq
            waterfall_with_impulses[t_idx, f_idx] += np.random.normal(0, 20)
        
        # Aplica filtro de impulsos
        filtered_waterfall = self.rfi_mitigator.impulse_filter(waterfall_with_impulses)
        
        # Verifica que se redujeron los impulsos
        original_outliers = np.sum(np.abs(waterfall_with_impulses) > 5)
        filtered_outliers = np.sum(np.abs(filtered_waterfall) > 5)
        
        assert filtered_outliers < original_outliers
        assert self.rfi_mitigator.rfi_stats['impulses_flagged'] > 0
    
    def test_polarization_filter(self):
        """Test filtrado de polarización."""
        stokes_i = self.clean_waterfall.copy()
        
        # Crea datos de polarización con RFI altamente polarizado
        stokes_v = np.random.normal(0, 0.1, (self.n_time, self.n_freq))
        
        # Añade RFI altamente polarizado
        rfi_regions = np.random.choice(
            self.n_time * self.n_freq, 
            self.n_time * self.n_freq // 10, 
            replace=False
        )
        
        for pos in rfi_regions:
            t_idx = pos // self.n_freq
            f_idx = pos % self.n_freq
            stokes_v[t_idx, f_idx] = np.random.normal(0, 2)  # Alta polarización
        
        # Aplica filtro de polarización
        filtered_stokes_i = self.rfi_mitigator.polarization_filter(
            stokes_i, stokes_v=stokes_v
        )
        
        # Verifica que se aplicó filtrado
        assert self.rfi_mitigator.rfi_stats['high_pol_flagged'] > 0
        assert self.rfi_mitigator.rfi_stats['pol_fraction'] > 0
    
    def test_apply_masks(self):
        """Test aplicación de máscaras."""
        waterfall = self.clean_waterfall.copy()
        
        # Crea máscaras
        freq_mask = np.ones(self.n_freq, dtype=bool)
        freq_mask[10:20] = False  # Marca algunos canales como malos
        
        time_mask = np.ones(self.n_time, dtype=bool)
        time_mask[100:110] = False  # Marca algunas muestras como malas
        
        # Aplica máscaras
        masked_waterfall = self.rfi_mitigator.apply_masks(
            waterfall, freq_mask, time_mask, interpolate=True
        )
        
        # Verifica que se aplicaron las máscaras
        assert not np.isnan(masked_waterfall).any()  # Interpolación funcionó
        assert masked_waterfall.shape == waterfall.shape
    
    def test_complete_cleaning_pipeline(self):
        """Test pipeline completo de limpieza."""
        # Crea waterfall con múltiples tipos de RFI
        contaminated_waterfall = self.clean_waterfall.copy()
        
        # Añade FRB sintético
        frb_waterfall = inject_synthetic_frb(
            contaminated_waterfall,
            peak_time_idx=self.n_time // 2,
            peak_freq_idx=self.n_freq // 2,
            amplitude=8.0
        )
        
        # Añade varios tipos de RFI
        # Canales malos
        bad_channels = [10, 25, 50]
        for ch in bad_channels:
            frb_waterfall[:, ch] += np.random.normal(0, 5, self.n_time)
        
        # Muestras temporales malas
        bad_times = [100, 200, 300]
        for t in bad_times:
            frb_waterfall[t, :] += np.random.normal(0, 8, self.n_freq)
        
        # Impulsos
        n_impulses = 20
        impulse_positions = np.random.randint(0, self.n_time * self.n_freq, n_impulses)
        for pos in impulse_positions:
            t_idx = pos // self.n_freq
            f_idx = pos % self.n_freq
            frb_waterfall[t_idx, f_idx] += np.random.normal(0, 15)
        
        # Aplica limpieza completa
        cleaned_waterfall, rfi_stats = self.rfi_mitigator.clean_waterfall(
            frb_waterfall,
            apply_all_filters=True
        )
        
        # Verifica que la limpieza mejoró la detección del FRB
        snr_original, _ = compute_snr_profile(frb_waterfall)
        snr_cleaned, _ = compute_snr_profile(cleaned_waterfall)
        
        peak_snr_orig, _, _ = find_snr_peak(snr_original)
        peak_snr_clean, _, _ = find_snr_peak(snr_cleaned)
        
        # El SNR debería mejorar o al menos mantenerse
        assert peak_snr_clean >= peak_snr_orig * 0.8  # Permite pequeña degradación
        
        # Verifica que se detectó RFI
        assert rfi_stats['total_flagged_fraction'] > 0
        assert 'bad_channels' in rfi_stats
        assert 'bad_time_samples' in rfi_stats
    
    def test_preservation_of_frb_signal(self):
        """Test que la limpieza preserva señales de FRB."""
        # Crea waterfall con FRB fuerte
        waterfall_with_frb = inject_synthetic_frb(
            self.clean_waterfall,
            peak_time_idx=self.n_time // 2,
            peak_freq_idx=self.n_freq // 2,
            amplitude=12.0
        )
        
        # Añade RFI moderado
        rfi_waterfall = waterfall_with_frb.copy()
        
        # Canales contaminados (lejos del FRB)
        bad_channels = [5, 10, 15, 120, 125]
        for ch in bad_channels:
            rfi_waterfall[:, ch] += np.random.normal(0, 3, self.n_time)
        
        # Aplica limpieza
        cleaned_waterfall, _ = self.rfi_mitigator.clean_waterfall(
            rfi_waterfall,
            apply_all_filters=True
        )
        
        # Verifica que el FRB se mantiene fuerte
        snr_original, _ = compute_snr_profile(waterfall_with_frb)
        snr_cleaned, _ = compute_snr_profile(cleaned_waterfall)
        
        peak_snr_orig, _, _ = find_snr_peak(snr_original)
        peak_snr_clean, _, _ = find_snr_peak(snr_cleaned)
        
        # El FRB debería mantenerse fuerte
        assert peak_snr_clean >= peak_snr_orig * 0.9  # Máximo 10% de degradación
        assert peak_snr_clean > 8.0  # Debe mantenerse por encima de umbral típico
    
    def test_rfi_diagnostics_plot(self):
        """Test generación de gráficos de diagnóstico."""
        # Crea datos con RFI
        waterfall_with_rfi = self.clean_waterfall.copy()
        waterfall_with_rfi[:, 10] += np.random.normal(0, 5, self.n_time)
        
        # Aplica limpieza
        cleaned_waterfall, _ = self.rfi_mitigator.clean_waterfall(
            waterfall_with_rfi,
            apply_all_filters=True
        )
        
        # Verifica que se puede generar el gráfico sin errores
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_diagnostics.png"
            
            # No debería lanzar excepción
            self.rfi_mitigator.plot_rfi_diagnostics(
                waterfall_with_rfi,
                cleaned_waterfall,
                output_path
            )
            
            # Verifica que se creó el archivo
            assert output_path.exists()
            assert output_path.stat().st_size > 0


def test_rfi_config_integration():
    """Test integración con configuración del pipeline."""
    # Verifica que la configuración de RFI se puede importar
    from DRAFTS import config
    
    # Verifica que las configuraciones de RFI existen
    assert hasattr(config, 'RFI_FREQ_SIGMA_THRESH')
    assert hasattr(config, 'RFI_TIME_SIGMA_THRESH')
    assert hasattr(config, 'RFI_ZERO_DM_SIGMA_THRESH')
    assert hasattr(config, 'RFI_IMPULSE_SIGMA_THRESH')
    assert hasattr(config, 'RFI_POLARIZATION_THRESH')
    assert hasattr(config, 'RFI_ENABLE_ALL_FILTERS')
    
    # Verifica que los valores son razonables
    assert 0 < config.RFI_FREQ_SIGMA_THRESH < 10
    assert 0 < config.RFI_TIME_SIGMA_THRESH < 10
    assert 0 < config.RFI_ZERO_DM_SIGMA_THRESH < 10
    assert 0 < config.RFI_IMPULSE_SIGMA_THRESH < 10
    assert 0 < config.RFI_POLARIZATION_THRESH < 1


if __name__ == "__main__":
    # Ejecuta los tests
    pytest.main([__file__, "-v"])
