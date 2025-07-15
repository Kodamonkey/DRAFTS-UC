"""
Script de ejemplo para integrar limpieza de RFI en el pipeline DRAFTS.

Este script muestra cómo usar el módulo de mitigación de RFI para limpiar
datos DM vs. Time antes del análisis de SNR y detección de FRBs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

# Importa módulos del pipeline
from DRAFTS.rfi_mitigation import RFIMitigator
from DRAFTS.snr_utils import compute_snr_profile, find_snr_peak
from DRAFTS import config


def integrate_rfi_cleaning_pipeline(
    waterfall: np.ndarray,
    stokes_v: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Pipeline integrado de limpieza de RFI y análisis de SNR.
    
    Parameters
    ----------
    waterfall : np.ndarray
        Datos waterfall DM vs. Time (tiempo, frecuencia)
    stokes_v : np.ndarray, optional
        Datos de polarización circular (Stokes V)
    output_dir : Path, optional
        Directorio para guardar resultados
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        Waterfall limpio, perfil SNR, estadísticas de RFI
    """
    print("[INFO] === PIPELINE DE LIMPIEZA DE RFI ===")
    
    # Configura el mitigador de RFI
    rfi_mitigator = RFIMitigator(
        freq_sigma_thresh=config.RFI_FREQ_SIGMA_THRESH,
        time_sigma_thresh=config.RFI_TIME_SIGMA_THRESH,
        zero_dm_sigma_thresh=config.RFI_ZERO_DM_SIGMA_THRESH,
        impulse_sigma_thresh=config.RFI_IMPULSE_SIGMA_THRESH,
        polarization_thresh=config.RFI_POLARIZATION_THRESH
    )
    
    # Aplica limpieza de RFI
    cleaned_waterfall, rfi_stats = rfi_mitigator.clean_waterfall(
        waterfall,
        stokes_v=stokes_v,
        apply_all_filters=config.RFI_ENABLE_ALL_FILTERS
    )
    
    # Calcula SNR después de limpieza
    print("[INFO] Calculando SNR en datos limpios...")
    snr_profile, sigma_estimate = compute_snr_profile(
        cleaned_waterfall,
        off_regions=config.SNR_OFF_REGIONS
    )
    
    # Encuentra pico de SNR
    peak_snr, peak_time, peak_idx = find_snr_peak(snr_profile)
    
    print(f"[INFO] Pico de SNR detectado: {peak_snr:.2f} en tiempo {peak_time}")
    print(f"[INFO] Fracción de datos flagged por RFI: {rfi_stats.get('total_flagged_fraction', 0):.3f}")
    
    # Guarda diagnósticos si está configurado
    if config.RFI_SAVE_DIAGNOSTICS and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Gráficos de diagnóstico de RFI
        diagnostic_path = output_dir / "rfi_diagnostics.png"
        rfi_mitigator.plot_rfi_diagnostics(
            waterfall, cleaned_waterfall, diagnostic_path
        )
        
        # Gráfico comparativo de SNR
        plot_snr_comparison(waterfall, cleaned_waterfall, snr_profile, output_dir)
    
    return cleaned_waterfall, snr_profile, rfi_stats


def plot_snr_comparison(
    original_waterfall: np.ndarray,
    cleaned_waterfall: np.ndarray,
    snr_profile: np.ndarray,
    output_dir: Path
) -> None:
    """
    Genera gráfico comparativo de SNR antes y después de limpieza RFI.
    
    Parameters
    ----------
    original_waterfall : np.ndarray
        Datos originales
    cleaned_waterfall : np.ndarray
        Datos limpiados
    snr_profile : np.ndarray
        Perfil SNR de datos limpios
    output_dir : Path
        Directorio de salida
    """
    # Calcula SNR en datos originales
    snr_original, _ = compute_snr_profile(
        original_waterfall,
        off_regions=config.SNR_OFF_REGIONS
    )
    
    # Encuentra picos
    peak_snr_orig, peak_time_orig, _ = find_snr_peak(snr_original)
    peak_snr_clean, peak_time_clean, _ = find_snr_peak(snr_profile)
    
    # Genera gráfico
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Waterfall original
    im1 = axes[0, 0].imshow(original_waterfall.T, aspect='auto', origin='lower',
                           cmap=config.SNR_COLORMAP)
    axes[0, 0].set_title('DM vs. Time - Original')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Dispersion Measure (pc cm⁻³)')
    axes[0, 0].axvline(peak_time_orig, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Waterfall limpio
    im2 = axes[0, 1].imshow(cleaned_waterfall.T, aspect='auto', origin='lower',
                           cmap=config.SNR_COLORMAP)
    axes[0, 1].set_title('DM vs. Time - Cleaned')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Dispersion Measure (pc cm⁻³)')
    axes[0, 1].axvline(peak_time_clean, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Comparación de perfiles SNR
    time_axis = np.arange(len(snr_original))
    axes[1, 0].plot(time_axis, snr_original, label=f'Original (pico: {peak_snr_orig:.2f})',
                    alpha=0.7, color='blue')
    axes[1, 0].plot(time_axis, snr_profile, label=f'Limpio (pico: {peak_snr_clean:.2f})',
                    alpha=0.7, color='red')
    axes[1, 0].axhline(config.SNR_THRESH, color='green', linestyle='--',
                      label=f'Umbral SNR = {config.SNR_THRESH}')
    axes[1, 0].axvline(peak_time_orig, color='blue', linestyle=':', alpha=0.5)
    axes[1, 0].axvline(peak_time_clean, color='red', linestyle=':', alpha=0.5)
    axes[1, 0].set_title('SNR Profile Comparison')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('SNR (σ)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histograma de mejora en SNR
    snr_improvement = snr_profile - snr_original
    axes[1, 1].hist(snr_improvement, bins=50, alpha=0.7, color='green')
    axes[1, 1].set_title('SNR Improvement by RFI Cleaning')
    axes[1, 1].set_xlabel('Δ SNR (Cleaned - Original)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].axvline(np.mean(snr_improvement), color='red', linestyle='--',
                      label=f'Media: {np.mean(snr_improvement):.3f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guarda gráfico
    comparison_path = output_dir / "snr_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Comparación SNR guardada en: {comparison_path}")
    
    plt.show()


def create_test_data_with_rfi(
    n_time: int = 1024,
    n_freq: int = 256,
    frb_strength: float = 10.0,
    rfi_fraction: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea datos de prueba con FRB sintético y RFI.
    
    Parameters
    ----------
    n_time : int
        Número de muestras temporales
    n_freq : int
        Número de canales de frecuencia
    frb_strength : float
        Intensidad del FRB sintético (en sigma)
    rfi_fraction : float
        Fracción de datos contaminados por RFI
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Waterfall con RFI, datos de polarización
    """
    print(f"[INFO] Creando datos de prueba ({n_time}x{n_freq}) con FRB y RFI...")
    
    # Genera ruido gaussiano base
    waterfall = np.random.normal(0, 1, (n_time, n_freq))
    
    # Inyecta FRB sintético (dispersado)
    frb_center_time = n_time // 2
    frb_width = 20
    dm_sweep = np.linspace(-30, 30, n_freq)  # Simulación de barrido DM
    
    for i, dm_delay in enumerate(dm_sweep):
        time_idx = int(frb_center_time + dm_delay)
        if 0 <= time_idx < n_time:
            # Pulso gaussiano con dispersión
            pulse_profile = frb_strength * np.exp(
                -0.5 * ((np.arange(n_time) - time_idx) / frb_width) ** 2
            )
            waterfall[:, i] += pulse_profile
    
    # Inyecta RFI
    n_rfi_samples = int(rfi_fraction * n_time * n_freq)
    
    # RFI impulsivo
    rfi_impulse_indices = np.random.choice(
        n_time * n_freq, n_rfi_samples // 2, replace=False
    )
    rfi_time_idx = rfi_impulse_indices // n_freq
    rfi_freq_idx = rfi_impulse_indices % n_freq
    waterfall[rfi_time_idx, rfi_freq_idx] += np.random.normal(
        0, 10, len(rfi_impulse_indices)
    )
    
    # RFI de canales persistentes
    n_bad_channels = int(0.05 * n_freq)
    bad_channels = np.random.choice(n_freq, n_bad_channels, replace=False)
    for ch in bad_channels:
        waterfall[:, ch] += np.random.normal(0, 5, n_time)
    
    # RFI temporal (interferencia de banda ancha)
    n_bad_times = int(0.02 * n_time)
    bad_times = np.random.choice(n_time, n_bad_times, replace=False)
    for t in bad_times:
        waterfall[t, :] += np.random.normal(0, 8, n_freq)
    
    # Genera datos de polarización sintéticos
    stokes_v = np.random.normal(0, 0.1, (n_time, n_freq))
    
    print(f"[INFO] Datos creados. FRB en tiempo {frb_center_time}, RFI: {rfi_fraction*100:.1f}%")
    
    return waterfall, stokes_v


def main_rfi_test():
    """
    Función principal para probar el pipeline de limpieza de RFI.
    """
    print("[INFO] === PRUEBA DE LIMPIEZA DE RFI ===")
    
    # Crea datos de prueba
    waterfall, stokes_v = create_test_data_with_rfi(
        n_time=1024,
        n_freq=256,
        frb_strength=8.0,
        rfi_fraction=0.15
    )
    
    # Configura directorio de salida
    output_dir = Path("./test_rfi_output")
    output_dir.mkdir(exist_ok=True)
    
    # Ejecuta pipeline de limpieza
    cleaned_waterfall, snr_profile, rfi_stats = integrate_rfi_cleaning_pipeline(
        waterfall,
        stokes_v=stokes_v,
        output_dir=output_dir
    )
    
    # Imprime estadísticas finales
    print("\n[INFO] === ESTADÍSTICAS FINALES ===")
    for key, value in rfi_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Encuentra pico final
    peak_snr, peak_time, peak_idx = find_snr_peak(snr_profile)
    print(f"\n[INFO] Pico SNR final: {peak_snr:.2f} en tiempo {peak_time}")
    
    # Calcula eficiencia de limpieza
    snr_original, _ = compute_snr_profile(waterfall, off_regions=config.SNR_OFF_REGIONS)
    peak_snr_orig, _, _ = find_snr_peak(snr_original)
    
    improvement = peak_snr - peak_snr_orig
    print(f"[INFO] Mejora en SNR: {improvement:.2f} ({improvement/peak_snr_orig*100:.1f}%)")
    
    print(f"\n[INFO] Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main_rfi_test()
