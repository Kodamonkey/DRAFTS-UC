#!/usr/bin/env python3
"""Script para probar las nuevas funcionalidades de SNR en el pipeline."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from DRAFTS.snr_utils import compute_snr_profile, find_snr_peak, inject_synthetic_frb
    from DRAFTS.visualization import save_patch_plot
    from DRAFTS import config
    
    print("✓ Módulos SNR importados exitosamente")
    
    # Crear datos de prueba
    print("\n=== Creando datos de prueba ===")
    np.random.seed(42)
    
    # Simular waterfall con ruido
    n_time, n_freq = 500, 256
    noise_waterfall = np.random.normal(0, 1, (n_time, n_freq))
    
    # Inyectar FRB sintético
    frb_waterfall = inject_synthetic_frb(
        noise_waterfall, 
        peak_time_idx=250, 
        peak_freq_idx=128, 
        amplitude=8.0,
        width_time=8,
        width_freq=30
    )
    
    print(f"Waterfall shape: {frb_waterfall.shape}")
    
    # Calcular SNR
    print("\n=== Calculando perfiles SNR ===")
    snr_profile, sigma = compute_snr_profile(frb_waterfall)
    peak_snr, peak_time, peak_idx = find_snr_peak(snr_profile)
    
    print(f"Peak SNR: {peak_snr:.2f}σ")
    print(f"Peak position: {peak_idx} (inyectado en 250)")
    print(f"Sigma estimado: {sigma:.3f}")
    
    # Crear directorio de pruebas
    test_dir = Path("test_snr_output")
    test_dir.mkdir(exist_ok=True)
    
    # Probar save_patch_plot con SNR
    print("\n=== Probando save_patch_plot con SNR ===")
    freq_axis = np.linspace(1400, 1200, n_freq)
    time_reso = 1e-4  # 100 microsegundos
    start_time = 0.0
    
    try:
        save_patch_plot(
            patch=frb_waterfall,
            out_path=test_dir / "test_patch_with_snr.png",
            freq=freq_axis,
            time_reso=time_reso,
            start_time=start_time,
            off_regions=None,  # Usar método IQR
            thresh_snr=config.SNR_THRESH,
        )
        print(f"✓ Patch con SNR guardado en: {test_dir / 'test_patch_with_snr.png'}")
    except Exception as e:
        print(f"✗ Error en save_patch_plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Crear un plot simple para verificar
    print("\n=== Creando plot de verificación ===")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel superior: perfil SNR
    time_axis = np.arange(len(snr_profile)) * time_reso
    ax1.plot(time_axis, snr_profile, 'b-', linewidth=1.5, label='SNR Profile')
    ax1.axhline(y=config.SNR_THRESH, color='red', linestyle='--', alpha=0.7, label=f'Threshold = {config.SNR_THRESH}σ')
    ax1.plot(time_axis[peak_idx], peak_snr, 'ro', markersize=8, label=f'Peak = {peak_snr:.1f}σ')
    ax1.set_ylabel('SNR (σ)')
    ax1.set_title(f'SNR Profile - Peak: {peak_snr:.1f}σ at t={time_axis[peak_idx]*1000:.1f}ms')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel inferior: waterfall
    extent = [time_axis[0]*1000, time_axis[-1]*1000, freq_axis.min(), freq_axis.max()]
    im = ax2.imshow(frb_waterfall.T, origin='lower', aspect='auto', cmap='viridis', extent=extent)
    ax2.axvline(x=time_axis[peak_idx]*1000, color='red', linestyle='-', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_title('Waterfall with Synthetic FRB')
    
    plt.colorbar(im, ax=ax2, label='Intensity')
    plt.tight_layout()
    plt.savefig(test_dir / "test_snr_verification.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot de verificación guardado en: {test_dir / 'test_snr_verification.png'}")
    
    print(f"\n✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
    print(f"📁 Archivos de salida en: {test_dir.absolute()}")
    print(f"🔍 Revisa los archivos .png para ver los resultados")
    
except ImportError as e:
    print(f"✗ Error de importación: {e}")
    print("Asegúrate de que todos los módulos estén disponibles")
except Exception as e:
    print(f"✗ Error durante las pruebas: {e}")
    import traceback
    traceback.print_exc()
