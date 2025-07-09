#!/usr/bin/env python3
"""
Script para experimentar con diferentes valores de SLICE_LEN y comparar resultados.

Este script automáticamente:
1. Guarda la configuración original
2. Prueba diferentes valores de SLICE_LEN
3. Ejecuta el pipeline con cada valor
4. Compara los resultados
5. Restaura la configuración original

Uso:
    python experiment_slice_len.py
"""

import sys
import os
import csv
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configurar path
sys.path.append(str(Path(__file__).parent))

from DRAFTS import config
from DRAFTS.pipeline import process_file

def backup_config():
    """Crear backup de la configuración actual."""
    config_path = Path(__file__).parent / "DRAFTS" / "config.py"
    backup_path = config_path.with_suffix(".py.backup")
    shutil.copy2(config_path, backup_path)
    print(f"✅ Backup creado: {backup_path}")
    return backup_path

def restore_config(backup_path: Path):
    """Restaurar configuración desde backup."""
    config_path = Path(__file__).parent / "DRAFTS" / "config.py"
    shutil.copy2(backup_path, config_path)
    backup_path.unlink()  # Eliminar backup
    print(f"✅ Configuración restaurada desde backup")

def modify_slice_len(new_value: int):
    """Modificar SLICE_LEN en config.py."""
    config_path = Path(__file__).parent / "DRAFTS" / "config.py"
    
    # Leer archivo actual
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar y reemplazar SLICE_LEN
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('SLICE_LEN:') and '=' in line:
            # Reemplazar la línea completa
            lines[i] = f'SLICE_LEN: int = {new_value} # Experimental value'
            break
    
    # Escribir archivo modificado
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ SLICE_LEN modificado a: {new_value}")

def count_candidates(results_dir: Path) -> Tuple[int, float]:
    """Contar candidatos en archivos CSV y calcular SNR promedio."""
    csv_files = list(results_dir.glob("*.csv"))
    total_candidates = 0
    total_snr = 0.0
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                file_candidates = 0
                file_snr = 0.0
                
                for row in reader:
                    file_candidates += 1
                    # Intentar obtener SNR si existe
                    for snr_col in ['snr', 'SNR', 'detection_snr', 'confidence']:
                        if snr_col in row and row[snr_col]:
                            try:
                                file_snr += float(row[snr_col])
                                break
                            except ValueError:
                                pass
                
                total_candidates += file_candidates
                total_snr += file_snr
                
        except Exception as e:
            print(f"⚠️  Error leyendo {csv_file}: {e}")
    
    avg_snr = total_snr / total_candidates if total_candidates > 0 else 0.0
    return total_candidates, avg_snr

def run_pipeline_experiment(slice_len_values: List[int], 
                          target_file: str = None) -> Dict[int, Dict[str, Any]]:
    """Ejecutar experimento con diferentes valores de SLICE_LEN."""
    
    print("🧪 === EXPERIMENTO SLICE_LEN ===\n")
    
    # Crear backup de configuración
    backup_path = backup_config()
    
    # Directorio base para resultados
    base_results_dir = Path(__file__).parent / "Results" / "slice_len_experiment"
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        for slice_len in slice_len_values:
            print(f"\n🔬 === EXPERIMENTO CON SLICE_LEN = {slice_len} ===")
            
            # Modificar configuración
            modify_slice_len(slice_len)
            
            # Recargar módulo config
            import importlib
            importlib.reload(config)
            
            # Directorio para resultados de este experimento
            exp_results_dir = base_results_dir / f"slice_len_{slice_len}"
            exp_results_dir.mkdir(exist_ok=True)
            
            # Medir tiempo de ejecución
            start_time = time.time()
            
            try:
                # Obtener archivo objetivo
                if target_file is None:
                    data_dir = Path(__file__).parent / "Data"
                    fil_files = list(data_dir.glob("*.fil"))
                    if fil_files:
                        target_file = fil_files[0].name
                    else:
                        fits_files = list(data_dir.glob("*.fits"))
                        if fits_files:
                            target_file = fits_files[0].name
                        else:
                            raise FileNotFoundError("No se encontraron archivos .fil o .fits en Data/")
                
                print(f"📁 Procesando archivo: {target_file}")
                
                # Ejecutar pipeline
                process_file(target_file, save_dir=exp_results_dir)
                
                execution_time = time.time() - start_time
                
                # Contar resultados
                candidates, avg_snr = count_candidates(exp_results_dir)
                
                # Contar archivos de composites
                composite_files = len(list(exp_results_dir.glob("*composite*.png")))
                
                results[slice_len] = {
                    'candidates': candidates,
                    'avg_snr': avg_snr,
                    'execution_time': execution_time,
                    'composite_files': composite_files,
                    'success': True,
                    'error': None
                }
                
                print(f"✅ SLICE_LEN {slice_len} completado:")
                print(f"   • Candidatos: {candidates}")
                print(f"   • SNR promedio: {avg_snr:.2f}")
                print(f"   • Tiempo: {execution_time:.1f}s")
                print(f"   • Composites: {composite_files}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                results[slice_len] = {
                    'candidates': 0,
                    'avg_snr': 0.0,
                    'execution_time': execution_time,
                    'composite_files': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ Error con SLICE_LEN {slice_len}: {e}")
        
    finally:
        # Restaurar configuración original
        restore_config(backup_path)
        
        # Recargar config original
        import importlib
        importlib.reload(config)
    
    return results

def print_comparison_table(results: Dict[int, Dict[str, Any]]):
    """Imprimir tabla comparativa de resultados."""
    
    print("\n" + "="*90)
    print("📊 COMPARACIÓN DE RESULTADOS POR SLICE_LEN")
    print("="*90)
    
    # Cabecera
    print("SLICE_LEN | Candidatos | SNR Prom | Tiempo(s) | Composites | Estado")
    print("-"*90)
    
    # Datos
    for slice_len in sorted(results.keys()):
        r = results[slice_len]
        status = "✅ OK" if r['success'] else "❌ ERROR"
        
        print(f"{slice_len:8d} | {r['candidates']:10d} | {r['avg_snr']:8.2f} | "
              f"{r['execution_time']:8.1f} | {r['composite_files']:9d} | {status}")
    
    print("="*90)
    
    # Análisis
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        best_candidates = max(successful_results.items(), key=lambda x: x[1]['candidates'])
        best_snr = max(successful_results.items(), key=lambda x: x[1]['avg_snr'])
        fastest = min(successful_results.items(), key=lambda x: x[1]['execution_time'])
        
        print("\n🏆 MEJORES RESULTADOS:")
        print(f"   • Más candidatos: SLICE_LEN = {best_candidates[0]} ({best_candidates[1]['candidates']} candidatos)")
        print(f"   • Mejor SNR promedio: SLICE_LEN = {best_snr[0]} (SNR = {best_snr[1]['avg_snr']:.2f})")
        print(f"   • Más rápido: SLICE_LEN = {fastest[0]} ({fastest[1]['execution_time']:.1f}s)")
        
        print(f"\n💡 RECOMENDACIONES:")
        
        # Analizar patrones
        slice_lens = sorted(successful_results.keys())
        candidates_by_len = [successful_results[sl]['candidates'] for sl in slice_lens]
        snr_by_len = [successful_results[sl]['avg_snr'] for sl in slice_lens]
        
        if len(slice_lens) >= 2:
            # Encontrar SLICE_LEN óptimo basado en candidatos * SNR
            scores = [(sl, successful_results[sl]['candidates'] * successful_results[sl]['avg_snr']) 
                     for sl in slice_lens]
            best_overall = max(scores, key=lambda x: x[1])
            
            print(f"   • Para máxima sensibilidad: SLICE_LEN = {best_overall[0]}")
            print(f"     (Mejor balance candidatos × SNR = {best_overall[1]:.1f})")
            
            # Recomendar basado en tendencias
            if candidates_by_len[0] > candidates_by_len[-1]:
                print(f"   • Valores menores detectan más candidatos → Probar SLICE_LEN < {slice_lens[0]}")
            elif candidates_by_len[-1] > candidates_by_len[0]:
                print(f"   • Valores mayores detectan más candidatos → Probar SLICE_LEN > {slice_lens[-1]}")

def save_results_csv(results: Dict[int, Dict[str, Any]], output_file: Path):
    """Guardar resultados en archivo CSV."""
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Cabecera
        writer.writerow(['SLICE_LEN', 'candidates', 'avg_snr', 'execution_time_s', 
                        'composite_files', 'success', 'error'])
        
        # Datos
        for slice_len in sorted(results.keys()):
            r = results[slice_len]
            writer.writerow([
                slice_len,
                r['candidates'],
                r['avg_snr'],
                r['execution_time'],
                r['composite_files'],
                r['success'],
                r['error'] if r['error'] else ''
            ])
    
    print(f"📊 Resultados guardados en: {output_file}")

def main():
    """Función principal del experimento."""
    
    print("🧪 EXPERIMENTO SLICE_LEN - PIPELINE DRAFTS")
    print("="*50)
    
    # Valores a probar (puedes modificar esta lista)
    slice_len_values = [32, 64, 128]  # Comenzar con rango pequeño
    
    print(f"🔬 Valores a probar: {slice_len_values}")
    print(f"📝 Configuración actual SLICE_LEN: {config.SLICE_LEN}")
    
    # Confirmar ejecución
    response = input("\n¿Continuar con el experimento? (y/N): ").strip().lower()
    if response not in ['y', 'yes', 'sí', 's']:
        print("❌ Experimento cancelado")
        return
    
    # Ejecutar experimento
    results = run_pipeline_experiment(slice_len_values)
    
    # Mostrar resultados
    print_comparison_table(results)
    
    # Guardar resultados
    output_file = Path(__file__).parent / "Results" / "slice_len_experiment_results.csv"
    save_results_csv(results, output_file)
    
    print(f"\n✅ Experimento completado")
    print(f"📁 Resultados detallados en: Results/slice_len_experiment/")
    print(f"📊 Resumen en: {output_file}")

if __name__ == "__main__":
    main()
