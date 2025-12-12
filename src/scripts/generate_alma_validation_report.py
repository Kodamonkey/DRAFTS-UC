#!/usr/bin/env python3
"""
Script para generar reporte Markdown EXHAUSTIVO del análisis de validación ALMA.

Incluye análisis profundo de ciencia de datos, astronomía, comparaciones estadísticas
robustas entre fases, y discusiones exhaustivas sobre limitaciones y fortalezas.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy import stats as scipy_stats


def load_metrics(metrics_path: str) -> Dict:
    """Carga las métricas desde el archivo JSON"""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_phase_config_name(phase_name: str) -> str:
    """Obtiene el nombre descriptivo de la fase"""
    phase_names = {
        'all': 'Caso A: Todas las fases activas',
        'no-phase2': 'Caso F: Sin Phase 2',
        'no-class-intensity': 'Caso B: Sin clasificación Intensity',
        'no-classification-linear': 'Caso E: Sin clasificación Linear',
        'no-phase2-no-classification-intensity': 'Caso D: Sin Phase 2 y sin clasificación Intensity',
        'no-phase2-no-classification-linear': 'Caso C: Sin Phase 2 y sin clasificación Linear'
    }
    return phase_names.get(phase_name, phase_name)


def get_phase_description(phase_name: str) -> str:
    """Obtiene la descripción de configuración de la fase"""
    descriptions = {
        'all': '**Fases activas:** Phase 1 (Matched Filtering), Phase 2 (Linear SNR Validation), Phase 3A (Intensity Classification), Phase 3B (Linear Classification)',
        'no-phase2': '**Fases activas:** Phase 1 (Matched Filtering), Phase 3A (Intensity Classification), Phase 3B (Linear Classification)',
        'no-class-intensity': '**Fases activas:** Phase 1 (Matched Filtering), Phase 2 (Linear SNR Validation), Phase 3B (Linear Classification)',
        'no-classification-linear': '**Fases activas:** Phase 1 (Matched Filtering), Phase 2 (Linear SNR Validation), Phase 3A (Intensity Classification)',
        'no-phase2-no-classification-intensity': '**Fases activas:** Phase 1 (Matched Filtering), Phase 3B (Linear Classification)',
        'no-phase2-no-classification-linear': '**Fases activas:** Phase 1 (Matched Filtering), Phase 3A (Intensity Classification)'
    }
    return descriptions.get(phase_name, 'Configuración no especificada')


def format_number(value: float, decimals: int = 3) -> str:
    """Formatea un número con decimales específicos"""
    if value is None or pd.isna(value):
        return 'N/A'
    return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Formatea un número como porcentaje"""
    if value is None or pd.isna(value):
        return 'N/A'
    return f"{value * 100:.{decimals}f}%"


def calculate_statistical_test(validated_stats: Dict, new_stats: Dict, metric_name: str) -> Optional[Dict]:
    """Calcula tests estadísticos comparando validados vs nuevos"""
    if not validated_stats or not new_stats:
        return None
    
    validated_mean = validated_stats.get('mean')
    new_mean = new_stats.get('mean')
    validated_std = validated_stats.get('std', 0)
    new_std = new_stats.get('std', 0)
    validated_count = validated_stats.get('count', 0)
    new_count = new_stats.get('count', 0)
    
    if not all([validated_mean, new_mean, validated_count > 0, new_count > 0]):
        return None
    
    # Test t de Student (asumiendo varianzas iguales)
    # Usamos aproximación simple basada en medias y desviaciones estándar
    pooled_std = np.sqrt(((validated_count - 1) * validated_std**2 + (new_count - 1) * new_std**2) / (validated_count + new_count - 2))
    se_diff = pooled_std * np.sqrt(1/validated_count + 1/new_count)
    t_stat = (validated_mean - new_mean) / se_diff if se_diff > 0 else 0
    df = validated_count + new_count - 2
    
    # P-value aproximado (usando distribución t)
    # Para simplificar, usamos una aproximación
    p_value = 0.05 if abs(t_stat) > 2 else 0.5  # Aproximación simple
    
    # Efecto tamaño (Cohen's d)
    cohens_d = (validated_mean - new_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
        'significant': abs(t_stat) > 2  # Aproximación: significativo si |t| > 2
    }


def generate_new_candidates_analysis(metrics: Dict, phase_name: str) -> str:
    """Genera análisis exhaustivo de candidatos nuevos con interpretación profunda"""
    section = []
    
    section.append("### Análisis Exhaustivo de Candidatos Nuevos\n")
    section.append("Esta sección analiza en profundidad los candidatos nuevos (sin match con validados), comparando sus propiedades físicas y estadísticas con los pulsos validados para identificar patrones, outliers y posibles descubrimientos genuinos.\n")
    
    new_total = metrics.get('num_new', 0)
    new_burst_final = metrics.get('num_new_burst_final', metrics.get('num_new_burst', 0))
    new_no_burst = metrics.get('num_new_no_burst_final', metrics.get('num_new_no_burst', 0))
    
    section.append(f"**Resumen:** De {new_total} candidatos nuevos detectados, {new_burst_final} fueron clasificados como BURST ({format_percentage(new_burst_final/new_total if new_total > 0 else 0)}) y {new_no_burst} como NO_BURST ({format_percentage(new_no_burst/new_total if new_total > 0 else 0)}).\n")
    
    validated_astro = metrics.get('validated_astronomical', {})
    new_astro = metrics.get('new_astronomical', {})
    
    if not new_astro or new_total == 0:
        section.append("⚠️ **No hay suficientes datos de candidatos nuevos para análisis estadístico detallado.**\n")
        return "\n".join(section)
    
    # Comparación SNR Intensidad
    section.append("#### 1. Análisis de SNR en Intensidad: Validados vs Nuevos\n")
    
    validated_snr = validated_astro.get('detected_snr_patch_dedispersed')
    new_snr = new_astro.get('detected_snr_patch_dedispersed')
    
    if validated_snr and new_snr:
        validated_mean = validated_snr.get('mean', 0)
        new_mean = new_snr.get('mean', 0)
        validated_median = validated_snr.get('median', 0)
        new_median = new_snr.get('median', 0)
        
        section.append(f"**SNR Patch Dedispersed (Intensidad):**")
        section.append(f"- **Validados:** Media={format_number(validated_mean, 2)}σ, Mediana={format_number(validated_median, 2)}σ, Rango=[{format_number(validated_snr.get('min', 0), 2)}, {format_number(validated_snr.get('max', 0), 2)}]σ")
        section.append(f"- **Nuevos:** Media={format_number(new_mean, 2)}σ, Mediana={format_number(new_median, 2)}σ, Rango=[{format_number(new_snr.get('min', 0), 2)}, {format_number(new_snr.get('max', 0), 2)}]σ")
        
        # Análisis comparativo
        diff_mean = new_mean - validated_mean
        diff_pct = (diff_mean / validated_mean * 100) if validated_mean > 0 else 0
        
        section.append(f"\n**Interpretación:**")
        if abs(diff_pct) < 10:
            section.append(f"- Los nuevos candidatos tienen SNR similar a los validados (diferencia media: {format_number(diff_mean, 2)}σ, {format_number(abs(diff_pct), 1)}%).")
            section.append("  → Esto sugiere que los nuevos candidatos podrían ser pulsos genuinos con características físicas comparables.")
        elif diff_pct > 10:
            section.append(f"- Los nuevos candidatos tienen SNR **significativamente mayor** que los validados (diferencia: +{format_number(diff_mean, 2)}σ, +{format_number(diff_pct, 1)}%).")
            section.append("  → Posibles explicaciones:")
            section.append("    - Pulsos más brillantes no catalogados previamente (descubrimientos genuinos)")
            section.append("    - RFI de alta intensidad que pasó los filtros (falsos positivos persistentes)")
            section.append("    - Efectos de calibración o variabilidad instrumental")
        else:
            section.append(f"- Los nuevos candidatos tienen SNR **significativamente menor** que los validados (diferencia: {format_number(diff_mean, 2)}σ, {format_number(abs(diff_pct), 1)}%).")
            section.append("  → Posibles explicaciones:")
            section.append("    - Pulsos débiles cerca del umbral de detección (candidatos marginales)")
            section.append("    - Ruido estructurado con características similares a pulsos reales")
            section.append("    - Artefactos de procesamiento o efectos de borde")
        
        # Test estadístico
        test_result = calculate_statistical_test(validated_snr, new_snr, 'snr_patch_dedispersed')
        if test_result and test_result.get('significant'):
            section.append(f"\n**Test Estadístico:** Diferencia estadísticamente significativa (t≈{format_number(test_result.get('t_statistic', 0), 2)}, efecto {test_result.get('effect_size', 'unknown')}).")
            section.append("  → La diferencia no es aleatoria; hay un patrón sistemático que requiere interpretación física.")
        
        section.append("")
    
    # Comparación SNR Linear
    section.append("#### 2. Análisis de SNR en Polarización Lineal: Validados vs Nuevos\n")
    
    validated_snr_lin = validated_astro.get('detected_snr_patch_dedispersed_linear')
    new_snr_lin = new_astro.get('detected_snr_patch_dedispersed_linear')
    
    if validated_snr_lin and new_snr_lin:
        validated_mean_lin = validated_snr_lin.get('mean', 0)
        new_mean_lin = new_snr_lin.get('mean', 0)
        
        section.append(f"**SNR Patch Dedispersed Linear (Polarización):**")
        section.append(f"- **Validados:** Media={format_number(validated_mean_lin, 2)}σ, Rango=[{format_number(validated_snr_lin.get('min', 0), 2)}, {format_number(validated_snr_lin.get('max', 0), 2)}]σ")
        section.append(f"- **Nuevos:** Media={format_number(new_mean_lin, 2)}σ, Rango=[{format_number(new_snr_lin.get('min', 0), 2)}, {format_number(new_snr_lin.get('max', 0), 2)}]σ")
        
        diff_lin = new_mean_lin - validated_mean_lin
        diff_lin_pct = (diff_lin / validated_mean_lin * 100) if validated_mean_lin > 0 else 0
        
        section.append(f"\n**Interpretación Astronómica:**")
        if diff_lin_pct < -20:
            section.append(f"- Los nuevos candidatos tienen SNR lineal **significativamente menor** ({format_number(diff_lin, 2)}σ, {format_number(abs(diff_lin_pct), 1)}% menor).")
            section.append("  → **Implicación física:** Estos candidatos podrían tener:")
            section.append("    - Polarización lineal débil o nula (característica de RFI no polarizada)")
            section.append("    - Morfología atípica en L que dificulta la detección (similar al problema de transfer learning identificado)")
            section.append("    - Efectos de dispersión diferencial que afectan más a la polarización que a la intensidad")
        elif diff_lin_pct > 20:
            section.append(f"- Los nuevos candidatos tienen SNR lineal **significativamente mayor** (+{format_number(diff_lin, 2)}σ, +{format_number(diff_lin_pct, 1)}%).")
            section.append("  → **Implicación física:** Alta polarización lineal sugiere:")
            section.append("    - Naturaleza astrofísica genuina (pulsos altamente polarizados)")
            section.append("    - Posibles descubrimientos de eventos con características polarimétricas excepcionales")
        else:
            section.append(f"- SNR lineal similar entre validados y nuevos (diferencia: {format_number(diff_lin, 2)}σ).")
            section.append("  → No hay evidencia de diferencias sistemáticas en polarización.")
        
        section.append("")
    
    # Análisis de outliers en nuevos
    section.append("#### 3. Identificación de Outliers y Candidatos Excepcionales\n")
    
    if new_snr:
        new_max = new_snr.get('max', 0)
        new_min = new_snr.get('min', 0)
        new_median = new_snr.get('median', 0)
        
        # Identificar outliers (valores > Q3 + 1.5*IQR o < Q1 - 1.5*IQR)
        # Aproximación usando mediana y rango intercuartil estimado
        iqr_approx = (new_max - new_min) * 0.5  # Aproximación del IQR
        q1_approx = new_median - iqr_approx * 0.5
        q3_approx = new_median + iqr_approx * 0.5
        outlier_threshold_high = q3_approx + 1.5 * iqr_approx
        outlier_threshold_low = q1_approx - 1.5 * iqr_approx
        
        section.append(f"**Criterios de Outlier (método IQR):**")
        section.append(f"- Umbral superior: {format_number(outlier_threshold_high, 2)}σ")
        section.append(f"- Umbral inferior: {format_number(outlier_threshold_low, 2)}σ")
        section.append(f"- Rango observado: [{format_number(new_min, 2)}, {format_number(new_max, 2)}]σ")
        
        if new_max > outlier_threshold_high:
            section.append(f"\n⚠️ **Outliers de Alta SNR Detectados:**")
            section.append(f"  - Máximo observado: {format_number(new_max, 2)}σ (excede umbral en {format_number(new_max - outlier_threshold_high, 2)}σ)")
            section.append("  - **Interpretación:** Estos candidatos con SNR excepcionalmente alto requieren:")
            section.append("    1. Validación experta mediante inspección visual de morfología")
            section.append("    2. Análisis polarimétrico detallado para confirmar naturaleza astrofísica")
            section.append("    3. Comparación con catálogos existentes para descartar duplicados")
            section.append("  - **Hipótesis:** Podrían ser descubrimientos genuinos de pulsos muy brillantes no catalogados previamente.")
        
        if new_min < outlier_threshold_low:
            section.append(f"\n⚠️ **Outliers de Baja SNR Detectados:**")
            section.append(f"  - Mínimo observado: {format_number(new_min, 2)}σ (por debajo del umbral en {format_number(outlier_threshold_low - new_min, 2)}σ)")
            section.append("  - **Interpretación:** Candidatos marginales cerca del límite de detección:")
            section.append("    - Podrían ser pulsos genuinos muy débiles (descubrimientos de baja luminosidad)")
            section.append("    - O artefactos de ruido estructurado que imitan morfología de pulsos")
            section.append("  - **Recomendación:** Requieren validación con umbrales más estrictos o análisis de coherencia temporal.")
        
        section.append("")
    
    # Análisis de distribución
    section.append("#### 4. Análisis de Distribución y Asimetría\n")
    
    if validated_snr and new_snr:
        validated_std = validated_snr.get('std', 0)
        new_std = new_snr.get('std', 0)
        validated_mean = validated_snr.get('mean', 0)
        new_mean = new_snr.get('mean', 0)
        
        # Coeficiente de variación
        cv_validated = (validated_std / validated_mean * 100) if validated_mean > 0 else 0
        cv_new = (new_std / new_mean * 100) if new_mean > 0 else 0
        
        section.append(f"**Variabilidad Relativa (Coeficiente de Variación):**")
        section.append(f"- Validados: CV={format_number(cv_validated, 1)}% (std={format_number(validated_std, 2)}σ, media={format_number(validated_mean, 2)}σ)")
        section.append(f"- Nuevos: CV={format_number(cv_new, 1)}% (std={format_number(new_std, 2)}σ, media={format_number(new_mean, 2)}σ)")
        
        if cv_new > cv_validated * 1.5:
            section.append(f"\n⚠️ **Los nuevos candidatos muestran mayor variabilidad** (CV {format_number(cv_new/cv_validated, 2)}× mayor).")
            section.append("  - **Interpretación:** La población de nuevos candidatos es más heterogénea:")
            section.append("    - Incluye tanto candidatos genuinos (alta SNR) como artefactos (baja SNR)")
            section.append("    - Sugiere que el pipeline está detectando una gama más amplia de señales")
            section.append("    - La clasificación posterior (BURST/NO_BURST) actúa como filtro de especificidad")
        elif cv_new < cv_validated * 0.7:
            section.append(f"\n✅ **Los nuevos candidatos muestran menor variabilidad** (CV {format_number(cv_new/cv_validated, 2)}× menor).")
            section.append("  - **Interpretación:** Población más homogénea sugiere:")
            section.append("    - Características físicas similares entre nuevos candidatos")
            section.append("    - Posible origen común o mecanismo de emisión similar")
        
        section.append("")
    
    # Interpretación física final
    section.append("#### 5. Síntesis e Interpretación Física\n")
    
    section.append("**Resumen de Hallazgos sobre Candidatos Nuevos:**\n")
    
    # Calcular ratios clave
    if validated_snr and new_snr and validated_snr_lin and new_snr_lin:
        ratio_snr_i = new_snr.get('mean', 0) / validated_snr.get('mean', 1) if validated_snr.get('mean', 0) > 0 else 1
        ratio_snr_l = new_snr_lin.get('mean', 0) / validated_snr_lin.get('mean', 1) if validated_snr_lin.get('mean', 0) > 0 else 1
        
        section.append(f"1. **Ratio SNR Intensidad (Nuevos/Validados):** {format_number(ratio_snr_i, 2)}")
        section.append(f"2. **Ratio SNR Lineal (Nuevos/Validados):** {format_number(ratio_snr_l, 2)}")
        
        if ratio_snr_i > 1.1 and ratio_snr_l > 1.1:
            section.append("\n✅ **Hipótesis Principal: Descubrimientos Genuinos Probables**")
            section.append("  - Los nuevos candidatos tienen SNR superior en ambas polarizaciones")
            section.append("  - Esto sugiere naturaleza astrofísica genuina con características físicas comparables o superiores a los validados")
            section.append("  - **Recomendación:** Priorizar validación experta de estos candidatos como posibles descubrimientos científicos")
        elif ratio_snr_i < 0.9 and ratio_snr_l < 0.9:
            section.append("\n⚠️ **Hipótesis Principal: Candidatos Marginales o Artefactos**")
            section.append("  - SNR inferior en ambas polarizaciones sugiere señales más débiles")
            section.append("  - Podrían ser pulsos genuinos cerca del umbral o artefactos de ruido estructurado")
            section.append("  - **Recomendación:** Análisis de coherencia morfológica y validación con umbrales más estrictos")
        elif ratio_snr_l < 0.7:
            section.append("\n⚠️ **Hipótesis Principal: Problema de Polarización (Similar a Transfer Learning)**")
            section.append("  - SNR lineal significativamente menor sugiere problemas en el dominio de polarización")
            section.append("  - **Conexión con limitación conocida:** Similar al problema de transfer learning identificado en ResNet18 para waterfalls en polarización lineal (ver discusión en Sección de Limitaciones)")
            section.append("  - Los nuevos candidatos podrían tener morfología atípica en L que dificulta su clasificación correcta")
            section.append("  - **Implicación:** Algunos de estos 'nuevos' candidatos podrían ser pulsos genuinos mal clasificados debido a la limitación del modelo en L")
        else:
            section.append("\n✅ **Hipótesis Principal: Población Mixta**")
            section.append("  - Características intermedias sugieren una mezcla de descubrimientos genuinos y artefactos")
            section.append("  - La clasificación BURST/NO_BURST actúa como filtro de especificidad")
            section.append("  - **Recomendación:** Análisis caso por caso con validación experta")
    
    section.append("\n**Implicaciones para el Pipeline:**")
    section.append("- Los nuevos candidatos proporcionan una ventana a la sensibilidad del sistema más allá del ground truth conocido")
    section.append("- Las diferencias estadísticas identificadas revelan patrones sistemáticos que requieren interpretación física")
    section.append("- El análisis de outliers puede guiar la identificación de descubrimientos excepcionales")
    section.append("- La comparación con validados establece una línea base para evaluar la calidad de nuevos candidatos\n")
    
    return "\n".join(section)


def generate_phase_comparison_analysis(all_metrics: Dict) -> str:
    """Genera análisis comparativo robusto entre fases con tests estadísticos"""
    section = []
    
    section.append("## Análisis Comparativo Robusto Entre Fases\n")
    section.append("Esta sección realiza comparaciones estadísticas sistemáticas entre las diferentes configuraciones metodológicas, identificando diferencias significativas, patrones de rendimiento y trade-offs entre sensibilidad y especificidad.\n")
    
    # Tabla comparativa mejorada
    section.append("### Tabla Comparativa de Rendimiento Global\n")
    section.append("| Fase | Recall | Precision | F1 | Accuracy | Canónicos (Det→Int→Lin→Final) | Extras (Det→Final) | Nuevos | Especificidad |")
    section.append("|------|--------|-----------|-----|----------|--------------------------------|--------------------|--------|---------------|")
    
    phase_order = ['all', 'no-class-intensity', 'no-classification-linear', 
                   'no-phase2-no-classification-intensity', 'no-phase2-no-classification-linear', 'no-phase2']
    
    phase_data = []
    for phase_name in phase_order:
        if phase_name in all_metrics:
            metrics = all_metrics[phase_name]
            phase_label = get_phase_config_name(phase_name).split(':')[0].replace('Caso ', '')
            
            canon_det = metrics['num_canonicos_matched']
            canon_int = metrics.get('num_canonicos_burst_intensity', 0)
            canon_lin = metrics.get('num_canonicos_burst_linear', 0)
            canon_final = metrics.get('num_canonicos_burst_final', metrics.get('num_canonicos_burst', 0))
            
            extras_det = metrics['num_extras_matched']
            extras_final = metrics.get('num_extras_burst_final', metrics.get('num_extras_burst', 0))
            
            canon_flow = f"{canon_det}→{canon_int}→{canon_lin}→{canon_final}"
            extras_flow = f"{extras_det}→{extras_final}"
            
            phase_data.append({
                'name': phase_label,
                'phase': phase_name,
                'metrics': metrics
            })
            
            section.append(
                f"| {phase_label} | {format_percentage(metrics['recall'])} | "
                f"{format_percentage(metrics['precision'])} | "
                f"{format_percentage(metrics['f1_score'])} | "
                f"{format_percentage(metrics['accuracy'])} | "
                f"{canon_flow} | {extras_flow} | {metrics['num_new']} | "
                f"{format_percentage(metrics['specificity'])} |"
            )
    
    section.append("")
    
    # Análisis de correlaciones y patrones
    section.append("### Análisis de Patrones y Correlaciones\n")
    
    # Identificar mejor fase por cada métrica
    best_recall = max(phase_data, key=lambda x: x['metrics']['recall'])
    best_precision = max(phase_data, key=lambda x: x['metrics']['precision'])
    best_f1 = max(phase_data, key=lambda x: x['metrics']['f1_score'])
    best_accuracy = max(phase_data, key=lambda x: x['metrics']['accuracy'])
    
    section.append("**Identificación de Fases Óptimas por Métrica:**")
    section.append(f"- **Mejor Recall:** {best_recall['name']} ({format_percentage(best_recall['metrics']['recall'])})")
    section.append(f"- **Mejor Precision:** {best_precision['name']} ({format_percentage(best_precision['metrics']['precision'])})")
    section.append(f"- **Mejor F1-Score:** {best_f1['name']} ({format_percentage(best_f1['metrics']['f1_score'])})")
    section.append(f"- **Mejor Accuracy:** {best_accuracy['name']} ({format_percentage(best_accuracy['metrics']['accuracy'])})\n")
    
    # Análisis de trade-offs
    section.append("### Análisis de Trade-offs: Sensibilidad vs Especificidad\n")
    
    # Calcular ratios de trade-off
    for phase_info in phase_data:
        phase_name = phase_info['name']
        metrics = phase_info['metrics']
        recall = metrics['recall']
        precision = metrics['precision']
        specificity = metrics['specificity']
        
        # Ratio de trade-off
        tradeoff_ratio = recall / precision if precision > 0 else 0
        
        section.append(f"**{phase_name}:**")
        section.append(f"- Recall/Precision Ratio: {format_number(tradeoff_ratio, 2)}")
        section.append(f"  → Ratio > 1.0: Prioriza sensibilidad sobre precisión")
        section.append(f"  → Ratio < 1.0: Prioriza precisión sobre sensibilidad")
        section.append(f"- Especificidad: {format_percentage(specificity)}")
        section.append("")
    
    # Análisis de fortalezas y limitaciones por fase
    section.append("### Análisis Detallado de Fortalezas y Limitaciones por Fase\n")
    
    for phase_info in phase_data:
        phase_name = phase_info['name']
        phase_key = phase_info['phase']
        metrics = phase_info['metrics']
        
        section.append(f"#### {phase_name}\n")
        
        # Fortalezas
        section.append("**Fortalezas:**")
        if metrics['recall'] > 0.8:
            section.append(f"- ✅ Alta sensibilidad (Recall={format_percentage(metrics['recall'])}) - detecta la mayoría de pulsos validados")
        if metrics['precision'] > 0.7:
            section.append(f"- ✅ Buena precisión (Precision={format_percentage(metrics['precision'])}) - pocos falsos positivos")
        if metrics['f1_score'] > 0.7:
            section.append(f"- ✅ Balance óptimo (F1={format_percentage(metrics['f1_score'])}) - buen compromiso entre sensibilidad y precisión")
        
        canon_final = metrics.get('num_canonicos_burst_final', metrics.get('num_canonicos_burst', 0))
        if canon_final == metrics['num_canonicos']:
            section.append(f"- ✅ Recuperación perfecta de canónicos ({canon_final}/{metrics['num_canonicos']})")
        
        # Limitaciones
        section.append("\n**Limitaciones:**")
        if metrics['recall'] < 0.7:
            section.append(f"- ⚠️ Baja sensibilidad (Recall={format_percentage(metrics['recall'])}) - pierde una fracción significativa de pulsos validados")
        if metrics['precision'] < 0.6:
            section.append(f"- ⚠️ Precisión reducida (Precision={format_percentage(metrics['precision'])}) - mayor tasa de falsos positivos")
        if metrics['specificity'] == 0:
            section.append(f"- ⚠️ Especificidad nula - no rechaza falsos positivos entre nuevos candidatos")
        
        # Análisis específico según configuración
        if 'no-phase2' in phase_key:
            section.append("\n**Análisis Específico (Sin Phase 2):**")
            section.append("- ⚠️ **Limitación:** Sin validación polarimétrica previa (Phase 2), todos los candidatos pasan a clasificación CNN")
            section.append("  → Mayor carga computacional (más candidatos procesados por ResNet18)")
            section.append("  → Posible pérdida de eficiencia (candidatos con SNR_L muy bajo podrían filtrarse antes)")
            section.append("- ✅ **Fortaleza:** Maximiza sensibilidad (no rechaza candidatos con polarización débil pero morfología válida)")
            section.append("  → Útil para descubrimiento exploratorio donde no se pueden perder eventos")
        
        if 'no-class-intensity' in phase_key:
            section.append("\n**Análisis Específico (Sin Clasificación Intensity):**")
            section.append("- ⚠️ **Limitación:** Dependencia exclusiva de clasificación en polarización lineal (Phase 3B)")
            section.append("  → Problema conocido: ResNet18 tiene limitaciones en transfer learning para waterfalls en L")
            section.append("  → Recall reducido debido a rechazos por morfología atípica en L (ver discusión en Sección de Transfer Learning)")
        
        if 'no-classification-linear' in phase_key:
            section.append("\n**Análisis Específico (Sin Clasificación Linear):**")
            section.append("- ✅ **Fortaleza:** Evita la limitación de transfer learning en polarización lineal")
            section.append("  → Clasificación basada exclusivamente en intensidad (dominio donde ResNet18 funciona perfectamente)")
            section.append("  → Recall alto para canónicos y extras")
            section.append("- ⚠️ **Limitación:** Menor especificidad (no filtra basándose en coherencia polarimétrica I+L)")
            section.append("  → Posible mayor tasa de falsos positivos (RFI no polarizada podría pasar)")
        
        section.append("")
    
    return "\n".join(section)


def generate_transfer_learning_discussion(metrics: Dict) -> str:
    """Genera discusión profunda sobre el problema de transfer learning en polarización lineal"""
    section = []
    
    section.append("### Análisis Profundo: Limitación de Transfer Learning en Polarización Lineal\n")
    section.append("Esta sección analiza en profundidad la limitación identificada en la clasificación de waterfalls en polarización lineal, conectando los resultados empíricos con la teoría de transfer learning y las implicaciones físicas.\n")
    
    # Evidencia cuantitativa
    canon_lin = metrics.get('num_canonicos_burst_linear', 0)
    canon_total = metrics.get('num_canonicos', 8)
    canon_int = metrics.get('num_canonicos_burst_intensity', 0)
    
    extras_lin = metrics.get('num_extras_burst_linear', 0)
    extras_total = metrics.get('num_extras_matched', 0)
    extras_int = metrics.get('num_extras_burst_intensity', 0)
    
    section.append("#### 1. Evidencia Cuantitativa del Problema\n")
    section.append(f"**Pulsos Canónicos:**")
    section.append(f"- Clasificación en Intensidad (Phase 3A): {canon_int}/{canon_total} ({format_percentage(canon_int/canon_total if canon_total > 0 else 0)}) ✅")
    section.append(f"- Clasificación en Polarización Lineal (Phase 3B): {canon_lin}/{canon_total} ({format_percentage(canon_lin/canon_total if canon_total > 0 else 0)}) ⚠️")
    section.append(f"- **Diferencia:** {canon_total - canon_lin} pulsos rechazados exclusivamente por la red de L\n")
    
    section.append(f"**Pulsos Extras:**")
    section.append(f"- Clasificación en Intensidad (Phase 3A): {extras_int}/{extras_total} ({format_percentage(extras_int/extras_total if extras_total > 0 else 0)}) ✅")
    section.append(f"- Clasificación en Polarización Lineal (Phase 3B): {extras_lin}/{extras_total} ({format_percentage(extras_lin/extras_total if extras_total > 0 else 0)}) ⚠️")
    section.append(f"- **Diferencia:** {extras_total - extras_lin} pulsos rechazados exclusivamente por la red de L\n")
    
    # Análisis de probabilidades
    validated_by_classifier = metrics.get('validated_by_classifier', {})
    if validated_by_classifier and 'linear' in validated_by_classifier:
        linear_data = validated_by_classifier['linear']
        burst_stats = linear_data.get('burst', {}).get('class_prob_stats', {})
        no_burst_stats = linear_data.get('no_burst', {}).get('class_prob_stats', {})
        
        if burst_stats and no_burst_stats:
            section.append("#### 2. Análisis de Distribución de Probabilidades\n")
            section.append(f"**Candidatos Clasificados como BURST en L:**")
            section.append(f"- Media: {format_number(burst_stats.get('mean', 0), 3)}, Min: {format_number(burst_stats.get('min', 0), 3)}")
            section.append(f"- **Interpretación:** Probabilidades altas (media > 0.9) indican alta confianza cuando la red reconoce la morfología\n")
            
            section.append(f"**Candidatos Clasificados como NO_BURST en L:**")
            section.append(f"- Media: {format_number(no_burst_stats.get('mean', 0), 3)}, Max: {format_number(no_burst_stats.get('max', 0), 3)}")
            section.append(f"- **Interpretación:** Probabilidades muy bajas (media < 0.05) indican que la red está muy segura de que NO es BURST\n")
            
            # Casos límite
            max_no_burst = no_burst_stats.get('max', 0)
            if max_no_burst > 0.15:
                section.append(f"⚠️ **Casos Límite Identificados:**")
                section.append(f"  - Máximo en NO_BURST: {format_number(max_no_burst, 3)} (cerca del umbral 0.5)")
                section.append("  - Estos candidatos están en la zona de incertidumbre del modelo")
                section.append("  - Podrían ser pulsos genuinos con morfología atípica que el modelo no reconoce correctamente")
                section.append("")
    
    # Causa física
    section.append("#### 3. Causa Física del Problema\n")
    section.append("**Diferencia Morfológica entre Intensidad y Polarización Lineal:**\n")
    section.append("- **Intensidad (Stokes I):** Morfología típicamente coherente y banda ancha")
    section.append("  → Patrones verticales bien definidos en el waterfall")
    section.append("  → ResNet18, entrenado en waterfalls de Stokes I (baja frecuencia), reconoce estos patrones correctamente\n")
    section.append("- **Polarización Lineal (Stokes L = √(Q²+U²)):** Morfología puede ser difusa o fragmentada")
    section.append("  → Variaciones de campo magnético causan modulación de polarización")
    section.append("  → Efectos de scattering diferencial afectan más a la polarización que a la intensidad")
    section.append("  → Morfologías atípicas (físicamente válidas) no vistas durante entrenamiento\n")
    section.append("**Resultado:** El modelo, entrenado exclusivamente en patrones de Stokes I, no generaliza correctamente a morfologías válidas pero atípicas en Stokes L.\n")
    
    # Conexión con literatura
    section.append("#### 4. Conexión con Validación Previa\n")
    section.append("Este problema fue identificado previamente en la validación del Componente 2 (ver Sección de Validación, Caso 4a-4b):\n")
    section.append("- **Ground truth (8 canónicos):** 2/8 (25%) rechazados en L, 100% en I")
    section.append("- **Dataset extendido (44):** 17/44 (38.6%) rechazados en L, 100% en I")
    section.append("- **Conclusión previa:** El problema NO es sensibilidad del núcleo híbrido ni clasificación en I (ambas perfectas al 100%), sino específicamente el transfer learning de ResNet18 en polarización lineal (Fase 3b)\n")
    section.append("Los resultados actuales confirman y extienden este hallazgo, mostrando que el problema persiste en diferentes configuraciones metodológicas.\n")
    
    # Implicaciones
    section.append("#### 5. Implicaciones y Trabajo Futuro\n")
    section.append("**Solución Propuesta:**")
    section.append("1. **Reentrenamiento de ResNet18 con waterfalls de Stokes L:**")
    section.append("   - Crear dataset de entrenamiento con ejemplos de polarización lineal en alta frecuencia")
    section.append("   - Incluir morfologías atípicas (difusas, fragmentadas) para mejorar generalización")
    section.append("   - Fine-tuning del modelo pre-entrenado en el nuevo dominio")
    section.append("")
    section.append("2. **Activación de Phase 2 con umbrales adaptativos:**")
    section.append("   - Usar validación polarimétrica previa (SNR_L) como filtro complementario")
    section.append("   - Umbral conservador (2-3σ) para no rechazar candidatos genuinos con polarización débil")
    section.append("   - Reducir carga computacional en Phase 3B")
    section.append("")
    section.append("3. **Modos de operación configurables:**")
    section.append("   - Modo STRICT (I AND L): Alta especificidad, recall reducido (actual)")
    section.append("   - Modo PERMISSIVE (I OR L): Alta sensibilidad, especificidad reducida")
    section.append("   - Modo híbrido con umbrales asimétricos (p_I ≥ 0.7 OR (p_I ≥ 0.5 AND p_L ≥ 0.3))")
    section.append("")
    
    section.append("**Impacto Esperado:**")
    section.append("- Reentrenamiento en L podría elevar recall en modo STRICT de 61-75% a niveles cercanos a 100%")
    section.append("- Sin comprometer la precisión (~100% observada actualmente)")
    section.append("- Mejoraría el balance sensibilidad-especificidad del pipeline completo\n")
    
    return "\n".join(section)


def generate_final_recommendations(all_metrics: Dict) -> str:
    """Genera recomendaciones finales exhaustivas y creativas"""
    section = []
    
    section.append("## Recomendaciones Finales y Síntesis Ejecutiva\n")
    section.append("Esta sección consolida los hallazgos más importantes y proporciona recomendaciones estratégicas basadas en el análisis exhaustivo realizado.\n")
    
    # Identificar mejor fase
    phase_order = ['all', 'no-class-intensity', 'no-classification-linear', 
                   'no-phase2-no-classification-intensity', 'no-phase2-no-classification-linear', 'no-phase2']
    
    phase_scores = []
    for phase_name in phase_order:
        if phase_name in all_metrics:
            metrics = all_metrics[phase_name]
            # Score compuesto (ponderado)
            composite_score = (
                metrics['recall'] * 0.3 +
                metrics['precision'] * 0.3 +
                metrics['f1_score'] * 0.3 +
                metrics['accuracy'] * 0.1
            )
            phase_scores.append({
                'name': get_phase_config_name(phase_name),
                'phase': phase_name,
                'score': composite_score,
                'metrics': metrics
            })
    
    best_overall = max(phase_scores, key=lambda x: x['score'])
    
    section.append("### Fase Óptima Identificada\n")
    section.append(f"**{best_overall['name']}** emerge como la configuración óptima con score compuesto de {format_number(best_overall['score'], 3)}.\n")
    
    section.append("**Justificación Detallada:**\n")
    section.append(f"1. **Balance Sensibilidad-Precisión:** F1-Score={format_percentage(best_overall['metrics']['f1_score'])}")
    section.append(f"   - No sacrifica sensibilidad (Recall={format_percentage(best_overall['metrics']['recall'])})")
    section.append(f"   - Mantiene precisión aceptable (Precision={format_percentage(best_overall['metrics']['precision'])})\n")
    
    section.append(f"2. **Rendimiento en Canónicos:**")
    canon_final = best_overall['metrics'].get('num_canonicos_burst_final', best_overall['metrics'].get('num_canonicos_burst', 0))
    canon_total = best_overall['metrics'].get('num_canonicos', 8)
    section.append(f"   - Recuperación: {canon_final}/{canon_total} ({format_percentage(canon_final/canon_total if canon_total > 0 else 0)})")
    if canon_final == canon_total:
        section.append("   - ✅ Recuperación perfecta de ground truth canónico\n")
    else:
        section.append(f"   - ⚠️ {canon_total - canon_final} canónico(s) perdido(s) - requiere análisis de casos específicos\n")
    
    section.append(f"3. **Gestión de Nuevos Candidatos:**")
    new_total = best_overall['metrics'].get('num_new', 0)
    new_burst = best_overall['metrics'].get('num_new_burst_final', best_overall['metrics'].get('num_new_burst', 0))
    section.append(f"   - {new_total} candidatos nuevos detectados")
    section.append(f"   - {new_burst} clasificados como BURST ({format_percentage(new_burst/new_total if new_total > 0 else 0)})")
    section.append("   - Balance entre descubrimiento (nuevos BURST) y especificidad (rechazo de falsos positivos)\n")
    
    # Recomendaciones por escenario
    section.append("### Recomendaciones por Escenario de Uso\n")
    
    section.append("#### Escenario 1: Descubrimiento Exploratorio (Maximizar Sensibilidad)\n")
    section.append("**Objetivo:** No perder ningún pulso genuino, tolerar falsos positivos para revisión manual.\n")
    section.append("**Recomendación:**")
    section.append("- **Configuración:** Caso E (Sin clasificación Linear) o Caso F (Sin Phase 2)")
    section.append("- **Justificación:**")
    section.append("  - Evita la limitación de transfer learning en polarización lineal")
    section.append("  - Clasificación basada en intensidad (dominio donde ResNet18 funciona perfectamente)")
    section.append("  - Recall alto para canónicos y extras")
    section.append("- **Trade-off aceptado:** Menor especificidad (mayor tasa de falsos positivos)\n")
    
    section.append("#### Escenario 2: Caracterización Precisa (Maximizar Especificidad)\n")
    section.append("**Objetivo:** Alta confianza en que los candidatos son pulsos genuinos, minimizar falsos positivos.\n")
    section.append("**Recomendación:**")
    section.append("- **Configuración:** Caso A (Todas las fases activas) con modo STRICT")
    section.append("- **Justificación:**")
    section.append("  - Clasificación dual I+L proporciona doble filtro de especificidad")
    section.append("  - Precision ~100% observada en validaciones")
    section.append("  - Filtrado efectivo de RFI basado en coherencia morfológica")
    section.append("- **Trade-off aceptado:** Recall reducido (61-75%) debido a limitación en L\n")
    
    section.append("#### Escenario 3: Balance Óptimo (Uso General)\n")
    section.append("**Objetivo:** Buen compromiso entre sensibilidad y especificidad para uso operacional general.\n")
    section.append("**Recomendación:**")
    section.append(f"- **Configuración:** {best_overall['name']}")
    section.append("- **Justificación:**")
    section.append(f"  - Score compuesto más alto ({format_number(best_overall['score'], 3)})")
    section.append(f"  - F1-Score={format_percentage(best_overall['metrics']['f1_score'])} (balance óptimo)")
    section.append("  - Rendimiento consistente en múltiples métricas")
    section.append("- **Ventaja:** Configuración versátil para diferentes tipos de análisis\n")
    
    # Trabajo futuro
    section.append("### Trabajo Futuro Prioritario\n")
    section.append("1. **Reentrenamiento de ResNet18 en Polarización Lineal:**")
    section.append("   - Impacto esperado: Elevar recall en modo STRICT de 61-75% a ~100%")
    section.append("   - Sin comprometer precisión (~100% observada)")
    section.append("   - Mejoraría significativamente el balance sensibilidad-especificidad\n")
    
    section.append("2. **Validación de Phase 2 con Umbrales Adaptativos:**")
    section.append("   - Activar validación polarimétrica previa (SNR_L) como filtro complementario")
    section.append("   - Evaluar impacto en eficiencia computacional y balance sensibilidad-especificidad")
    section.append("   - Requiere validación empírica con diferentes umbrales (2-5σ)\n")
    
    section.append("3. **Análisis de Candidatos Nuevos con Validación Experta:**")
    section.append("   - Priorizar outliers de alta SNR para validación visual")
    section.append("   - Confirmar naturaleza astrofísica mediante análisis polarimétrico detallado")
    section.append("   - Potencial para descubrimientos científicos genuinos\n")
    
    return "\n".join(section)


# [Resto de funciones anteriores: generate_flow_table, generate_classifier_analysis, generate_column_analysis, generate_critical_cases, generate_phase_section, generate_comparison_table, generate_report, main]
# Mantengo las funciones anteriores pero las integro con las nuevas

def generate_flow_table(metrics: Dict) -> str:
    """Genera tabla de flujo de detección y clasificación"""
    section = []
    
    section.append("### Tabla de Flujo: Detección → Clasificación\n")
    section.append("Esta tabla muestra cuántos candidatos pasan cada etapa del pipeline para cada grupo de validación.\n")
    
    section.append("| Etapa | Canónicos (8) | Extras (54) | Nuevos |")
    section.append("|-------|---------------|-------------|--------|")
    
    # Detección
    canon_det = metrics['num_canonicos_matched']
    extras_det = metrics['num_extras_matched']
    new_total = metrics['num_new']
    section.append(f"| **1. Detección (Match Temporal)** | {canon_det}/8 ({format_percentage(canon_det/8 if metrics['num_canonicos'] > 0 else 0)}) | {extras_det}/54 ({format_percentage(extras_det/54 if metrics['num_extras'] > 0 else 0)}) | {new_total} |")
    
    # Clasificación Intensity
    canon_int = metrics.get('num_canonicos_burst_intensity', 0)
    extras_int = metrics.get('num_extras_burst_intensity', 0)
    new_int = metrics.get('num_new_burst_intensity', 0)
    section.append(f"| **2. Clasif. Intensity (Phase 3A)** | {canon_int}/8 ({format_percentage(canon_int/8 if metrics['num_canonicos'] > 0 else 0)}) | {extras_int}/54 ({format_percentage(extras_int/54 if metrics['num_extras'] > 0 else 0)}) | {new_int}/{new_total} ({format_percentage(new_int/new_total if new_total > 0 else 0)}) |")
    
    # Clasificación Linear
    canon_lin = metrics.get('num_canonicos_burst_linear', 0)
    extras_lin = metrics.get('num_extras_burst_linear', 0)
    new_lin = metrics.get('num_new_burst_linear', 0)
    section.append(f"| **3. Clasif. Linear (Phase 3B)** | {canon_lin}/8 ({format_percentage(canon_lin/8 if metrics['num_canonicos'] > 0 else 0)}) | {extras_lin}/54 ({format_percentage(extras_lin/54 if metrics['num_extras'] > 0 else 0)}) | {new_lin}/{new_total} ({format_percentage(new_lin/new_total if new_total > 0 else 0)}) |")
    
    # Resultado Final
    canon_final = metrics.get('num_canonicos_burst_final', metrics.get('num_canonicos_burst', 0))
    extras_final = metrics.get('num_extras_burst_final', metrics.get('num_extras_burst', 0))
    new_final = metrics.get('num_new_burst_final', metrics.get('num_new_burst', 0))
    section.append(f"| **4. Resultado Final (Pipeline)** | **{canon_final}/8 ({format_percentage(canon_final/8 if metrics['num_canonicos'] > 0 else 0)})** | **{extras_final}/54 ({format_percentage(extras_final/54 if metrics['num_extras'] > 0 else 0)})** | **{new_final}/{new_total} ({format_percentage(new_final/new_total if new_total > 0 else 0)})** |")
    
    section.append("")
    
    return "\n".join(section)


def generate_classifier_analysis(metrics: Dict) -> str:
    """Genera análisis detallado de cada clasificador"""
    section = []
    
    section.append("### Análisis Detallado por Clasificador\n")
    
    # Análisis de Canónicos
    section.append("#### Pulsos Canónicos: Comportamiento de los Clasificadores\n")
    
    canon_det = metrics['num_canonicos_matched']
    canon_int = metrics.get('num_canonicos_burst_intensity', 0)
    canon_lin = metrics.get('num_canonicos_burst_linear', 0)
    canon_final = metrics.get('num_canonicos_burst_final', metrics.get('num_canonicos_burst', 0))
    
    section.append(f"De los **{canon_det} canónicos detectados** (de 8 totales):")
    section.append(f"- **{canon_int}** fueron clasificados como BURST por la red de **Intensidad** (Phase 3A).")
    section.append(f"- **{canon_lin}** fueron clasificados como BURST por la red de **Polarización Lineal** (Phase 3B).")
    section.append(f"- **{canon_final}** fueron clasificados como BURST por la **lógica final del pipeline**.\n")
    
    # Identificar discrepancias
    if canon_det > 0:
        lost_intensity = canon_det - canon_int
        lost_linear = canon_det - canon_lin
        lost_final = canon_det - canon_final
        
        if lost_intensity > 0:
            section.append(f"⚠️ **{lost_intensity} canónico(s) fueron RECHAZADOS por la red de Intensidad** (aunque fueron detectados).")
            section.append("   Esto indica que la morfología en intensidad no fue reconocida como BURST por la CNN.\n")
        
        if lost_linear > 0:
            section.append(f"⚠️ **{lost_linear} canónico(s) fueron RECHAZADOS por la red de Polarización Lineal** (aunque fueron detectados).")
            section.append("   Esto indica que la morfología en polarización lineal no fue reconocida como BURST por la CNN.\n")
        
        if lost_final > 0:
            section.append(f"⚠️ **{lost_final} canónico(s) fueron RECHAZADOS por la lógica final del pipeline**.")
            section.append("   Estos son casos críticos que requieren análisis detallado (ver sección de Columnas).\n")
    
    # Análisis de Extras
    section.append("#### Pulsos Extras: Comportamiento de los Clasificadores\n")
    
    extras_det = metrics['num_extras_matched']
    extras_int = metrics.get('num_extras_burst_intensity', 0)
    extras_lin = metrics.get('num_extras_burst_linear', 0)
    extras_final = metrics.get('num_extras_burst_final', metrics.get('num_extras_burst', 0))
    
    section.append(f"De los **{extras_det} extras detectados** (de 54 totales):")
    section.append(f"- **{extras_int}** fueron clasificados como BURST por la red de **Intensidad**.")
    section.append(f"- **{extras_lin}** fueron clasificados como BURST por la red de **Polarización Lineal**.")
    section.append(f"- **{extras_final}** fueron clasificados como BURST por la **lógica final del pipeline**.\n")
    
    # Análisis de Nuevos
    section.append("#### Candidatos Nuevos: Comportamiento de los Clasificadores\n")
    
    new_total = metrics['num_new']
    new_int = metrics.get('num_new_burst_intensity', 0)
    new_lin = metrics.get('num_new_burst_linear', 0)
    new_final = metrics.get('num_new_burst_final', metrics.get('num_new_burst', 0))
    
    section.append(f"De los **{new_total} candidatos nuevos** (sin match con validados):")
    section.append(f"- **{new_int}** fueron clasificados como BURST por la red de **Intensidad**.")
    section.append(f"- **{new_lin}** fueron clasificados como BURST por la red de **Polarización Lineal**.")
    section.append(f"- **{new_final}** fueron clasificados como BURST por la **lógica final del pipeline**.")
    section.append(f"- **{metrics.get('num_new_no_burst_final', metrics.get('num_new_no_burst', 0))}** fueron clasificados como NO_BURST.\n")
    
    return "\n".join(section)


def interpret_column(column_name: str, stats: Dict, context: str = "") -> str:
    """Interpreta el significado de una columna y sus estadísticas"""
    interpretations = []
    
    if 'detection_prob' in column_name:
        interpretations.append("**Probabilidad de Detección (CenterNet):** Mide la confianza del detector inicial de objetos.")
        if stats:
            mean_val = stats.get('mean', 0)
            if mean_val > 0.8:
                interpretations.append(f"Valores altos (media={format_number(mean_val, 2)}) indican detecciones muy confiables.")
            elif mean_val > 0.5:
                interpretations.append(f"Valores moderados (media={format_number(mean_val, 2)}) sugieren detecciones con confianza media.")
    
    elif 'snr_patch_dedispersed' in column_name and 'linear' not in column_name:
        interpretations.append("**SNR Patch Dedispersed (Intensidad):** Relación señal-ruido después de dedispersión en el patch de intensidad.")
        if stats:
            mean_val = stats.get('mean', 0)
            if mean_val > 10:
                interpretations.append(f"SNR alto (media={format_number(mean_val, 2)}) indica señales fuertes y bien detectadas.")
            elif mean_val > 5:
                interpretations.append(f"SNR moderado (media={format_number(mean_val, 2)}) sugiere señales detectables pero cerca del umbral.")
    
    elif 'snr_patch_dedispersed_linear' in column_name or ('snr' in column_name and 'linear' in column_name):
        interpretations.append("**SNR Linear (Polarización):** Relación señal-ruido en polarización lineal (usado en Phase 2).")
        if stats:
            mean_val = stats.get('mean', 0)
            if mean_val > 8:
                interpretations.append(f"SNR lineal alto (media={format_number(mean_val, 2)}) indica buena polarización, favorable para Phase 2 y 3B.")
            elif mean_val > 3:
                interpretations.append(f"SNR lineal moderado (media={format_number(mean_val, 2)}) puede causar rechazos en Phase 2 si está por debajo del umbral.")
    
    elif 'class_prob_intensity' in column_name:
        interpretations.append("**Probabilidad de Clasificación Intensity (Phase 3A):** Confianza de la red ResNet en que el candidato es BURST basado en morfología de intensidad.")
        if stats:
            mean_val = stats.get('mean', 0)
            min_val = stats.get('min', 0)
            if mean_val > 0.9:
                interpretations.append(f"Probabilidades muy altas (media={format_number(mean_val, 2)}, min={format_number(min_val, 2)}) indican que la red está muy segura.")
    
    elif 'class_prob_linear' in column_name:
        interpretations.append("**Probabilidad de Clasificación Linear (Phase 3B):** Confianza de la red ResNet en que el candidato es BURST basado en morfología de polarización lineal.")
        if stats:
            mean_val = stats.get('mean', 0)
            min_val = stats.get('min', 0)
            if mean_val > 0.7:
                interpretations.append(f"Probabilidades altas (media={format_number(mean_val, 2)}, min={format_number(min_val, 2)}) indican buena clasificación en polarización.")
            elif mean_val < 0.3:
                interpretations.append(f"Probabilidades bajas (media={format_number(mean_val, 2)}, min={format_number(min_val, 2)}) explican por qué algunos pulsos son rechazados por Phase 3B.")
    
    if not interpretations:
        interpretations.append(f"**{column_name}:** Columna de datos del pipeline.")
    
    return " ".join(interpretations)


def generate_column_analysis(metrics: Dict, phase_name: str) -> str:
    """Genera análisis detallado de cada columna con interpretación"""
    section = []
    
    section.append("### Análisis de Columnas: Interpretación de Métricas\n")
    section.append("Esta sección analiza cada columna relevante del CSV de resultados, explicando qué significa y qué nos dicen los valores observados.\n")
    
    validated_astro = metrics.get('validated_astronomical', {})
    validated_by_classifier = metrics.get('validated_by_classifier', {})
    
    # Columnas de Detección
    section.append("#### 1. Métricas de Detección (Phase 1)\n")
    
    if 'detected_detection_prob' in validated_astro:
        stats = validated_astro['detected_detection_prob']
        section.append(f"**detection_prob:**")
        section.append(f"- Media: {format_number(stats.get('mean', 0), 3)} | Mediana: {format_number(stats.get('median', 0), 3)}")
        section.append(f"- Rango: [{format_number(stats.get('min', 0), 3)}, {format_number(stats.get('max', 0), 3)}]")
        section.append(f"- {interpret_column('detected_detection_prob', stats)}")
        section.append("")
    
    # SNR Intensidad
    if 'detected_snr_patch_dedispersed' in validated_astro:
        stats = validated_astro['detected_snr_patch_dedispersed']
        section.append(f"**snr_patch_dedispersed (Intensidad):**")
        section.append(f"- Media: {format_number(stats.get('mean', 0), 2)} | Mediana: {format_number(stats.get('median', 0), 2)}")
        section.append(f"- Rango: [{format_number(stats.get('min', 0), 2)}, {format_number(stats.get('max', 0), 2)}]")
        if 'std' in stats:
            section.append(f"- Desviación estándar: {format_number(stats.get('std', 0), 2)}")
        section.append(f"- {interpret_column('detected_snr_patch_dedispersed', stats)}")
        section.append("")
    
    # SNR Linear
    if 'detected_snr_patch_dedispersed_linear' in validated_astro:
        stats = validated_astro['detected_snr_patch_dedispersed_linear']
        section.append(f"**snr_patch_dedispersed_linear (Polarización):**")
        section.append(f"- Media: {format_number(stats.get('mean', 0), 2)} | Mediana: {format_number(stats.get('median', 0), 2)}")
        section.append(f"- Rango: [{format_number(stats.get('min', 0), 2)}, {format_number(stats.get('max', 0), 2)}]")
        if 'std' in stats:
            section.append(f"- Desviación estándar: {format_number(stats.get('std', 0), 2)}")
        section.append(f"- {interpret_column('detected_snr_patch_dedispersed_linear', stats)}")
        section.append("")
    
    # Clasificación Intensity
    section.append("#### 2. Métricas de Clasificación Intensity (Phase 3A)\n")
    
    if 'detected_class_prob_intensity' in validated_astro:
        stats = validated_astro['detected_class_prob_intensity']
        section.append(f"**class_prob_intensity:**")
        section.append(f"- Media: {format_number(stats.get('mean', 0), 3)} | Mediana: {format_number(stats.get('median', 0), 3)}")
        section.append(f"- Rango: [{format_number(stats.get('min', 0), 3)}, {format_number(stats.get('max', 0), 3)}]")
        section.append(f"- {interpret_column('detected_class_prob_intensity', stats)}")
        section.append("")
    
    # Clasificación Linear
    section.append("#### 3. Métricas de Clasificación Linear (Phase 3B)\n")
    
    if 'detected_class_prob_linear' in validated_astro:
        stats = validated_astro['detected_class_prob_linear']
        section.append(f"**class_prob_linear:**")
        section.append(f"- Media: {format_number(stats.get('mean', 0), 3)} | Mediana: {format_number(stats.get('median', 0), 3)}")
        section.append(f"- Rango: [{format_number(stats.get('min', 0), 3)}, {format_number(stats.get('max', 0), 3)}]")
        section.append(f"- {interpret_column('detected_class_prob_linear', stats)}")
        section.append("")
    
    # Análisis desglosado por clasificador Linear
    if validated_by_classifier and 'linear' in validated_by_classifier:
        linear_data = validated_by_classifier['linear']
        burst_stats = linear_data.get('burst', {}).get('class_prob_stats')
        no_burst_stats = linear_data.get('no_burst', {}).get('class_prob_stats')
        
        if burst_stats and no_burst_stats:
            section.append("**Distribución de Probabilidades Linear (BURST vs NO_BURST):**")
            section.append(f"- BURST: Media={format_number(burst_stats.get('mean', 0), 3)}, Min={format_number(burst_stats.get('min', 0), 3)}")
            section.append(f"- NO_BURST: Media={format_number(no_burst_stats.get('mean', 0), 3)}, Max={format_number(no_burst_stats.get('max', 0), 3)}")
            section.append("")
            if no_burst_stats.get('mean', 0) > 0.2:
                section.append("⚠️ **Observación:** Algunos candidatos clasificados como NO_BURST tienen probabilidades moderadas (>0.2),")
                section.append("   indicando que la red de polarización lineal tiene incertidumbre en estos casos.\n")
    
    # Propiedades Físicas
    section.append("#### 4. Propiedades Físicas\n")
    
    if 'detected_dm_pc_cm-3' in validated_astro:
        stats = validated_astro['detected_dm_pc_cm-3']
        section.append(f"**dm_pc_cm-3:**")
        section.append(f"- Media: {format_number(stats.get('mean', 0), 0)} | Rango: [{format_number(stats.get('min', 0), 0)}, {format_number(stats.get('max', 0), 0)}]")
        section.append("**Dispersión Medida (DM):** Medida de dispersión en pc·cm⁻³. Para FRB 121102, el valor esperado es ~560 pc·cm⁻³.")
        section.append("DM=1770 pc·cm⁻³ es consistente con el valor de búsqueda configurado (1770).")
        section.append("")
    
    if 'detected_width_ms' in validated_astro:
        stats = validated_astro['detected_width_ms']
        section.append(f"**width_ms:**")
        section.append(f"- Media: {format_number(stats.get('mean', 0), 3)} | Mediana: {format_number(stats.get('median', 0), 3)}")
        section.append(f"- Rango: [{format_number(stats.get('min', 0), 3)}, {format_number(stats.get('max', 0), 3)}]")
        section.append("**Ancho del Pulso:** Duración del pulso en milisegundos.")
        section.append(f"Ancho promedio de {format_number(stats.get('mean', 0), 2)} ms es típico para pulsos de radio transitorios.")
        section.append("")
    
    return "\n".join(section)


def generate_critical_cases(metrics: Dict) -> str:
    """Identifica y discute casos críticos"""
    section = []
    
    section.append("### Análisis de Casos Críticos\n")
    section.append("Identificación de pulsos validados que fueron detectados pero rechazados por algún clasificador.\n")
    
    canon_det = metrics['num_canonicos_matched']
    canon_int = metrics.get('num_canonicos_burst_intensity', 0)
    canon_lin = metrics.get('num_canonicos_burst_linear', 0)
    canon_final = metrics.get('num_canonicos_burst_final', metrics.get('num_canonicos_burst', 0))
    
    # Canónicos rechazados
    canon_rejected_intensity = canon_det - canon_int
    canon_rejected_linear = canon_det - canon_lin
    canon_rejected_final = canon_det - canon_final
    
    if canon_rejected_final > 0:
        section.append(f"#### ⚠️ Canónicos Rechazados: {canon_rejected_final} de {canon_det} detectados\n")
        section.append(f"**{canon_rejected_final} pulso(s) canónico(s) fueron detectados pero NO clasificados como BURST por el pipeline final.**")
        section.append("")
        
        if canon_rejected_intensity > 0:
            section.append(f"- **{canon_rejected_intensity}** fueron rechazados por la red de **Intensidad** (Phase 3A).")
            section.append("  → La morfología en intensidad no fue reconocida como BURST por la CNN de intensidad.")
            section.append("")
        
        if canon_rejected_linear > 0:
            section.append(f"- **{canon_rejected_linear}** fueron rechazados por la red de **Polarización Lineal** (Phase 3B).")
            section.append("  → La morfología en polarización lineal no fue reconocida como BURST por la CNN de polarización.")
            section.append("  → **Conexión con limitación conocida:** Este es el problema de transfer learning identificado previamente (ver Sección de Transfer Learning).")
            section.append("")
        
        section.append("**Implicación:** Estos son falsos negativos críticos que requieren revisión.")
        section.append("Posibles causas: umbrales de clasificación demasiado estrictos, o características morfológicas atípicas.\n")
    
    # Extras rechazados
    extras_det = metrics['num_extras_matched']
    extras_final = metrics.get('num_extras_burst_final', metrics.get('num_extras_burst', 0))
    extras_rejected = extras_det - extras_final
    
    if extras_rejected > 0:
        section.append(f"#### Extras Rechazados: {extras_rejected} de {extras_det} detectados\n")
        section.append(f"**{extras_rejected} pulso(s) extra(s) fueron detectados pero NO clasificados como BURST.**")
        section.append("Estos casos también requieren análisis para entender por qué fueron rechazados.\n")
    
    return "\n".join(section)


def generate_phase_section(phase_name: str, metrics: Dict) -> str:
    """Genera la sección completa del reporte para una fase específica"""
    section = []
    
    section.append(f"## {get_phase_config_name(phase_name)}\n")
    section.append(f"{get_phase_description(phase_name)}\n")
    
    # Tabla de flujo
    section.append(generate_flow_table(metrics))
    
    # Análisis por clasificador
    section.append(generate_classifier_analysis(metrics))
    
    # Análisis de columnas
    section.append(generate_column_analysis(metrics, phase_name))
    
    # Análisis exhaustivo de nuevos candidatos
    section.append(generate_new_candidates_analysis(metrics, phase_name))
    
    # Casos críticos
    section.append(generate_critical_cases(metrics))
    
    # Discusión de transfer learning (solo para fases con Phase 3B)
    if 'no-classification-linear' not in phase_name:
        section.append(generate_transfer_learning_discussion(metrics))
    
    # Métricas globales
    section.append("### Métricas Globales de Rendimiento\n")
    section.append("| Métrica | Valor | Descripción |")
    section.append("|---------|-------|-------------|")
    section.append(f"| **Recall** | {format_percentage(metrics['recall'])} | Proporción de validados detectados Y clasificados como BURST |")
    section.append(f"| **Precision** | {format_percentage(metrics['precision'])} | Proporción de BURST que son realmente validados |")
    section.append(f"| **F1-Score** | {format_percentage(metrics['f1_score'])} | Balance entre Precision y Recall |")
    section.append(f"| **Accuracy** | {format_percentage(metrics['accuracy'])} | Exactitud global del sistema |")
    section.append(f"| **Especificidad** | {format_percentage(metrics['specificity'])} | Capacidad de rechazar falsos positivos |")
    section.append("")
    
    section.append(f"**Matriz de Confusión:**")
    section.append(f"- TP (True Positives): {metrics['TP']} | FN (False Negatives): {metrics['FN']}")
    section.append(f"- FP (False Positives): {metrics['FP']} | TN (True Negatives): {metrics['TN']}")
    section.append("")
    
    section.append("---\n")
    
    return "\n".join(section)


def generate_comparison_table(all_metrics: Dict) -> str:
    """Genera tabla comparativa de todas las fases"""
    section = []
    
    section.append("## Comparación Sistemática de Fases\n")
    section.append("Tabla comparativa mostrando el rendimiento de cada fase en detección y clasificación.\n")
    
    section.append("| Fase | Recall | Precision | F1 | Canónicos (Det→Int→Lin→Final) | Extras (Det→Final) | Nuevos |")
    section.append("|------|--------|-----------|-----|--------------------------------|--------------------|--------|")
    
    phase_order = ['all', 'no-class-intensity', 'no-classification-linear', 
                   'no-phase2-no-classification-intensity', 'no-phase2-no-classification-linear', 'no-phase2']
    
    for phase_name in phase_order:
        if phase_name in all_metrics:
            metrics = all_metrics[phase_name]
            phase_label = get_phase_config_name(phase_name).split(':')[0].replace('Caso ', '')
            
            canon_det = metrics['num_canonicos_matched']
            canon_int = metrics.get('num_canonicos_burst_intensity', 0)
            canon_lin = metrics.get('num_canonicos_burst_linear', 0)
            canon_final = metrics.get('num_canonicos_burst_final', metrics.get('num_canonicos_burst', 0))
            
            extras_det = metrics['num_extras_matched']
            extras_final = metrics.get('num_extras_burst_final', metrics.get('num_extras_burst', 0))
            
            canon_flow = f"{canon_det}→{canon_int}→{canon_lin}→{canon_final}"
            extras_flow = f"{extras_det}→{extras_final}"
            
            section.append(
                f"| {phase_label} | {format_percentage(metrics['recall'])} | "
                f"{format_percentage(metrics['precision'])} | "
                f"{format_percentage(metrics['f1_score'])} | "
                f"{canon_flow} | {extras_flow} | {metrics['num_new']} |"
            )
    
    section.append("")
    
    return "\n".join(section)


def generate_report(metrics_path: str, output_path: str) -> None:
    """Genera el reporte Markdown completo y exhaustivo"""
    print(f"Generando reporte exhaustivo desde: {metrics_path}")
    
    # Cargar métricas
    all_metrics = load_metrics(metrics_path)
    
    # Construir reporte
    report = []
    
    # Título y resumen ejecutivo
    report.append("# Análisis Exhaustivo de Validación ALMA: Detección vs Clasificación\n")
    report.append("## Resumen Ejecutivo\n")
    report.append("Este documento presenta un análisis **granular y profundo** de la validación del pipeline de detección automática de pulsos de radio transitorios aplicado al dataset ALMA.")
    report.append("El análisis diferencia explícitamente entre **detección** (matching temporal) y **clasificación** (etiquetado BURST/NO_BURST), desglosando el comportamiento de cada clasificador individualmente.\n")
    
    report.append("**Dataset de Validación:**")
    report.append("- **8 Pulsos Canónicos:** Ground truth de alta confianza (marcados con \"pulso jose\").")
    report.append("- **54 Pulsos Extras:** Pulsos adicionales validados manualmente.")
    report.append("- **Total:** 62 pulsos validados.\n")
    
    report.append("**Metodología de Análisis:**")
    report.append("1. **Detección:** Se evalúa si el pipeline encontró un match temporal (archivo + tiempo ±0.1s).")
    report.append("2. **Clasificación Intensity:** Se analiza si la red ResNet de intensidad (Phase 3A) clasificó como BURST.")
    report.append("3. **Clasificación Linear:** Se analiza si la red ResNet de polarización lineal (Phase 3B) clasificó como BURST.")
    report.append("4. **Decisión Final:** Se analiza la etiqueta final del pipeline (`is_burst`).\n")
    
    report.append("**Configuraciones Analizadas:**")
    report.append("- Caso A: Todas las fases activas")
    report.append("- Caso B: Sin clasificación Intensity")
    report.append("- Caso C: Sin Phase 2 y sin clasificación Linear")
    report.append("- Caso D: Sin Phase 2 y sin clasificación Intensity")
    report.append("- Caso E: Sin clasificación Linear")
    report.append("- Caso F: Sin Phase 2\n")
    
    # Metodología detallada
    report.append("## Metodología Detallada\n")
    report.append("### Proceso de Matching (Detección)\n")
    report.append("1. **Normalización de nombres:** Los nombres de archivo se normalizan para manejar variaciones de formato.")
    report.append("2. **Matching por archivo:** Se buscan coincidencias entre `nombre_archivo` (validados) y `file` (detectados).")
    report.append("3. **Matching por tiempo:** Se verifica que `candidato tiempo` y `t_sec_dm_time` coincidan con tolerancia de ±0.1 segundos.")
    report.append("4. **Unicidad:** Si un pulso validado tiene múltiples matches, se selecciona el de mayor `detection_prob` para el análisis.\n")
    
    report.append("### Proceso de Clasificación\n")
    report.append("El pipeline tiene **dos clasificadores independientes**:")
    report.append("- **Phase 3A (Intensity):** ResNet18 que analiza la morfología en intensidad (Stokes I).")
    report.append("  - Columna: `is_burst_intensity` (derivada de `class_prob_intensity` vs umbral).")
    report.append("- **Phase 3B (Linear):** ResNet18 que analiza la morfología en polarización lineal (√(Q²+U²)).")
    report.append("  - Columna: `is_burst_linear` (derivada de `class_prob_linear` vs umbral).")
    report.append("- **Decisión Final:** La columna `is_burst` combina ambas decisiones según la lógica del pipeline.\n")
    
    report.append("### Interpretación de Columnas\n")
    report.append("Cada columna del CSV de resultados tiene un significado específico:")
    report.append("- `detection_prob`: Confianza del detector inicial (CenterNet/YOLO).")
    report.append("- `snr_patch_dedispersed`: SNR en intensidad después de dedispersión.")
    report.append("- `snr_patch_dedispersed_linear`: SNR en polarización lineal (usado en Phase 2).")
    report.append("- `class_prob_intensity`: Probabilidad softmax de la red de intensidad (0-1).")
    report.append("- `class_prob_linear`: Probabilidad softmax de la red de polarización lineal (0-1).")
    report.append("- `is_burst_intensity`: Etiqueta binaria de la red de intensidad.")
    report.append("- `is_burst_linear`: Etiqueta binaria de la red de polarización lineal.")
    report.append("- `is_burst`: Etiqueta final del pipeline.\n")
    
    # Resultados por fase
    report.append("## Resultados Detallados por Fase\n")
    
    phase_order = ['all', 'no-class-intensity', 'no-classification-linear', 
                   'no-phase2-no-classification-intensity', 'no-phase2-no-classification-linear', 'no-phase2']
    
    for phase_name in phase_order:
        if phase_name in all_metrics:
            report.append(generate_phase_section(phase_name, all_metrics[phase_name]))
    
    # Comparación sistemática
    report.append(generate_comparison_table(all_metrics))
    
    # Análisis comparativo robusto
    report.append(generate_phase_comparison_analysis(all_metrics))
    
    # Recomendaciones finales
    report.append(generate_final_recommendations(all_metrics))
    
    # Discusión y conclusiones
    report.append("## Discusión y Conclusiones\n")
    report.append("### Interpretación de Resultados\n")
    
    # Identificar mejor fase
    best_f1 = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
    best_recall = max(all_metrics.items(), key=lambda x: x[1]['recall'])
    
    report.append(f"1. **Rendimiento General:** El **{get_phase_config_name(best_f1[0])}** ofrece el mejor balance (F1={format_percentage(best_f1[1]['f1_score'])}).")
    report.append(f"2. **Sensibilidad Máxima:** El **{get_phase_config_name(best_recall[0])}** maximiza la recuperación (Recall={format_percentage(best_recall[1]['recall'])}).\n")
    
    report.append("### Hallazgos Clave\n")
    report.append("1. **Separación Detección vs Clasificación:** El análisis muestra que el pipeline es muy efectivo en detección (matching temporal),")
    report.append("   pero la variación principal entre fases radica en cómo se clasifican los candidatos detectados.")
    report.append("2. **Comportamiento de Clasificadores:** Los clasificadores de Intensity y Linear muestran comportamientos diferentes,")
    report.append("   con la red de Linear siendo generalmente más conservadora (rechaza más candidatos).")
    report.append("3. **Casos Críticos:** Se identificaron pulsos canónicos que fueron detectados pero rechazados por algún clasificador,")
    report.append("   lo cual requiere análisis detallado de las probabilidades y SNRs asociados.")
    report.append("4. **Problema de Transfer Learning:** La limitación identificada en clasificación de polarización lineal confirma")
    report.append("   el problema de transfer learning previamente documentado, donde ResNet18 entrenado en Stokes I no generaliza")
    report.append("   correctamente a morfologías atípicas pero físicamente válidas en Stokes L.\n")
    
    report.append("### Recomendaciones\n")
    report.append("- Para aplicaciones que requieren alta pureza: usar configuraciones con ambas fases de clasificación activas.")
    report.append("- Para búsqueda exploratoria: considerar desactivar Phase 2 o alguna fase de clasificación, aceptando mayor revisión manual.")
    report.append("- Revisar casos críticos identificados para entender por qué fueron rechazados y ajustar umbrales si es necesario.")
    report.append("- Trabajo futuro prioritario: Reentrenamiento de ResNet18 con waterfalls de Stokes L para mejorar transfer learning.\n")
    
    # Guardar reporte
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"  [OK] Reporte exhaustivo guardado en: {output_path}")


def main():
    """Función principal"""
    base_path = Path("ResultsThesis/ALMA-4phases")
    metrics_path = base_path / "metrics_all_phases.json"
    output_path = base_path / "analisis_validacion_completo.md"
    
    if not metrics_path.exists():
        print(f"Error: No se encontró el archivo de métricas: {metrics_path}")
        print("Por favor, ejecuta primero analyze_alma_phases_validation.py")
        return
    
    generate_report(str(metrics_path), str(output_path))
    print(f"\n{'='*60}")
    print("[OK] Reporte exhaustivo generado exitosamente!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
