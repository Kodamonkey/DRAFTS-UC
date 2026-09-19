#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Case 3: FRB 121102 Scalability Validation
===================================================

Script to analyze FRB 121102 case study files (3096-3102) and validate
scalability, memory management, and discovery capabilities.

Usage:
    python analyze_case_frb121102.py [--results-dir ResultsThesis] [--output report.json]

Author: DRAFTS System
Version: 1.0
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Plomería compartida con los demás analyze_case_*.py (REF-18). El directorio
# del script ya está en sys.path al ejecutarlo como
# `python src/scripts/analyze_case_frb121102.py`.
from _case_common import (
    build_arg_parser,
    emit_report,
    find_candidates_csv,
    find_validation_json,
    load_validation_metrics,
    logger,
    resolve_results_dir,
)

# FRB 121102 files
FRB121102_FILES = ['3096_0001_00_8bit', '3097_0001_00_8bit', '3098_0001_00_8bit',
                   '3099_0001_00_8bit', '3100_0001_00_8bit', '3101_0001_00_8bit',
                   '3102_0001_00_8bit']

# Ground truth: 24 known events
GROUND_TRUTH_COUNT = 24


def extract_scalability_metrics(metrics: Dict) -> Dict[str, Any]:
    """Extract scalability and memory metrics."""
    result = {
        'file_size_gb': None,
        'peak_memory_gb': None,
        'available_memory_gb': None,
        'chunks_processed': 0,
        'oom_errors': 0,
        'processing_successful': True,
    }
    
    actual = metrics.get('actual_processing', {})
    result['peak_memory_gb'] = actual.get('peak_memory_gb', None)
    result['chunks_processed'] = actual.get('chunks_processed', 0)
    result['oom_errors'] = actual.get('oom_errors', 0)
    result['processing_successful'] = (result['oom_errors'] == 0)
    
    mem_budget = metrics.get('memory_budget', {})
    result['available_memory_gb'] = mem_budget.get('ram_available_gb', None)
    
    data_char = metrics.get('data_characteristics', {})
    total_samples = data_char.get('total_samples_decimated', None)
    bytes_per_sample = data_char.get('bytes_per_sample', None)
    if total_samples and bytes_per_sample:
        result['file_size_gb'] = (total_samples * bytes_per_sample) / (1024**3)
    
    return result


def analyze_detections(csv_path: Path) -> Dict[str, Any]:
    """Analyze detections from candidates CSV."""
    result = {
        'total_detections': 0,
        'burst_detections': 0,
        'ground_truth_recovered': 0,
        'new_candidates': 0,
        'recall_percent': None,
    }
    
    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return result
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            detections = list(reader)
        
        result['total_detections'] = len(detections)
        
        # Count burst detections
        burst_count = 0
        for row in detections:
            is_burst = row.get('is_burst', '').lower()
            if is_burst in ['true', '1', 'yes']:
                burst_count += 1
        
        result['burst_detections'] = burst_count
        
        # Note: Ground truth matching would require additional logic
        # to match timestamps/DM with known events
        # For now, we assume all detections are candidates
        result['new_candidates'] = result['total_detections']
    
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
    
    return result


def analyze_file(results_dir: Path, file_stem: str) -> Optional[Dict[str, Any]]:
    """Analyze a single FRB 121102 file."""
    logger.info(f"Analyzing {file_stem}...")
    
    json_path = find_validation_json(results_dir, file_stem)
    metrics = {}
    
    if json_path:
        logger.info(f"Found validation JSON: {json_path}")
        metrics = load_validation_metrics(json_path)
    else:
        logger.warning(f"No validation JSON found for {file_stem}")
    
    # Find candidates CSV
    csv_path = find_candidates_csv(results_dir, file_stem)
    detections = {}
    if csv_path:
        logger.info(f"Found candidates CSV: {csv_path}")
        detections = analyze_detections(csv_path)
    else:
        logger.warning(f"No candidates CSV found for {file_stem}")
    
    # Extract metrics
    analysis = {
        'file_stem': file_stem,
        'scalability': extract_scalability_metrics(metrics) if metrics else {},
        'detections': detections,
    }
    
    return analysis


def generate_report(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive report from analyses."""
    # Aggregate statistics
    total_files = len(analyses)
    total_size_gb = sum(a['scalability'].get('file_size_gb', 0) or 0 for a in analyses)
    total_detections = sum(a['detections'].get('total_detections', 0) for a in analyses)
    total_oom = sum(a['scalability'].get('oom_errors', 0) for a in analyses)
    all_successful = all(a['scalability'].get('processing_successful', False) for a in analyses)
    
    report = {
        'case': 'FRB 121102',
        'files_analyzed': total_files,
        'analyses': analyses,
        'summary': {
            'total_files': total_files,
            'total_size_gb': total_size_gb,
            'total_detections': total_detections,
            'ground_truth_count': GROUND_TRUTH_COUNT,
            'total_oom_errors': total_oom,
            'all_processed_successfully': all_successful,
            'scalability_metrics': [],
        }
    }
    
    # Build scalability metrics table
    for analysis in analyses:
        scal = analysis['scalability']
        report['summary']['scalability_metrics'].append({
            'file': analysis['file_stem'],
            'size_gb': scal.get('file_size_gb'),
            'peak_memory_gb': scal.get('peak_memory_gb'),
            'chunks': scal.get('chunks_processed', 0),
            'oom_errors': scal.get('oom_errors', 0),
            'detections': analysis['detections'].get('total_detections', 0),
        })
    
    return report


def print_summary(report: Dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "="*70)
    print("FRB 121102 SCALABILITY VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\nFiles analyzed: {report['files_analyzed']}")
    print(f"Total size: {report['summary']['total_size_gb']:.2f} GB")
    print(f"Total detections: {report['summary']['total_detections']}")
    print(f"Ground truth events: {report['summary']['ground_truth_count']}")
    print(f"OOM errors: {report['summary']['total_oom_errors']}")
    print(f"All processed successfully: {'✓' if report['summary']['all_processed_successfully'] else '✗'}")
    
    print("\n--- Per-File Metrics ---")
    for metric in report['summary']['scalability_metrics']:
        print(f"\n{metric['file']}:")
        print(f"  Size: {metric['size_gb']:.2f} GB" if metric['size_gb'] else "  Size: N/A")
        print(f"  Peak memory: {metric['peak_memory_gb']:.2f} GB" if metric['peak_memory_gb'] else "  Peak memory: N/A")
        print(f"  Chunks: {metric['chunks']}")
        print(f"  OOM errors: {metric['oom_errors']}")
        print(f"  Detections: {metric['detections']}")
    
    print("="*70 + "\n")


def main():
    parser = build_arg_parser(
        description='Analyze FRB 121102 case study scalability',
        default_results_dir='Results-polarization-finales',
    )
    args = parser.parse_args()

    # Resolve paths
    project_root, results_dir = resolve_results_dir(args.results_dir)

    # Analyze each FRB 121102 file
    analyses = []
    for file_stem in FRB121102_FILES:
        analysis = analyze_file(results_dir, file_stem)
        if analysis:
            analyses.append(analysis)
    
    if not analyses:
        logger.warning("No valid analyses found!")
        logger.info("This is expected if files haven't been processed yet")
        sys.exit(0)
    
    # Generate report
    report = generate_report(analyses)

    emit_report(report, args, project_root, print_summary,
                f"Analyzed {len(analyses)} files")


if __name__ == '__main__':
    main()

