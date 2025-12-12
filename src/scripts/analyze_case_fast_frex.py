#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Case 1: FAST-FREX Validation
=====================================

Script to analyze FAST-FREX case study files (FRB20180301_0001, FRB20201124_0009)
and extract validation metrics from JSON files for quantitative validation.

Usage:
    python analyze_case_fast_frex.py [--results-dir ResultsThesis] [--output report.json]

Author: DRAFTS System
Version: 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Setup path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FAST-FREX files
FAST_FREX_FILES = ['FRB20180301_0001', 'FRB20201124_0009']


def find_validation_json(results_dir: Path, file_stem: str) -> Optional[Path]:
    """Find validation JSON file for a given file stem."""
    # Look in Validation/ directory (direct files)
    validation_dir = results_dir / 'Validation'
    if validation_dir.exists():
        # Try direct match
        for json_file in validation_dir.glob(f'*{file_stem}*.json'):
            return json_file
        # Try in subdirectories
        for subdir in validation_dir.iterdir():
            if subdir.is_dir() and file_stem in subdir.name:
                for json_file in subdir.glob(f'*{file_stem}*.json'):
                    return json_file
    
    # Look in Summary/*/Validation/validation_metrics.json (legacy structure)
    for summary_dir in results_dir.glob('Summary/*'):
        if file_stem in summary_dir.name:
            validation_dir = summary_dir / 'Validation'
            if validation_dir.exists():
                json_file = validation_dir / 'validation_metrics.json'
                if json_file.exists():
                    return json_file
    return None


def load_validation_metrics(json_path: Path) -> Dict[str, Any]:
    """Load validation metrics from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return {}


def extract_resource_planning_metrics(metrics: Dict) -> Dict[str, Any]:
    """Extract resource planning metrics from validation JSON."""
    result = {
        'N_0': None,  # Original samples
        'N_d': None,  # Decimated samples
        'b_p': None,  # Bytes per sample
        'M_d': None,  # Available memory (GB)
        'M_u': None,  # Usable memory (GB)
        'N_c': None,  # Final chunk size
        'aligned': False,  # Is N_c aligned to L_s?
    }
    
    # Extract from data_characteristics
    data_char = metrics.get('data_characteristics', {})
    result['N_0'] = data_char.get('file_length_samples', data_char.get('total_samples_raw', None))
    result['N_d'] = data_char.get('decimated_samples', data_char.get('total_samples_decimated', None))
    result['b_p'] = data_char.get('bytes_per_sample', None)
    
    # Extract from memory_budget
    mem_budget = metrics.get('memory_budget', {})
    result['M_d'] = mem_budget.get('available_ram_gb', mem_budget.get('ram_available_gb', None))
    result['M_u'] = mem_budget.get('total_usable_gb', mem_budget.get('memory_utilizable_gb', None))
    
    # Extract from chunk_calculation
    chunk_calc = metrics.get('chunk_calculation', {})
    result['N_c'] = chunk_calc.get('final_chunk_samples', None)
    
    # Check alignment (N_c should be multiple of slice_len)
    slice_len = data_char.get('slice_length_samples', chunk_calc.get('phase_c', {}).get('slice_len', None))
    if result['N_c'] and slice_len:
        result['aligned'] = (result['N_c'] % slice_len == 0) if slice_len > 0 else False
    
    return result


def extract_budget_phases_metrics(metrics: Dict) -> Dict[str, Any]:
    """Extract three-phase budget metrics from validation JSON."""
    result = {
        'C_s_kb': None,  # Cost per sample (KB)
        'N_max': None,  # Maximum capacity
        'N_min': None,  # Minimum required
        'scenario': None,  # Ideal or Extreme
        'N_c_final': None,  # Final chunk size
    }
    
    chunk_calc = metrics.get('chunk_calculation', {})
    
    # Phase A: Cost per sample
    phase_a = chunk_calc.get('phase_a', {})
    cost_per_sample_bytes = phase_a.get('cost_per_sample_bytes', None)
    if cost_per_sample_bytes:
        result['C_s_kb'] = cost_per_sample_bytes / 1024.0
    else:
        # Try from dm_cube
        dm_cube = metrics.get('dm_cube', {})
        cost_per_sample_bytes = dm_cube.get('cost_per_sample_bytes', None)
        if cost_per_sample_bytes:
            result['C_s_kb'] = cost_per_sample_bytes / 1024.0
    
    # Phase B: Maximum capacity
    phase_b = chunk_calc.get('phase_b', {})
    result['N_max'] = phase_b.get('max_samples', phase_b.get('max_capacity_samples', None))
    
    # Phase C: Minimum required
    phase_c = chunk_calc.get('phase_c', {})
    result['N_min'] = phase_c.get('required_min_size', phase_c.get('min_required_samples', None))
    
    # Scenario and final chunk
    result['scenario'] = chunk_calc.get('scenario', None)
    result['N_c_final'] = chunk_calc.get('final_chunk_samples', None)
    
    return result


def extract_memory_usage_metrics(metrics: Dict) -> Dict[str, Any]:
    """Extract memory usage metrics from validation JSON."""
    result = {
        'peak_memory_gb': None,
        'available_memory_gb': None,
        'usable_memory_gb': None,
        'ratio': None,
        'oom_errors': 0,
    }
    
    actual = metrics.get('actual_processing', {})
    result['peak_memory_gb'] = actual.get('peak_memory_usage_gb', actual.get('peak_memory_gb', None))
    result['oom_errors'] = actual.get('oom_errors', 0)
    
    mem_budget = metrics.get('memory_budget', {})
    result['available_memory_gb'] = mem_budget.get('available_ram_gb', mem_budget.get('ram_available_gb', None))
    result['usable_memory_gb'] = mem_budget.get('total_usable_gb', mem_budget.get('memory_utilizable_gb', None))
    
    # Calculate ratio
    if result['peak_memory_gb'] and result['usable_memory_gb']:
        result['ratio'] = result['peak_memory_gb'] / result['usable_memory_gb']
    
    return result


def extract_overlap_metrics(metrics: Dict) -> Dict[str, Any]:
    """Extract overlap and temporal continuity metrics."""
    result = {
        'delta_t_max_seconds': None,
        'overlap_decimated_samples': None,
        'num_chunks': 0,
        'continuity_validated': False,
    }
    
    dm_cube = metrics.get('dm_cube', {})
    result['delta_t_max_seconds'] = dm_cube.get('delta_t_max_seconds', None)
    
    overlap_validation = metrics.get('overlap_validation', {})
    result['overlap_decimated_samples'] = overlap_validation.get('overlap_decimated', overlap_validation.get('overlap_decimated_samples', None))
    result['continuity_validated'] = overlap_validation.get('no_edge_losses', False)
    
    actual = metrics.get('actual_processing', {})
    result['num_chunks'] = actual.get('chunks_processed', 0)
    
    return result


def analyze_file(results_dir: Path, file_stem: str) -> Optional[Dict[str, Any]]:
    """Analyze a single FAST-FREX file."""
    logger.info(f"Analyzing {file_stem}...")
    
    json_path = find_validation_json(results_dir, file_stem)
    if not json_path:
        logger.warning(f"No validation JSON found for {file_stem}")
        return None
    
    logger.info(f"Found validation JSON: {json_path}")
    metrics = load_validation_metrics(json_path)
    
    if not metrics:
        logger.warning(f"Empty or invalid metrics for {file_stem}")
        return None
    
    # Extract all metrics
    analysis = {
        'file_stem': file_stem,
        'resource_planning': extract_resource_planning_metrics(metrics),
        'budget_phases': extract_budget_phases_metrics(metrics),
        'memory_usage': extract_memory_usage_metrics(metrics),
        'overlap': extract_overlap_metrics(metrics),
    }
    
    return analysis


def generate_report(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive report from analyses."""
    report = {
        'case': 'FAST-FREX',
        'files_analyzed': len(analyses),
        'analyses': analyses,
        'summary': {
            'resource_planning_table': [],
            'budget_phases_table': [],
            'memory_usage_summary': [],
            'overlap_summary': [],
        }
    }
    
    # Build tables
    for analysis in analyses:
        file_stem = analysis['file_stem']
        rp = analysis['resource_planning']
        bp = analysis['budget_phases']
        mu = analysis['memory_usage']
        ov = analysis['overlap']
        
        # Resource planning table row
        report['summary']['resource_planning_table'].append({
            'file': file_stem,
            'N_0': rp.get('N_0'),
            'N_d': rp.get('N_d'),
            'b_p_bytes': rp.get('b_p'),
            'M_d_gb': rp.get('M_d'),
            'M_u_gb': rp.get('M_u'),
            'N_c': rp.get('N_c'),
            'aligned': rp.get('aligned'),
        })
        
        # Budget phases table row
        report['summary']['budget_phases_table'].append({
            'file': file_stem,
            'C_s_kb': bp.get('C_s_kb'),
            'N_max': bp.get('N_max'),
            'N_min': bp.get('N_min'),
            'scenario': bp.get('scenario'),
            'N_c_final': bp.get('N_c_final'),
        })
        
        # Memory usage summary
        report['summary']['memory_usage_summary'].append({
            'file': file_stem,
            'peak_gb': mu.get('peak_memory_gb'),
            'available_gb': mu.get('available_memory_gb'),
            'usable_gb': mu.get('usable_memory_gb'),
            'ratio': mu.get('ratio'),
            'oom_errors': mu.get('oom_errors'),
        })
        
        # Overlap summary
        report['summary']['overlap_summary'].append({
            'file': file_stem,
            'delta_t_max_s': ov.get('delta_t_max_seconds'),
            'overlap_samples': ov.get('overlap_decimated_samples'),
            'num_chunks': ov.get('num_chunks'),
            'continuity_validated': ov.get('continuity_validated'),
        })
    
    return report


def print_summary(report: Dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "="*70)
    print("FAST-FREX VALIDATION ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nFiles analyzed: {report['files_analyzed']}")
    
    print("\n--- Resource Planning ---")
    for row in report['summary']['resource_planning_table']:
        print(f"\n{row['file']}:")
        print(f"  N_0: {row['N_0']:,}" if row['N_0'] else "  N_0: N/A")
        print(f"  N_d: {row['N_d']:,}" if row['N_d'] else "  N_d: N/A")
        print(f"  b_p: {row['b_p_bytes']} bytes" if row['b_p_bytes'] else "  b_p: N/A")
        print(f"  M_d: {row['M_d_gb']:.2f} GB" if row['M_d_gb'] else "  M_d: N/A")
        print(f"  M_u: {row['M_u_gb']:.2f} GB" if row['M_u_gb'] else "  M_u: N/A")
        print(f"  N_c: {row['N_c']:,}" if row['N_c'] else "  N_c: N/A")
        print(f"  Aligned: {'✓' if row['aligned'] else '✗'}")
    
    print("\n--- Budget Phases ---")
    for row in report['summary']['budget_phases_table']:
        print(f"\n{row['file']}:")
        print(f"  C_s: {row['C_s_kb']:.2f} KB" if row['C_s_kb'] else "  C_s: N/A")
        print(f"  N_max: {row['N_max']:,}" if row['N_max'] else "  N_max: N/A")
        print(f"  N_min: {row['N_min']:,}" if row['N_min'] else "  N_min: N/A")
        print(f"  Scenario: {row['scenario']}" if row['scenario'] else "  Scenario: N/A")
        print(f"  N_c final: {row['N_c_final']:,}" if row['N_c_final'] else "  N_c final: N/A")
    
    print("\n--- Memory Usage ---")
    for row in report['summary']['memory_usage_summary']:
        print(f"\n{row['file']}:")
        print(f"  Peak: {row['peak_gb']:.2f} GB" if row['peak_gb'] else "  Peak: N/A")
        print(f"  Available: {row['available_gb']:.2f} GB" if row['available_gb'] else "  Available: N/A")
        print(f"  Usable: {row['usable_gb']:.2f} GB" if row['usable_gb'] else "  Usable: N/A")
        print(f"  Ratio: {row['ratio']:.2f}" if row['ratio'] else "  Ratio: N/A")
        print(f"  OOM errors: {row['oom_errors']}")
    
    print("\n--- Overlap & Continuity ---")
    for row in report['summary']['overlap_summary']:
        print(f"\n{row['file']}:")
        print(f"  Δt_max: {row['delta_t_max_s']:.6f} s" if row['delta_t_max_s'] else "  Δt_max: N/A")
        print(f"  Overlap: {row['overlap_samples']:,} samples" if row['overlap_samples'] else "  Overlap: N/A")
        print(f"  Chunks: {row['num_chunks']}")
        print(f"  Continuity: {'✓' if row['continuity_validated'] else '✗'}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze FAST-FREX case study validation metrics'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='ResultsThesis',
        help='Results directory (default: ResultsThesis)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON report file (optional)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    results_dir = project_root / args.results_dir
    
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    # Analyze each FAST-FREX file
    analyses = []
    for file_stem in FAST_FREX_FILES:
        analysis = analyze_file(results_dir, file_stem)
        if analysis:
            analyses.append(analysis)
    
    if not analyses:
        logger.error("No valid analyses found!")
        sys.exit(1)
    
    # Generate report
    report = generate_report(analyses)
    
    # Save report if requested
    if args.output:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {output_path}")
    
    # Print summary
    if args.summary:
        print_summary(report)
    else:
        logger.info(f"Analyzed {len(analyses)} files")
        logger.info("Use --summary to see detailed statistics")


if __name__ == '__main__':
    main()

