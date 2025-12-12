#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Case 2: B0355+54 Temporal Robustness Validation
========================================================

Script to analyze B0355+54 case study file and validate temporal continuity
between chunks, overlap calculations, and pulse detection statistics.

Usage:
    python analyze_case_b0355.py [--results-dir ResultsThesis] [--output report.json]

Author: DRAFTS System
Version: 1.0
"""

from __future__ import annotations

import argparse
import json
import csv
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

# B0355+54 file
B0355_FILE = 'B0355+54_FB_20220918'


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


def find_candidates_csv(results_dir: Path, file_stem: str) -> Optional[Path]:
    """Find candidates CSV file for a given file stem."""
    for summary_dir in results_dir.glob('Summary/*'):
        if file_stem in summary_dir.name:
            csv_file = summary_dir / f"{summary_dir.name}.candidates.csv"
            if csv_file.exists():
                return csv_file
    return None


def load_validation_metrics(json_path: Path) -> Dict[str, Any]:
    """Load validation metrics from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return {}


def extract_overlap_metrics(metrics: Dict) -> Dict[str, Any]:
    """Extract overlap calculation metrics."""
    result = {
        'dm_max': None,
        'delta_t_max_seconds': None,
        'overlap_decimated_samples': None,
        'num_chunks': 0,
    }
    
    dm_cube = metrics.get('dm_cube', {})
    result['dm_max'] = dm_cube.get('dm_max', None)
    result['delta_t_max_seconds'] = dm_cube.get('delta_t_max_seconds', None)
    
    overlap_validation = metrics.get('overlap_validation', {})
    result['overlap_decimated_samples'] = overlap_validation.get('overlap_decimated', overlap_validation.get('overlap_decimated_samples', None))
    
    actual = metrics.get('actual_processing', {})
    result['num_chunks'] = actual.get('chunks_processed', 0)
    
    return result


def extract_chunk_continuity_metrics(metrics: Dict) -> List[Dict[str, Any]]:
    """Extract continuity metrics for each chunk."""
    chunks_data = []
    
    chunks = metrics.get('chunks', [])
    for i, chunk in enumerate(chunks):
        temporal_info = chunk.get('temporal_info', {})
        chunk_metrics = {
            'chunk_index': chunk.get('chunk_idx', i),
            'overlap_left': temporal_info.get('overlap_left_decimated', chunk.get('overlap_left_decimated', 0)),
            'overlap_right': temporal_info.get('overlap_right_decimated', chunk.get('overlap_right_decimated', 0)),
            'valid_start': temporal_info.get('valid_start_decimated', chunk.get('valid_start_decimated', None)),
            'valid_end': temporal_info.get('valid_end_decimated', chunk.get('valid_end_decimated', None)),
            'valid_samples': temporal_info.get('valid_samples', chunk.get('valid_samples_decimated', None)),
            'overlap_sufficient': chunk.get('overlap_sufficient', False),
            'overlap_ratio': chunk.get('overlap_vs_delay_ratio', chunk.get('overlap_ratio', 0.0)),
            'continuity_with_previous': chunk.get('continuity_with_previous', False),
        }
        chunks_data.append(chunk_metrics)
    
    return chunks_data


def extract_global_continuity_metrics(metrics: Dict) -> Dict[str, Any]:
    """Extract global continuity validation metrics."""
    overlap_validation = metrics.get('overlap_validation', {})
    
    return {
        'no_edge_losses': overlap_validation.get('no_edge_losses', False),
        'total_valid_samples': None,
        'total_decimated_samples': None,
        'coverage_percent': None,
    }


def analyze_pulse_detections(csv_path: Path) -> Dict[str, Any]:
    """Analyze pulse detections from candidates CSV."""
    result = {
        'total_detections': 0,
        'burst_classifications': 0,
        'expected_pulses': 752,  # From validation document
        'recall_percent': None,
        'classification_precision_percent': None,
    }
    
    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return result
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            detections = list(reader)
        
        result['total_detections'] = len(detections)
        
        # Count burst classifications
        burst_count = 0
        for row in detections:
            is_burst = row.get('is_burst', '').lower()
            if is_burst in ['true', '1', 'yes']:
                burst_count += 1
        
        result['burst_classifications'] = burst_count
        
        # Calculate recall
        if result['expected_pulses'] > 0:
            result['recall_percent'] = (result['total_detections'] / result['expected_pulses']) * 100
        
        # Calculate classification precision
        if result['total_detections'] > 0:
            result['classification_precision_percent'] = (burst_count / result['total_detections']) * 100
    
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
    
    return result


def analyze_file(results_dir: Path, file_stem: str) -> Optional[Dict[str, Any]]:
    """Analyze B0355+54 file."""
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
    
    # Find candidates CSV
    csv_path = find_candidates_csv(results_dir, file_stem)
    pulse_stats = {}
    if csv_path:
        logger.info(f"Found candidates CSV: {csv_path}")
        pulse_stats = analyze_pulse_detections(csv_path)
    else:
        logger.warning(f"No candidates CSV found for {file_stem}")
    
    # Extract all metrics
    analysis = {
        'file_stem': file_stem,
        'overlap': extract_overlap_metrics(metrics),
        'chunk_continuity': extract_chunk_continuity_metrics(metrics),
        'global_continuity': extract_global_continuity_metrics(metrics),
        'pulse_detections': pulse_stats,
    }
    
    # Calculate total valid samples
    total_valid = sum(c.get('valid_samples', 0) or 0 for c in analysis['chunk_continuity'])
    analysis['global_continuity']['total_valid_samples'] = total_valid if total_valid > 0 else None
    
    # Get total decimated samples from data characteristics
    data_char = metrics.get('data_characteristics', {})
    total_decimated = data_char.get('decimated_samples', data_char.get('total_samples_decimated', None))
    analysis['global_continuity']['total_decimated_samples'] = total_decimated
    
    # Calculate coverage
    if total_decimated and total_decimated > 0:
        coverage = (total_valid / total_decimated) * 100
        analysis['global_continuity']['coverage_percent'] = coverage
    
    return analysis


def generate_report(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive report from analysis."""
    report = {
        'case': 'B0355+54',
        'file': analysis['file_stem'],
        'analysis': analysis,
        'summary': {
            'overlap_table': {
                'dm_max': analysis['overlap'].get('dm_max'),
                'delta_t_max_s': analysis['overlap'].get('delta_t_max_seconds'),
                'overlap_samples': analysis['overlap'].get('overlap_decimated_samples'),
                'num_chunks': analysis['overlap'].get('num_chunks'),
            },
            'continuity_table': analysis['chunk_continuity'],
            'global_continuity': analysis['global_continuity'],
            'pulse_detection': analysis['pulse_detections'],
        }
    }
    
    return report


def print_summary(report: Dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "="*70)
    print("B0355+54 TEMPORAL CONTINUITY VALIDATION SUMMARY")
    print("="*70)
    
    analysis = report['analysis']
    
    print(f"\nFile: {analysis['file_stem']}")
    
    print("\n--- Overlap Calculation ---")
    overlap = analysis['overlap']
    print(f"DM_max: {overlap.get('dm_max')} pc cm⁻³")
    print(f"Δt_max: {overlap.get('delta_t_max_seconds'):.6f} s" if overlap.get('delta_t_max_seconds') else "Δt_max: N/A")
    print(f"Overlap (O_d): {overlap.get('overlap_decimated_samples'):,} samples" if overlap.get('overlap_decimated_samples') else "Overlap: N/A")
    print(f"Number of chunks: {overlap.get('num_chunks')}")
    
    print("\n--- Chunk Continuity ---")
    for chunk in analysis['chunk_continuity']:
        idx = chunk['chunk_index']
        print(f"\nChunk {idx}:")
        print(f"  Overlap left: {chunk['overlap_left']}")
        print(f"  Overlap right: {chunk['overlap_right']}")
        print(f"  Valid start: {chunk['valid_start']}")
        print(f"  Valid end: {chunk['valid_end']}")
        print(f"  Valid samples: {chunk['valid_samples']:,}" if chunk['valid_samples'] else "  Valid samples: N/A")
        print(f"  Overlap sufficient: {'✓' if chunk['overlap_sufficient'] else '✗'}")
        print(f"  Overlap ratio: {chunk['overlap_ratio']:.2f}")
        print(f"  Continuity with previous: {'✓' if chunk['continuity_with_previous'] else '✗'}")
    
    print("\n--- Global Continuity ---")
    global_cont = analysis['global_continuity']
    print(f"No edge losses: {'✓' if global_cont.get('no_edge_losses') else '✗'}")
    print(f"Total valid samples: {global_cont.get('total_valid_samples'):,}" if global_cont.get('total_valid_samples') else "Total valid: N/A")
    print(f"Total decimated samples: {global_cont.get('total_decimated_samples'):,}" if global_cont.get('total_decimated_samples') else "Total decimated: N/A")
    print(f"Coverage: {global_cont.get('coverage_percent'):.2f}%" if global_cont.get('coverage_percent') else "Coverage: N/A")
    
    print("\n--- Pulse Detection ---")
    pulse = analysis['pulse_detections']
    if pulse.get('total_detections', 0) > 0:
        print(f"Total detections: {pulse['total_detections']}")
        print(f"Burst classifications: {pulse['burst_classifications']}")
        print(f"Expected pulses: {pulse['expected_pulses']}")
        if pulse.get('recall_percent'):
            print(f"Recall: {pulse['recall_percent']:.1f}%")
        if pulse.get('classification_precision_percent'):
            print(f"Classification precision: {pulse['classification_precision_percent']:.1f}%")
    else:
        print("No detection data available")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze B0355+54 case study temporal continuity'
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
    
    # Analyze B0355+54 file
    analysis = analyze_file(results_dir, B0355_FILE)
    
    if not analysis:
        logger.error("No valid analysis found!")
        sys.exit(1)
    
    # Generate report
    report = generate_report(analysis)
    
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
        logger.info(f"Analysis complete for {B0355_FILE}")
        logger.info("Use --summary to see detailed statistics")


if __name__ == '__main__':
    main()

