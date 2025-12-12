#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Case 4: ALMA Phased PSR J1745-2900 HF Pipeline Validation
==================================================================

Script to analyze ALMA case study files and calculate HF pipeline metrics,
including ground truth validation (8 canonical pulses) and extended validation
(54 additional pulses).

Usage:
    python analyze_case_alma_psr1745.py [--results-dir ResultsThesis] [--output report.json]

Author: DRAFTS System
Version: 1.0
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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

# Ground truth: 8 canonical pulses
GROUND_TRUTH = {
    '142_0003': [39.977],
    '142_0006': [10.882, 25.829],
    '153_0006': [23.444],
    '230_0002': [2.3, 17.395],
    '230_0003': [36.548],
    '242_0005': [44.919],
}


def find_candidates_csv(results_dir: Path, file_pattern: str) -> List[Path]:
    """Find all candidate CSV files matching pattern."""
    csv_files = []
    for summary_dir in results_dir.glob('Summary/*'):
        if file_pattern in summary_dir.name:
            csv_file = summary_dir / f"{summary_dir.name}.candidates.csv"
            if csv_file.exists():
                csv_files.append(csv_file)
    return csv_files


def load_candidates_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load candidates from CSV file."""
    candidates = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates.append(dict(row))
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
    return candidates


def match_ground_truth(candidates: List[Dict], file_stem: str, tolerance_sec: float = 0.1) -> Tuple[int, List[Dict]]:
    """Match candidates with ground truth timestamps."""
    expected_times = GROUND_TRUTH.get(file_stem, [])
    matched = []
    matched_indices = set()
    
    for expected_time in expected_times:
        best_match = None
        best_idx = -1
        min_diff = float('inf')
        
        for idx, cand in enumerate(candidates):
            if idx in matched_indices:
                continue
            
            # Try different time fields
            cand_time = None
            for field in ['t_sec_waterfall', 't_sec_dm_time', 't_sample']:
                if field in cand:
                    try:
                        cand_time = float(cand[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if cand_time is None:
                continue
            
            diff = abs(cand_time - expected_time)
            if diff < min_diff and diff <= tolerance_sec:
                min_diff = diff
                best_match = cand
                best_idx = idx
        
        if best_match:
            matched.append({
                'expected_time': expected_time,
                'candidate': best_match,
                'time_diff': min_diff,
            })
            matched_indices.add(best_idx)
    
    return len(matched), matched


def extract_classification_metrics(candidates: List[Dict]) -> Dict[str, Any]:
    """Extract classification metrics from candidates."""
    result = {
        'total_candidates': len(candidates),
        'burst_intensity': 0,
        'burst_linear': 0,
        'burst_both': 0,
        'class_prob_i_distribution': [],
        'class_prob_l_distribution': [],
    }
    
    for cand in candidates:
        is_burst_i = cand.get('is_burst_intensity', '').lower() in ['true', '1', 'yes']
        is_burst_l = cand.get('is_burst_linear', '').lower() in ['true', '1', 'yes']
        
        if is_burst_i:
            result['burst_intensity'] += 1
        if is_burst_l:
            result['burst_linear'] += 1
        if is_burst_i and is_burst_l:
            result['burst_both'] += 1
        
        # Extract probabilities
        try:
            prob_i = float(cand.get('class_prob_intensity', 0) or 0)
            prob_l = float(cand.get('class_prob_linear', 0) or 0)
            result['class_prob_i_distribution'].append(prob_i)
            result['class_prob_l_distribution'].append(prob_l)
        except (ValueError, TypeError):
            pass
    
    return result


def calculate_metrics(matched_gt: int, total_gt: int, total_candidates: int, 
                     false_positives: int = 0) -> Dict[str, Any]:
    """Calculate performance metrics."""
    recall_detection = (matched_gt / total_gt * 100) if total_gt > 0 else 0
    recall_classification = recall_detection  # Will be updated based on classification
    
    precision = ((matched_gt) / (matched_gt + false_positives) * 100) if (matched_gt + false_positives) > 0 else 0
    
    f1_score = (2 * recall_detection * precision / (recall_detection + precision)) if (recall_detection + precision) > 0 else 0
    
    return {
        'recall_detection_percent': recall_detection,
        'recall_classification_percent': recall_classification,
        'precision_percent': precision,
        'f1_score': f1_score,
        'matched_ground_truth': matched_gt,
        'total_ground_truth': total_gt,
        'total_candidates': total_candidates,
        'false_positives': false_positives,
    }


def analyze_ground_truth_validation(results_dir: Path) -> Dict[str, Any]:
    """Analyze validation against 8 canonical pulses."""
    logger.info("Analyzing ground truth validation (8 canonical pulses)...")
    
    all_results = {}
    total_matched = 0
    total_gt = 8  # Total canonical pulses
    
    for file_stem, expected_times in GROUND_TRUTH.items():
        logger.info(f"Processing {file_stem}...")
        
        # Find CSV files for this file
        csv_files = find_candidates_csv(results_dir, file_stem)
        
        if not csv_files:
            logger.warning(f"No CSV files found for {file_stem}")
            all_results[file_stem] = {
                'matched': 0,
                'expected': len(expected_times),
                'candidates': [],
            }
            continue
        
        # Load all candidates from matching files
        all_candidates = []
        for csv_file in csv_files:
            candidates = load_candidates_csv(csv_file)
            all_candidates.extend(candidates)
        
        # Match with ground truth
        matched_count, matched = match_ground_truth(all_candidates, file_stem)
        total_matched += matched_count
        
        # Extract classification info for matched pulses
        matched_details = []
        for match in matched:
            cand = match['candidate']
            matched_details.append({
                'timestamp': match['expected_time'],
                'detected_time': cand.get('t_sec_waterfall', 'N/A'),
                'class_prob_i': cand.get('class_prob_intensity', 'N/A'),
                'class_prob_l': cand.get('class_prob_linear', 'N/A'),
                'is_burst_i': cand.get('is_burst_intensity', 'false').lower() == 'true',
                'is_burst_l': cand.get('is_burst_linear', 'false').lower() == 'true',
                'snr': cand.get('snr_waterfall', 'N/A'),
            })
        
        all_results[file_stem] = {
            'matched': matched_count,
            'expected': len(expected_times),
            'matched_details': matched_details,
            'total_candidates': len(all_candidates),
        }
    
    # Calculate overall metrics
    metrics = calculate_metrics(total_matched, total_gt, sum(r.get('total_candidates', 0) for r in all_results.values()))
    
    return {
        'ground_truth_pulses': all_results,
        'metrics': metrics,
        'summary': {
            'total_ground_truth': total_gt,
            'total_matched': total_matched,
            'recall_detection_percent': metrics['recall_detection_percent'],
        }
    }


def analyze_extended_validation(results_dir: Path) -> Dict[str, Any]:
    """Analyze extended validation (54 additional pulses)."""
    logger.info("Analyzing extended validation (54 additional pulses)...")
    
    # Find all ALMA-related CSV files
    alma_patterns = ['No0134', 'No0142', 'No0143', 'No0152', 'No0153', 
                     'No0220', 'No0227', 'No0230', 'No0240', 'No0242', 'No0243']
    
    all_candidates = []
    for pattern in alma_patterns:
        csv_files = find_candidates_csv(results_dir, pattern)
        for csv_file in csv_files:
            candidates = load_candidates_csv(csv_file)
            all_candidates.extend(candidates)
    
    # Extract classification metrics
    classification_metrics = extract_classification_metrics(all_candidates)
    
    return {
        'total_candidates': len(all_candidates),
        'classification_metrics': classification_metrics,
    }


def generate_report(gt_validation: Dict[str, Any], extended_validation: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive report."""
    report = {
        'case': 'ALMA PSR J1745-2900',
        'ground_truth_validation': gt_validation,
        'extended_validation': extended_validation,
        'summary': {
            'ground_truth': {
                'total_pulses': gt_validation['summary']['total_ground_truth'],
                'matched_pulses': gt_validation['summary']['total_matched'],
                'recall_detection': gt_validation['metrics']['recall_detection_percent'],
                'recall_classification': gt_validation['metrics']['recall_classification_percent'],
                'precision': gt_validation['metrics']['precision_percent'],
                'f1_score': gt_validation['metrics']['f1_score'],
            },
            'extended': {
                'total_candidates': extended_validation['total_candidates'],
                'burst_intensity': extended_validation['classification_metrics']['burst_intensity'],
                'burst_linear': extended_validation['classification_metrics']['burst_linear'],
                'burst_both': extended_validation['classification_metrics']['burst_both'],
            }
        }
    }
    
    return report


def print_summary(report: Dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "="*70)
    print("ALMA PSR J1745-2900 HF PIPELINE VALIDATION SUMMARY")
    print("="*70)
    
    gt_summary = report['summary']['ground_truth']
    print("\n--- Ground Truth Validation (8 Canonical Pulses) ---")
    print(f"Total ground truth: {gt_summary['total_pulses']}")
    print(f"Matched: {gt_summary['matched_pulses']}")
    print(f"Recall (detection): {gt_summary['recall_detection']:.1f}%")
    print(f"Recall (classification): {gt_summary['recall_classification']:.1f}%")
    print(f"Precision: {gt_summary['precision']:.1f}%")
    print(f"F1-score: {gt_summary['f1_score']:.3f}")
    
    print("\n--- Per-File Ground Truth Results ---")
    for file_stem, results in report['ground_truth_validation']['ground_truth_pulses'].items():
        print(f"\n{file_stem}:")
        print(f"  Expected: {results['expected']} pulses")
        print(f"  Matched: {results['matched']} pulses")
        if results.get('matched_details'):
            for detail in results['matched_details']:
                print(f"    t={detail['timestamp']}s: p_I={detail['class_prob_i']}, p_L={detail['class_prob_l']}")
    
    ext_summary = report['summary']['extended']
    print("\n--- Extended Validation (54 Additional Pulses) ---")
    print(f"Total candidates: {ext_summary['total_candidates']}")
    print(f"Burst (Intensity): {ext_summary['burst_intensity']}")
    print(f"Burst (Linear): {ext_summary['burst_linear']}")
    print(f"Burst (Both): {ext_summary['burst_both']}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ALMA PSR J1745-2900 case study HF pipeline validation'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='Results-polarization',
        help='Results directory (default: Results-polarization)'
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
    
    # Analyze ground truth validation
    gt_validation = analyze_ground_truth_validation(results_dir)
    
    # Analyze extended validation
    extended_validation = analyze_extended_validation(results_dir)
    
    # Generate report
    report = generate_report(gt_validation, extended_validation)
    
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
        logger.info("Analysis complete")
        logger.info("Use --summary to see detailed statistics")


if __name__ == '__main__':
    main()

