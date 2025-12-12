#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Catalog Data Files for Validation Cases
========================================

Script to scan and catalog all files in Data/raw/ directory, classifying them
by validation case study and extracting basic metadata.

Usage:
    python catalog_data_files.py [--output catalog.csv] [--json]

Author: DRAFTS System
Version: 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
from astropy.io import fits

# Setup path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Case study patterns
CASE_1_PATTERNS = ['FRB20180301', 'FRB20201124']  # FAST-FREX
CASE_2_PATTERNS = ['B0355+54', 'B0355']  # B0355+54
CASE_3_PATTERNS = ['3096', '3097', '3098', '3099', '3100', '3101', '3102']  # FRB 121102
CASE_4_PATTERNS = ['2017-04-03', 'No0134', 'No0142', 'No0143', 'No0152', 'No0153', 
                   'No0220', 'No0227', 'No0230', 'No0240', 'No0242', 'No0243']  # ALMA PSR J1745


def classify_file(filename: str, path: Path) -> Tuple[int, str]:
    """
    Classify a file into a validation case study.
    
    Returns:
        (case_number, case_name): Case number (1-4) or 0 for "Other", and case name
    """
    filename_lower = filename.lower()
    path_str = str(path).lower()
    
    # Case 1: FAST-FREX
    for pattern in CASE_1_PATTERNS:
        if pattern.lower() in filename_lower or pattern.lower() in path_str:
            return (1, "FAST-FREX")
    
    # Case 2: B0355+54
    for pattern in CASE_2_PATTERNS:
        if pattern.lower() in filename_lower or pattern.lower() in path_str:
            return (2, "B0355+54")
    
    # Case 3: FRB 121102
    for pattern in CASE_3_PATTERNS:
        if pattern.lower() in filename_lower or pattern.lower() in path_str:
            return (3, "FRB 121102")
    
    # Case 4: ALMA PSR J1745
    for pattern in CASE_4_PATTERNS:
        if pattern.lower() in filename_lower or pattern.lower() in path_str:
            return (4, "ALMA PSR J1745-2900")
    
    return (0, "Otro")


def extract_file_metadata(file_path: Path) -> Dict:
    """Extract basic metadata from a file."""
    metadata = {
        'filename': file_path.name,
        'path': str(file_path.relative_to(file_path.parents[len(file_path.parts) - file_path.parts.index('raw') - 1])),
        'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
        'size_mb': 0,
        'format': 'unknown',
        'telescope': 'unknown',
        'frequency_ghz': None,
        'has_header': False,
    }
    
    # Calculate size in MB
    if metadata['size_bytes'] > 0:
        metadata['size_mb'] = metadata['size_bytes'] / (1024 * 1024)
    
    # Determine format
    ext = file_path.suffix.lower()
    if ext in ['.fits', '.fit']:
        metadata['format'] = 'FITS/PSRFITS'
    elif ext == '.fil':
        metadata['format'] = 'SIGPROC Filterbank'
    else:
        metadata['format'] = ext[1:] if ext else 'unknown'
    
    # Try to read header for FITS files
    if ext in ['.fits', '.fit'] and file_path.exists():
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                metadata['has_header'] = True
                
                # Extract telescope
                telescope = header.get('TELESCOP', header.get('TELESCOPE', 'unknown'))
                if telescope and telescope != 'unknown':
                    metadata['telescope'] = str(telescope).strip()
                
                # Extract frequency
                freq_min = header.get('FREQMIN', None)
                freq_max = header.get('FREQMAX', None)
                freq_center = header.get('FREQ', None)
                
                if freq_center:
                    # Convert to GHz if in MHz or Hz
                    freq_val = float(freq_center)
                    if freq_val > 1e9:
                        metadata['frequency_ghz'] = freq_val / 1e9
                    elif freq_val > 1e6:
                        metadata['frequency_ghz'] = freq_val / 1e6
                    else:
                        metadata['frequency_ghz'] = freq_val
                elif freq_min and freq_max:
                    freq_center_val = (float(freq_min) + float(freq_max)) / 2
                    if freq_center_val > 1e9:
                        metadata['frequency_ghz'] = freq_center_val / 1e9
                    elif freq_center_val > 1e6:
                        metadata['frequency_ghz'] = freq_center_val / 1e6
                    else:
                        metadata['frequency_ghz'] = freq_center_val
                
        except Exception as e:
            logger.debug(f"Could not read header from {file_path}: {e}")
            metadata['has_header'] = False
    
    return metadata


def scan_directory(data_dir: Path) -> List[Dict]:
    """Scan directory recursively and catalog all data files."""
    catalog = []
    
    # Supported extensions
    data_extensions = ['.fits', '.fit', '.fil']
    
    logger.info(f"Scanning directory: {data_dir}")
    
    for root, dirs, files in os.walk(data_dir):
        root_path = Path(root)
        
        for file in files:
            file_path = root_path / file
            
            # Check if it's a data file
            if file_path.suffix.lower() not in data_extensions:
                continue
            
            # Classify file
            case_num, case_name = classify_file(file, file_path)
            
            # Extract metadata
            metadata = extract_file_metadata(file_path)
            metadata['case_number'] = case_num
            metadata['case_name'] = case_name
            
            catalog.append(metadata)
            logger.debug(f"Cataloged: {file} -> Case {case_num} ({case_name})")
    
    logger.info(f"Total files cataloged: {len(catalog)}")
    return catalog


def save_catalog_csv(catalog: List[Dict], output_path: Path):
    """Save catalog to CSV file."""
    if not catalog:
        logger.warning("Catalog is empty, nothing to save")
        return
    
    fieldnames = [
        'filename', 'path', 'case_number', 'case_name', 'format',
        'size_mb', 'telescope', 'frequency_ghz', 'has_header'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in catalog:
            row = {k: item.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    logger.info(f"Catalog saved to {output_path}")


def save_catalog_json(catalog: List[Dict], output_path: Path):
    """Save catalog to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Catalog saved to {output_path}")


def print_summary(catalog: List[Dict]):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("CATALOG SUMMARY")
    print("="*70)
    
    # Count by case
    case_counts = {}
    total_size = 0
    
    for item in catalog:
        case_name = item['case_name']
        case_counts[case_name] = case_counts.get(case_name, 0) + 1
        total_size += item.get('size_mb', 0)
    
    print(f"\nTotal files: {len(catalog)}")
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    print("\nFiles by case:")
    for case_name, count in sorted(case_counts.items()):
        print(f"  {case_name}: {count} files")
    
    # Count by format
    format_counts = {}
    for item in catalog:
        fmt = item.get('format', 'unknown')
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    print("\nFiles by format:")
    for fmt, count in sorted(format_counts.items()):
        print(f"  {fmt}: {count} files")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Catalog data files in Data/raw/ directory'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='Data/raw',
        help='Directory to scan (default: Data/raw)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Data/catalog.csv',
        help='Output CSV file (default: Data/catalog.csv)'
    )
    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Also save as JSON file (optional)'
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
    data_dir = project_root / args.data_dir
    output_path = project_root / args.output
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Scan and catalog
    catalog = scan_directory(data_dir)
    
    if not catalog:
        logger.warning("No data files found!")
        sys.exit(1)
    
    # Save catalog
    save_catalog_csv(catalog, output_path)
    
    if args.json:
        json_path = project_root / args.json
        save_catalog_json(catalog, json_path)
    
    # Print summary
    if args.summary:
        print_summary(catalog)
    else:
        logger.info(f"Cataloged {len(catalog)} files")
        logger.info(f"Use --summary to see detailed statistics")


if __name__ == '__main__':
    main()

