#!/usr/bin/env python3
"""
Script CLI para procesar archivos grandes usando chunking.

Este script divide archivos .fil o .fits grandes en trozos manejables,
procesa cada trozo con el pipeline estándar, y combina los resultados.
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import fire
import psutil
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.stream_fil import stream_filterbank, write_temp_fil, cleanup_temp_files
from utils.stream_fits import stream_fits, write_temp_fits, estimate_fits_memory_usage
from DRAFTS.core.pipeline import run_pipeline
from DRAFTS.core import config

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('chunking_process.log')
        ]
    )


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def calculate_chunk_size(
    file_path: Path,
    chunk_samples: Optional[int] = None,
    chunk_sec: Optional[float] = None,
    max_memory_gb: float = 2.0
) -> int:
    """
    Calculate optimal chunk size based on parameters.
    
    Args:
        file_path: Path to input file
        chunk_samples: Number of samples per chunk
        chunk_sec: Duration in seconds per chunk
        max_memory_gb: Maximum memory usage per chunk in GB
    
    Returns:
        int: Optimal chunk size in samples
    """
    if chunk_samples is not None:
        return chunk_samples
    
    if chunk_sec is not None:
        # Estimate tsamp from file
        if file_path.suffix.lower() == '.fits':
            try:
                from utils.stream_fits import get_fits_info
                info = get_fits_info(file_path)
                tsamp = info['tsamp']
            except Exception:
                tsamp = 0.000064  # Default value
        else:
            # For .fil files, we'll need to read header
            try:
                with open(file_path, 'rb') as f:
                    # Simple header reading for tsamp
                    # This is a simplified approach
                    tsamp = 0.000064  # Default value
            except Exception:
                tsamp = 0.000064  # Default value
        
        return int(chunk_sec / tsamp)
    
    # Default chunk size based on memory
    return int(max_memory_gb * 1024**3 / (512 * 4))  # Rough estimate


def process_large_file(
    input_path: str,
    chunk_samples: Optional[int] = None,
    chunk_sec: Optional[float] = None,
    out_dir: Optional[str] = None,
    max_memory_gb: float = 2.0,
    temp_dir: Optional[str] = None,
    log_level: str = "INFO"
) -> None:
    """
    Process a large file using chunking.
    
    Args:
        input_path: Path to input file (.fil or .fits)
        chunk_samples: Number of samples per chunk
        chunk_sec: Duration in seconds per chunk
        out_dir: Output directory for results
        max_memory_gb: Maximum memory usage per chunk in GB
        temp_dir: Directory for temporary files
        log_level: Logging level
    """
    setup_logging(log_level)
    
    file_path = Path(input_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output directory
    if out_dir is None:
        out_dir = Path("./Results/Chunked") / file_path.stem
    else:
        out_dir = Path(out_dir) / file_path.stem
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine temporary directory
    if temp_dir is None:
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 INICIANDO PROCESAMIENTO DE ARCHIVO GRANDE")
    logger.info(f"📁 Archivo de entrada: {file_path}")
    logger.info(f"📊 Directorio de salida: {out_dir}")
    logger.info(f"🗂️  Directorio temporal: {temp_dir}")
    
    # Calculate chunk size
    chunk_size = calculate_chunk_size(file_path, chunk_samples, chunk_sec, max_memory_gb)
    logger.info(f"📏 Tamaño de trozo: {chunk_size} muestras")
    
    # Estimate memory usage
    initial_memory = get_memory_usage()
    logger.info(f"💾 Memoria inicial: {initial_memory:.2f} GB")
    
    # Determine file type and setup streaming
    if file_path.suffix.lower() == '.fil':
        logger.info("📋 Detectado archivo SIGPROC filterbank (.fil)")
        stream_gen = stream_filterbank(file_path, chunk_size)
        write_temp_func = write_temp_fil
    elif file_path.suffix.lower() == '.fits':
        logger.info("📋 Detectado archivo PSRFITS (.fits)")
        stream_gen = stream_fits(file_path, chunk_size)
        write_temp_func = write_temp_fits
    else:
        raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")
    
    # Process chunks
    temp_files = []
    chunk_results = []
    start_time = time.time()
    
    try:
        for chunk_idx, block in enumerate(stream_gen):
            chunk_start_time = time.time()
            chunk_memory_before = get_memory_usage()
            
            logger.info(f"🔄 Procesando trozo {chunk_idx + 1}")
            logger.info(f"   📊 Forma del bloque: {block.shape}")
            logger.info(f"   💾 Memoria antes: {chunk_memory_before:.2f} GB")
            
            # Write temporary file
            temp_file = write_temp_func(block, {}, chunk_idx, temp_dir=temp_dir)
            temp_files.append(temp_file)
            
            # Create chunk-specific output directory
            chunk_out_dir = out_dir / f"chunk_{chunk_idx:04d}"
            chunk_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Process with standard pipeline
            try:
                # Temporarily modify config for this chunk
                original_targets = config.FRB_TARGETS.copy()
                config.FRB_TARGETS = [temp_file.stem]
                
                # Run pipeline
                run_pipeline()
                
                # Restore original config
                config.FRB_TARGETS = original_targets
                
                chunk_results.append({
                    'chunk_idx': chunk_idx,
                    'status': 'success',
                    'temp_file': temp_file,
                    'output_dir': chunk_out_dir
                })
                
                logger.info(f"✅ Trozo {chunk_idx + 1} procesado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error procesando trozo {chunk_idx + 1}: {e}")
                chunk_results.append({
                    'chunk_idx': chunk_idx,
                    'status': 'error',
                    'error': str(e),
                    'temp_file': temp_file
                })
            
            # Cleanup
            del block
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            chunk_memory_after = get_memory_usage()
            chunk_time = time.time() - chunk_start_time
            
            logger.info(f"   💾 Memoria después: {chunk_memory_after:.2f} GB")
            logger.info(f"   ⏱️  Tiempo del trozo: {chunk_time:.1f} s")
            
            # Clean up temporary file
            try:
                temp_file.unlink()
                temp_files.remove(temp_file)
                logger.debug(f"🗑️  Archivo temporal eliminado: {temp_file}")
            except Exception as e:
                logger.warning(f"⚠️  No se pudo eliminar {temp_file}: {e}")
        
        # Get header from generator (PEP 380)
        header = stream_gen.send(None)
        logger.info(f"📋 Header del archivo: {len(header)} parámetros")
        
    except Exception as e:
        logger.error(f"❌ Error durante el procesamiento: {e}")
        raise
    
    finally:
        # Final cleanup
        cleanup_temp_files(temp_files)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    successful_chunks = sum(1 for r in chunk_results if r['status'] == 'success')
    failed_chunks = len(chunk_results) - successful_chunks
    
    logger.info("🎉 PROCESAMIENTO COMPLETADO")
    logger.info(f"   📊 Total de trozos: {len(chunk_results)}")
    logger.info(f"   ✅ Trozos exitosos: {successful_chunks}")
    logger.info(f"   ❌ Trozos fallidos: {failed_chunks}")
    logger.info(f"   ⏱️  Tiempo total: {total_time:.1f} s")
    logger.info(f"   💾 Memoria final: {final_memory:.2f} GB")
    logger.info(f"   📁 Resultados en: {out_dir}")
    
    # Write summary report
    summary_file = out_dir / "chunking_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CHUNKING PROCESS SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input file: {file_path}\n")
        f.write(f"Chunk size: {chunk_size} samples\n")
        f.write(f"Total chunks: {len(chunk_results)}\n")
        f.write(f"Successful chunks: {successful_chunks}\n")
        f.write(f"Failed chunks: {failed_chunks}\n")
        f.write(f"Total time: {total_time:.1f} s\n")
        f.write(f"Initial memory: {initial_memory:.2f} GB\n")
        f.write(f"Final memory: {final_memory:.2f} GB\n")
        f.write("\nCHUNK DETAILS:\n")
        for result in chunk_results:
            f.write(f"Chunk {result['chunk_idx']}: {result['status']}\n")
            if result['status'] == 'error':
                f.write(f"  Error: {result['error']}\n")
    
    logger.info(f"📋 Resumen guardado en: {summary_file}")


def main():
    """Main CLI entry point using fire."""
    fire.Fire(process_large_file)


if __name__ == "__main__":
    main() 
