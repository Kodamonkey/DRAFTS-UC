# This module manages GPU-related logging helpers.

"""
Manejo de Logging para GPU en DRAFTS
===================================

Este módulo proporciona funciones para manejar los mensajes de GPU de manera
limpia y configurable, evitando el spam de mensajes técnicos.
"""

                          
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

                                        
GPU_VERBOSE = False                                                  

# This function sets GPU verbose.
def set_gpu_verbose(verbose: bool = False):
    """Configura si mostrar mensajes detallados de GPU."""
    global GPU_VERBOSE
    GPU_VERBOSE = verbose

# This function manages the GPU logging context.
def gpu_context(operation: str, suppress_messages: bool = True):
    """
    Context manager para operaciones GPU que suprime mensajes innecesarios.
    
    Args:
        operation: Descripción de la operación GPU
        suppress_messages: Si suprimir mensajes de CUDA
    """
    if suppress_messages:
                                                                    
        import os
        import sys
        from io import StringIO
        
                                 
        original_stderr = sys.stderr
        
        try:
                                                                
            sys.stderr = StringIO()
            yield
        finally:
                              
            sys.stderr = original_stderr
    else:
        yield

# This function logs GPU operation.
def log_gpu_operation(operation: str, success: bool = True, details: Optional[str] = None):
    """
    Log de operaciones GPU de manera limpia.
    
    Args:
        operation: Descripción de la operación
        success: Si la operación fue exitosa
        details: Detalles adicionales (solo si GPU_VERBOSE=True)
    """
    try:
        from .logging_config import get_global_logger
        global_logger = get_global_logger()
        
        if success:
            global_logger.gpu_info(f"{operation}")
            if GPU_VERBOSE and details:
                global_logger.gpu_info(f"{details}", level="DEBUG")
        else:
            global_logger.gpu_info(f"{operation}", level="ERROR")
    except ImportError:
                                  
        if success:
            logger.info(f"{operation}")
        else:
            logger.error(f"{operation}")

# This function logs GPU memory operation.
def log_gpu_memory_operation(operation: str, bytes_allocated: int = 0):
    """
    Log de operaciones de memoria GPU de manera simplificada.
    
    Args:
        operation: Tipo de operación ('alloc', 'free', 'init')
        bytes_allocated: Bytes involucrados
    """
    if not GPU_VERBOSE:
        return
    
    try:
        from .logging_config import get_global_logger
        global_logger = get_global_logger()
        
        if bytes_allocated > 0:
            size_mb = bytes_allocated / (1024 * 1024)
            global_logger.gpu_info(f"{operation}: {size_mb:.1f} MB", level="DEBUG")
        else:
            global_logger.gpu_info(f"{operation}", level="DEBUG")
    except ImportError:
                                  
        if bytes_allocated > 0:
            size_mb = bytes_allocated / (1024 * 1024)
            logger.debug(f"{operation}: {size_mb:.1f} MB")
        else:
            logger.debug(f"{operation}")

                                                
# This function filters cuda messages.
def filter_cuda_messages(message: str) -> bool:
    """
    Filtra mensajes CUDA para mostrar solo los relevantes.
    
    Args:
        message: Mensaje a filtrar
        
    Returns:
        True si el mensaje debe mostrarse, False si debe suprimirse
    """
                                           
    important_messages = [
        "CUDA error",
        "out of memory",
        "cudaMalloc failed",
        "cudaMemcpy failed"
    ]
    
                                    
    spam_messages = [
        "add pending dealloc",
        "dealloc: cuMemFree_v2",
        "init",
        "cudaMemcpy",
        "cudaMalloc"
    ]
    
                                            
    if any(important in message.lower() for important in important_messages):
        return True
    
                                          
    if any(spam in message.lower() for spam in spam_messages):
        return False
    
                                                            
    return GPU_VERBOSE 
