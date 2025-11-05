# This module manages GPU-related logging helpers.

"""
GPU logging utilities for DRAFTS
================================

This module provides functions to handle GPU messages cleanly and
configurably, avoiding spammy technical logs.
"""

                          
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

                                        
GPU_VERBOSE = False                                                  

def set_gpu_verbose(verbose: bool = False):
    """Configure whether to display detailed GPU messages."""
    global GPU_VERBOSE
    GPU_VERBOSE = verbose

def gpu_context(operation: str, suppress_messages: bool = True):
    """Context manager for GPU operations that suppresses unnecessary messages.

    Args:
        operation: Description of the GPU operation
        suppress_messages: Whether to suppress CUDA messages
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

def log_gpu_operation(
    operation: str, success: bool = True, details: Optional[str] = None
):
    """Cleanly log GPU operations.

    Args:
        operation: Description of the operation
        success: Whether the operation succeeded
        details: Additional details (only if ``GPU_VERBOSE=True``)
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

def log_gpu_memory_operation(operation: str, bytes_allocated: int = 0):
    """Log GPU memory operations in a simplified way.

    Args:
        operation: Type of operation ('alloc', 'free', 'init')
        bytes_allocated: Bytes involved
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

                                                
def filter_cuda_messages(message: str) -> bool:
    """Filter CUDA messages to show only the relevant ones.

    Args:
        message: Message to filter

    Returns:
        True if the message should be shown, False if it should be suppressed
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
