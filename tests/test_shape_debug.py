#!/usr/bin/env python3

import sys
sys.path.append('.')
import numpy as np
from DRAFTS.pipeline import _load_fil_chunk

def test_data_shape():
    print("=== TESTING DATA SHAPE ===")
    
    # Load a small chunk for testing
    print("Loading test chunk...")
    try:
        chunk_size = 1000  # Small chunk for testing
        data_chunk = _load_fil_chunk('Data/3100_0001_00_8bit.fil', 0, chunk_size)
        print(f"✅ Chunk loaded: shape {data_chunk.shape}")
        print(f"   Expected: (time, nifs, nchans)")
        print(f"   Actual: {data_chunk.shape}")
        print(f"   Number of dimensions: {data_chunk.ndim}")
        
        if data_chunk.ndim == 3:
            print(f"   Time samples: {data_chunk.shape[0]}")
            print(f"   IFs (polarizations): {data_chunk.shape[1]}")
            print(f"   Channels: {data_chunk.shape[2]}")
        elif data_chunk.ndim == 2:
            print(f"   Time samples: {data_chunk.shape[0]}")
            print(f"   Channels: {data_chunk.shape[1]}")
            print("   ❌ Missing IFs dimension!")
        
        return True
    except Exception as e:
        print(f"❌ Error loading chunk: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_shape()
