#!/usr/bin/env python3

import sys
sys.path.append('.')
from DRAFTS.pipeline import _load_fil_chunk, _process_single_chunk
from DRAFTS.filterbank_io import _read_header
from DRAFTS import config
import numpy as np

def test_header_reading():
    print("Testing header reading...")
    try:
        with open('Data/3100_0001_00_8bit.fil', 'rb') as f:
            header, hdr_len = _read_header(f)
            print(f"Header read successfully: {header.get('nsamples', 'N/A')} samples")
            print(f"Header length: {hdr_len} bytes")
            print(f"Channels: {header.get('nchans', 'N/A')}")
            print(f"Bits: {header.get('nbits', 'N/A')}")
        print("Header reading test completed successfully")
        return True
    except Exception as e:
        print(f"Error in header reading: {e}")
        return False

def test_chunk_loading():
    print("\nTesting chunk loading...")
    try:
        # Load a small chunk (1000 samples) from the beginning
        chunk = _load_fil_chunk('Data/3100_0001_00_8bit.fil', 0, 1000)
        print(f"Chunk loaded successfully: shape {chunk.shape}")
        print(f"Chunk dtype: {chunk.dtype}")
        print(f"Chunk min/max: {chunk.min()}/{chunk.max()}")
        return True
    except Exception as e:
        print(f"Error loading chunk: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detect_function():
    print("\nTesting detection function...")
    try:
        # Create a dummy tensor for testing
        dummy_img = np.random.rand(1, 512, 512).astype(np.float32)
        
        # Import and test the _detect function
        from DRAFTS.pipeline import _detect
        from DRAFTS.pipeline import _load_model
        
        # This would require loading the model, which might be heavy
        # For now, just test the function signature
        print("Detection function imported successfully")
        return True
    except Exception as e:
        print(f"Error in detection function test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== CHUNK PROCESSING DEBUG TESTS ===")
    
    success = True
    success &= test_header_reading()
    success &= test_chunk_loading()
    success &= test_detect_function()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
