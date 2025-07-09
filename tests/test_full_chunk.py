#!/usr/bin/env python3

import sys
sys.path.append('.')
import numpy as np
from pathlib import Path

def test_chunk_processing():
    print("=== TESTING CHUNK PROCESSING PIPELINE ===")
    
    # Import pipeline functions
    from DRAFTS.pipeline import _load_fil_chunk, _process_single_chunk, _load_model, _load_class_model
    from DRAFTS.filterbank_io import get_obparams_fil
    from DRAFTS import config
    
    fits_path = Path("Data/3100_0001_00_8bit.fil")
    save_dir = Path("test_chunk_processing")
    save_dir.mkdir(exist_ok=True)
    
    # Load observation parameters
    print("Loading observation parameters...")
    try:
        get_obparams_fil(str(fits_path))
        print("✅ Observation parameters loaded")
        print(f"   FILE_LENG: {config.FILE_LENG}")
        print(f"   DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
        print(f"   DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        return False
    
    # Load models
    print("\nLoading models...")
    try:
        det_model = _load_model()
        cls_model = _load_class_model()
        print("✅ Models loaded")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False
    
    # Load chunk
    print("\nLoading chunk...")
    try:
        chunk_size = 10000  # Small chunk
        data_chunk = _load_fil_chunk(str(fits_path), 0, chunk_size)
        print(f"✅ Chunk loaded: shape {data_chunk.shape}")
    except Exception as e:
        print(f"❌ Error loading chunk: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test chunk processing
    print("\nTesting chunk processing...")
    try:
        csv_file = save_dir / "test.csv"
        
        result = _process_single_chunk(
            det_model, cls_model, data_chunk, fits_path, save_dir,
            chunk_idx=0, start_sample_global=0, csv_file=csv_file
        )
        
        print(f"✅ Chunk processing completed")
        print(f"   Result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Error in chunk processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_chunk_processing()
