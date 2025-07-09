#!/usr/bin/env python3

import sys
sys.path.append('.')
import numpy as np
from pathlib import Path

def test_basic_chunk_processing():
    print("=== TESTING BASIC CHUNK PROCESSING ===")
    
    # Import required modules
    from DRAFTS.pipeline import _load_fil_chunk
    from DRAFTS.filterbank_io import get_obparams_fil
    from DRAFTS.preprocessing import downsample_data
    from DRAFTS import config
    
    fits_path = Path("Data/3100_0001_00_8bit.fil")
    
    # Load observation parameters
    print("Loading observation parameters...")
    try:
        get_obparams_fil(str(fits_path))
        print("✅ Observation parameters loaded")
        print(f"   FILE_LENG: {config.FILE_LENG}")
        print(f"   DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
        print(f"   DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
        print(f"   FREQ shape: {config.FREQ.shape if hasattr(config, 'FREQ') and config.FREQ is not None else 'N/A'}")
        print(f"   FREQ_RESO: {config.FREQ_RESO}")
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load chunk
    print("\nLoading chunk...")
    try:
        chunk_size = 5000  # Small chunk
        data_chunk = _load_fil_chunk(str(fits_path), 0, chunk_size)
        print(f"✅ Chunk loaded: shape {data_chunk.shape}")
    except Exception as e:
        print(f"❌ Error loading chunk: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test vstack
    print("\nTesting vstack...")
    try:
        data_chunk = np.vstack([data_chunk, data_chunk[::-1, :]])
        print(f"✅ Vstack completed: shape {data_chunk.shape}")
    except Exception as e:
        print(f"❌ Error in vstack: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test downsample
    print("\nTesting downsample...")
    try:
        data_chunk = downsample_data(data_chunk)
        print(f"✅ Downsample completed: shape {data_chunk.shape}")
    except Exception as e:
        print(f"❌ Error in downsample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test basic DM calculation (without numba)
    print("\nTesting basic processing...")
    try:
        height = config.DM_max - config.DM_min + 1
        width_total = data_chunk.shape[0] // config.DOWN_TIME_RATE
        print(f"✅ Basic calculations: height={height}, width_total={width_total}")
        
        # Don't call d_dm_time_g yet - it uses numba
        print("✅ Basic processing completed")
        return True
        
    except Exception as e:
        print(f"❌ Error in basic processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_chunk_processing()
