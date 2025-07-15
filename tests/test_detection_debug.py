#!/usr/bin/env python3

import sys
sys.path.append('.')
import torch
import numpy as np
from pathlib import Path
from DRAFTS.pipeline import _load_fil_chunk, _load_model, _load_class_model, _detect, _process_single_chunk
from DRAFTS.filterbank_io import get_obparams_fil
from DRAFTS import config

def test_detection_pipeline():
    print("=== TESTING DETECTION PIPELINE ===")
    
    # Set up configuration
    fits_path = Path("Data/3100_0001_00_8bit.fil")
    print(f"Testing with file: {fits_path}")
    
    # Load observation parameters
    print("Loading observation parameters...")
    try:
        get_obparams_fil(str(fits_path))
        print("✅ Observation parameters loaded successfully")
        print(f"FILE_LENG: {config.FILE_LENG}")
        print(f"FREQ shape: {len(config.FREQ) if hasattr(config, 'FREQ') else 'N/A'}")
    except Exception as e:
        print(f"❌ Error loading observation parameters: {e}")
        return False
    
    # Load models
    print("\nLoading models...")
    try:
        det_model = _load_model()
        cls_model = _load_class_model()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False
    
    # Load a small chunk for testing
    print("\nLoading test chunk...")
    try:
        chunk_size = 10000  # Small chunk for testing
        data_chunk = _load_fil_chunk(str(fits_path), 0, chunk_size)
        print(f"✅ Chunk loaded: shape {data_chunk.shape}")
    except Exception as e:
        print(f"❌ Error loading chunk: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test single detection
    print("\nTesting detection on chunk...")
    try:
        # Create a small subset for testing
        test_data = data_chunk[:1000]  # Even smaller for quick test
        print(f"Original test_data shape: {test_data.shape}")
        
        # Apply preprocessing similar to _process_single_chunk
        test_data = np.vstack([test_data, test_data[::-1, :]])
        print(f"After vstack test_data shape: {test_data.shape}")
        
        # Import required functions
        from DRAFTS.preprocessing import downsample_data
        from DRAFTS.dedispersion import d_dm_time_g
        from DRAFTS.image_utils import preprocess_img
        
        print(f"Before downsample_data: {test_data.shape}")
        print(f"config.DOWN_FREQ_RATE: {getattr(config, 'DOWN_FREQ_RATE', 'N/A')}")
        print(f"Expected shape for downsample_data: (time, pol, freq)")
        
        test_data = downsample_data(test_data)
        print(f"✅ Data preprocessed: shape {test_data.shape}")
        
        # Calculate DM time
        height = config.DM_max - config.DM_min + 1
        width_total = test_data.shape[0] // config.DOWN_TIME_RATE
        dm_time = d_dm_time_g(test_data, height=height, width=width_total)
        
        print(f"✅ DM time calculated: shape {dm_time.shape}")
        
        # Test detection on first band
        if dm_time.shape[0] > 0:
            band_img = dm_time[0]  # First band
            print(f"Band image shape: {band_img.shape}")
            
            # Test preprocessing
            img_tensor = preprocess_img(band_img)
            print(f"✅ Image tensor preprocessed: shape {img_tensor.shape}")
            
            # Test detection
            top_conf, top_boxes = _detect(det_model, img_tensor)
            print(f"✅ Detection completed")
            print(f"   Confidences: {len(top_conf) if top_conf else 0}")
            print(f"   Boxes: {len(top_boxes) if top_boxes else 0}")
            
            if top_boxes and len(top_boxes) > 0:
                print(f"   First box: {top_boxes[0]}")
                print(f"   First box type: {type(top_boxes[0])}")
                print(f"   First box length: {len(top_boxes[0])}")
                
                # Test the problematic operations
                box = top_boxes[0]
                try:
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    print(f"   ✅ Box center calculation successful: ({center_x}, {center_y})")
                    
                    box_int = tuple(map(int, box))
                    print(f"   ✅ Box integer conversion successful: {box_int}")
                    
                except Exception as e:
                    print(f"   ❌ Error in box operations: {e}")
                    print(f"   Box details: {box}, type: {type(box)}, len: {len(box) if hasattr(box, '__len__') else 'N/A'}")
                    return False
            else:
                print("   No detections found (this is normal)")
                
        print("✅ Detection pipeline test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in detection pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detection_pipeline()
    if success:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ TESTS FAILED!")
