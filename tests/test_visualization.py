#!/usr/bin/env python3

import sys
sys.path.append('.')
import numpy as np
from pathlib import Path

def test_visualization_generation():
    print("=== TESTING VISUALIZATION GENERATION ===")
    
    # Import required modules
    from DRAFTS.pipeline import _load_fil_chunk, _process_single_chunk, _load_model, _load_class_model
    from DRAFTS.filterbank_io import get_obparams_fil
    from DRAFTS import config
    
    fits_path = Path("Data/3100_0001_00_8bit.fil")
    save_dir = Path("test_visualization_debug")
    save_dir.mkdir(exist_ok=True)
    
    # Load observation parameters
    print("Loading observation parameters...")
    try:
        get_obparams_fil(str(fits_path))
        print("✅ Observation parameters loaded")
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        return False
    
    # Load models (but we'll skip them for a quick test)
    print("\nTesting visualization functions directly...")
    try:
        # Test import of visualization functions
        from DRAFTS.visualization import (
            save_plot,
            save_patch_plot,
            save_slice_summary,
            plot_waterfall_block,
        )
        from DRAFTS.image_utils import preprocess_img, postprocess_img
        print("✅ Visualization functions imported successfully")
        
        # Create test directories
        test_dirs = [
            save_dir / "waterfall_dispersion" / "test",
            save_dir / "waterfall_dedispersion" / "test", 
            save_dir / "Composite" / "test",
            save_dir / "Detections" / "test",
            save_dir / "Patches" / "test"
        ]
        
        for test_dir in test_dirs:
            test_dir.mkdir(parents=True, exist_ok=True)
            
        print("✅ Test directories created")
        
        # Test basic waterfall plot generation
        print("\nTesting waterfall plot generation...")
        test_data = np.random.rand(512, 100).astype(np.float32)  # Simple test data
        test_freq = np.linspace(1000, 1500, 100)
        
        plot_waterfall_block(
            data_block=test_data,
            freq=test_freq,
            time_reso=0.001,
            block_size=test_data.shape[0],
            block_idx=0,
            save_dir=test_dirs[0],
            filename="test_waterfall",
            normalize=True,
        )
        
        print("✅ Waterfall plot generated successfully")
        
        # Test detection plot generation
        print("\nTesting detection plot generation...")
        test_img = np.random.rand(512, 512).astype(np.float32)
        test_img_rgb = preprocess_img(test_img)
        test_img_rgb = postprocess_img(test_img_rgb)
        
        # Create fake detection data
        fake_conf = [0.8, 0.6]
        fake_boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
        fake_class_probs = [0.9, 0.7]
        
        save_plot(
            test_img_rgb,
            fake_conf,
            fake_boxes,
            fake_class_probs,
            test_dirs[3] / "test_detection.png",
            slice_idx=0,
            time_slice=1,
            band_name="Test Band",
            band_suffix="test",
            fits_stem="test",
            slice_len=512,
        )
        
        print("✅ Detection plot generated successfully")
        
        # Check if files were created
        created_files = list(save_dir.rglob("*.png"))
        print(f"\n✅ Created {len(created_files)} visualization files:")
        for file in created_files:
            print(f"   - {file.relative_to(save_dir)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization_generation()
    if success:
        print("\n✅ VISUALIZATION TEST PASSED!")
        print("The visualization functions are working correctly.")
        print("You can now run the full pipeline with confidence.")
    else:
        print("\n❌ VISUALIZATION TEST FAILED!")
        print("There are issues with the visualization functions.")
