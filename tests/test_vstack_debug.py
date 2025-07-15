#!/usr/bin/env python3

import sys
sys.path.append('.')
import numpy as np

def test_vstack_operations():
    print("=== TESTING VSTACK OPERATIONS ===")
    
    # Simulate 3D data: (time, nifs, nchans)
    print("Creating test data...")
    test_data = np.random.rand(100, 1, 512)  # 100 time samples, 1 IF, 512 channels
    print(f"Original data shape: {test_data.shape}")
    
    # Test vstack operation
    print("\nTesting vstack operation...")
    try:
        # This is what the original code does
        stacked_data = np.vstack([test_data, test_data[::-1, :]])
        print(f"After vstack shape: {stacked_data.shape}")
        print(f"Expected: (200, 1, 512)")
        
        # Test vstack with different approaches
        print("\nTesting alternative vstack approaches...")
        
        # Approach 1: concatenate along time axis (axis=0)
        concat_data = np.concatenate([test_data, test_data[::-1, :]], axis=0)
        print(f"Concatenate axis=0 shape: {concat_data.shape}")
        
        # Approach 2: vstack might be wrong for 3D data
        print("Testing vstack on 3D data...")
        # vstack is equivalent to concatenate along axis=0 for 3D
        # But let's see what happens
        
        return True
    except Exception as e:
        print(f"❌ Error in vstack operations: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_downsample_data_shape():
    print("\n=== TESTING DOWNSAMPLE_DATA EXPECTATIONS ===")
    
    from DRAFTS.preprocessing import downsample_data
    
    # Test what shape downsample_data expects
    print("Testing downsample_data with different shapes...")
    
    # Test 1: 3D data (time, nifs, nchans)
    try:
        test_data_3d = np.random.rand(100, 1, 512)
        print(f"Testing 3D data shape: {test_data_3d.shape}")
        result = downsample_data(test_data_3d)
        print(f"✅ 3D data worked, result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"❌ 3D data failed: {e}")
        
    # Test 2: 2D data (time, freq)
    try:
        test_data_2d = np.random.rand(100, 512)
        print(f"Testing 2D data shape: {test_data_2d.shape}")
        result = downsample_data(test_data_2d)
        print(f"✅ 2D data worked, result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"❌ 2D data failed: {e}")
        
    return False

if __name__ == "__main__":
    success = True
    success &= test_vstack_operations()
    success &= test_downsample_data_shape()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
