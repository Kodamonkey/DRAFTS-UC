#!/usr/bin/env python3

import sys
sys.path.append('.')

def test_imports():
    print("=== TESTING IMPORTS ===")
    
    try:
        print("Importing numpy...")
        import numpy as np
        print("✅ numpy imported")
        
        print("Importing config...")
        from DRAFTS import config
        print("✅ config imported")
        
        print("Importing filterbank_io...")
        from DRAFTS.filterbank_io import get_obparams_fil
        print("✅ filterbank_io imported")
        
        print("Importing preprocessing...")
        from DRAFTS.preprocessing import downsample_data
        print("✅ preprocessing imported")
        
        print("Importing dedispersion...")
        from DRAFTS.dedispersion import d_dm_time_g
        print("✅ dedispersion imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
