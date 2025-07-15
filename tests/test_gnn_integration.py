#!/usr/bin/env python3
"""
Test script to verify GNN integration with WhiskiWrap pipeline
"""

import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

def create_test_data():
    """Create synthetic whisker tracking data for testing"""
    np.random.seed(42)
    
    # Create test data with temporal consistency issues
    n_frames = 100
    n_whiskers = 5
    
    data = []
    for frame in range(n_frames):
        for whisker in range(n_whiskers):
            # Create whisker positions with some noise
            base_x = 50 + whisker * 20 + np.random.normal(0, 2)
            base_y = 100 + whisker * 15 + np.random.normal(0, 2)
            
            # Create some ID swapping in the middle frames to test reassignment
            if 30 <= frame <= 40 and whisker in [1, 2]:
                wid = 2 if whisker == 1 else 1  # Swap IDs
            else:
                wid = whisker
            
            data.append({
                'fid': frame,
                'wid': wid,
                'follicle_x': base_x,
                'follicle_y': base_y,
                'tip_x': base_x + 50 + np.random.normal(0, 5),
                'tip_y': base_y + 30 + np.random.normal(0, 5),
                'angle': 45 + whisker * 10 + np.random.normal(0, 5),
                'curvature': 0.1 + whisker * 0.05 + np.random.normal(0, 0.02),
                'length': 60 + whisker * 5 + np.random.normal(0, 3),
                'face_side': 'left' if whisker < 3 else 'right',
                'score': 0.8 + np.random.normal(0, 0.1)
            })
    
    return pd.DataFrame(data)

def test_gnn_integration():
    """Test the GNN integration"""
    print("Testing GNN integration...")
    
    # Create test data
    df = create_test_data()
    print(f"Created test data with {len(df)} rows")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        df.to_parquet(tmp.name, index=False)
        test_file = tmp.name
    
    try:
        # Test import
        print("Testing import...")
        from wwutils.classifiers.gnn_classifier import reassign_whisker_ids_gnn
        print("✓ Import successful")
        
        # Test function call
        print("Testing GNN reassignment...")
        df_reassigned = reassign_whisker_ids_gnn(
            df, 
            model_path=None,
            n_epochs=5,  # Small number for testing
            temporal_window=5,
            train_split=0.8,
            verbose=True
        )
        
        print(f"✓ GNN reassignment successful")
        print(f"  Original shape: {df.shape}")
        print(f"  Reassigned shape: {df_reassigned.shape}")
        
        # Check if IDs were reassigned
        original_ids = df.groupby('fid')['wid'].apply(list)
        reassigned_ids = df_reassigned.groupby('fid')['wid'].apply(list)
        
        changes = 0
        for fid in original_ids.index:
            if fid in reassigned_ids.index:
                if original_ids[fid] != reassigned_ids[fid]:
                    changes += 1
        
        print(f"  Frames with ID changes: {changes}")
        
        # Save results
        output_file = test_file.replace('.parquet', '_gnn.parquet')
        df_reassigned.to_parquet(output_file, index=False)
        print(f"  Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == "__main__":
    success = test_gnn_integration()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
