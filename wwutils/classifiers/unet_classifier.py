"""Utilities to assign whisker IDs using a U-Net model.

This module provides a thin wrapper around a semantic segmentation model. The
model should take a single video frame as input and output a mask with a unique
label for each whisker. Whiskers are ordered from posterior to anterior with the
most posterior whisker receiving ID 1.

The U-Net model is assumed to be stored on disk in PyTorch format. Only very
small portions of videos are processed, so this step should complete quickly
once the model weights are loaded.

This module intentionally keeps the implementation lightweight. If PyTorch or the
required model file is not available, ``assign_whisker_ids`` will simply return
``None`` so that callers can fall back to other approaches.
"""

from __future__ import annotations

import os
from typing import Optional


def _load_model(model_path: str):
    """Load a U-Net style model if torch is available."""
    try:
        import torch
        from torch import nn  # noqa: F401
        
        # Try to load as TorchScript first, then as regular PyTorch model
        try:
            model = torch.jit.load(model_path)
        except:
            # Load regular PyTorch model - need to import UNet class
            # Import UNet from train_unet.py
            import sys
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            from train_unet import UNet
            
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model info from checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Determine number of classes from final layer
                final_layer_key = 'outc.weight'
                if final_layer_key in state_dict:
                    num_classes = state_dict[final_layer_key].shape[0]
                else:
                    num_classes = 4  # fallback default
                
                # Get class mapping if available
                class_mapping = checkpoint.get('class_to_wid_mapping', None)
                wid_mapping = checkpoint.get('wid_to_class_mapping', None)
            else:
                # Direct state dict
                state_dict = checkpoint
                # Try to infer number of classes
                for key in state_dict.keys():
                    if 'outc' in key and 'weight' in key:
                        num_classes = state_dict[key].shape[0]
                        break
                else:
                    num_classes = 4
                class_mapping = None
                wid_mapping = None
                
            model = UNet(3, num_classes)
            model.load_state_dict(state_dict)
            
            # Store mapping info in model for later use
            model.class_to_wid_mapping = class_mapping
            model.wid_to_class_mapping = wid_mapping
            
        model.eval()
        return model
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "U-Net model could not be loaded. Ensure PyTorch is installed and "
            f"the model exists at {model_path}. Error: {exc}"
        ) from exc


def assign_whisker_ids(video_file: str, tracked_parquet: str,
                        model_path: Optional[str] = None) -> Optional[str]:
    """Assign whisker IDs using a pre-trained U-Net model.

    Parameters
    ----------
    video_file:
        Path to the source video that was tracked.
    tracked_parquet:
        Path to the combined whisker tracking parquet file.
    model_path:
        Optional path to a PyTorch U-Net model. If ``None`` the function
        looks for ``unet_model.pt`` next to this script.

    Returns
    -------
    str or None
        Path to a new parquet file with updated ``wid`` column if the
        model was applied successfully. ``None`` is returned if the model cannot
        be loaded or an error occurs so that callers can fall back to another
        method.
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "unet_model.pt")

    try:
        model = _load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    try:
        import pandas as pd
        import torch
        import cv2
        import numpy as np

        df = pd.read_parquet(tracked_parquet)
        cap = cv2.VideoCapture(video_file)
        
        # Check if model has class mapping
        has_mapping = hasattr(model, 'class_to_wid_mapping') and model.class_to_wid_mapping is not None
        if has_mapping:
            print(f"✅ Using class-to-wid mapping: {model.class_to_wid_mapping}")
        else:
            print("⚠️  No class mapping found - using direct class values as wid")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        predicted_wids = []
        processed_frames = 0
        total_frames = len(df["fid"].unique())
        
        print(f"Processing {total_frames} unique frames...")
        
        last_frame = -1
        frame = None
        label_mask = None
        
        for i, fid in enumerate(sorted(df["fid"].unique())):
            if fid != last_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️  Could not read frame {fid}")
                    break
                    
                # Preprocess frame to match training
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(device)
                
                # Get model prediction
                with torch.no_grad():
                    logits = model(frame_tensor)
                    label_mask = logits.argmax(1).squeeze().cpu().numpy()
                
                last_frame = fid
                processed_frames += 1
                
                if processed_frames % 50 == 0:
                    print(f"  Processed {processed_frames}/{total_frames} frames ({processed_frames/total_frames*100:.1f}%)")
            
            # Extract prediction for each whisker in this frame
            frame_rows = df[df["fid"] == fid]
            for _, row in frame_rows.iterrows():
                # Sample along the entire whisker trajectory instead of just follicle
                whisker_pixels = []
                
                # Get whisker trajectory points from pixels_x and pixels_y
                try:
                    pixels_x = row["pixels_x"]
                    pixels_y = row["pixels_y"]
                    
                    # Check if we have valid pixel data
                    if pixels_x is not None and pixels_y is not None:
                        # Parse the pixel coordinates (they might be strings or arrays)
                        if isinstance(pixels_x, str):
                            pixels_x = eval(pixels_x)  # Convert string representation to list
                            pixels_y = eval(pixels_y)
                        # Handle numpy arrays or lists
                        elif hasattr(pixels_x, '__iter__'):
                            pixels_x = list(pixels_x)
                            pixels_y = list(pixels_y)
                        
                        # Sample mask values along the whisker trajectory
                        for px, py in zip(pixels_x, pixels_y):
                            px, py = int(px), int(py)
                            if 0 <= py < label_mask.shape[0] and 0 <= px < label_mask.shape[1]:
                                whisker_pixels.append(int(label_mask[py, px]))
                    else:
                        # Fallback to follicle position if no pixel trajectory available
                        x, y = int(row["follicle_x"]), int(row["follicle_y"])
                        if 0 <= y < label_mask.shape[0] and 0 <= x < label_mask.shape[1]:
                            whisker_pixels.append(int(label_mask[y, x]))
                except Exception as e:
                    # Fallback to follicle position if whisker pixels parsing fails
                    print(f"⚠️  Error parsing whisker pixels for frame {fid}: {e}")
                    x, y = int(row["follicle_x"]), int(row["follicle_y"])
                    if 0 <= y < label_mask.shape[0] and 0 <= x < label_mask.shape[1]:
                        whisker_pixels.append(int(label_mask[y, x]))
                
                if whisker_pixels:
                    # Use majority vote across all whisker pixels (excluding background class 0)
                    non_bg_pixels = [p for p in whisker_pixels if p != 0]  # Remove background predictions
                    if non_bg_pixels:
                        # Find most common non-background prediction
                        predicted_class = max(set(non_bg_pixels), key=non_bg_pixels.count)
                    else:
                        # All pixels were background, use most frequent overall
                        predicted_class = max(set(whisker_pixels), key=whisker_pixels.count)
                    
                    # Map class back to wid if mapping exists
                    if has_mapping and predicted_class in model.class_to_wid_mapping:
                        predicted_wid = model.class_to_wid_mapping[predicted_class]
                        # Handle special cases of string values
                        if isinstance(predicted_wid, str):
                            if "background" in predicted_wid.lower():
                                predicted_wid = -1  # Background -> invalid/no whisker (-1)
                            elif "unclassified" in predicted_wid.lower():
                                predicted_wid = -1  # Unclassified -> invalid/no whisker (-1)
                            else:
                                predicted_wid = -1  # Other string values -> invalid/no whisker (-1)
                    else:
                        predicted_wid = predicted_class
                        
                    predicted_wids.append(predicted_wid)
                else:
                    print(f"⚠️  No valid pixels found for whisker in frame {fid}")
                    predicted_wids.append(-1)  # Use -1 for no valid pixels
        
        cap.release()
        
        if len(predicted_wids) != len(df):
            print(f"❌ Mismatch: {len(predicted_wids)} predictions vs {len(df)} rows")
            return None

        # Create output dataframe with updated wid
        df_out = df.copy()
        df_out["wid"] = predicted_wids
        
        # Also keep original wid for comparison
        df_out["wid_original"] = df["wid"]
        
        out_file = tracked_parquet.replace(".parquet", "_unet.parquet")
        df_out.to_parquet(out_file)
        
        print(f"✅ Predictions saved to {out_file}")
        print(f"   Processed {processed_frames} frames with {len(predicted_wids)} whisker detections")
        
        # Show some statistics
        unique_original = df["wid"].unique()
        unique_predicted = df_out["wid"].unique()
        print(f"   Original WIDs: {sorted(unique_original)}")
        print(f"   Predicted WIDs: {sorted(unique_predicted)}")
        
        return out_file
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Command line interface for whisker ID assignment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Assign whisker IDs using a U-Net model")
    parser.add_argument("video_file", help="Path to video file")
    parser.add_argument("parquet_file", help="Path to tracking parquet file") 
    parser.add_argument("--model", help="Path to U-Net model file (default: unet_model.pt in script directory)")
    
    args = parser.parse_args()
    
    print(f"🎯 Assigning whisker IDs using U-Net classifier")
    print(f"   Video: {args.video_file}")
    print(f"   Parquet: {args.parquet_file}")
    print(f"   Model: {args.model or 'default (unet_model.pt)'}")
    print()
    
    result = assign_whisker_ids(args.video_file, args.parquet_file, args.model)
    
    if result:
        print(f"\n🎉 Success! Results saved to: {result}")
    else:
        print(f"\n❌ Failed to assign whisker IDs")
        exit(1)


if __name__ == "__main__":
    main()
