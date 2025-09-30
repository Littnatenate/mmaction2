# This is a simple script to check the shapes of keypoints in a .pkl file.
# It does not take any command-line arguments.

import pickle
import numpy as np

print("--- Starting data inspection script (v2) ---")

# --- IMPORTANT: Change this path to your annotation file ---
# You will need to run this twice: once for train and once for val.
annotation_path = 'data/5xSTS/val_annotations.pkl'

print(f"Checking file: {annotation_path}")

try:
    with open(annotation_path, 'rb') as f:
        # Load the pickle file
        annotations = pickle.load(f)

    # --- SCRIPT FIX ---
    # The error showed that the .pkl file is a LIST directly.
    # The previous script assumed it was a dictionary. This is now corrected.
    # We now directly use the loaded data as our list of annotations.

    print(f"Found {len(annotations)} samples.")
    is_problem_found = False

    for i, sample in enumerate(annotations):
        # Check if 'keypoint' data exists for the sample
        if 'keypoint' not in sample:
            print(f"--> WARNING: Sample at index {i} has no 'keypoint' data.")
            continue
            
        keypoint_shape = sample['keypoint'].shape
        
        # A correct shape should have 4 dimensions.
        if len(keypoint_shape) != 4:
            print(f"--> PROBLEM FOUND: Sample at index {i} has an incorrect shape: {keypoint_shape}")
            is_problem_found = True
            # You can also print the video name if you stored it
            if 'frame_dir' in sample:
                print(f"    Video source: {sample['frame_dir']}")

    if not is_problem_found:
        print("\n✅ All samples in this file have the correct 4-dimensional shape.")
    else:
        print("\n❌ Problems were found. Please check the output above.")


except FileNotFoundError:
    print(f"--> ERROR: The file was not found at: {annotation_path}")
    print("    Please make sure the path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- Script finished ---")