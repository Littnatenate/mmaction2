import os
import numpy as np
import pickle

# --- Configuration ---
# You will need to run this script TWICE.

# RUN 1: For TRAINING data
DATA_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\FYP NYP PROJECT FINAL\FYP-NYP-PROJECT\mmaction2\data\5xSTS\processed_val"
OUTPUT_FILE = "val_annotations.pkl"

# RUN 2: For VALIDATION data (uncomment the two lines below and comment out the two lines above)
# DATA_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\FYP NYP PROJECT FINAL\FYP-NYP-PROJECT\mmaction2\data\5xSTS\processed_val"
# OUTPUT_FILE = "val_annotations.pkl"

# This is the folder where the final .pkl file will be saved.
SAVE_DIR = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\FYP NYP PROJECT FINAL\FYP-NYP-PROJECT\mmaction2\data\5xSTS"

# --- Main Script Logic ---
def generate_annotations(root_path, out_file):
    annotations = []
    action_labels = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    label_map = {label: i for i, label in enumerate(action_labels)}

    print(f"Found {len(action_labels)} actions: {action_labels}")
    print("Starting 3-channel (X, Y, Z) annotation generation...")

    for action_name in action_labels:
        action_path = os.path.join(root_path, action_name)
        print(f"Processing action: {action_name}")

        for npy_file in os.listdir(action_path):
            if not npy_file.endswith('.npy'):
                continue

            video_name = os.path.splitext(npy_file)[0]
            npy_path = os.path.join(action_path, npy_file)
            
            raw_data = np.load(npy_path)
            num_frames = raw_data.shape[0]

            # Reshape to (num_frames, 25 joints, 3 coords)
            keypoints_3d = raw_data.reshape((num_frames, 25, 3))

            # --- THIS IS THE CHANGE ---
            # We now use the full keypoints_3d array to include X, Y, and Z.
            # Final shape: (1 person, num_frames, 25 joints, 3 coords)
            final_keypoints = np.expand_dims(keypoints_3d, axis=0).astype(np.float32)

            sample = {
                'frame_dir': video_name,
                'total_frames': num_frames,
                'keypoint': final_keypoints,
                'label': label_map[action_name]
            }
            annotations.append(sample)

    print(f"\nGenerated {len(annotations)} total annotations.")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    final_output_path = os.path.join(SAVE_DIR, out_file)

    with open(final_output_path, 'wb') as f:
        pickle.dump(annotations, f)
    
    print(f"✅ Successfully saved annotations to {final_output_path}")

if __name__ == '__main__':
    generate_annotations(DATA_PATH, OUTPUT_FILE)