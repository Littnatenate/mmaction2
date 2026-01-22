import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm

# --- Configuration ---
# Define where your dataset lives
DATA_ROOT = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\FYP NYP PROJECT FINAL\FYP-NYP-PROJECT\mmaction2\data\5xSTS"

# Define your input folder names and output file names
SPLITS = {
    "train": ("processed_train", "train_annotations.pkl"),
    "val":   ("processed_val", "val_annotations.pkl")
}

def extract_pose_from_shishao(data):
    """
    Handles data loading from the Shi_Shao format (List of Dicts) 
    OR standard numpy arrays.
    """
    # Case 1: Standard Numpy Array (Old format)
    # Shape: (Frames, 99) or (Frames, 33, 3)
    if isinstance(data, np.ndarray):
        # Flattened? Reshape it.
        if len(data.shape) == 2 and data.shape[1] == 99:
            return data.reshape(-1, 33, 3)
        # Already 3D? Return it.
        elif len(data.shape) == 3 and data.shape[1] == 33:
            return data
        else:
            # Fallback for 25 joints if older data exists
            if data.shape[1] == 75 or (len(data.shape)==3 and data.shape[1]==25):
                print(f"Warning: Found 25-joint data. Padding to 33.")
                # Logic to pad would go here, but for now we skip/warn
                return None 
            return data

    # Case 2: Shi_Shao Format (List of Dictionaries)
    # [ {'pose': (33,3), 'face': ...}, {'pose': ...} ]
    elif isinstance(data, list) or isinstance(data, np.ndarray): # sometimes np.load returns an object array wrapper
        frames_pose = []
        try:
            # If it's a numpy object array, convert to list
            if isinstance(data, np.ndarray): 
                data = data.tolist()
                
            for frame in data:
                # Extract 'pose' key
                if 'pose' in frame:
                    frames_pose.append(frame['pose'])
                else:
                    # Handle case where keys might be missing
                    frames_pose.append(np.zeros((33, 3)))
            
            return np.array(frames_pose) # Shape: (Frames, 33, 3)
        except Exception as e:
            print(f"Error parsing Shi_Shao format: {e}")
            return None

    return None

def generate_split(split_name, input_folder_name, output_filename):
    full_input_path = os.path.join(DATA_ROOT, input_folder_name)
    full_output_path = os.path.join(DATA_ROOT, output_filename)

    if not os.path.exists(full_input_path):
        print(f"⚠️  Skipping {split_name}: Folder '{full_input_path}' not found.")
        return

    print(f"\n🚀 Processing {split_name} set from: {input_folder_name}")
    
    # 1. Get Class Labels (Folder Names)
    # Filter out non-folders
    action_labels = sorted([d for d in os.listdir(full_input_path) if os.path.isdir(os.path.join(full_input_path, d))])
    label_map = {label: i for i, label in enumerate(action_labels)}
    
    print(f"   Found {len(action_labels)} classes: {action_labels}")
    
    annotations = []
    
    # 2. Process Each Class
    for action_name in action_labels:
        action_path = os.path.join(full_input_path, action_name)
        file_list = [f for f in os.listdir(action_path) if f.endswith('.npy')]
        
        for npy_file in tqdm(file_list, desc=f"   Class '{action_name}'"):
            npy_path = os.path.join(action_path, npy_file)
            
            try:
                # Load the raw data
                raw_data = np.load(npy_path, allow_pickle=True)
                
                # Extract CLEAN (Frames, 33, 3) pose data
                pose_data = extract_pose_from_shishao(raw_data)
                
                if pose_data is None:
                    print(f"   ❌ Skipping bad file: {npy_file}")
                    continue
                
                num_frames = pose_data.shape[0]
                
                # MMAction2 expects: (Person, Frames, Joints, Coords)
                # We add the 'Person' dimension (Axis 0)
                # Shape: (1, T, 33, 3)
                final_keypoints = np.expand_dims(pose_data, axis=0).astype(np.float32)
                
                # Create the Annotation Dictionary
                video_id = os.path.splitext(npy_file)[0]
                sample = {
                    'frame_dir': video_id,
                    'total_frames': num_frames,
                    'img_shape': (1080, 1920), # Placeholder resolution
                    'original_shape': (1080, 1920),
                    'label': label_map[action_name],
                    'keypoint': final_keypoints
                }
                
                annotations.append(sample)
                
            except Exception as e:
                print(f"   ❌ Error processing {npy_file}: {e}")

    # 3. Save the Result
    with open(full_output_path, 'wb') as f:
        pickle.dump(annotations, f)
    
    print(f"✅ Saved {len(annotations)} samples to {output_filename}")

if __name__ == "__main__":
    print("=== MMAction2 Data Preparer (Universal) ===")
    print(f"Root Directory: {DATA_ROOT}")
    
    # Run both splits automatically
    for split, (folder, outfile) in SPLITS.items():
        generate_split(split, folder, outfile)
    
    print("\n🎉 All Done!")
