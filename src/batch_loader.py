import os
import numpy as np
from src.preprocessing import load_actiheart_data

def load_all_participants(data_dir='data/', window_size=4):
    """
    Loads and processes Actiheart data from all valid participant CSVs.

    Parameters:
        data_dir (str): Path to the folder containing all participant data files.
        window_size (int): How many time steps to include in each input sequence.

    Returns:
        X_all (np.ndarray): Combined input features from all participants.
                            Shape = (total_samples, window_size, features)
        y_all (np.ndarray): Combined target glucose values.
                            Shape = (total_samples,)
    """

    X_all, y_all = [], []

    # Loop through every file in the data directory
    for filename in os.listdir(data_dir):
        # Only load files that are CSV and contain "glucose_actiheart" in name
        if filename.endswith(".csv") and "glucose_actiheart" in filename:
            csv_path = os.path.join(data_dir, filename)
            print(f"📂 Loading {filename}...")

            try:
                # Load and process one participant
                X, y = load_actiheart_data(csv_path, window_size=window_size)
                X_all.append(X)
                y_all.append(y)
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    # Combine all participants' data
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Summary
    print("✅ Finished loading all participants.")
    print("🧠 Final X shape:", X_all.shape)
    print("🎯 Final y shape:", y_all.shape)

    return X_all, y_all
