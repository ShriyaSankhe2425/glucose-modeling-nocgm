import os
import numpy as np
from src.preprocessing import load_actiheart_data

def load_all_participants(data_dir='data/', window_size=4, model_name="nocgm_base"):
    """
    Loads and combines data for all participants in the dataset folder.

    Parameters:
        data_dir (str): Path to directory containing CSV files (1 per participant)
        window_size (int): Number of time steps per input sequence
        model_name (str): Controls feature selection in preprocessing

    Returns:
        X_all (np.array): Shape = (total_samples, window_size, num_features)
        y_all (np.array): Shape = (total_samples,)
    """
    X_all, y_all = [], []

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and "glucose_actiheart_integrated" in filename:
            csv_path = os.path.join(data_dir, filename)
            print(f"Loading {filename}...")

            try:
                # Call preprocessing function with model-specific feature selection
                X, y = load_actiheart_data(
                    csv_path,
                    window_size=window_size,
                    model_name=model_name
                )
                X_all.append(X)
                y_all.append(y)

            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    if len(X_all) == 0:
        raise RuntimeError("No participant data loaded — check file paths or errors above.")

    # Concatenate data from all participants into one dataset
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    print("Finished loading all participants.")
    print("Final X shape:", X_all.shape)
    print("Final y shape:", y_all.shape)

    return X_all, y_all
