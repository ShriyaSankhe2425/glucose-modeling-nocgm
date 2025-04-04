import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_actiheart_data(csv_path, window_size=4, normalize=True):
    """
    Loads one participant's Actiheart data and prepares it for machine learning.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        window_size (int): Number of 15-minute time steps in one input sequence.
        normalize (bool): Whether to normalize input features using z-score.

    Returns:
        X (np.ndarray): 3D array of shape (samples, window_size, features)
                        Used as input to models like LSTM.
        y (np.ndarray): 1D array of glucose values (Detrended) as target.
    """

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Keep only valid rows (mask == False)
    df = df[df['mask'] == False].reset_index(drop=True)

    # Define input features and target column
    features = ['Activity', 'BPM', 'RMSSD']
    target = 'Detrended'

    # Normalize features to mean=0, std=1
    if normalize:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

    # Create sliding windows of size `window_size`
    X, y = [], []
    for i in range(window_size, len(df)):
        # Take last `window_size` rows of features as input
        window = df[features].iloc[i - window_size:i].values
        # Predict the glucose value at current time
        glucose = df[target].iloc[i]

        X.append(window)
        y.append(glucose)

    return np.array(X), np.array(y)
