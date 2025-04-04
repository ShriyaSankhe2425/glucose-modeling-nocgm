import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_actiheart_data(csv_path, window_size=4, normalize=True, include_glucose=False):
    """
    Loads a participant's Actiheart data and optionally includes CGM as an input feature.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        window_size (int): Number of time steps to use for each input sequence.
        normalize (bool): Whether to normalize features.
        include_glucose (bool): If True, includes past glucose values in the input features.

    Returns:
        X (np.ndarray): Input data (samples, time_steps, features)
        y (np.ndarray): Target glucose value at each time step
    """
    df = pd.read_csv(csv_path)
    df = df[df['mask'] == False].reset_index(drop=True)

    features = ['Activity', 'BPM', 'RMSSD']
    if include_glucose:
        features.append('Detrended')  # Include glucose in X

    target = 'Detrended'

    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(window_size, len(df)):
        window = df[features].iloc[i - window_size:i].values
        X.append(window)
        y.append(df[target].iloc[i])  # still predict next glucose value

    return np.array(X), np.array(y)
