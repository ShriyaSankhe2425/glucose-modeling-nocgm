import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.meal_flag_generator import load_meal_times, add_meal_features
import os

def load_actiheart_data(csv_path, window_size=4, normalize=True, model_name="nocgm_base"):
    """
    Loads a single participant's glucose + Actiheart CSV file,
    adds contextual features, and returns windowed inputs + targets for model training.

    Parameters:
        csv_path (str): Path to the integrated participant CSV file
        window_size (int): Number of time steps in each LSTM input sequence
        normalize (bool): Whether to apply StandardScaler to features
        model_name (str): Name of model variant used for feature selection

    Returns:
        X (np.array): Shape = (samples, window_size, num_features)
        y (np.array): Shape = (samples,)
    """

    # --- Load and clean CSV ---
    df = pd.read_csv(csv_path)
    df = df[df['mask'] == False].reset_index(drop=True)  # remove masked rows

    # --- Add Circadian Features (Time of Day as sin/cos wave) ---
    df['TimeOfDay'] = df['abs_time_hours'] % 24
    df['TimeOfDay_sin'] = np.sin(2 * np.pi * df['TimeOfDay'] / 24)
    df['TimeOfDay_cos'] = np.cos(2 * np.pi * df['TimeOfDay'] / 24)

    # --- Add Meal Ingestion Features (MealFlag, TimeSinceLastMeal, MealDensity) ---
    base_name = os.path.basename(csv_path).replace('glucose_actiheart_integrated_', '').replace('.csv', '')
    food_path = os.path.join('data', f'food_{base_name}.xlsx')

    if os.path.exists(food_path):
        meal_times = load_meal_times(food_path)
        df = add_meal_features(df, meal_times)
    else:
        # If food log not available, default to no meal info
        df['MealFlag'] = 0
        df['TimeSinceLastMeal'] = np.nan
        df['MealDensity'] = 0

    # --- Dynamically select features based on model type ---
    if model_name == "nocgm_base":
        # Baseline model: only biosignals
        features = ['Activity', 'BPM', 'RMSSD']

    elif model_name == "nocgm_meals":
        # Add binary MealFlag (1 = meal in window)
        features = ['Activity', 'BPM', 'RMSSD', 'MealFlag']

    elif model_name == "nocgm_mealscirc":
        # Add full food + circadian context (no CGM input)
        features = [
            'Activity', 'BPM', 'RMSSD',
            'MealFlag', 'TimeSinceLastMeal', 'MealDensity',
            'TimeOfDay_sin', 'TimeOfDay_cos'
        ]

    elif model_name == "cgm_mealscirc":
        # Full-featured model: biosignals + food + circadian + CGM input
        features = [
            'Activity', 'BPM', 'RMSSD',
            'MealFlag', 'TimeSinceLastMeal', 'MealDensity',
            'TimeOfDay_sin', 'TimeOfDay_cos',
            'Detrended'  # Past CGM values as input
        ]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Target to predict (always Detrended glucose)
    target = 'Detrended'

    # --- Normalize selected input features ---
    if normalize:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

    # --- Create sliding windows for LSTM ---
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[features].iloc[i - window_size:i].values)  # shape: (window, num_features)
        y.append(df[target].iloc[i])  # shape: scalar

    return np.array(X), np.array(y)
