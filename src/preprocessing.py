import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from src.meal_flag_generator import load_meal_times, add_meal_features

def load_actiheart_data(csv_path, window_size=4, normalize=True, include_glucose=False, include_meal_flag=True):
    df = pd.read_csv(csv_path)
    df = df[df['mask'] == False].reset_index(drop=True)
    # Add circadian (time of day) features
    df['TimeOfDay'] = df['abs_time_hours'] % 24
    df['TimeOfDay_sin'] = np.sin(2 * np.pi * df['TimeOfDay'] / 24)
    df['TimeOfDay_cos'] = np.cos(2 * np.pi * df['TimeOfDay'] / 24)

    # Auto match food XLSX
    base_name = os.path.basename(csv_path).replace('glucose_actiheart_integrated_', '').replace('.csv', '')
    food_path = os.path.join('data', f'food_{base_name}.xlsx')

    if include_meal_flag and os.path.exists(food_path):
        meal_times = load_meal_times(food_path)
        df = add_meal_features(df, meal_times)
    else:
        df['MealFlag'] = 0

    # Features to include
    features = [
        'Activity', 'BPM', 'RMSSD',
        'MealFlag', 'TimeSinceLastMeal', 'MealDensity',
        'TimeOfDay_sin', 'TimeOfDay_cos'
    ]
    if include_meal_flag:
        features.append('MealFlag')
    if include_glucose:
        features.append('Detrended')

    target = 'Detrended'

    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[features].iloc[i - window_size:i].values)
        y.append(df[target].iloc[i])

    return np.array(X), np.array(y)
