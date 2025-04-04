import pandas as pd
from datetime import timedelta
import os

def load_meal_times(xlsx_path):
    """
    Loads ingestion timestamps from the food log Excel file.
    Assumes a 'time' column exists (may vary slightly per file).
    """
    df = pd.read_excel(xlsx_path)
    # Try to infer the datetime column
    time_col = [col for col in df.columns if 'time' in col.lower()][0]
    df[time_col] = pd.to_datetime(df[time_col])
    return df[time_col].tolist()

def add_meal_features(glucose_df, meal_times, tolerance_minutes=15, density_window_minutes=120):
    """
    For each row in the glucose dataframe, adds:
      - MealFlag: 1 if a meal occurred ±tolerance_minutes
      - TimeSinceLastMeal: Minutes since last meal (NaN if no meal yet)
      - MealDensity: Number of meals in past density_window_minutes
    """
    from bisect import bisect_right
    import numpy as np
    from datetime import timedelta

    start_date = pd.Timestamp("2022-01-01")
    glucose_df['abs_datetime'] = glucose_df['abs_time_hours'].apply(
        lambda x: start_date + timedelta(hours=x)
    )

    meal_times = sorted(meal_times)
    glucose_df['MealFlag'] = 0
    glucose_df['TimeSinceLastMeal'] = np.nan
    glucose_df['MealDensity'] = 0

    for i, row_time in enumerate(glucose_df['abs_datetime']):
        # Meal flag (± tolerance)
        for meal_time in meal_times:
            if abs((row_time - meal_time).total_seconds()) <= tolerance_minutes * 60:
                glucose_df.at[i, 'MealFlag'] = 1
                break

        # Time since last meal
        earlier_meals = [m for m in meal_times if m <= row_time]
        if earlier_meals:
            last_meal = max(earlier_meals)
            delta = (row_time - last_meal).total_seconds() / 60
            glucose_df.at[i, 'TimeSinceLastMeal'] = delta

        # Meal density (within past X mins)
        count = sum([(row_time - meal_time).total_seconds() <= density_window_minutes * 60 and
                     (row_time - meal_time).total_seconds() > 0 for meal_time in meal_times])
        glucose_df.at[i, 'MealDensity'] = count

    return glucose_df