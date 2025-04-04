from src.batch_loader import load_all_participants

# Load all participant data
X, y = load_all_participants(data_dir='data/', window_size=4)

# Print basic shapes
print("Preprocessing complete!")
print(f"X shape: {X.shape}")  # Expected: (number_of_samples, 4, 3)
print(f"y shape: {y.shape}")  # Expected: (number_of_samples,)
