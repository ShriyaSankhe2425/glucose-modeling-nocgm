import os
import numpy as np
import matplotlib.pyplot as plt
import json

# List your models here
model_names = [
    "nocgm_base",
    "nocgm_meals",
    "nocgm_mealscirc",
    "cgm_mealscirc"
]

results_dir = "results"

metrics_dict = {}
all_preds = {}
all_true = {}

# Load all saved predictions + metrics
for name in model_names:
    y_pred = np.load(os.path.join(results_dir, f"{name}_y_pred.npy"))
    y_true = np.load(os.path.join(results_dir, f"{name}_y_true.npy"))
    with open(os.path.join(results_dir, f"{name}_metrics.json")) as f:
        metrics = json.load(f)
    metrics_dict[name] = metrics
    all_preds[name] = y_pred
    all_true[name] = y_true

# ---------- 1️⃣ Bar Plot: RMSE, MAE, R² ----------
labels = model_names
rmse_vals = [metrics_dict[m]["RMSE"] for m in model_names]
mae_vals  = [metrics_dict[m]["MAE"] for m in model_names]
r2_vals   = [metrics_dict[m]["R2"] for m in model_names]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, rmse_vals, width, label='RMSE')
plt.bar(x, mae_vals, width, label='MAE')
plt.bar(x + width, r2_vals, width, label='R²')
plt.xticks(x, labels, rotation=15)
plt.ylabel("Score")
plt.title("📊 Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 2️⃣ Scatter: Predicted vs Actual ----------
plt.figure(figsize=(12, 5))
for i, name in enumerate(model_names):
    plt.subplot(1, 4, i + 1)
    plt.scatter(all_true[name], all_preds[name], alpha=0.3, s=5)
    plt.plot([all_true[name].min(), all_true[name].max()],
             [all_true[name].min(), all_true[name].max()], 'r--')
    plt.title(name)
    plt.xlabel("Actual")
    if i == 0:
        plt.ylabel("Predicted")
plt.suptitle("🔵 Predicted vs Actual Glucose")
plt.tight_layout()
plt.show()

# ---------- 3️⃣ Time Series: Prediction Over Time ----------
plt.figure(figsize=(14, 6))
for name in model_names:
    plt.plot(all_true[name][:300], label=f"{name} (true)", linestyle='--', alpha=0.5)
    plt.plot(all_preds[name][:300], label=f"{name} (pred)", alpha=0.9)
plt.legend()
plt.title("📈 Glucose Predictions Over Time (First 300 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Glucose (normalized)")
plt.tight_layout()
plt.show()
