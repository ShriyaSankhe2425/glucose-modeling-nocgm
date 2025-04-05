import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import json
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from src.batch_loader import load_all_participants

# 1️⃣ Hyperparameters
WINDOW_SIZE = 4       # 1 hour (15-min steps)
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
HIDDEN_DIM = 64

# to name the model variant
model_name = "nocgm_meals"  # <-- edit for each run

# 2️⃣ Load preprocessed data
X, y = load_all_participants(data_dir='data/', window_size=WINDOW_SIZE, model_name=model_name)


# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

# Create dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split into 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# 3️⃣ Define the LSTM model
class GlucoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GlucoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        return self.fc(out)

model = GlucoseLSTM(input_dim=4, hidden_dim=HIDDEN_DIM)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 4️⃣ Train the model
print("🟢 Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
print("✅ Training complete.")

# 5️⃣ Evaluate on test set
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).squeeze().numpy()
        targets = yb.squeeze().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

# Compute metrics
rmse = root_mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print("\n📊 Evaluation on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")


# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Save predictions
np.save(f"results/{model_name}_y_true.npy", np.array(all_targets))
np.save(f"results/{model_name}_y_pred.npy", np.array(all_preds))

# Save metrics
metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
with open(f"results/{model_name}_metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Results saved for model: {model_name}")
