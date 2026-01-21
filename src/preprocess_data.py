import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("data/telemetry.csv")

nominal_data = data[data.is_anomaly == 0].copy()
features = ["temperature", "voltage", "pressure"]
X_nominal = nominal_data[features].values  # shape (n_samples, n_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nominal)

joblib.dump(scaler, "scaler.save")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

batch_size = 16
train_dataset = torch.utils.data.TensorDataset(X_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

print(f"Training on {len(train_dataset)} samples with batch size {batch_size}")
