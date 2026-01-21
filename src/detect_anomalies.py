import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from autoencoder import Autoencoder
import joblib

data = pd.read_csv("data/telemetry.csv")
features = ["temperature", "voltage", "pressure"]
X = data[features].values

scaler = joblib.load("scaler.save")
X_scaled = scaler.transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

with torch.no_grad():
    reconstructions = model(X_tensor.to(device))
reconstruction_error = torch.mean((X_tensor.to(device) - reconstructions) ** 2, dim=1)

threshold = reconstruction_error.mean() + 3 * reconstruction_error.std()
predicted_anomalies = reconstruction_error > threshold
data["predicted_anomaly"] = predicted_anomalies.int()

print(f"Anomaly detection threshold: {threshold:.4f}")
print(f"Detected {predicted_anomalies.sum()} anomalies out of {len(data)} points")

data.to_csv("data/telemetry_with_predictions.csv", index=False)
print("Results saved to data/telemetry_with_predictions.csv")

plt.figure(figsize=(12, 5))
plt.plot(data["time"], data["temperature"], label="Temperature")
plt.scatter(
    data.loc[data.predicted_anomaly == 1, "time"],
    data.loc[data.predicted_anomaly == 1, "temperature"],
    color="red",
    label="Detected Anomalies"
)
plt.xlabel("Time step")
plt.ylabel("Temperature (°C)")
plt.title("Telemetry Temperature with Detected Anomalies")
plt.legend()
plt.tight_layout()
plt.savefig("data/temperature_anomalies.png")
print("Plot saved as data/temperature_anomalies.png")
