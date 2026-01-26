# Anomaly Detection in Telemetry Using PyTorch Autoencoder

This project demonstrates anomaly detection in synthetic telemetry using a PyTorch autoencoder. 


## Requirements

- Python 3.12+, PyTorch, pandas, numpy, scikit-learn, matplotlib, joblib  
- `uv` package manager


## Usage

1. uv setup
```bash
uv sync
```

2. Generate synthetic data
```bash
uv run python src/generate_data.py
```

3. Preprocess data
```bash
uv run python src/preprocess_data.py
```
Standardizes telemetry data, saves scaler.save.

4. Train autoencoder
```bash
uv run python src/autoencoder.py
```
Trains on nominal data, saves autoencoder.pth

5. Detect anomalies
```bash
uv run python src/detect_anomalies.py
```
Computes reconstruction errors, flags anomalies.
