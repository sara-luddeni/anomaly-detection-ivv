import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_nominal_data(n_steps=1000, seed=42):
    rng = np.random.default_rng(seed)
    time = np.arange(n_steps)

    temperature = (
        20
        + 2 * np.sin(0.02 * time)
        + rng.normal(0, 0.2, n_steps)
    )

    voltage = 3.3 + rng.normal(0, 0.05, n_steps)

    pressure = (
        101
        + 0.01 * time
        + rng.normal(0, 0.1, n_steps)
    )

    data = pd.DataFrame({
        "time": time,
        "temperature": temperature,
        "voltage": voltage,
        "pressure": pressure,
        "is_anomaly": 0
    })

    return data


def inject_point_anomalies(data, n_anomalies=15, seed=123):
    rng = np.random.default_rng(seed)
    anomaly_indices = rng.choice(
        data.index, size=n_anomalies, replace=False
    )

    data.loc[anomaly_indices, "temperature"] += rng.uniform(8, 12, n_anomalies)
    data.loc[anomaly_indices, "voltage"] -= rng.uniform(0.8, 1.2, n_anomalies)
    data.loc[anomaly_indices, "pressure"] += rng.uniform(5, 8, n_anomalies)
    data.loc[anomaly_indices, "is_anomaly"] = 1

    return data, anomaly_indices


if __name__ == "__main__":
    data = generate_nominal_data()
    data, anomaly_indices = inject_point_anomalies(data)

    data.to_csv("data/telemetry.csv", index=False)

    print(f"Generated data with {len(anomaly_indices)} point anomalies.")
    
    plt.plot(data["time"], data["temperature"], label="Temperature")
    plt.scatter(
        data.loc[data.is_anomaly == 1, "time"],
        data.loc[data.is_anomaly == 1, "temperature"],
        color="red",
        label="Anomalies"
    )
    plt.xlabel("Time step")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.savefig("data/temperature_plot.png")
    print("Plot saved as data/temperature_plot.png")



    


