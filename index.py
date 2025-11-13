# Streamlit AI Health Anomaly Detection System (Large-Scale Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta

ANOMALY_LABEL = "anomaly_label"

# ---------------------------
# SECTION 1: Data Simulation
# ---------------------------
def simulate_health_data(
    num_users=5,
    minutes=500,
    anomaly_ratio=0.05,
    seed=42,
):
    rng = np.random.default_rng(seed)
    user_ids = [f"user_{i+1}" for i in range(num_users)]
    start_time = datetime.now()
    data = []

    for user in user_ids:
        timestamp = start_time
        for _ in range(minutes):
            is_anomaly = rng.uniform() < anomaly_ratio

            heart_rate = rng.integers(60, 100)
            blood_oxygen = rng.integers(93, 100)
            temperature = rng.normal(36.6, 0.3)
            respiration_rate = rng.integers(12, 20)
            activity_level = rng.choice(["low", "moderate", "high"])

            if is_anomaly:
                heart_rate = rng.integers(40, 180)
                blood_oxygen = rng.integers(80, 95)
                temperature = rng.normal(38.0, 0.6)
                respiration_rate = rng.integers(8, 30)
                activity_level = rng.choice(["low", "moderate", "high"])

            data.append(
                {
                    "user_id": user,
                    "timestamp": timestamp,
                    "heart_rate": heart_rate,
                    "blood_oxygen": blood_oxygen,
                    "temperature": round(temperature, 2),
                    "respiration_rate": respiration_rate,
                    "activity_level": activity_level,
                    ANOMALY_LABEL: int(is_anomaly),
                }
            )
            timestamp += timedelta(minutes=1)

    return pd.DataFrame(data)

# -----------------------------
# SECTION 2: Preprocessing
# -----------------------------
def preprocess_data(df):
    df = df.copy()
    df["activity_level"] = df["activity_level"].map({"low": 0, "moderate": 1, "high": 2})
    features = [
        "heart_rate",
        "blood_oxygen",
        "temperature",
        "respiration_rate",
        "activity_level",
    ]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features,
        index=df.index,
    )
    return df, df_scaled, features

# -----------------------------
# SECTION 3: Anomaly Detection
# -----------------------------
def detect_anomalies(df_scaled, contamination=0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(df_scaled)
    return preds, model

# -----------------------------
# SECTION 4: Evaluation
# -----------------------------
def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()

# -----------------------------
# SECTION 5: Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="AI Health Anomaly Detection", layout="wide")
    st.title("🧠 AI-Powered Health Anomaly Detection")

# Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    num_users = st.sidebar.slider("Number of Users", 1, 10, 3)
    num_minutes = st.sidebar.slider("Minutes of Data per User", 100, 1000, 300)
    contamination = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.2, 0.05)
    simulated_anomaly_ratio = st.sidebar.slider(
        "Simulated anomaly ratio", min_value=0.01, max_value=0.3, value=0.08, step=0.01
    )
    random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

# Data simulation and display
    st.header("📊 Simulated Health Data")
    df = simulate_health_data(
        num_users=num_users,
        minutes=num_minutes,
        anomaly_ratio=simulated_anomaly_ratio,
        seed=random_seed,
    )
    st.caption("First 100 rows of simulated wearable telemetry.")
    st.dataframe(df.head(100))

# Preprocessing and anomaly detection
    st.header("🧪 Anomaly Detection")
    df_processed, df_scaled, feature_cols = preprocess_data(df)
    preds_full, model = detect_anomalies(df_scaled, contamination)
    df_processed["predicted_anomaly"] = np.where(preds_full == -1, 1, 0)
    df_processed["prediction_label"] = df_processed["predicted_anomaly"].map(
        {0: "Normal", 1: "Anomaly"}
    )
    st.caption("Predicted anomalies across the first 100 samples.")
    st.dataframe(
        df_processed[
            [
                "user_id",
                "timestamp",
                "heart_rate",
                "blood_oxygen",
                "temperature",
                ANOMALY_LABEL,
                "prediction_label",
            ]
        ].head(100)
    )

# Evaluation setup
    st.header("📈 Model Evaluation")
    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled,
        df_processed[ANOMALY_LABEL],
        test_size=0.2,
        stratify=df_processed[ANOMALY_LABEL],
        random_state=random_seed,
    )
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)

    eval_df = evaluate_model(y_test, y_pred)
    st.caption(
        "Classification report comparing simulated ground-truth anomalies to model predictions."
    )
    st.dataframe(eval_df)

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

# Visualize anomalies
    st.header("📉 Anomaly Visualization")
    fig, ax = plt.subplots()
    anomaly_points = df_processed[df_processed["prediction_label"] == "Anomaly"]
    sns.lineplot(data=df_processed, x="timestamp", y="heart_rate", hue="user_id", ax=ax)
    plt.scatter(
        anomaly_points["timestamp"],
        anomaly_points["heart_rate"],
        color="red",
        label="Predicted anomaly",
        zorder=5,
    )
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

# Optional save
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Export Anomaly Report"):
        output_path = "anomaly_report.csv"
        df_processed.to_csv(output_path, index=False)
        st.success(f"Report saved as {output_path}")


if __name__ == "__main__":
    main()
