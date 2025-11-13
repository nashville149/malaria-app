# Streamlit AI Health Anomaly Detection System (Large-Scale Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta

ANOMALY_LABEL = "anomaly_label"
UNSUPERVISED_MODEL_LABEL = "Isolation Forest (unsupervised)"
SUPERVISED_MODEL_LABEL = "Random Forest (supervised)"
MODEL_OPTIONS = [UNSUPERVISED_MODEL_LABEL, SUPERVISED_MODEL_LABEL]

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


def run_isolation_forest_pipeline(
    df_scaled,
    labels,
    contamination,
    random_seed,
):
    preds_raw, fitted_model = detect_anomalies(df_scaled, contamination)
    predictions = np.where(preds_raw == -1, 1, 0)

    eval_df = None
    cm = None

    if labels.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            df_scaled,
            labels,
            test_size=0.2,
            stratify=labels,
            random_state=random_seed,
        )
        eval_model = IsolationForest(contamination=contamination, random_state=random_seed)
        eval_model.fit(X_train)
        y_pred = eval_model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
        eval_df = evaluate_model(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

    return predictions, eval_df, cm, fitted_model


def run_random_forest_pipeline(
    df_scaled,
    labels,
    random_seed,
):
    stratify = labels if labels.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled,
        labels,
        test_size=0.2,
        stratify=stratify,
        random_state=random_seed,
    )
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_seed,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    eval_df = evaluate_model(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    full_predictions = model.predict(df_scaled)
    return full_predictions, eval_df, cm, model

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
    st.caption(
        "Simulate wearable telemetry, detect anomalies automatically, and audit model performance."
    )

    st.sidebar.header("⚙️ Configuration")
    model_choice = st.sidebar.selectbox("Detection model", MODEL_OPTIONS, index=0)
    num_users = st.sidebar.slider("Number of Users", 1, 10, 3)
    num_minutes = st.sidebar.slider("Minutes of Data per User", 100, 1000, 300)
    contamination = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.2, 0.05)
    simulated_anomaly_ratio = st.sidebar.slider(
        "Simulated anomaly ratio", min_value=0.01, max_value=0.3, value=0.08, step=0.01
    )
    random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    df = simulate_health_data(
        num_users=num_users,
        minutes=num_minutes,
        anomaly_ratio=simulated_anomaly_ratio,
        seed=random_seed,
    )

    total_records = len(df)
    total_anomalies = int(df[ANOMALY_LABEL].sum())
    anomaly_rate_display = (
        f"{(total_anomalies / total_records * 100):.2f}%" if total_records else "0%"
    )

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    overview_col1.metric("Users simulated", df["user_id"].nunique())
    overview_col2.metric("Records generated", f"{total_records:,}")
    overview_col3.metric("Ground-truth anomalies", f"{total_anomalies:,}")
    overview_col4.metric("Anomaly prevalence", anomaly_rate_display)

    st.markdown("### 🎯 Focus Controls")
    selected_user = st.selectbox(
        "Select a user to explore",
        options=["All users"] + sorted(df["user_id"].unique()),
        index=0,
    )
    df_focus = df[df["user_id"] == selected_user].copy() if selected_user != "All users" else df.copy()

    if df_focus.empty:
        st.warning("No records available for the current configuration.")
        return

    df_processed, df_scaled, feature_cols = preprocess_data(df_focus)

    if model_choice == SUPERVISED_MODEL_LABEL:
        predictions, eval_df, cm, trained_model = run_random_forest_pipeline(
            df_scaled, df_processed[ANOMALY_LABEL], random_seed
        )
    else:
        predictions, eval_df, cm, trained_model = run_isolation_forest_pipeline(
            df_scaled, df_processed[ANOMALY_LABEL], contamination, random_seed
        )

    predictions = np.asarray(predictions).astype(int)
    df_processed["predicted_anomaly"] = predictions
    df_processed["prediction_label"] = df_processed["predicted_anomaly"].map(
        {0: "Normal", 1: "Anomaly"}
    )
    predicted_anomalies = int(df_processed["predicted_anomaly"].sum())

    tabs = st.tabs(["Overview", "Data Explorer", "Model Diagnostics", "Visual Analytics"])

    with tabs[0]:
        st.subheader("System Snapshot")
        match_rate = (
            (df_processed[ANOMALY_LABEL] == df_processed["predicted_anomaly"]).mean()
            if len(df_processed)
            else 0.0
        )
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric(
            "Predicted anomalies (current view)", f"{predicted_anomalies:,}"
        )
        summary_col2.metric("Match rate vs. ground truth", f"{match_rate:.2%}")
        summary_col3.metric("Features used", len(feature_cols))
        st.markdown(f"**Active model:** {model_choice}")
        st.markdown(
            "Use the tabs to explore raw telemetry, inspect diagnostics, and visualize anomaly timelines. "
            "Switch the user focus above to compare individual profiles."
        )

    with tabs[1]:
        st.subheader("📊 Data Explorer")
        st.caption(
            "Simulated telemetry with ground-truth anomaly tags and predictions from the selected model."
        )
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
            ].head(300)
        )

    with tabs[2]:
        st.subheader("🧪 Model Diagnostics")
        st.markdown(f"**Model:** {model_choice}")
        if eval_df is None or cm is None:
            st.info(
                "Insufficient class diversity to compute evaluation metrics. Try increasing simulation duration or anomaly ratio."
            )
        else:
            st.caption(
                "Classification report comparing simulated ground-truth anomalies to model predictions."
            )
            st.dataframe(eval_df)

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

            if model_choice == SUPERVISED_MODEL_LABEL and hasattr(trained_model, "feature_importances_"):
                st.markdown("#### Feature Importance")
                importances = (
                    pd.Series(trained_model.feature_importances_, index=feature_cols)
                    .sort_values(ascending=False)
                )
                st.bar_chart(importances)

    with tabs[3]:
        st.subheader("📉 Visual Analytics")
        timeline_col1, timeline_col2 = st.columns([3, 1])
        with timeline_col1:
            fig, ax = plt.subplots()
            sns.lineplot(
                data=df_processed,
                x="timestamp",
                y="heart_rate",
                hue="user_id",
                ax=ax,
            )
            anomaly_points = df_processed[df_processed["prediction_label"] == "Anomaly"]
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
        with timeline_col2:
            st.markdown("#### Feature Distributions")
            selected_feature = st.selectbox("Feature", feature_cols, index=0)
            fig_feat, ax_feat = plt.subplots()
            sns.kdeplot(
                data=df_processed,
                x=selected_feature,
                hue="prediction_label",
                fill=True,
                common_norm=False,
                ax=ax_feat,
            )
            ax_feat.set_title(f"{selected_feature.title()} Distribution")
            st.pyplot(fig_feat)

    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Export Anomaly Report"):
        output_path = "anomaly_report.csv"
        df_processed.to_csv(output_path, index=False)
        st.success(f"Report saved as {output_path}")


if __name__ == "__main__":
    main()
