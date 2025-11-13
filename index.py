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
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple
import os

from data_sources.fitbit_client import (
    FitbitClient,
    FitbitClientError,
    FitbitCredentials,
    FitbitToken,
)

ANOMALY_LABEL = "anomaly_label"
UNSUPERVISED_MODEL_LABEL = "Isolation Forest (unsupervised)"
SUPERVISED_MODEL_LABEL = "Random Forest (supervised)"
MODEL_OPTIONS = [UNSUPERVISED_MODEL_LABEL, SUPERVISED_MODEL_LABEL]
DATA_SOURCE_SIMULATED = "Simulated data"
DATA_SOURCE_FITBIT = "Fitbit API"
DATA_SOURCE_OPTIONS = [DATA_SOURCE_SIMULATED, DATA_SOURCE_FITBIT]

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
    required_columns = {
        "heart_rate": np.nan,
        "blood_oxygen": np.nan,
        "temperature": np.nan,
        "respiration_rate": np.nan,
        "activity_level": "moderate",
    }
    for column, default in required_columns.items():
        if column not in df.columns:
            df[column] = default

    df["activity_level_label"] = df["activity_level"]
    df["activity_level"] = (
        df["activity_level"].map({"low": 0, "moderate": 1, "high": 2}).fillna(1)
    )

    numeric_features = ["heart_rate", "blood_oxygen", "temperature", "respiration_rate"]
    df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors="coerce")

    df[numeric_features] = df[numeric_features].ffill().bfill()
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

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
def classify_activity_level_from_hr(heart_rate: float, resting_rate: Optional[float] = None) -> str:
    if pd.isna(heart_rate):
        return "moderate"
    baseline = resting_rate or 65
    if heart_rate <= baseline + 10:
        return "low"
    if heart_rate <= baseline + 30:
        return "moderate"
    return "high"


def build_fitbit_dataframe(
    dataset: List[Dict[str, Any]],
    user_id: str,
    resting_rate: Optional[float] = None,
) -> pd.DataFrame:
    if not dataset:
        return pd.DataFrame()

    df = pd.DataFrame(dataset)
    df.rename(columns={"value": "heart_rate"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["user_id"] = user_id or "fitbit_user"
    df["blood_oxygen"] = np.nan
    df["temperature"] = np.nan
    df["respiration_rate"] = 16

    df["activity_level"] = df["heart_rate"].apply(
        lambda hr: classify_activity_level_from_hr(hr, resting_rate)
    )
    df[ANOMALY_LABEL] = df["heart_rate"].apply(
        lambda hr: 1 if (pd.notna(hr) and (hr < 50 or hr > 120)) else 0
    )

    columns = [
        "user_id",
        "timestamp",
        "heart_rate",
        "blood_oxygen",
        "temperature",
        "respiration_rate",
        "activity_level",
        ANOMALY_LABEL,
    ]
    df = df[columns].dropna(subset=["timestamp", "heart_rate"])
    df.sort_values("timestamp", inplace=True)
    return df.reset_index(drop=True)


def load_fitbit_dataframe(
    credentials: FitbitCredentials,
    stored_token: Optional[Dict[str, Any]],
    target_date: date,
    detail_level: str,
) -> Tuple[pd.DataFrame, FitbitToken]:
    token = FitbitToken.from_dict(stored_token) if stored_token else None
    client = FitbitClient(credentials, token=token)

    if token and token.is_expired:
        token = client.refresh_access_token()

    intraday = client.get_heart_rate_intraday(datetime.combine(target_date, datetime.min.time()), detail_level)
    summary = client.get_daily_activity_summary(datetime.combine(target_date, datetime.min.time()))
    resting_rate = summary.get("summary", {}).get("restingHeartRate")

    df = build_fitbit_dataframe(intraday, client.token.user_id if client.token else "fitbit_user", resting_rate)
    if df.empty:
        raise FitbitClientError("No heart rate samples returned for the selected date.")

    return df, client.token


def main():
    st.set_page_config(page_title="AI Health Anomaly Detection", layout="wide")
    st.title("🧠 AI-Powered Health Anomaly Detection")
    st.caption(
        "Simulate wearable telemetry, detect anomalies automatically, and audit model performance."
    )

    if "fitbit_token" not in st.session_state:
        st.session_state["fitbit_token"] = None
    if "fitbit_df" not in st.session_state:
        st.session_state["fitbit_df"] = None
    if "fitbit_auth_url" not in st.session_state:
        st.session_state["fitbit_auth_url"] = None

    st.sidebar.header("⚙️ Configuration")
    data_source = st.sidebar.selectbox("Data source", DATA_SOURCE_OPTIONS, index=0)
    model_choice = st.sidebar.selectbox("Detection model", MODEL_OPTIONS, index=0)

    df = pd.DataFrame()

    if data_source == DATA_SOURCE_SIMULATED:
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
    else:
        st.sidebar.subheader("Fitbit OAuth")
        client_id = st.sidebar.text_input("Client ID", value=os.getenv("FITBIT_CLIENT_ID", ""))
        client_secret = st.sidebar.text_input(
            "Client Secret", value=os.getenv("FITBIT_CLIENT_SECRET", ""), type="password"
        )
        redirect_uri = st.sidebar.text_input(
            "Redirect URI", value=os.getenv("FITBIT_REDIRECT_URI", "http://localhost:8501")
        )
        scope = st.sidebar.text_input("Scope", value="activity heartrate sleep")
        contamination = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.2, 0.05)
        random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

        credentials = FitbitCredentials(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
        )

        client = FitbitClient(credentials, token=(
            FitbitToken.from_dict(st.session_state["fitbit_token"])
            if st.session_state["fitbit_token"]
            else None
        ))

        if st.sidebar.button("Generate authorization URL", key="fitbit_auth_link"):
            auth_url = client.get_authorize_url()
            st.session_state["fitbit_auth_url"] = auth_url

        if st.session_state.get("fitbit_auth_url"):
            st.sidebar.markdown(
                f"[Open Fitbit consent page]({st.session_state['fitbit_auth_url']})"
            )

        authorization_code = st.sidebar.text_input("Authorization code (paste after consent)")
        exchange_clicked = st.sidebar.button("Exchange code", key="fitbit_exchange")
        refresh_clicked = st.sidebar.button("Refresh token", key="fitbit_refresh")

        if exchange_clicked and authorization_code:
            try:
                token = client.exchange_code_for_token(authorization_code.strip())
                st.session_state["fitbit_token"] = token.to_dict()
                st.sidebar.success("Fitbit access token stored in session.")
            except FitbitClientError as exc:
                st.sidebar.error(f"Authorization failed: {exc}")
        elif exchange_clicked and not authorization_code:
            st.sidebar.warning("Provide the authorization code obtained after consent.")

        if refresh_clicked:
            try:
                token = client.refresh_access_token()
                st.session_state["fitbit_token"] = token.to_dict()
                st.sidebar.success("Access token refreshed.")
            except FitbitClientError as exc:
                st.sidebar.error(f"Token refresh failed: {exc}")

        st.sidebar.subheader("Data retrieval")
        data_date = st.sidebar.date_input("Sample date", value=date.today())
        detail_level = st.sidebar.selectbox("Granularity", ["1sec", "1min", "5min", "15min"], index=1)
        fetch_clicked = st.sidebar.button("Fetch Fitbit data", key="fitbit_fetch")

        if fetch_clicked:
            if not st.session_state["fitbit_token"]:
                st.sidebar.warning("Authenticate with Fitbit before fetching data.")
            elif not client_id or not client_secret:
                st.sidebar.warning("Client ID and Client Secret are required.")
            else:
                try:
                    df_fitbit, token = load_fitbit_dataframe(
                        credentials,
                        st.session_state["fitbit_token"],
                        data_date,
                        detail_level,
                    )
                    st.session_state["fitbit_df"] = df_fitbit
                    st.session_state["fitbit_token"] = token.to_dict()
                    st.sidebar.success(
                        f"Retrieved {len(df_fitbit):,} heart rate samples for {data_date}."
                    )
                except FitbitClientError as exc:
                    st.sidebar.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(f"Unexpected error loading Fitbit data: {exc}")

        df = st.session_state.get("fitbit_df")

    if df is None or df.empty:
        st.info("No data loaded yet. Configure inputs in the sidebar to begin analysis.")
        return

    total_records = len(df)
    total_anomalies = int(df.get(ANOMALY_LABEL, pd.Series(dtype=int)).sum())
    anomaly_rate_display = (
        f"{(total_anomalies / total_records * 100):.2f}%" if total_records else "0%"
    )

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    overview_col1.metric("Users observed", df["user_id"].nunique())
    overview_col2.metric("Records analyzed", f"{total_records:,}")
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
        st.markdown(f"**Data source:** {data_source}  ")
        st.markdown(f"**Active model:** {model_choice}")
        st.markdown(
            "Use the tabs to explore raw telemetry, inspect diagnostics, and visualize anomaly timelines. "
            "Switch the user focus above to compare individual profiles."
        )

    with tabs[1]:
        st.subheader("📊 Data Explorer")
        st.caption(
            "Telemetry with ground-truth anomaly tags and predictions from the selected model."
        )
        st.dataframe(
            df_processed[
                [
                    "user_id",
                    "timestamp",
                    "heart_rate",
                    "blood_oxygen",
                    "temperature",
                    "respiration_rate",
                    "activity_level_label",
                    ANOMALY_LABEL,
                    "prediction_label",
                ]
            ].head(500)
        )

    with tabs[2]:
        st.subheader("🧪 Model Diagnostics")
        st.markdown(f"**Model:** {model_choice}")
        if eval_df is None or cm is None:
            st.info(
                "Insufficient class diversity to compute evaluation metrics. Try expanding the dataset or adjusting anomaly thresholds."
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
