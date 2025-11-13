# AI-Powered Health Anomaly Detection Project Report

## 1. Overview

- **Goal:** Support SDG 3 (Good Health and Well‑Being) by enabling early detection of abnormal vital signs through an AI-powered dashboard.
- **Scope:** Simulated wearable telemetry feeds an anomaly detection engine built with unsupervised machine learning. Results are visualized in a Streamlit web application to support clinicians, caregivers, or researchers.
- **Team Focus:** Software engineering best practices including modular design, integration testing, containerization, and ethical considerations.

## 2. Problem Statement & SDG Alignment

- **Challenge:** Continuous monitoring of heart rate, blood oxygen, temperature, and respiration data leads to large streams of information. Manual inspection is infeasible and delays response time when anomalies occur.
- **SDG Target Links:** SDG 3.4 (reduce premature mortality from non-communicable diseases) and SDG 3.d (strengthen early warning systems for health risks).
- **Impact Hypothesis:** Automated anomaly detection empowers faster escalations to professionals, potentially reducing adverse health events.

## 3. Technical Solution

- **Data Simulation:** Synthetic sensor readings for multiple users with injectable anomaly rates allow rapid experimentation without privacy risks.
- **Preprocessing:** Numeric encoding, scaling, and modular pipelines enable rapid substitution of models or features.
- **Model:** Isolation Forest (unsupervised) detects outliers without requiring extensive labelled datasets. Hyperparameters (e.g., contamination) are configurable through the UI.
- **Visualization:** Streamlit dashboard surfaces anomalies, evaluation metrics, and time-series plots to aid interpretation.

## 4. Architecture Overview

1. `simulate_health_data` generates time-series telemetry with ground-truth anomaly labels.
2. `preprocess_data` standardizes features for the model.
3. `detect_anomalies` fits and applies an Isolation Forest to detect outliers.
4. Results feed into evaluation (classification report, confusion matrix) and visualizations.
5. Streamlit exports the enriched dataset for downstream review.


```startLine:endLine:index.py
def simulate_health_data(
    num_users=5,
    minutes=500,
    anomaly_ratio=0.05,
    seed=42,
):
// ... existing code ...
```

## 5. Testing Strategy

- **Integration Tests:** `pytest` suite in `tests/test_integration.py` exercises the end-to-end pipeline from simulation to anomaly predictions, ensuring code remains reliable as modules evolve.
- **Manual UI Validation:** Streamlit app launched locally (`streamlit run index.py`) to inspect controls, metrics, and charts.
- **Future Enhancements:**
  - Extend tests to cover regression metrics and data export functionality.
  - Add synthetic drift scenarios to validate resiliency and monitoring.

## 6. Deployment & Operations

- **Local:** `pip install -r requirements.txt` followed by `streamlit run index.py`.
- **Container:** `Dockerfile` supplies a reproducible environment. `.dockerignore` keeps the image lean.
- **Cloud:** Detailed Azure App Service deployment instructions in `deployment/azure/README.md`, including container registry setup and scaling guidance.
- **Monitoring:** Recommend enabling Azure Monitor logs and alerts. For production, integrate model drift dashboards and automated retraining triggers.

## 7. Ethical & Sustainability Considerations

- **Fairness:** Although simulated, real deployments must audit demographic and physiological coverage to prevent bias. Logging ground-truth labels and false alarm rates per cohort is recommended.
- **Privacy:** Use encryption in transit/at rest, de-identify user data, and adhere to HIPAA/GDPR when integrating real patient data.
- **Energy Efficiency:** Isolation Forest provides a lightweight baseline. Consider inference on edge devices to reduce cloud compute when feasible.
- **Accessibility:** Provide clear explanations and escalation workflows to limit alarm fatigue and ensure actionable intelligence.

## 8. Future Work

- Integrate LSTM forecasting for trend analysis and predictive alerts.
- Ingest real wearable APIs (e.g., Fitbit, Apple HealthKit) with consented data.
- Add role-based dashboards for clinicians vs. patients.
- Automate CI/CD pipelines to rebuild and redeploy the container on every main-branch update.

## 9. Conclusion

The project demonstrates how AI for Software Engineering practices accelerate SDG-aligned innovation. Through reproducible infrastructure, integration tests, and ethical guardrails, the system is well-positioned for iterative enhancements and real-world pilots.

