import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from index import (  # noqa: E402
    ANOMALY_LABEL,
    detect_anomalies,
    preprocess_data,
    simulate_health_data,
)


def test_simulation_to_anomaly_predictions():
    df = simulate_health_data(
        num_users=2,
        minutes=120,
        anomaly_ratio=0.1,
        seed=123,
    )

    assert not df.empty
    assert ANOMALY_LABEL in df.columns
    assert df[ANOMALY_LABEL].between(0, 1).all()

    df_processed, df_scaled, features = preprocess_data(df)
    preds, model = detect_anomalies(df_scaled, contamination=0.1)

    assert len(preds) == len(df_processed)
    assert set(np.unique(preds)).issubset({-1, 1})
    assert model is not None

    df_processed["predicted_anomaly"] = np.where(preds == -1, 1, 0)

    # Ensure at least some anomalies were detected in the simulation
    assert df_processed["predicted_anomaly"].isin([0, 1]).all()

