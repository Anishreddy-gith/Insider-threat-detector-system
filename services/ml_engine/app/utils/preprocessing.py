from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: list[str] = [
    "login_frequency",
    "after_hours_activity",
    "file_access_deviation",
    "email_anomalies",
    "session_duration",
]


class FeaturePreprocessor:
    def __init__(self, feature_columns: Iterable[str] | None = None) -> None:
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        x = self._extract_features(df)
        return self.scaler.fit_transform(x)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._extract_features(df)
        return self.scaler.transform(x)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._extract_features(df)
        return self.scaler.fit_transform(x)

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        x = df[self.feature_columns].copy()
        x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return x.to_numpy(dtype=np.float32)

    def save(self, output_path: str | Path) -> None:
        import pickle

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, input_path: str | Path) -> "FeaturePreprocessor":
        import pickle

        with open(input_path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(feature_columns=payload["feature_columns"])
        instance.scaler = payload["scaler"]
        return instance
