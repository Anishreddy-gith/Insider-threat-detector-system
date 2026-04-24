from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray) -> None:
        self.model.fit(x)

    def score(self, x: np.ndarray) -> np.ndarray:
        # decision_function: higher is more normal, lower is more anomalous.
        raw = self.model.decision_function(x)
        # Convert to anomaly score: higher is more anomalous.
        return (-raw).astype(np.float32)

    def save(self, path: str | Path) -> None:
        import pickle

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestModel":
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)

        instance = cls()
        instance.model = model
        return instance
