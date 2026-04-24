from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from services.ml_engine.app.models.autoencoder import TabularAutoencoder
from services.ml_engine.app.models.isolation_forest import IsolationForestModel
from services.ml_engine.app.utils.metrics import compute_classification_metrics
from services.ml_engine.app.utils.preprocessing import FEATURE_COLUMNS, FeaturePreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoencoder + isolation forest models")
    parser.add_argument("--data-path", required=True, help="Processed feature dataset CSV/Parquet path")
    parser.add_argument("--artifact-dir", required=True, help="Directory to save models and thresholds")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--label-column", default="label", help="Optional label column for metrics")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def train_autoencoder(
    x_train: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> TabularAutoencoder:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabularAutoencoder(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.from_numpy(x_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for (batch_x,) in loader:
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch_x)
            loss = F.mse_loss(recon, batch_x)
            loss.backward()
            optimizer.step()

    return model


def compute_autoencoder_scores(model: TabularAutoencoder, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x)
        recon = model(x_tensor)
        mse = F.mse_loss(recon, x_tensor, reduction="none").mean(dim=1)
        return mse.cpu().numpy().astype(np.float32)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_v = float(np.min(scores))
    max_v = float(np.max(scores))
    if max_v - min_v < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - min_v) / (max_v - min_v)).astype(np.float32)


def main() -> None:
    args = parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.data_path)

    preprocessor = FeaturePreprocessor(feature_columns=FEATURE_COLUMNS)
    x_scaled = preprocessor.fit_transform(df)

    ae_model = train_autoencoder(
        x_train=x_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    ae_scores = compute_autoencoder_scores(ae_model, x_scaled)

    if_model = IsolationForestModel(contamination=max(0.001, 1.0 - args.threshold_percentile / 100.0))
    if_model.fit(x_scaled)
    if_scores = if_model.score(x_scaled)

    ae_norm = normalize_scores(ae_scores)
    if_norm = normalize_scores(if_scores)
    final_scores = (0.5 * ae_norm + 0.5 * if_norm).astype(np.float32)

    threshold = float(np.percentile(final_scores, args.threshold_percentile))

    torch.save(
        {
            "state_dict": ae_model.state_dict(),
            "input_dim": x_scaled.shape[1],
            "feature_columns": FEATURE_COLUMNS,
        },
        artifact_dir / "autoencoder.pt",
    )
    if_model.save(artifact_dir / "isolation_forest.pkl")
    preprocessor.save(artifact_dir / "preprocessor.pkl")

    threshold_payload = {
        "threshold_percentile": float(args.threshold_percentile),
        "threshold": threshold,
    }
    with open(artifact_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, indent=2)

    if args.label_column in df.columns:
        y_true = pd.to_numeric(df[args.label_column], errors="coerce").fillna(0).astype(int).to_numpy()
        metrics = compute_classification_metrics(y_true=y_true, y_score=final_scores, threshold=threshold)
        with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

