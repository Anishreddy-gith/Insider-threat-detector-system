from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from services.ml_engine.app.graph.graph_features import build_graph_from_cert_directory
from services.ml_engine.app.graph.graph_model import GraphAnomalyScorer
from services.ml_engine.app.models.autoencoder import TabularAutoencoder
from services.ml_engine.app.models.isolation_forest import IsolationForestModel
from services.ml_engine.app.utils.preprocessing import FeaturePreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anomaly inference with trained models")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing trained model artifacts")
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON string with feature values or path to JSON file",
    )
    parser.add_argument(
        "--graph-cert-dir",
        required=False,
        default=None,
        help="Directory containing CERT CSVs (logon.csv/file.csv/email.csv) for graph scoring",
    )
    parser.add_argument(
        "--graph-behavior-path",
        required=False,
        default=None,
        help="Behavioral feature dataframe path (csv/parquet) used for graph node features",
    )
    parser.add_argument(
        "--graph-target-user",
        required=False,
        default=None,
        help="Target user_id for graph anomaly score; defaults to input_json.user_id if present",
    )
    parser.add_argument("--w-autoencoder", type=float, default=0.4)
    parser.add_argument("--w-isolation", type=float, default=0.4)
    parser.add_argument("--w-gnn", type=float, default=0.2)
    return parser.parse_args()


def _load_json_input(value: str) -> dict[str, Any]:
    candidate = Path(value)
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_v = float(np.min(scores))
    max_v = float(np.max(scores))
    if max_v - min_v < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - min_v) / (max_v - min_v)).astype(np.float32)


def _sigmoid_score(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _read_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _compute_gnn_score(
    artifact_dir: Path,
    cert_dir: str | None,
    behavior_path: str | None,
    target_user: str | None,
) -> float:
    gnn_ckpt = artifact_dir / "graph_gnn.pt"
    if not gnn_ckpt.exists() or not cert_dir or not behavior_path:
        return 0.0

    behavior_df = _read_df(behavior_path)
    graph = build_graph_from_cert_directory(cert_dir=cert_dir, behavior_df=behavior_df)

    encoder, centroid = GraphAnomalyScorer.load_checkpoint(gnn_ckpt)
    scorer = GraphAnomalyScorer(encoder)

    candidate_user = (target_user or "").strip().lower()
    if "@" in candidate_user:
        candidate_user = candidate_user.split("@", 1)[0]
    if not candidate_user:
        return 0.0

    node_key = f"u:{candidate_user}"
    if node_key not in graph.node_index_map:
        return 0.0

    user_indices = torch.tensor(
        [i for i, nid in enumerate(graph.node_ids) if nid.startswith("u:")],
        dtype=torch.long,
    )
    if user_indices.numel() == 0:
        return 0.0

    node_index = graph.node_index_map[node_key]
    return float(
        scorer.score_user(
            data=graph.data,
            node_index=node_index,
            user_indices=user_indices,
            centroid=centroid,
        )
    )


def infer_one(
    feature_input: dict[str, Any],
    artifact_dir: str | Path,
    graph_cert_dir: str | None = None,
    graph_behavior_path: str | None = None,
    graph_target_user: str | None = None,
    w_autoencoder: float = 0.4,
    w_isolation: float = 0.4,
    w_gnn: float = 0.2,
) -> dict[str, float]:
    ad = Path(artifact_dir)

    preprocessor = FeaturePreprocessor.load(ad / "preprocessor.pkl")

    ae_checkpoint = torch.load(ad / "autoencoder.pt", map_location="cpu")
    ae_model = TabularAutoencoder(input_dim=int(ae_checkpoint["input_dim"]))
    ae_model.load_state_dict(ae_checkpoint["state_dict"])
    ae_model.eval()

    if_model = IsolationForestModel.load(ad / "isolation_forest.pkl")

    x_df = pd.DataFrame([feature_input])
    x_scaled = preprocessor.transform(x_df)

    x_tensor = torch.from_numpy(x_scaled)
    with torch.no_grad():
        recon = ae_model(x_tensor)
        ae_score = F.mse_loss(recon, x_tensor, reduction="none").mean(dim=1).cpu().numpy().astype(np.float32)

    if_score = if_model.score(x_scaled)

    target_user = graph_target_user
    if target_user is None and "user_id" in feature_input:
        target_user = str(feature_input.get("user_id", "")).strip()
    gnn_score = _compute_gnn_score(
        artifact_dir=ad,
        cert_dir=graph_cert_dir,
        behavior_path=graph_behavior_path,
        target_user=target_user,
    )

    ae_blend = _sigmoid_score(float(ae_score[0]))
    if_blend = _sigmoid_score(float(if_score[0]))
    gnn_blend = float(np.clip(gnn_score, 0.0, 1.0))

    weight_sum = max(float(w_autoencoder + w_isolation + w_gnn), 1e-8)
    wa = float(w_autoencoder) / weight_sum
    wi = float(w_isolation) / weight_sum
    wg = float(w_gnn) / weight_sum
    final_score = float((wa * ae_blend) + (wi * if_blend) + (wg * gnn_blend))

    return {
        "autoencoder_score": float(ae_score[0]),
        "isolation_score": float(if_score[0]),
        "gnn_score": float(gnn_blend),
        "final_score": final_score,
    }


def main() -> None:
    args = parse_args()
    feature_input = _load_json_input(args.input_json)
    result = infer_one(
        feature_input=feature_input,
        artifact_dir=args.artifact_dir,
        graph_cert_dir=args.graph_cert_dir,
        graph_behavior_path=args.graph_behavior_path,
        graph_target_user=args.graph_target_user,
        w_autoencoder=args.w_autoencoder,
        w_isolation=args.w_isolation,
        w_gnn=args.w_gnn,
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()

