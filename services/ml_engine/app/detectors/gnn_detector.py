"""
Graph Neural Network detector using PyTorch Geometric (GraphSAGE).

Best for: **relational / lateral-movement anomalies** â€” detecting when
a user accesses resources outside their normal peer-group neighbourhood.
Example: a developer who has never touched HR files suddenly reads
``/restricted/salary_data.xlsx`` â€” the *point* features are plausible
(normal working hours, moderate file size), but the *graph edge* is
unprecedented.

Graph construction
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Bipartite graph: **Users â†” Resources**

    â€¢ User nodes:     behavioural embeddings (D-dim feature vectors)
    â€¢ Resource nodes: one-hot type + sensitivity level + access frequency
    â€¢ Edge weight:    historical access count in the baseline window

GraphSAGE convolution aggregates neighbourhood information inductively,
meaning new users or resources can be scored without re-training the
full graph.

Why GraphSAGE?
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GCN requires the full graph Laplacian and is transductive.  GraphSAGE
samples and aggregates, making it:
  - inductive (new nodes at inference time)
  - scalable (mini-batch training)
  - faster at inference (no eigenvalue decomposition)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from services.ml_engine.app.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


# â”€â”€ Network definition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _GraphSAGEAnomalyNet(nn.Module):
    """
    2-layer GraphSAGE â†’ anomaly head.

    Forward pass returns per-node anomaly scores in [0, 1].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # Skip connection
        self.skip = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # Anomaly scoring head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = h + residual                         # skip connection
        scores = torch.sigmoid(self.head(h))     # (N, 1)
        return scores.squeeze(-1)                # (N,)

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return learned node embeddings (before the anomaly head)."""
        residual = self.skip(x)
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h + residual


# â”€â”€ Detector class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class GNNDetector(BaseDetector):
    """
    GraphSAGE-based relational anomaly detector for Userâ†”Resource graphs.

    Parameters
    ----------
    input_dim : int
        Node feature dimensionality.
    hidden_dim : int
        GraphSAGE hidden layer size.
    dropout : float
        Dropout rate.
    lr : float
        Adam learning rate.
    epochs : int
        Training epochs.
    anomaly_threshold : float
        Score above which a node is flagged anomalous.
    """

    name = "gnn_relational"

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        anomaly_threshold: float = 0.5,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.anomaly_threshold = anomaly_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _GraphSAGEAnomalyNet(
            input_dim, hidden_dim, dropout
        ).to(self.device)
        self._optimizer = Adam(self._model.parameters(), lr=lr)

        # Stored graph for reference during inference
        self._train_data: Data | None = None

    # â”€â”€ graph construction helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def build_graph(
        user_features: np.ndarray,
        resource_features: np.ndarray,
        access_matrix: np.ndarray,
    ) -> Data:
        """
        Build a bipartite Userâ†”Resource graph.

        Parameters
        ----------
        user_features : (U, D) â€“ one row per user
        resource_features : (R, D) â€“ one row per resource (padded to D)
        access_matrix : (U, R) â€“ access counts (0 = no edge)

        Returns
        -------
        A ``torch_geometric.data.Data`` object with:
            x           â€“ (U+R, D) node features
            edge_index  â€“ (2, E) bipartite edges (both directions)
            edge_weight â€“ (E,) access-frequency weights
            node_type   â€“ 0=user, 1=resource
        """
        n_users = user_features.shape[0]
        n_resources = resource_features.shape[0]

        # Concatenate features
        x = np.vstack([user_features, resource_features])

        # Build edges from access matrix
        src, dst, weights = [], [], []
        for u in range(n_users):
            for r in range(n_resources):
                if access_matrix[u, r] > 0:
                    r_idx = n_users + r
                    # Bidirectional edges
                    src.extend([u, r_idx])
                    dst.extend([r_idx, u])
                    w = float(access_matrix[u, r])
                    weights.extend([w, w])

        # Handle empty graph
        if not src:
            src, dst, weights = [0], [0], [1.0]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)

        node_type = torch.zeros(n_users + n_resources, dtype=torch.long)
        node_type[n_users:] = 1

        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_type=node_type,
            num_users=n_users,
            num_resources=n_resources,
        )

    # â”€â”€ training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        resource_features: np.ndarray | None = None,
        access_matrix: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train the GNN in semi-supervised mode.

        If labels ``y`` are provided (1=anomalous, 0=normal for user
        nodes), uses BCE loss.  Otherwise falls back to an unsupervised
        objective: minimise reconstruction of edge weights (link
        prediction proxy).

        Parameters
        ----------
        X : (U, D) user feature matrix
        y : (U,) optional labels
        resource_features : (R, D) resource features
        access_matrix : (U, R) access count matrix
        """
        if resource_features is None:
            resource_features = np.zeros((1, X.shape[1]), dtype=np.float32)
        if access_matrix is None:
            access_matrix = np.ones((X.shape[0], resource_features.shape[0]))

        data = self.build_graph(X, resource_features, access_matrix)
        data = data.to(self.device)
        self._train_data = data
        n_users = int(data.num_users)

        self._model.train()
        epoch_losses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            self._optimizer.zero_grad()
            scores = self._model(data.x, data.edge_index, data.edge_weight)

            if y is not None:
                # Supervised: BCE on user nodes
                labels = torch.tensor(y, dtype=torch.float32).to(self.device)
                loss = F.binary_cross_entropy(scores[:n_users], labels)
            else:
                # Unsupervised: push scores of all nodes toward 0
                # (treat the entire training graph as "normal")
                loss = scores[:n_users].pow(2).mean()

            loss.backward()
            self._optimizer.step()
            epoch_losses.append(loss.item())

        self._mark_fitted()
        logger.info(
            "gnn_trained",
            extra={
                "n_users": n_users,
                "n_resources": int(data.num_resources),
                "n_edges": data.edge_index.shape[1],
                "final_loss": round(epoch_losses[-1], 6),
            },
        )
        return {
            "n_users": n_users,
            "n_resources": int(data.num_resources),
            "n_edges": data.edge_index.shape[1],
            "epochs": self.epochs,
            "final_loss": epoch_losses[-1],
        }

    # â”€â”€ prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def predict(self, X: np.ndarray) -> list[dict[str, Any]]:
        """
        Score user nodes on the stored graph.

        If new users need scoring, call ``predict_with_graph()`` instead.
        """
        if not self.is_fitted or self._train_data is None:
            raise RuntimeError("Detector not fitted â€“ call train() first")
        return self.predict_with_graph(X, self._train_data)

    def predict_with_graph(
        self,
        user_features: np.ndarray,
        graph: Data,
    ) -> list[dict[str, Any]]:
        """
        Score user nodes given an explicit graph.

        Parameters
        ----------
        user_features : (U, D) â€“ will replace the first U rows of graph.x
        graph : torch_geometric.data.Data
        """
        self._model.eval()
        data = graph.clone().to(self.device)
        n_users = user_features.shape[0]

        # Replace user features (resource features stay)
        new_x = data.x.clone()
        new_x[:n_users] = torch.tensor(
            user_features, dtype=torch.float32
        ).to(self.device)
        data.x = new_x

        with torch.no_grad():
            scores = self._model(data.x, data.edge_index, data.edge_weight)
            embeddings = self._model.get_embeddings(data.x, data.edge_index)
            user_scores = scores[:n_users].cpu().numpy()
            user_embeds = embeddings[:n_users].cpu().numpy()

        results: list[dict[str, Any]] = []
        for i in range(n_users):
            s = float(user_scores[i])

            # Find connected resource indices and their edge weights
            edge_src = data.edge_index[0].cpu().numpy()
            edge_dst = data.edge_index[1].cpu().numpy()
            edge_w = data.edge_weight.cpu().numpy()

            # Edges originating from this user
            mask = edge_src == i
            connected_resources = edge_dst[mask]
            conn_weights = edge_w[mask]

            resource_links: list[dict[str, Any]] = []
            for r_idx, w in zip(connected_resources, conn_weights):
                r_score = float(scores[r_idx].cpu()) if r_idx < len(scores) else 0.0
                resource_links.append({
                    "resource_idx": int(r_idx) - n_users,
                    "access_weight": round(float(w), 4),
                    "resource_anomaly_score": round(r_score, 6),
                })
            # Sort links by resource anomaly score descending
            resource_links.sort(
                key=lambda rl: rl["resource_anomaly_score"], reverse=True
            )

            results.append(
                {
                    "anomaly_score": round(s, 6),
                    "is_anomaly": s > self.anomaly_threshold,
                    "detector": self.name,
                    "embedding_norm": round(float(np.linalg.norm(user_embeds[i])), 4),
                    "top_resource_links": resource_links[:5],
                }
            )
        return results

    # â”€â”€ persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "gnn_detector.pt"
        save_dict: dict[str, Any] = {
            "state_dict": self._model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "anomaly_threshold": self.anomaly_threshold,
            "meta": self._meta(),
        }
        if self._train_data is not None:
            save_dict["train_data"] = {
                "x": self._train_data.x.cpu(),
                "edge_index": self._train_data.edge_index.cpu(),
                "edge_weight": self._train_data.edge_weight.cpu(),
                "node_type": self._train_data.node_type.cpu(),
                "num_users": int(self._train_data.num_users),
                "num_resources": int(self._train_data.num_resources),
            }
        torch.save(save_dict, path)
        logger.info("gnn_saved", extra={"path": str(path)})
        return path

    def load(self, directory: str | Path) -> None:
        path = Path(directory) / "gnn_detector.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = ckpt["input_dim"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.dropout = ckpt["dropout"]
        self.anomaly_threshold = ckpt["anomaly_threshold"]
        self._model = _GraphSAGEAnomalyNet(
            self.input_dim, self.hidden_dim, self.dropout
        ).to(self.device)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()

        if "train_data" in ckpt:
            td = ckpt["train_data"]
            self._train_data = Data(
                x=td["x"],
                edge_index=td["edge_index"],
                edge_weight=td["edge_weight"],
                node_type=td["node_type"],
                num_users=td["num_users"],
                num_resources=td["num_resources"],
            )

        self._mark_fitted()
        logger.info("gnn_loaded", extra={"path": str(path)})

