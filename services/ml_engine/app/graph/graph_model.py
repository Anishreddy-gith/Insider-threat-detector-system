from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

USER_NODE = 0


class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        model_type: str = "graphsage",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_type = model_type.lower()
        self.dropout = nn.Dropout(dropout)

        if self.model_type == "gcn":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, embedding_dim)
        else:
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        z = self.conv2(h, edge_index)
        return z


class GraphAnomalyScorer:
    def __init__(self, encoder: GraphEncoder) -> None:
        self.encoder = encoder

    def embeddings(self, data: Data) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(data.x, data.edge_index)

    def score_user(
        self,
        data: Data,
        node_index: int,
        user_indices: torch.Tensor,
        centroid: torch.Tensor | None = None,
    ) -> float:
        z = self.embeddings(data)

        if centroid is None:
            centroid = z[user_indices].mean(dim=0)

        distances = torch.norm(z[user_indices] - centroid.unsqueeze(0), dim=1)
        mean_d = distances.mean()
        std_d = distances.std(unbiased=False)

        target_d = torch.norm(z[node_index] - centroid, dim=0)
        if float(std_d.item()) < 1e-12:
            score = torch.sigmoid(target_d - mean_d)
        else:
            zscore = (target_d - mean_d) / (std_d + 1e-8)
            score = torch.sigmoid(zscore)
        return float(score.item())

    @staticmethod
    def save_checkpoint(
        output_path: str | Path,
        encoder: GraphEncoder,
        centroid: torch.Tensor | None = None,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": encoder.model_type,
            "input_dim": encoder.conv1.in_channels,
            "hidden_dim": encoder.conv1.out_channels,
            "embedding_dim": encoder.conv2.out_channels,
            "state_dict": encoder.state_dict(),
            "centroid": centroid.detach().cpu() if centroid is not None else None,
        }
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: str | Path) -> tuple[GraphEncoder, torch.Tensor | None]:
        payload = torch.load(path, map_location="cpu")
        encoder = GraphEncoder(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
            model_type=str(payload.get("model_type", "graphsage")),
        )
        encoder.load_state_dict(payload["state_dict"])
        encoder.eval()
        centroid = payload.get("centroid")
        return encoder, centroid
