"""
GraphSAGE-based GNN for relational anomaly detection.

Entities (users) are graph nodes; edges represent interactions (shared
file access, email exchanges, IM conversations, shared-printer usage).
GraphSAGE aggregates neighbourhood features to detect collusion patterns
and lateral movement that per-entity models would miss.

WHY GraphSAGE over GCN?
───────────────────────
GraphSAGE uses *sampling* + aggregation, making it inductive – it can
score new users who weren't in the training graph.  Vanilla GCN is
transductive and must re-train when the graph structure changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class InsiderThreatGNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 45,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Skip connection: project input_dim → hidden_dim
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Anomaly head: maps hidden → scalar anomaly score
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return per-node anomaly scores in [0, 1]."""
        residual = self.skip(x)
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + residual                             # skip connection
        return self.anomaly_head(h).squeeze(-1)       # (N,)
