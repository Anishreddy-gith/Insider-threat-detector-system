"""
Graph Neural Network for Entity Relationship Modelling
========================================================
Uses PyTorch Geometric to model relationships between users, devices,
files, and network resources as a heterogeneous graph.

Why GNNs for Insider Threat Detection?
  1. **Lateral movement**: An insider who compromises multiple accounts
     creates unusual edges in the access graph.  GNNs naturally propagate
     this signal through the neighbourhood.
  2. **Collusion detection**: Two users who rarely interact suddenly
     sharing sensitive files creates a new edge — the GNN detects this
     as a structural anomaly.
  3. **Context-aware scoring**: A file access that's normal for an engineer
     is anomalous for HR.  The GNN encodes role-based context via the
     graph topology.

Architecture:
  • GraphSAGE-based message passing (inductive — handles new nodes without
    retraining, critical for a dynamic employee population).
  • 2-layer GNN with skip connections for gradient stability.
  • Node-level anomaly scoring via a decoder MLP.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class InsiderThreatGNN(nn.Module):
    """
    GraphSAGE-based GNN for entity-level anomaly detection in the
    user-resource access graph.

    Node types: user, device, file, network_resource
    Edge types: accessed, logged_into, transferred_to, communicated_with

    Args:
        in_channels:     Dimensionality of input node features.
        hidden_channels: Hidden layer width.
        out_channels:    Output embedding size (fed to anomaly decoder).
        num_layers:      Number of message-passing layers.
        dropout:         Dropout rate for regularisation.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Build message-passing layers.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in → hidden.
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Middle layers: hidden → hidden.
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Last conv layer: hidden → out.
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
            self.norms.append(nn.LayerNorm(out_channels))

        # Skip connection projection (if dimensions differ).
        self.skip_proj = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Anomaly scoring head — predicts anomaly probability per node.
        self.anomaly_head = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x:          (num_nodes, in_channels) — node feature matrix.
            edge_index: (2, num_edges) — COO edge list.

        Returns:
            embeddings:    (num_nodes, out_channels) — learned node representations.
            anomaly_scores: (num_nodes, 1) — per-node anomaly probability.
        """
        h = x
        x_skip = self.skip_proj(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index)
            h = norm(h)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Residual / skip connection — helps with gradient flow and
        # preserves raw input features that may be diagnostic.
        embeddings = h + x_skip

        anomaly_scores = self.anomaly_head(embeddings)

        return embeddings, anomaly_scores

    def get_node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return just the node embeddings (for downstream clustering / viz)."""
        embeddings, _ = self.forward(x, edge_index)
        return embeddings


class GraphBuilder:
    """
    Utility class to construct PyTorch Geometric graphs from raw event data.

    Converts tabular event logs into a graph where:
      • Nodes = users, devices, files, IP addresses
      • Edges = interactions (login, file access, network connection)
      • Node features = aggregated behavioural statistics
    """

    def __init__(self, feature_dim: int = 64) -> None:
        self.feature_dim = feature_dim
        self._node_index: dict[str, int] = {}
        self._edges: list[tuple[int, int]] = []
        self._node_features: list[list[float]] = []

    def add_node(self, node_id: str, features: list[float] | None = None) -> int:
        """Add a node (or return existing index)."""
        if node_id not in self._node_index:
            idx = len(self._node_index)
            self._node_index[node_id] = idx
            self._node_features.append(
                features or [0.0] * self.feature_dim
            )
        return self._node_index[node_id]

    def add_edge(self, src_id: str, dst_id: str) -> None:
        """Add a directed edge between two nodes."""
        src_idx = self.add_node(src_id)
        dst_idx = self.add_node(dst_id)
        self._edges.append((src_idx, dst_idx))

    def build(self) -> dict[str, torch.Tensor]:
        """
        Build the PyTorch Geometric data dict.

        Returns:
            dict with 'x' (node features) and 'edge_index' tensors.
        """
        x = torch.tensor(self._node_features, dtype=torch.float32)

        if self._edges:
            edge_index = torch.tensor(self._edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return {"x": x, "edge_index": edge_index}

    def reset(self) -> None:
        """Clear the builder for the next snapshot."""
        self._node_index.clear()
        self._edges.clear()
        self._node_features.clear()
