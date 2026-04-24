from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch_geometric.data import Data

from services.ml_engine.app.utils.preprocessing import FEATURE_COLUMNS

USER_NODE = 0
DEVICE_NODE = 1
FILE_NODE = 2


@dataclass(slots=True)
class GraphBuildResult:
    data: Data
    node_index_map: dict[str, int]
    node_types: torch.Tensor
    node_ids: list[str]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    if required:
        raise KeyError(f"Missing required columns. Expected one of {list(candidates)}")
    return None


def _parse_recipients(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    for sep in [";", ","]:
        text = text.replace(sep, "|")
    return [t.strip() for t in text.split("|") if t.strip()]


def _build_node_maps(
    logon_df: pd.DataFrame,
    file_df: pd.DataFrame,
    email_df: pd.DataFrame,
) -> tuple[dict[str, int], list[str], torch.Tensor]:
    user_ids: set[str] = set()
    device_ids: set[str] = set()
    file_ids: set[str] = set()

    if not logon_df.empty:
        user_col = _resolve_column(logon_df, ["user", "user_id", "userid"])
        device_col = _resolve_column(logon_df, ["pc", "device", "host", "machine"], required=False)
        user_ids.update(logon_df[user_col].astype(str).str.strip().tolist())
        if device_col:
            device_ids.update(logon_df[device_col].astype(str).str.strip().tolist())

    if not file_df.empty:
        user_col = _resolve_column(file_df, ["user", "user_id", "userid"])
        file_col = _resolve_column(file_df, ["filename", "file_name", "path"])
        user_ids.update(file_df[user_col].astype(str).str.strip().tolist())
        file_ids.update(file_df[file_col].astype(str).str.strip().tolist())

    if not email_df.empty:
        user_col = _resolve_column(email_df, ["user", "user_id", "userid"])
        to_col = _resolve_column(email_df, ["to", "recipient", "recipients"], required=False)
        cc_col = _resolve_column(email_df, ["cc"], required=False)
        bcc_col = _resolve_column(email_df, ["bcc"], required=False)

        user_ids.update(email_df[user_col].astype(str).str.strip().tolist())
        for col in [to_col, cc_col, bcc_col]:
            if not col:
                continue
            for value in email_df[col].tolist():
                for recipient in _parse_recipients(value):
                    local = recipient.split("@", 1)[0].strip().lower()
                    if local:
                        user_ids.add(local)

    user_nodes = [f"u:{u}" for u in sorted(x for x in user_ids if x)]
    device_nodes = [f"d:{d}" for d in sorted(x for x in device_ids if x)]
    file_nodes = [f"f:{f}" for f in sorted(x for x in file_ids if x)]

    node_ids = user_nodes + device_nodes + file_nodes
    node_index_map = {nid: idx for idx, nid in enumerate(node_ids)}

    node_types = torch.empty(len(node_ids), dtype=torch.long)
    node_types[: len(user_nodes)] = USER_NODE
    node_types[len(user_nodes) : len(user_nodes) + len(device_nodes)] = DEVICE_NODE
    node_types[len(user_nodes) + len(device_nodes) :] = FILE_NODE

    return node_index_map, node_ids, node_types


def _build_edge_index(
    logon_df: pd.DataFrame,
    file_df: pd.DataFrame,
    email_df: pd.DataFrame,
    node_index_map: dict[str, int],
) -> torch.Tensor:
    edges: list[tuple[int, int]] = []

    if not logon_df.empty:
        user_col = _resolve_column(logon_df, ["user", "user_id", "userid"])
        device_col = _resolve_column(logon_df, ["pc", "device", "host", "machine"], required=False)
        if device_col:
            for row in logon_df[[user_col, device_col]].itertuples(index=False):
                u = f"u:{str(row[0]).strip()}"
                d = f"d:{str(row[1]).strip()}"
                if u in node_index_map and d in node_index_map:
                    edges.append((node_index_map[u], node_index_map[d]))

    if not file_df.empty:
        user_col = _resolve_column(file_df, ["user", "user_id", "userid"])
        file_col = _resolve_column(file_df, ["filename", "file_name", "path"])
        for row in file_df[[user_col, file_col]].itertuples(index=False):
            u = f"u:{str(row[0]).strip()}"
            f = f"f:{str(row[1]).strip()}"
            if u in node_index_map and f in node_index_map:
                edges.append((node_index_map[u], node_index_map[f]))

    if not email_df.empty:
        user_col = _resolve_column(email_df, ["user", "user_id", "userid"])
        to_col = _resolve_column(email_df, ["to", "recipient", "recipients"], required=False)
        cc_col = _resolve_column(email_df, ["cc"], required=False)
        bcc_col = _resolve_column(email_df, ["bcc"], required=False)

        cols = [c for c in [to_col, cc_col, bcc_col] if c]
        for row in email_df[[user_col, *cols]].itertuples(index=False):
            src = f"u:{str(row[0]).strip()}"
            recipients: list[str] = []
            for part in row[1:]:
                recipients.extend(_parse_recipients(part))
            for recipient in recipients:
                local = recipient.split("@", 1)[0].strip().lower()
                dst = f"u:{local}"
                if src in node_index_map and dst in node_index_map:
                    edges.append((node_index_map[src], node_index_map[dst]))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def _user_feature_table(behavior_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["user_id", *FEATURE_COLUMNS]
    missing = [c for c in cols if c not in behavior_df.columns]
    if missing:
        raise ValueError(f"Behavior dataframe missing required columns: {missing}")

    user_df = behavior_df[cols].copy()
    user_df["user_id"] = user_df["user_id"].astype(str).str.strip()
    for c in FEATURE_COLUMNS:
        user_df[c] = pd.to_numeric(user_df[c], errors="coerce").fillna(0.0)

    user_agg = user_df.groupby("user_id", as_index=False)[FEATURE_COLUMNS].mean()
    return user_agg


def build_graph_from_cert_frames(
    logon_df: pd.DataFrame,
    file_df: pd.DataFrame,
    email_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> GraphBuildResult:
    node_index_map, node_ids, node_types = _build_node_maps(logon_df, file_df, email_df)
    edge_index = _build_edge_index(logon_df, file_df, email_df, node_index_map)

    behavior_agg = _user_feature_table(behavior_df)
    behavior_map = {
        str(row["user_id"]).strip(): row[FEATURE_COLUMNS].to_numpy(dtype="float32")
        for _, row in behavior_agg.iterrows()
    }

    node_count = len(node_ids)
    feature_dim = len(FEATURE_COLUMNS) + 5
    x = torch.zeros((node_count, feature_dim), dtype=torch.float32)

    if edge_index.numel() > 0:
        out_deg = torch.bincount(edge_index[0], minlength=node_count).float()
        in_deg = torch.bincount(edge_index[1], minlength=node_count).float()
    else:
        out_deg = torch.zeros(node_count, dtype=torch.float32)
        in_deg = torch.zeros(node_count, dtype=torch.float32)

    interaction_total = in_deg + out_deg

    user_device_count = torch.zeros(node_count, dtype=torch.float32)
    user_file_count = torch.zeros(node_count, dtype=torch.float32)
    user_user_count = torch.zeros(node_count, dtype=torch.float32)

    for src, dst in edge_index.t().tolist() if edge_index.numel() > 0 else []:
        src_type = int(node_types[src].item())
        dst_type = int(node_types[dst].item())
        if src_type == USER_NODE and dst_type == DEVICE_NODE:
            user_device_count[src] += 1.0
        elif src_type == USER_NODE and dst_type == FILE_NODE:
            user_file_count[src] += 1.0
        elif src_type == USER_NODE and dst_type == USER_NODE:
            user_user_count[src] += 1.0

    for idx, nid in enumerate(node_ids):
        if nid.startswith("u:"):
            raw_user = nid[2:]
            x[idx, : len(FEATURE_COLUMNS)] = torch.tensor(
                behavior_map.get(raw_user, [0.0] * len(FEATURE_COLUMNS)),
                dtype=torch.float32,
            )

        x[idx, len(FEATURE_COLUMNS) + 0] = in_deg[idx]
        x[idx, len(FEATURE_COLUMNS) + 1] = out_deg[idx]
        x[idx, len(FEATURE_COLUMNS) + 2] = interaction_total[idx]
        x[idx, len(FEATURE_COLUMNS) + 3] = user_device_count[idx] + user_file_count[idx]
        x[idx, len(FEATURE_COLUMNS) + 4] = user_user_count[idx]

    data = Data(x=x, edge_index=edge_index)
    return GraphBuildResult(
        data=data,
        node_index_map=node_index_map,
        node_types=node_types,
        node_ids=node_ids,
    )


def build_graph_from_cert_directory(cert_dir: str | Path, behavior_df: pd.DataFrame) -> GraphBuildResult:
    cert_path = Path(cert_dir)
    logon_df = _safe_read_csv(cert_path / "logon.csv")
    file_df = _safe_read_csv(cert_path / "file.csv")
    email_df = _safe_read_csv(cert_path / "email.csv")

    if logon_df.empty and file_df.empty and email_df.empty:
        raise FileNotFoundError(
            f"No CERT graph source files found in {cert_path}. Expected logon.csv/file.csv/email.csv"
        )

    return build_graph_from_cert_frames(
        logon_df=logon_df,
        file_df=file_df,
        email_df=email_df,
        behavior_df=behavior_df,
    )

