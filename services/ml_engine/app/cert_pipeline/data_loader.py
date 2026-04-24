from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class CertDataLoader:
    base_path: str | Path

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path)

    def load_logon(self) -> pd.DataFrame:
        df = self._read_csv("logon")
        return self._normalize_logon(df)

    def load_device(self) -> pd.DataFrame:
        df = self._read_csv("device")
        return self._normalize_device(df)

    def load_file(self) -> pd.DataFrame:
        df = self._read_csv("file")
        return self._normalize_file(df)

    def load_email(self) -> pd.DataFrame:
        df = self._read_csv("email")
        return self._normalize_email(df)

    def load_all(self) -> dict[str, pd.DataFrame]:
        return {
            "logon": self.load_logon(),
            "device": self.load_device(),
            "file": self.load_file(),
            "email": self.load_email(),
        }

    def build_unified_timeline(self) -> pd.DataFrame:
        frames = list(self.load_all().values())
        timeline = pd.concat(frames, ignore_index=True, sort=False)
        timeline = timeline.dropna(subset=["user_id", "timestamp"]).copy()
        timeline["user_id"] = timeline["user_id"].astype(str).str.strip()
        timeline["event_type"] = timeline["event_type"].astype(str).str.strip().str.lower()
        timeline = timeline.sort_values(["user_id", "timestamp", "event_type"], kind="mergesort")
        timeline = timeline.reset_index(drop=True)
        return timeline

    def _read_csv(self, stem: str) -> pd.DataFrame:
        for candidate in (f"{stem}.csv", f"{stem.capitalize()}.csv", f"{stem.upper()}.csv"):
            file_path = self.base_path / candidate
            if file_path.exists():
                df = pd.read_csv(file_path, low_memory=False)
                return self._normalize_columns(df)
        raise FileNotFoundError(f"Missing CERT file for '{stem}' in {self.base_path}")

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        norm = {c: str(c).strip().lower() for c in df.columns}
        return df.rename(columns=norm)

    @staticmethod
    def _resolve_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
        cols = {c.lower(): c for c in df.columns}
        for name in candidates:
            if name in cols:
                return cols[name]
        if required:
            raise KeyError(f"Could not find required column among: {candidates}")
        return None

    @staticmethod
    def _to_timestamp(series: pd.Series) -> pd.Series:
        ts = pd.to_datetime(series, errors="coerce", utc=True)
        if ts.isna().all():
            ts = pd.to_datetime(series, errors="coerce")
            ts = ts.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        return ts

    def _base_event_frame(self, df: pd.DataFrame, event_type: str) -> pd.DataFrame:
        user_col = self._resolve_column(df, ["user", "user_id", "userid"])
        time_col = self._resolve_column(df, ["date", "timestamp", "time"])
        out = pd.DataFrame(
            {
                "user_id": df[user_col].astype(str).str.strip(),
                "timestamp": self._to_timestamp(df[time_col]),
                "event_type": event_type,
            }
        )
        return out

    def _normalize_logon(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._base_event_frame(df, "logon")
        activity_col = self._resolve_column(df, ["activity", "action"], required=False)
        pc_col = self._resolve_column(df, ["pc", "host", "machine"], required=False)

        out["activity"] = (
            df[activity_col].astype(str).str.strip().str.lower() if activity_col else "unknown"
        )
        out["pc"] = df[pc_col].astype(str).str.strip() if pc_col else pd.NA
        return out

    def _normalize_device(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._base_event_frame(df, "device")
        activity_col = self._resolve_column(df, ["activity", "action"], required=False)
        pc_col = self._resolve_column(df, ["pc", "host", "machine"], required=False)

        out["activity"] = (
            df[activity_col].astype(str).str.strip().str.lower() if activity_col else "unknown"
        )
        out["pc"] = df[pc_col].astype(str).str.strip() if pc_col else pd.NA
        return out

    def _normalize_file(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._base_event_frame(df, "file")
        pc_col = self._resolve_column(df, ["pc", "host", "machine"], required=False)
        filename_col = self._resolve_column(df, ["filename", "file_name", "path"], required=False)
        to_media_col = self._resolve_column(df, ["to_removable_media", "to_removable", "to_usb"], required=False)
        from_media_col = self._resolve_column(df, ["from_removable_media", "from_removable", "from_usb"], required=False)

        out["pc"] = df[pc_col].astype(str).str.strip() if pc_col else pd.NA
        out["filename"] = df[filename_col].astype(str).str.strip() if filename_col else pd.NA
        out["to_removable_media"] = self._to_bool(df[to_media_col]) if to_media_col else False
        out["from_removable_media"] = self._to_bool(df[from_media_col]) if from_media_col else False
        out["activity"] = "file_access"
        return out

    def _normalize_email(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._base_event_frame(df, "email")
        pc_col = self._resolve_column(df, ["pc", "host", "machine"], required=False)
        to_col = self._resolve_column(df, ["to", "recipient", "recipients"], required=False)
        cc_col = self._resolve_column(df, ["cc"], required=False)
        bcc_col = self._resolve_column(df, ["bcc"], required=False)
        size_col = self._resolve_column(df, ["size", "bytes", "email_size"], required=False)

        out["pc"] = df[pc_col].astype(str).str.strip() if pc_col else pd.NA
        out["to"] = df[to_col].fillna("").astype(str) if to_col else ""
        out["cc"] = df[cc_col].fillna("").astype(str) if cc_col else ""
        out["bcc"] = df[bcc_col].fillna("").astype(str) if bcc_col else ""
        out["email_size"] = pd.to_numeric(df[size_col], errors="coerce").fillna(0.0) if size_col else 0.0
        out["activity"] = "email_send"
        return out

    @staticmethod
    def _to_bool(series: pd.Series) -> pd.Series:
        values = series.astype(str).str.strip().str.lower()
        return values.isin({"1", "true", "t", "yes", "y"})


def load_and_merge_cert_logs(base_path: str | Path) -> pd.DataFrame:
    loader = CertDataLoader(base_path=base_path)
    return loader.build_unified_timeline()
