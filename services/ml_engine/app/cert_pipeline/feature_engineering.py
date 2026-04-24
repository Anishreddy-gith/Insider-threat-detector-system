from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS: list[str] = [
    "login_frequency",
    "after_hours_activity",
    "file_access_deviation",
    "email_anomalies",
    "session_duration",
]


def create_feature_dataframe(timeline: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "timestamp", "event_type"}
    missing = required.difference(timeline.columns)
    if missing:
        raise ValueError(f"Timeline missing required columns: {sorted(missing)}")

    df = timeline.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["user_id", "timestamp"]).copy()

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    if "activity" in df.columns:
        df["activity"] = df["activity"].fillna("").astype(str).str.strip().str.lower()
    else:
        df["activity"] = ""

    df = df.sort_values(["user_id", "timestamp", "event_type"], kind="mergesort")
    df = df.reset_index(drop=True)

    df = _add_login_frequency(df)
    df = _add_after_hours_activity(df)
    df = _add_file_access_deviation(df)
    df = _add_email_anomalies(df)
    df = _add_session_duration(df)

    output_columns = ["user_id", "timestamp", *FEATURE_COLUMNS]
    return df[output_columns].copy()


def _add_login_frequency(df: pd.DataFrame) -> pd.DataFrame:
    is_login = (df["event_type"] == "logon").astype("int8")
    rolling_login = (
        is_login.groupby(df["user_id"])
        .rolling(window=24, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["login_frequency"] = rolling_login.astype(float)
    return df


def _add_after_hours_activity(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["timestamp"].dt.hour
    is_after_hours = ((hour < 8) | (hour >= 18)).astype("int8")
    after_hours_ratio = (
        is_after_hours.groupby(df["user_id"])
        .rolling(window=48, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["after_hours_activity"] = after_hours_ratio.astype(float)
    return df


def _add_file_access_deviation(df: pd.DataFrame) -> pd.DataFrame:
    is_file = (df["event_type"] == "file").astype(float)

    # User baseline from historical behavior only.
    hist_mean = (
        is_file.groupby(df["user_id"])
        .expanding(min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
        .shift(1)
    )
    hist_std = (
        is_file.groupby(df["user_id"])
        .expanding(min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
        .shift(1)
    )

    z = (is_file - hist_mean) / (hist_std.replace(0, np.nan))
    df["file_access_deviation"] = z.fillna(0.0).clip(-10.0, 10.0)
    return df


def _add_email_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    is_email = (df["event_type"] == "email").astype(float)

    to_col = df["to"].fillna("").astype(str) if "to" in df.columns else pd.Series("", index=df.index)
    cc_col = df["cc"].fillna("").astype(str) if "cc" in df.columns else pd.Series("", index=df.index)
    bcc_col = df["bcc"].fillna("").astype(str) if "bcc" in df.columns else pd.Series("", index=df.index)

    has_external = (
        to_col.str.contains("@", regex=False)
        & ~to_col.str.contains("@company.com", case=False, regex=False)
    ) | (
        cc_col.str.contains("@", regex=False)
        & ~cc_col.str.contains("@company.com", case=False, regex=False)
    ) | (
        bcc_col.str.contains("@", regex=False)
        & ~bcc_col.str.contains("@company.com", case=False, regex=False)
    )

    external_flag = has_external.astype(float)

    email_rate = (
        is_email.groupby(df["user_id"])
        .rolling(window=24, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    external_rate = (
        external_flag.groupby(df["user_id"])
        .rolling(window=24, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    anomaly_score = (0.7 * email_rate) + (0.3 * external_rate)
    df["email_anomalies"] = anomaly_score.astype(float)
    return df


def _add_session_duration(df: pd.DataFrame) -> pd.DataFrame:
    if "activity" not in df.columns:
        df["session_duration"] = 0.0
        return df

    df["session_duration"] = 0.0

    work = df[df["event_type"] == "logon"].copy()
    if work.empty:
        return df

    work["pc"] = work.get("pc", pd.Series(index=work.index, dtype=object)).astype(str)
    work["is_logon_start"] = work["activity"].isin({"logon", "login"})
    work["is_logon_end"] = work["activity"].isin({"logoff", "logout"})

    sessions: list[tuple[str, pd.Timestamp, float]] = []
    for (_, _), grp in work.groupby(["user_id", "pc"], dropna=False):
        starts: list[pd.Timestamp] = []
        for row in grp.itertuples(index=False):
            ts = row.timestamp
            if row.is_logon_start:
                starts.append(ts)
                continue
            if row.is_logon_end and starts:
                start_ts = starts.pop(0)
                dur = max((ts - start_ts).total_seconds(), 0.0)
                sessions.append((row.user_id, ts, dur))

    if not sessions:
        return df

    session_df = pd.DataFrame(sessions, columns=["user_id", "timestamp", "session_duration"])
    session_df = session_df.sort_values(["user_id", "timestamp"])

    sorted_df = df.sort_values(["user_id", "timestamp"]).copy()
    merged = pd.merge_asof(
        sorted_df,
        session_df,
        on="timestamp",
        by="user_id",
        direction="backward",
        suffixes=("", "_calc"),
    )

    merged = merged.sort_index()
    df["session_duration"] = merged["session_duration_calc"].fillna(0.0).astype(float).values
    return df
