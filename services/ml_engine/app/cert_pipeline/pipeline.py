from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_loader import CertDataLoader
from .feature_engineering import create_feature_dataframe


def run_pipeline(input_dir: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    loader = CertDataLoader(input_dir)
    timeline = loader.build_unified_timeline()
    feature_df = create_feature_dataframe(timeline)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".parquet":
            feature_df.to_parquet(out, index=False)
        else:
            feature_df.to_csv(out, index=False)

    return feature_df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CERT insider threat feature pipeline")
    parser.add_argument("--input-dir", required=True, help="Directory containing CERT CSV files")
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output file path (.csv or .parquet) for clean feature dataframe",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_pipeline(input_dir=args.input_dir, output_path=args.output_path)


if __name__ == "__main__":
    main()
