"""
SHAP Explainer for Insider Threat Models
==========================================
Generates SHAP (SHapley Additive exPlanations) values to explain *why*
a particular user was flagged as anomalous.

Why SHAP for Insider Threat XAI?
  1. **Regulatory requirement**: GDPR Art. 22 gives individuals the right
     not to be subject to "solely automated" decisions.  SHAP explanations
     enable human-in-the-loop review by showing contributing factors.
  2. **Analyst trust**: SOC analysts won't act on opaque ML scores.
     SHAP provides feature-level attributions (e.g. "75% of the anomaly
     score comes from after-hours file downloads").
  3. **Model debugging**: SHAP reveals spurious correlations in the model
     (e.g. flagging users just because they're in a different timezone).

Implementation:
  • We use ``shap.DeepExplainer`` for the LSTM autoencoder (works with
    PyTorch models) and ``shap.KernelExplainer`` as a model-agnostic
    fallback for the ensemble.
  • Background dataset is a sample of "normal" behaviour used as the
    baseline for Shapley value computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from app.features.extractor import FEATURE_NAMES
from shared.utils.logging import get_logger

log = get_logger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for the anomaly detection models.

    Args:
        model:            The PyTorch model to explain.
        background_data:  np.ndarray of shape (n_samples, seq_len, n_features)
                          representing "normal" behaviour.  Used as the
                          baseline for Shapley value computation.
        feature_names:    Human-readable feature names.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES
        self.background_data = background_data
        self._explainer = None

    def _lazy_init(self) -> None:
        """Lazily initialise the SHAP explainer (avoid import-time overhead)."""
        if self._explainer is not None:
            return

        import shap

        if self.background_data is not None:
            bg_tensor = torch.tensor(self.background_data, dtype=torch.float32)
            try:
                # DeepExplainer is faster but requires a PyTorch/TF model.
                self._explainer = shap.DeepExplainer(self.model, bg_tensor)
                log.info("shap.deep_explainer_initialized")
            except Exception:
                # Fallback to KernelExplainer (model-agnostic but slower).
                def _predict_fn(x: np.ndarray) -> np.ndarray:
                    with torch.no_grad():
                        t = torch.tensor(x, dtype=torch.float32)
                        out = self.model.get_anomaly_score(t)
                    return out.numpy()

                self._explainer = shap.KernelExplainer(_predict_fn, self.background_data)
                log.info("shap.kernel_explainer_initialized")

    def explain(
        self, features: np.ndarray, top_k: int = 10
    ) -> dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            features: (1, seq_len, n_features) or (1, n_features) input.
            top_k:    Return the top-K most influential features.

        Returns:
            dict with:
              - top_features: [{name, shap_value, direction}]
              - natural_language_explanation: human-readable summary
              - raw_shap_values: full SHAP value array
        """
        self._lazy_init()

        if self._explainer is None:
            return self._fallback_explanation(features)

        try:
            import shap
            if isinstance(self._explainer, shap.DeepExplainer):
                tensor_input = torch.tensor(features, dtype=torch.float32)
                shap_values = self._explainer.shap_values(tensor_input)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_array = np.array(shap_values)
            else:
                shap_values = self._explainer.shap_values(features)
                shap_array = np.array(shap_values)

            # Aggregate over time dimension if 3D.
            if shap_array.ndim == 3:
                shap_per_feature = np.mean(np.abs(shap_array[0]), axis=0)
            else:
                shap_per_feature = np.abs(shap_array[0])

            return self._format_explanation(shap_per_feature, features, top_k)

        except Exception as exc:
            log.warning("shap.explain_failed", error=str(exc))
            return self._fallback_explanation(features)

    def _format_explanation(
        self,
        shap_per_feature: np.ndarray,
        features: np.ndarray,
        top_k: int,
    ) -> dict[str, Any]:
        """Format SHAP values into a structured explanation."""
        # Sort by absolute SHAP value descending.
        indices = np.argsort(shap_per_feature)[::-1][:top_k]

        top_features = []
        for idx in indices:
            name = (
                self.feature_names[idx]
                if idx < len(self.feature_names)
                else f"feature_{idx}"
            )
            top_features.append({
                "name": name,
                "shap_value": float(shap_per_feature[idx]),
                "direction": "increases_risk" if shap_per_feature[idx] > 0 else "decreases_risk",
            })

        # Generate natural language explanation.
        explanation = self._generate_nl_explanation(top_features)

        return {
            "top_features": top_features,
            "natural_language_explanation": explanation,
            "raw_shap_values": shap_per_feature.tolist(),
        }

    def _fallback_explanation(self, features: np.ndarray) -> dict[str, Any]:
        """
        Fallback when SHAP is unavailable — use per-feature reconstruction
        error as a proxy for feature importance.
        """
        if hasattr(self.model, "get_feature_reconstruction_error"):
            tensor = torch.tensor(features, dtype=torch.float32)
            errors = self.model.get_feature_reconstruction_error(tensor)[0].numpy()
        else:
            errors = np.zeros(len(self.feature_names))

        return self._format_explanation(errors, features, top_k=10)

    @staticmethod
    def _generate_nl_explanation(top_features: list[dict[str, Any]]) -> str:
        """
        Create a human-readable summary from the top SHAP features.

        Example output:
            "This user was flagged primarily due to: after-hours file
            downloads (42% contribution), access to restricted files
            (28% contribution), and unusual network destinations
            (15% contribution)."
        """
        if not top_features:
            return "No significant behavioural deviations detected."

        total_shap = sum(f["shap_value"] for f in top_features) or 1.0

        parts = []
        for feat in top_features[:3]:
            pct = int(100 * feat["shap_value"] / total_shap)
            readable_name = feat["name"].replace("_", " ")
            parts.append(f"{readable_name} ({pct}% contribution)")

        return (
            "This user was flagged primarily due to: "
            + ", ".join(parts[:-1])
            + (f", and {parts[-1]}" if len(parts) > 1 else parts[0])
            + "."
        )
