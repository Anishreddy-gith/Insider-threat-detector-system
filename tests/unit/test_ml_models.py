"""
Unit tests for the Behavioral Analytics models.
"""

import torch
import pytest
from services.behavioral_analytics_models import LSTMAutoencoder, EnsembleScorer


class TestLSTMAutoencoder:
    @pytest.fixture
    def model(self):
        return LSTMAutoencoder(
            input_dim=45,
            hidden_dim=64,
            num_layers=2,
            seq_len=24,
        )

    def test_output_shape(self, model):
        """Reconstruction should match input shape."""
        batch = torch.randn(8, 24, 45)
        reconstructed = model(batch)
        assert reconstructed.shape == batch.shape

    def test_anomaly_score_range(self, model):
        """Anomaly scores should be in [0, 1] after sigmoid."""
        batch = torch.randn(4, 24, 45)
        scores = model.anomaly_score(batch)
        assert scores.shape == (4,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_encoder_latent_dim(self, model):
        batch = torch.randn(4, 24, 45)
        latent = model.encode(batch)
        assert latent.shape == (4, 64)


class TestEnsembleScorer:
    def test_weighted_average_fallback(self):
        """Before meta-learner training, should use weighted average."""
        scorer = EnsembleScorer()
        scores = {"autoencoder": 0.8, "gnn": 0.6}
        result = scorer.score(scores)
        assert 0 <= result <= 1

    def test_consistent_output(self):
        scorer = EnsembleScorer()
        scores = {"autoencoder": 0.5, "gnn": 0.5}
        r1 = scorer.score(scores)
        r2 = scorer.score(scores)
        assert r1 == r2
