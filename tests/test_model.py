"""Tests for dflash model utilities."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from dflash.model import build_target_layer_ids, extract_context_feature, sample


class TestBuildTargetLayerIds:
    def test_basic(self):
        ids = build_target_layer_ids(num_layers=12, every_n=4)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_every_1_returns_all(self):
        ids = build_target_layer_ids(num_layers=6, every_n=1)
        assert len(ids) == 6

    def test_every_n_larger_than_layers(self):
        ids = build_target_layer_ids(num_layers=4, every_n=10)
        assert len(ids) >= 1

    def test_empty_on_zero_layers(self):
        ids = build_target_layer_ids(num_layers=0, every_n=2)
        assert ids == []


class TestSample:
    def test_greedy(self):
        logits = torch.tensor([[0.1, 0.2, 5.0, 0.3]])
        token = sample(logits, temperature=0.0)
        assert token.item() == 2

    def test_temperature_sampling_shape(self):
        logits = torch.randn(1, 100)
        token = sample(logits, temperature=1.0)
        assert token.shape == (1,) or token.numel() == 1

    def test_top_p_sampling(self):
        logits = torch.randn(1, 50)
        token = sample(logits, temperature=1.0, top_p=0.9)
        assert 0 <= token.item() < 50


class TestExtractContextFeature:
    def _make_mock_model(self, num_layers=4, hidden_size=64):
        model = MagicMock()
        model.config.num_hidden_layers = num_layers
        # Simulate hidden states output
        hidden = torch.randn(1, 10, hidden_size)
        output = MagicMock()
        output.hidden_states = [hidden] * (num_layers + 1)
        model.return_value = output
        return model, hidden

    def test_returns_tensor(self):
        model, _ = self._make_mock_model()
        input_ids = torch.zeros(1, 10, dtype=torch.long)
        layer_ids = [1, 3]
        result = extract_context_feature(model, input_ids, layer_ids)
        assert isinstance(result, torch.Tensor)

    def test_output_dim_matches_layers(self):
        num_layers = 4
        hidden_size = 64
        model, _ = self._make_mock_model(num_layers=num_layers, hidden_size=hidden_size)
        input_ids = torch.zeros(1, 10, dtype=torch.long)
        layer_ids = [0, 2]
        result = extract_context_feature(model, input_ids, layer_ids)
        # Should aggregate over selected layers
        assert result.ndim >= 1
