import unittest
import torch
from pyhealth.models.feature_attention_lstm import FeatureAttentionLSTM


class MockDataset:
    """Mock dataset class to spoof PyHealth dataset tracking."""

    def __init__(self):
        self.input_info = {"feature_val": {"input_dim": 50}}
        self.output_info = {"label_val": {"output_dim": 2}}


def test_feature_attention_lstm_forward():
    # Instantiate with dummy architecture configurations
    mock_ds = MockDataset()
    model = FeatureAttentionLSTM(
        dataset=mock_ds,
        feature_keys=["feature_val"],
        label_key="label_val",
        mode="binary",
    )

    # Generate synthetic input tensors (Batch=4, TimeSteps=24, Features=50)
    mock_batch = {
        "feature_val": torch.randn(4, 24, 50),
        "label_val": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }

    output = model(**mock_batch)

    # Enforce strict structural dimension assertions
    assert "loss" in output
    assert output["y_hat"].shape == (4, 2)
    assert output["attention_weights"].shape == (4, 24, 50)

    # Softmax Verification: Check that features sum up to 1.0 at each timestep
    summed_weights = torch.sum(output["attention_weights"], dim=-1)
    torch.testing.assert_close(
        summed_weights, torch.ones_like(summed_weights), rtol=1e-5, atol=1e-5
    )

    # Ensure gradients propagate backward seamlessly
    output["loss"].backward()
    assert model.attention_matrix.weight.grad is not None


