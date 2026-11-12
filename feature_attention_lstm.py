from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from pyhealth.models import BaseModel


class FeatureAttentionLSTM(BaseModel):
    """Feature-level Attention LSTM for clinical time-series prediction.

    This model implements a dense attention layer across the feature dimension
    before passing the weighted representation into an LSTM network, as
    detailed in Yadav & Subbian (2025).

    Attributes:
        dataset: The PyHealth dataset object tracker.
        feature_keys: List of string keys representing input features.
        label_key: String key representing the target label.
        mode: Type of task, 'binary', 'multiclass', or 'multilabel'.
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super(FeatureAttentionLSTM, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # In PyHealth, input dimensions are inferred from the dataset input info
        # We assume a combined numerical/embedded feature dimension D
        self.feature_dim = self.get_input_info(feature_keys[0])["input_dim"]

        # Attention Weight Matrix W (Dimension: D x D)
        self.attention_matrix = nn.Linear(
            self.feature_dim, self.feature_dim, bias=False
        )

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output Layer matching PyHealth standard classification heads
        self.fc = nn.Linear(hidden_dim, self.get_output_dim(label_key))

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            **kwargs: Dictionary containing batch data from PyHealth DataLoader.

        Returns:
            Dict containing 'loss', 'y_hat', 'y_true', and 'attention_weights'.
        """
        # PyHealth inputs shape: (batch_size, time_steps, feature_dim)
        x = kwargs[self.feature_keys[0]]
        y_true = kwargs[self.label_key]

        # 1. Compute Attention Scores: S = W * X
        # Shape remains (batch_size, time_steps, feature_dim)
        attention_scores = self.attention_matrix(x)

        # 2. Normalize via Softmax along the Feature Dimension (dim=-1)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 3. Apply Element-wise Multiplication: X_new = X * A
        x_new = x * attention_weights

        # 4. Pass through Sequential LSTM
        lstm_out, (h_n, _) = self.lstm(x_new)

        # Use the last time-step hidden state for classification
        last_hidden = lstm_out[:, -1, :]

        # 5. Output Prediction Head
        y_hat = self.fc(last_hidden)

        # PyHealth internal loss computation support
        loss = self.get_loss_function()(y_hat, y_true)

        return {
            "loss": loss,
            "y_hat": y_hat,
            "y_true": y_true,
            "attention_weights": attention_weights,
        }

