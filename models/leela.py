# models/leela.py
import torch
import torch.nn as nn

from models.engru import ENGRU
from models.attention import LTLSelfAttention

class LEELA(nn.Module):
    """
    LEELA: Neural-Symbolic Fault Predictor

    Combines:
    - Temporal learning (ENGRU)
    - Semantic conditioning (LLM-based LTL embeddings)
    - Attention-based interpretability
    """

    def __init__(
        self,
        state_dim,
        hidden_dim,
        ltl_dim,
        num_layers=1
    ):
        super().__init__()

        # Temporal encoder
        self.engru = ENGRU(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # LTL-conditioned attention
        self.attention = LTLSelfAttention(
            hidden_dim=hidden_dim,
            ltl_dim=ltl_dim
        )

        # Final fault predictor
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, trace, ltl_embedding):
        """
        Args:
            trace: (B, T, D_state)
            ltl_embedding: (B, D_ltl)

        Returns:
            y_hat: (B,) fault probability
            alpha: (B, T) attention weights
        """
        hidden_states, _ = self.engru(trace)
        context, alpha = self.attention(hidden_states, ltl_embedding)

        logits = self.classifier(context)
        y_hat = torch.sigmoid(logits).squeeze(-1)

        return y_hat, alpha