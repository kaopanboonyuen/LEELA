# models/engru.py
import torch
import torch.nn as nn

class ENGRU(nn.Module):
    """
    Enhanced GRU (ENGRU)

    This module models temporal dynamics of CPN state-space traces.
    It is intentionally kept close to the standard GRU formulation
    to preserve interpretability and alignment with formal semantics.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Standard GRU block
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x, h0=None):
        """
        Args:
            x: Tensor of shape (B, T, D)
               B = batch size
               T = trace length
               D = encoded CPN state dimension

            h0: Optional initial hidden state

        Returns:
            h: All hidden states (B, T, H)
            h_T: Final hidden state (B, H)
        """
        h, h_T = self.gru(x, h0)
        return h, h_T[-1]