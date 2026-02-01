# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LTLSelfAttention(nn.Module):
    """
    LTL-conditioned self-attention.

    Aligns temporal hidden states with an LTL semantic embedding
    obtained from a pretrained LLM.
    """

    def __init__(self, hidden_dim, ltl_dim):
        super().__init__()

        # Project LTL embedding into hidden space
        self.ltl_proj = nn.Linear(ltl_dim, hidden_dim)

    def forward(self, hidden_states, ltl_embedding):
        """
        Args:
            hidden_states: (B, T, H)
            ltl_embedding: (B, D_ltl)

        Returns:
            context: (B, H)   weighted trace representation
            alpha:   (B, T)   attention weights (interpretable)
        """
        # Project LTL embedding
        ltl_query = self.ltl_proj(ltl_embedding).unsqueeze(1)  # (B, 1, H)

        # Dot-product attention
        scores = torch.sum(hidden_states * ltl_query, dim=-1)  # (B, T)
        alpha = F.softmax(scores, dim=1)

        # Weighted sum of hidden states
        context = torch.sum(hidden_states * alpha.unsqueeze(-1), dim=1)

        return context, alpha