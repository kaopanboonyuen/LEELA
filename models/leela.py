"""
===============================================================================
LEELA: Leveraging Large Language Models and Neural Approximations
for Fault Prediction in Colored Petri Net Models
-------------------------------------------------------------------------------
This file is part of the LEELA research artifact accompanying the paper:

  "LEELA: Leveraging Large Language Models and Neural Approximations
   for Fault Prediction in Colored Petri Net Models"

LEELA is a neural–symbolic framework that integrates:
  - Temporal neural modeling (GRU-based architectures)
  - Attention mechanisms for interpretability
  - Large Language Models (LLMs) for semantic reasoning over LTL specifications

The goal is to enable scalable, interpretable fault prediction in
reactive systems modeled by Colored Petri Nets (CPNs), alleviating
state-space explosion in classical model checking.

Repository:
  https://github.com/kaopanboonyuen/LEELA

Authors:
  KKU AI Vision Research Group

License:
  Research use only (see LICENSE file)

===============================================================================
"""

# models/leela.py

import torch
import torch.nn as nn
from typing import Tuple

from models.engru import ENGRU
from models.attention import LTLSelfAttention


class LEELA(nn.Module):
    """
    LEELA: Neural-Symbolic Fault Predictor

    This module serves as the central reasoning unit of the LEELA framework.
    It replaces classical binary model checking with *probabilistic temporal
    reasoning* while preserving interpretability and alignment with formal
    semantics.

    Unlike purely neural classifiers, LEELA is explicitly structured to
    reflect the reasoning pipeline of temporal logic verification.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        ltl_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """
        Args:
            state_dim:
                Dimensionality of encoded CPN states.
                Each state represents a snapshot of markings and variables.

            hidden_dim:
                Dimensionality of the temporal latent space.
                Acts as the neural analogue of abstract system states.

            ltl_dim:
                Dimensionality of LTL semantic embeddings produced by the LLM.

            num_layers:
                Number of recurrent layers in the ENGRU.
                Kept small to preserve interpretability and stability.

            dropout:
                Optional dropout for temporal regularization.
        """
        super().__init__()

        # ---------------------------------------------------------------------
        # (1) Temporal Encoder
        # ---------------------------------------------------------------------
        # ENGRU learns how system states evolve over time.
        # This replaces explicit state-space traversal with learned dynamics.
        self.temporal_encoder = ENGRU(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # ---------------------------------------------------------------------
        # (2) LTL-Conditioned Attention
        # ---------------------------------------------------------------------
        # Attention softly aligns execution traces with temporal logic intent.
        # This is the key neural–symbolic bridge in LEELA.
        self.ltl_attention = LTLSelfAttention(
            hidden_dim=hidden_dim,
            ltl_dim=ltl_dim
        )

        # ---------------------------------------------------------------------
        # (3) Fault Prediction Head
        # ---------------------------------------------------------------------
        # Outputs a probabilistic fault likelihood rather than a hard decision,
        # enabling ranking, thresholding, and risk-aware verification.
        self.fault_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        trace: torch.Tensor,
        ltl_embedding: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LEELA.

        Args:
            trace:
                Tensor of shape (B, T, D_state)
                where B = batch size,
                      T = length of execution trace,
                      D_state = encoded CPN state dimension.

            ltl_embedding:
                Tensor of shape (B, D_ltl)
                Semantic embedding of LTL specifications obtained from an LLM.

            return_attention:
                If True, returns attention weights for interpretability.

        Returns:
            y_hat:
                Tensor of shape (B,)
                Probabilistic fault likelihood in [0, 1].

            alpha (optional):
                Tensor of shape (B, T)
                Attention weights indicating which time steps
                contributed most to the decision.
        """

        # ---------------------------------------------------------------------
        # Step 1: Temporal Reasoning over Execution Trace
        # ---------------------------------------------------------------------
        # The ENGRU compresses raw traces into a sequence of latent states
        # representing abstract system behavior over time.
        hidden_states, _ = self.temporal_encoder(trace)
        # hidden_states: (B, T, H)

        # ---------------------------------------------------------------------
        # Step 2: Neural-Symbolic Alignment with LTL Semantics
        # ---------------------------------------------------------------------
        # Attention computes a soft alignment between each time step
        # and the LTL semantic embedding.
        context, attention_weights = self.ltl_attention(
            hidden_states,
            ltl_embedding
        )
        # context: (B, H)
        # attention_weights: (B, T)

        # ---------------------------------------------------------------------
        # Step 3: Fault Likelihood Estimation
        # ---------------------------------------------------------------------
        # The context vector summarizes the trace *as viewed through*
        # the lens of the LTL specification.
        logits = self.fault_head(context)
        y_hat = torch.sigmoid(logits).squeeze(-1)

        if return_attention:
            return y_hat, attention_weights
        else:
            return y_hat