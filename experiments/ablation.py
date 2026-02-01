# experiments/ablation.py
"""
Ablation Study for LEELA

This script disables components to measure their impact:
- Temporal modeling
- Attention
- LLM semantic knowledge
"""

from copy import deepcopy
from models.leela import LEELA

def disable_attention(model):
    model.attention = None
    return model

def disable_llm(model):
    model.use_llm = False
    return model

def run_ablation(base_model, config_name):
    print(f"Running ablation: {config_name}")
    # reuse training / evaluation pipeline
    ...

if __name__ == "__main__":
    base_model = LEELA(...)

    run_ablation(deepcopy(base_model), "Full LEELA")
    run_ablation(disable_attention(deepcopy(base_model)), "w/o Attention")
    run_ablation(disable_llm(deepcopy(base_model)), "w/o LLM")