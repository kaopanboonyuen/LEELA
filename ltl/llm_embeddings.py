# ltl/llm_embeddings.py
import torch
from transformers import AutoTokenizer, AutoModel

class LTLEmbedder:
    """
    Encodes LTL templates using a pretrained LLM.

    This enables semantic generalization across
    syntactically different but logically similar formulas.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed(self, ltl_strings):
        """
        Args:
            ltl_strings: List[str] of LTL templates

        Returns:
            embeddings: (B, D)
        """
        tokens = self.tokenizer(
            ltl_strings,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings