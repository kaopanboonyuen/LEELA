# experiments/evaluate.py
import torch
from sklearn.metrics import accuracy_score, f1_score

from models.leela import LEELA
from ltl.llm_embeddings import LTLEmbedder

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for trace, ltl_text, label in dataloader:
            trace = trace.to(device)
            ltl_embed = embedder.embed(ltl_text).to(device)

            y_hat, attention = model(trace, ltl_embed)
            preds.extend((y_hat > 0.5).cpu().numpy())
            labels.extend(label.numpy())

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LEELA(...)
    model.load_state_dict(torch.load("checkpoints/leela.pt"))
    model.to(device)

    embedder = LTLEmbedder()
    test_loader = DataLoader(...)

    metrics = evaluate(model, test_loader, device)
    print(metrics)