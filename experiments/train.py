# experiments/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.leela import LEELA
from ltl.llm_embeddings import LTLEmbedder

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    criterion = nn.BCELoss()

    for trace, ltl_text, label in dataloader:
        trace = trace.to(device)
        label = label.to(device)

        ltl_embed = embedder.embed(ltl_text).to(device)

        optimizer.zero_grad()
        y_hat, _ = model(trace, ltl_embed)

        loss = criterion(y_hat, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LEELA(
        state_dim=128,
        hidden_dim=256,
        ltl_dim=384
    ).to(device)

    embedder = LTLEmbedder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dataset assumed to return:
    # (trace_tensor, ltl_string, label)
    train_loader = DataLoader(...)

    for epoch in range(50):
        loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")