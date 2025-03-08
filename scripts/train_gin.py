import torch
from torch_geometric.loader import DataLoader
from models.gin_model import GIN
from data.mutag_dataset import load_mutag_dataset

# Training and testing functions here (as you provided originally)

# Training function
def train_gin(model, train_loader, optimizer, criterion, device, train_dataset):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)

# Testing function
def test_gin(model, full_loader, device, return_hidden=True, layer=1):
    model.eval()
    correct = 0
    all_hidden = []
    with torch.no_grad():
        for data in full_loader:
            data = data.to(device)
            if return_hidden:
                hidden, out = model(data, return_hidden=True, layer=layer)
                all_hidden.append(hidden.cpu())
            else:
                out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    acc = correct / len(full_loader.dataset)
    if return_hidden:
        return acc, torch.cat(all_hidden, dim=0)
    return acc