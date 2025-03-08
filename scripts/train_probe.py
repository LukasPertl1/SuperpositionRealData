import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.linear_probe import LinearProbe


def train_probe(probe, loader, optimizer, criterion, device, which_feature=0):
    probe.train()
    total_loss = 0.0
    total_samples = 0
    for embeddings, targets in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = probe(embeddings)  # shape: [batch_size, 1]
        target = targets.unsqueeze(1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * embeddings.size(0)
        total_samples += embeddings.size(0)
    return total_loss / total_samples

def evaluate_probe(probe, loader, device, criterion, which_feature=0):
    probe.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for embeddings, targets in loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            outputs = probe(embeddings)
            target = targets.unsqueeze(1)
            loss = criterion(outputs, target)
            total_loss += loss.item() * embeddings.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == target).float().sum().item()
            total_samples += embeddings.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f'Bias {probe.fc.bias}')
    return avg_loss, accuracy, probe.fc.weight.data