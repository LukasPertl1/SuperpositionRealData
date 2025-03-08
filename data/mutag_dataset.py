from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.loader import DataLoader

def load_mutag_dataset():
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=32, shuffle=False) 
    return dataset, train_dataset, train_loader, test_loader, full_loader