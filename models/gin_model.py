import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes):
        super(GIN, self).__init__()
        # First GIN layer: uses hidden_dims[0]
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dims[0])

        # Second GIN layer: uses hidden_dims[1]
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[1])
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dims[1])

        # Final linear layer for graph classification uses the last hidden dim.
        self.linear = torch.nn.Linear(hidden_dims[1], num_classes)

    def forward(self, data, return_hidden=False, layer=1):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        hidden1 = x.clone()  # store hidden representation after layer 1

        # Second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        hidden2 = x.clone()  # store hidden representation after layer 2

        # Global pooling to get graph-level embedding
        out = global_add_pool(x, batch)
        logits = self.linear(out)

        if return_hidden:
            if layer == 1:
                return hidden1, logits
            elif layer == 2:
                return hidden2, logits
        return logits