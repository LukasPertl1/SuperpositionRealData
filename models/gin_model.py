import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn as nn

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes, layer=2):
        """
        Initializes the GIN model with an arbitrary number of layers.

        Parameters:
            num_features (int): Number of input features.
            hidden_dims (list of int): A list where each entry specifies the output dimension of a GIN layer.
            num_classes (int): Number of output classes.
        """
        super(GIN, self).__init__()

        self.num_layers = len(hidden_dims)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Build the GIN layers based on hidden_dims
        for i in range(self.num_layers):
            in_dim = num_features if i == 0 else hidden_dims[i-1]
            out_dim = hidden_dims[i]
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            conv = GINConv(mlp)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(out_dim))

        # Final linear layer for graph-level classification
        self.linear = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, data, return_hidden=False, layer=3):
        """
        Forward pass of the GIN model.

        Parameters:
            data: A PyTorch Geometric data object with x, edge_index, and batch.
            return_hidden (bool): If True, returns the hidden representation of the specified layer.
            layer (int): Which hidden layer representation to return (1-indexed).

        Returns:
            If return_hidden is True:
                (hidden_rep, logits), where hidden_rep is the output from the specified GIN layer.
            Otherwise:
                logits.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        hidden_reps = []  # To store the output of each layer

        # Apply each GIN layer sequentially
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
            hidden_reps.append(x.clone())

        # Global pooling to get graph-level embedding from the last layer
        out = global_add_pool(x, batch)
        logits = self.linear(out)

        if return_hidden:
            if layer < 1 or layer > self.num_layers:
                raise ValueError(f"Layer must be between 1 and {self.num_layers}, got {layer}")
            # Return the hidden representation from the specified layer (1-indexed)
            return hidden_reps[layer-1], logits

        return logits