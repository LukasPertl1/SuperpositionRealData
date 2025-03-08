from collections import defaultdict
import torch

def edge_index_to_adj_matrix(edge_index, x=None):
    num_nodes = x.shape[0] if x is not None else torch.unique(edge_index).max().item() + 1

    A = torch.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        A[edge_index[0][i].item(), edge_index[1][i].item()] = 1
    return A

def edge_index_to_adj_list(edge_index, hops=1, exclude_self=False):
    adj = defaultdict(list)
    if edge_index.shape[1] == 0:
        return adj
    if hops == 1:
        for i in range(edge_index.shape[1]):
            adj[edge_index[0][i].item()].append(edge_index[1][i].item())
    else:
        A = edge_index_to_adj_matrix(edge_index)
        A_ = A
        B = A
        for _ in range(1, hops):
            A = A @ A_
            B = B + A  # do not use +=
        A = B
        for i in range(edge_index.shape[1]):
            u = edge_index[0][i].item()
            adj[u] = list(A[u].nonzero().squeeze(1).cpu().detach().numpy())
    if exclude_self and hops > 1:
        for v, k in adj.items():
            if v in k:
                k.remove(v)
    return adj

def edge_index_to_tuples(edge_index):
    return [(pair[0].item(), pair[1].item()) for pair in edge_index.T]