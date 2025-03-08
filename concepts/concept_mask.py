import torch
import numpy as np
from utils.graph_utils import edge_index_to_tuples

class ConceptMask:
    def __init__(self, graph, edges, nodes):
        self.edges = edges
        self.nodes = nodes
        self.graph = graph

        self.node_mask = np.zeros(graph.x.shape[0])
        self.node_mask[list(self.nodes)] = 1
        self.node_mask = torch.FloatTensor(self.node_mask)

    def union(self, concept_mask):
        edges_new = self.edges.union(concept_mask.edges)
        nodes_new = self.nodes.union(concept_mask.nodes)
        return ConceptMask(self.graph, edges_new, nodes_new)

    def inter(self, concept_mask):
        edges_new = self.edges.intersection(concept_mask.edges)
        nodes_new = self.nodes.intersection(concept_mask.nodes)
        return ConceptMask(self.graph, edges_new, nodes_new)

    def inv(self):
        edge_tuples = set(edge_index_to_tuples(self.graph.edge_index))
        edges_new = edge_tuples - self.edges
        nodes_new = set(list(np.arange(self.graph.x.shape[0]))) - self.nodes
        return ConceptMask(self.graph, edges_new, nodes_new)
