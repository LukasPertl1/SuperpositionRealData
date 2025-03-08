from concepts.concept_mask import ConceptMask
from utils.graph_utils import edge_index_to_adj_list

class BasicConcepts:
    def degree(self, graph, k, operator='<'):
        """
        Nodes with degree in range.
        """
        adj_list = edge_index_to_adj_list(graph.edge_index)
        node_set = set()

        for node_idx in range(graph.x.shape[0]):
            if operator == '<' and len(adj_list[node_idx]) < k:
                node_set.add(node_idx)
            elif operator == '>' and len(adj_list[node_idx]) > k:
                node_set.add(node_idx)
            elif operator == '=' and len(adj_list[node_idx]) == k:
                node_set.add(node_idx)
        return ConceptMask(graph, set(), node_set)

    def feature(self, graph, j, k, operator='<'):
        node_set = set()
        for node_idx in range(graph.x.shape[0]):
            if operator == '<' and graph.x[node_idx, j] < k:
                node_set.add(node_idx)
            elif operator == '>' and graph.x[node_idx, j] > k:
                node_set.add(node_idx)
            elif operator == '=' and graph.x[node_idx, j] == k:
                node_set.add(node_idx)
        return ConceptMask(graph, set(), node_set)

    def neigh_degree(self, graph, k, operator='<', require=1, hops=1):
        """
        Nodes where neighbour satisfies degree condition.
        """
        adj_list = edge_index_to_adj_list(graph.edge_index, hops=hops, exclude_self=True)

        mask = []
        node_set = set()

        for node_idx in range(graph.x.shape[0]):
            ns = []
            for neigh in adj_list[node_idx]:
                if operator == '<' and len(adj_list[neigh]) < k or \
                   operator == '>' and len(adj_list[neigh]) > k or \
                   operator == '=' and len(adj_list[neigh]) == k:
                    ns.append(neigh)
            if len(ns) >= require:
                for neigh in ns:
                    mask.append((node_idx, neigh))
                    mask.append((neigh, node_idx))
                    node_set.add(node_idx)

        return ConceptMask(graph, set(), node_set)

    def neigh_feature(self, graph, j, k, operator='<', require=1, hops=1):
        adj_list = edge_index_to_adj_list(graph.edge_index, hops=hops, exclude_self=True)

        mask = []
        node_set = set()

        for node_idx in range(graph.x.shape[0]):
            ns = []
            for neigh in adj_list[node_idx]:
                if operator == '<' and graph.x[neigh, j] < k or \
                   operator == '>' and graph.x[neigh, j] > k or \
                   operator == '=' and graph.x[neigh, j] == k:
                    ns.append(neigh)
            if len(ns) >= require:
                for neigh in ns:
                    mask.append((node_idx, neigh))
                    mask.append((neigh, node_idx))
                    node_set.add(node_idx)

        return ConceptMask(graph, set(), node_set)