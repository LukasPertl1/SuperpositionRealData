from concepts.concept_mask import *
from utils.graph_utils import edge_index_to_adj_list, edge_index_to_tuples

class MoleculeConcepts:
    def __init__(self, labels):
        self.labels = labels
        self.labels_inv = {pair[1]: pair[0] for pair in self.labels.items()}
        self.n_atoms = len(self.labels)

    def is_atom(self, x, atom):
        if atom == 'X':
            return True
        return not (x != torch.nn.functional.one_hot(torch.tensor(self.labels[atom]), self.n_atoms)).any()

    def element(self, x):
        return self.labels_inv[torch.argmax(x, dim=0).item()]

    def is_element(self, graph, ele):
        node_set = set()

        for node_idx in range(graph.x.shape[0]):
            if self.is_atom(graph.x[node_idx], ele):
                node_set.add(node_idx)

        return ConceptMask(graph, set(), node_set)

    def AB_K(self, graph, a, b, k):
        adj_list = edge_index_to_adj_list(graph.edge_index)
        mask = []

        assert a in self.labels.keys()
        assert b in self.labels.keys()

        for node_idx in range(graph.x.shape[0]):
            if self.is_atom(graph.x[node_idx], a):
                ox_edges = []
                for neigh in adj_list[node_idx]:
                    if self.is_atom(graph.x[neigh], b) and len(adj_list[neigh]) == 1:
                        ox_edges.append((node_idx, neigh))
                        ox_edges.append((neigh, node_idx))
                if len(ox_edges) == 2 * k:
                    mask.extend(ox_edges)

        return ConceptMask(graph, set(mask), set(np.unique([[pair[0], pair[1]] for pair in mask])))

    def element_neighbour(self, graph, ele_self, *eles_nb, strict=True, hops=1):
        """
        All instances of atom being only connected to elements specified.
        """
        assert all([ele in self.labels.keys() for ele in eles_nb])

        adj_list = edge_index_to_adj_list(graph.edge_index, hops=hops, exclude_self=True)
        mask = []
        node_set = set()

        def is_subseq(x, y):
            it = iter(y)
            return all(c in it for c in x)

        for node_idx in range(graph.x.shape[0]):
            if self.is_atom(graph.x[node_idx], ele_self):
                neighbs = []
                for neigh in adj_list[node_idx]:
                    neighbs.append(self.element(graph.x[neigh]))
                str_a = ''.join(sorted(neighbs))
                str_b = ''.join(sorted(eles_nb))
                if strict and str_a != str_b:
                    continue
                if not strict and not is_subseq(str_b, str_a):
                    continue
                for neigh in adj_list[node_idx]:
                    mask.append((node_idx, neigh))
                    mask.append((neigh, node_idx))
                node_set.add(node_idx)

        return ConceptMask(graph, set(mask), node_set)

    def element_same(self, graph, ele):
        """
        All edges connecting an element to the same element.
        """
        assert ele in self.labels.keys()

        edge_tuples = edge_index_to_tuples(graph.edge_index)
        mask = []

        carbon_indices = [node_idx for node_idx in range(graph.x.shape[0])
                          if self.is_atom(graph.x[node_idx], ele)]
        for idx in carbon_indices:
            for idx_ in carbon_indices:
                if idx != idx_ and (idx, idx_) in edge_tuples:
                    mask.append((idx, idx_))

        return ConceptMask(graph, set(mask), set(np.unique([[pair[0], pair[1]] for pair in mask])))

    def element_carbon(self, graph, ele):
        """
        All instances of atom being only connected to a carbon.
        """
        assert ele in self.labels.keys()

        adj_list = edge_index_to_adj_list(graph.edge_index)
        mask = []

        for node_idx in range(graph.x.shape[0]):
            if self.is_atom(graph.x[node_idx], ele) and len(adj_list[node_idx]) == 1:
                neigh = adj_list[node_idx][0]
                if self.is_atom(graph.x[neigh], 'C'):
                    mask.append((node_idx, neigh))
                    mask.append((neigh, node_idx))

        return ConceptMask(graph, set(mask), set(np.unique([[pair[0], pair[1]] for pair in mask])))