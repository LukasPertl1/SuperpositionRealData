from torch_geometric.datasets import TUDataset
from concepts.molecule_concepts import MoleculeConcepts
from concepts.basic_concepts import BasicConcepts
import torch


def create_concept_mask(concepts, combination, graph_index):
    """
    Create a complex concept mask by processing multiple concept brackets and combining them internally.

    Parameters:
      - concepts: A list of concept brackets. Each bracket can be either:
          a) A three-element list: [ [prefix, element_symbol, inversion_flag], operator, [prefix, element_symbol, inversion_flag] ]
             e.g. [['nx', 'C', True], 'AND', ['nx', 'O', False]]
          b) A one-element list: [ [prefix, element_symbol, inversion_flag] ]
             e.g. [['is', 'C', False]]
      - combination: A string ('AND' or 'OR') used to combine the masks from each concept bracket.

    Returns:
      - The final combined concept mask.
    """
    # Load the MUTAG dataset
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

    # Extract graphs (assuming we're always working with the first graph)
    graphs = dataset[:]

    # Initialize the concept classes
    concepts_mutag = MoleculeConcepts({'C': 0, 'N': 1, 'O': 2, 'F': 3, 'I': 4, 'Cl': 5, 'Br': 6})
    concepts_basic = BasicConcepts()  # Not used but allows other types of concepts to be included

    def get_mask(elem_info):
        """
        Given an element info list of the form [prefix, element_symbol, inversion_flag],
        call the correct function.
        For 'nx' prefix, calls element_neighbour with the arguments (g, 'X', element_symbol, strict=False).
        For 'is' prefix, calls is_element.
        """
        if not (isinstance(elem_info, list) and len(elem_info) == 3):
            print("Incorrect format for element info:", elem_info)
            return None

        prefix, element_symbol, inv_flag = elem_info[0], elem_info[1], elem_info[2]

        if prefix == 'nx':
            mask = concepts_mutag.element_neighbour(graphs[graph_index], 'X', element_symbol, strict=False, hops=1)
        elif prefix == 'is':
            mask = concepts_mutag.is_element(graphs[graph_index], element_symbol)
        elif prefix == "deg=1":
            mask = concepts_basic.degree(graphs[graph_index], 1, operator='=')
        elif prefix == "deg=2":
            mask = concepts_basic.degree(graphs[graph_index], 2, operator='=')
        elif prefix == "deg=3":
            mask = concepts_basic.degree(graphs[graph_index], 3, operator='=')
        elif prefix == "ndeg=1":
            mask = concepts_basic.neigh_degree(graphs[graph_index], 1, operator='=', require=1)
        elif prefix == "ndeg=2":
            mask = concepts_basic.neigh_degree(graphs[graph_index], 2, operator='=', require=1)
        elif prefix == "ndeg=3":
            mask = concepts_basic.neigh_degree(graphs[graph_index], 3, operator='=', require=1)
        elif prefix == "nx1C":
            mask = concepts_mutag.element_neighbour(graphs[graph_index], 'X', 'C', strict=True)
        elif prefix == "nx2C":
            mask = concepts_mutag.element_neighbour(graphs[graph_index], 'X', 'C', 'C', strict=True)
        elif prefix == "nx3C":
            mask = concepts_mutag.element_neighbour(graphs[graph_index], 'X', 'C', 'C', 'C', strict=True)
        else:
            print("Unknown prefix in element name:", prefix)
            mask = None

        if inv_flag and mask is not None:
            mask = mask.inv()
        return mask

    # Process each concept bracket and store the resulting masks.
    masks = []
    for bracket in concepts:
        # Process a one-element bracket: e.g. [['is', 'C', False]]
        if len(bracket) == 1:
            mask = get_mask(bracket[0])
            if mask is not None:
                masks.append(mask)
        # Process a three-element bracket: e.g. [['nx', 'C', True], 'AND', ['nx', 'O', False]]
        elif len(bracket) == 3:
            mask_a1 = get_mask(bracket[0])
            operator = bracket[1]
            mask_a2 = get_mask(bracket[2])
            if mask_a1 is None or mask_a2 is None:
                print("Error processing one of the elements in bracket:", bracket)
                continue

            if operator == 'AND':
                mask = mask_a1.inter(mask_a2)
            elif operator == 'OR':
                mask = mask_a1.union(mask_a2)
            else:
                print("Incorrect operator in bracket:", operator)
                continue
            masks.append(mask)
        else:
            print("Concept bracket format not recognized:", bracket)

    # Combine all the masks using the provided combination operator
    if not masks:
        print("No valid masks were generated.")
        return None

    final_mask = masks[0]
    for mask in masks[1:]:
        if combination == 'AND':
            final_mask = final_mask.inter(mask)
        elif combination == 'OR':
            final_mask = final_mask.union(mask)
        else:
            print("Incorrect combination operator:", combination)
            return None

    return final_mask


def concatenate_masks(concepts, combination_operator, dataset):
    masks = []

    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    graphs = dataset[:]

    for i, g in enumerate(graphs):
        # Pass the graph index i to create_concept_mask
        concept_mask_obj = create_concept_mask(concepts, combination_operator, i)
        if concept_mask_obj is not None:
            # Append the node_mask (assumed to be a 1D tensor)
            masks.append(concept_mask_obj.node_mask)

    # Concatenate all masks into one 1D tensor
    final_tensor = torch.cat(masks, dim=0)
    # Convert to a numpy array if desired:
    final_array = final_tensor.cpu().detach().numpy()
    return final_array