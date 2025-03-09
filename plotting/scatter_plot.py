import matplotlib.pyplot as plt
import torch
import numpy as np
from concepts.create_mask import concatenate_masks
from configs.concept_configs import concept_configs
from scripts.train_gin import test_gin

import matplotlib.pyplot as plt
import numpy as np
from configs.concept_configs import concept_configs
from concepts.create_mask import concatenate_masks
from scripts.train_gin import test_gin

def scatter_plot(model, full_loader, device, desired_layer=3, concept_number=10):
    """
    Creates a scatter plot of hidden embeddings from the model.
    
    It obtains the concept mask for a selected concept configuration and then
    gets the hidden embeddings from the model using test_gin. The plot displays 
    nodes (hidden embeddings) that satisfy the concept (mask==True) in red and 
    those that do not (mask==False) in blue.
    
    For 2D embeddings, a 2D scatter plot is produced; for 3D embeddings, a 3D scatter plot is produced.
    """
    # Choose a concept configuration to examine
    config = concept_configs[concept_number]
    print(config)
    concepts = config["concepts"]
    combination_operator = config["combination_operator"]
    
    # Get the mask via concatenate_masks. The mask is assumed to be a numpy array 
    # with one entry per node (e.g. 0 or 1).
    mask = concatenate_masks(concepts, combination_operator)
    print(mask)
    # Ensure mask is boolean (assuming nonzero means concept active)
    mask_bool = mask.astype(bool)
    print(f'Number of nodes with concept active: {np.sum(mask_bool)}')

    # Get the hidden embeddings via test_gin (assumed to be a tensor of shape [num_nodes, hidden_dim])
    acc, hidden_embeddings = test_gin(model, full_loader, device, return_hidden=True, layer=desired_layer)
    print(f'Hidden embeddings shape: {hidden_embeddings.shape}')
    
    # Convert hidden embeddings to numpy array for plotting.
    embeddings_np = hidden_embeddings.cpu().detach().numpy()
    print(embeddings_np)
    
    # Check that the number of mask entries matches the number of embeddings.
    if embeddings_np.shape[0] != mask_bool.shape[0]:
        raise ValueError("Mismatch between number of nodes in hidden embeddings and mask.")
    
    # If embeddings are 2D, create a 2D scatter plot.
    if embeddings_np.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_np[mask_bool, 0], embeddings_np[mask_bool, 1],
                    color='red', s=5, label='Concept Active')
        plt.scatter(embeddings_np[~mask_bool, 0], embeddings_np[~mask_bool, 1],
                    color='blue', s=5, alpha=0.1, label='Concept Inactive')
        plt.title("Scatter Plot of Hidden Embeddings with Concept Mask (2D)")
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # If embeddings are 3D, create a 3D scatter plot.
    elif embeddings_np.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(embeddings_np[mask_bool, 0], embeddings_np[mask_bool, 1],
                   embeddings_np[mask_bool, 2], color='red', s=5, label='Concept Active')
        ax.scatter(embeddings_np[~mask_bool, 0], embeddings_np[~mask_bool, 1],
                   embeddings_np[~mask_bool, 2], color='blue', s=5, label='Concept Inactive')
        ax.set_title("Scatter Plot of Hidden Embeddings with Concept Mask (3D)")
        ax.set_xlabel("Embedding Dimension 1")
        ax.set_ylabel("Embedding Dimension 2")
        ax.set_zlabel("Embedding Dimension 3")
        ax.legend()
        plt.show()
    
    else:
        print("Embeddings have dimensionality greater than 3. Consider using a dimensionality reduction technique.")
    
