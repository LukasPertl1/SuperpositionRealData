import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_probe_direction_angles(probe_directions):
    """
    Given a list of probe direction tensors, compute pairwise angles between the vectors
    and plot a heatmap of these angles in degrees.

    Parameters:
        probe_directions (list[torch.Tensor]): List of tensors representing probe direction vectors.
    """
    # Concatenate the list of tensors into one tensor matrix
    vectors = torch.cat(probe_directions, dim=0)  # Shape: (num_vectors, vector_dim)
    
    # Normalize vectors to unit length
    norms = torch.norm(vectors, dim=1, keepdim=True)
    unit_vectors = vectors / norms
    
    # Compute cosine similarity matrix
    cosine_similarity = torch.mm(unit_vectors, unit_vectors.T)
    
    # Compute angles in degrees (using clamp to ensure valid range for arccos)
    angles = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0)) * (180 / np.pi)
    
    # Convert to DataFrame for visualization
    angle_matrix = pd.DataFrame(angles.numpy())
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(angle_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 6})
    plt.title("Pairwise Angles Between Probe Direction Vectors")
    plt.xlabel("Vector Index")
    plt.ylabel("Vector Index")
    plt.show()