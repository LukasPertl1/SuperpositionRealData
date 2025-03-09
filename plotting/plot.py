import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_probe_directions(probe_directions):
    """
    Plots probe direction unit vectors. If the vectors are 2D,
    it produces a 2D quiver plot; if 3D, a 3D quiver plot.

    Special indices:
      - Vector at index 1 is plotted in red.
      - Vector at index 10 is plotted in blue.
      - Vector at index 15 is plotted in green.

    Parameters:
        probe_directions (list[torch.Tensor] or torch.Tensor): A list of tensors
            or a single tensor of shape (num_vectors, vector_dim).
    """
    # If probe_directions is a list, concatenate it to form a single tensor.
    if isinstance(probe_directions, list):
        try:
            vectors = torch.cat(probe_directions, dim=0)
        except Exception as e:
            print("Error concatenating probe_directions:", e)
            return
    else:
        vectors = probe_directions

    # Normalize vectors to unit length
    norms = torch.norm(vectors, dim=1, keepdim=True)
    unit_vectors = vectors / norms

    # Determine the dimensionality of the vectors
    dim = unit_vectors.shape[1]

    if dim == 2:
        x = unit_vectors[:, 0].numpy()
        y = unit_vectors[:, 1].numpy()

        plt.figure(figsize=(8, 6))
        for i in range(len(x)):
            if i == 1:
                color = 'red'
                label = 'Vector 1'
            elif i == 10:
                color = 'blue'
                label = 'Vector 10'
            elif i == 15:
                color = 'green'
                label = 'Vector 15'
            else:
                color = 'gray'
                label = None
            plt.quiver(0, 0, x[i], y[i], angles='xy', scale_units='xy', scale=1, 
                       color=color, width=0.005, label=label)

        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("2D Plot of Probe Direction Unit Vectors\n(Vector 1 in red, Vector 10 in blue)")
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys())
        plt.grid()
        plt.show()

    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D

        x = unit_vectors[:, 0].numpy()
        y = unit_vectors[:, 1].numpy()
        z = unit_vectors[:, 2].numpy()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(x)):
            if i == 1:
                color = 'red'
                label = 'Vector 1'
            elif i == 10:
                color = 'blue'
                label = 'Vector 10'
            elif i == 15:
                color = 'green'
                label = 'Vector 15'
            else:
                color = 'gray'
                label = None
            ax.quiver(0, 0, 0, x[i], y[i], z[i], color=color, arrow_length_ratio=0.1, linewidth=1.5, label=label)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Plot of Probe Direction Unit Vectors\n(Vector 1 in red, Vector 10 in blue)")
        plt.show()
    else:
        print("Dim too high:", dim)