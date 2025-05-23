import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_probe_directions(
    probe_directions,
    weights,
    special_indices=None,
    figsize=(6, 6),
    arrow_kwargs=None
):
    """
    Plot unit probe direction vectors in 2D or 3D with enhanced styling for publication.

    Parameters
    ----------
    probe_directions : list[torch.Tensor] | torch.Tensor
        A list of tensors or a single tensor of shape (N, D) containing vectors.
    special_indices : dict[int, dict], optional
        Mapping from vector index to dict of styling options, e.g.
        {1: {'color': 'red', 'label': 'Concept A'}, 10: {'color': 'blue'}}.
    figsize : tuple[int, int], default=(6, 6)
        Figure size in inches.
    arrow_kwargs : dict, optional
        Default keyword arguments for arrows (passed to quiver), e.g.
        {'pivot': 'mid', 'width': 0.005, 'headwidth': 3, 'headlength': 5}.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes._subplots.AxesSubplot | mpl_toolkits.mplot3d.axes3d.Axes3D
    """
    # extract and log readout vector
    readout_vec = weights[0].detach().cpu().numpy()
    print(f'Readout direction: {readout_vec}')

    # handle input types
    if isinstance(probe_directions, list):
        vectors = torch.cat(probe_directions, dim=0)
    else:
        vectors = probe_directions

    # normalize to unit vectors
    norms = vectors.norm(dim=1, keepdim=True)
    unit_vectors = (vectors / norms).cpu().numpy()

    # default styling
    special_indices = special_indices or {}
    default_arrow_kwargs = {
        'angles': 'xy',
        'scale_units': 'xy',
        'scale': 1,
        'pivot': 'mid',
        'width': 0.005,
        'headwidth': 3,
        'headlength': 5
    }
    if arrow_kwargs is not None:
        default_arrow_kwargs.update(arrow_kwargs)

    dim = unit_vectors.shape[1]

    # create figure and axes
    fig = plt.figure(figsize=figsize)
    if dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
    else:
        raise ValueError(f"Only 2D or 3D vectors supported, got dimension {dim}.")

    # plot all probe vectors
    for idx, vec in enumerate(unit_vectors):
        style = special_indices.get(idx, {})
        if dim == 2:
            ax.quiver(
                0, 0, vec[0], vec[1],
                angles=default_arrow_kwargs['angles'],
                scale_units=default_arrow_kwargs['scale_units'],
                scale=default_arrow_kwargs['scale'],
                pivot=default_arrow_kwargs['pivot'],
                width=default_arrow_kwargs['width'],
                headwidth=default_arrow_kwargs['headwidth'],
                headlength=default_arrow_kwargs['headlength'],
                color=style.get('color', 'gray')
            )
        else:
            ax.quiver(
                0, 0, 0, vec[0], vec[1], vec[2],
                length=1.0,
                arrow_length_ratio=0.1,
                normalize=True,
                color=style.get('color', 'gray'),
                linewidth=1.2
            )

    # add the read-out arrow
    if dim == 2:
        ax.quiver(
            0, 0,
            readout_vec[0], readout_vec[1],
            angles=default_arrow_kwargs['angles'],
            scale_units=default_arrow_kwargs['scale_units'],
            scale=default_arrow_kwargs['scale'],
            pivot=default_arrow_kwargs['pivot'],
            width=default_arrow_kwargs['width'] * 1.5,
            headwidth=default_arrow_kwargs['headwidth'] * 1.2,
            headlength=default_arrow_kwargs['headlength'] * 1.2,
            color='black',
            linewidths=2.5,
            label='Read-out'
        )
    elif dim == 3:
        ax.quiver(
            0, 0, 0,
            readout_vec[0], readout_vec[1], readout_vec[2],
            length=1.0,
            arrow_length_ratio=0.1,
            normalize=True,
            color='black',
            linewidth=2.5,
            label='Read-out'
        )

    # legend for special + read-out vectors
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', frameon=False)

    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    return fig, ax