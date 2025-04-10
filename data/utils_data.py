# Utility functions for batching, logging, etc.
import jax
import jax.numpy as jnp
import numpy as np

EDGE_PATTERNS = {
    "tetrahedron": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    "hexahedron": [(0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
             (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
             (0, 4), (1, 5), (2, 6), (3, 7)],  # Vertical edges
    "triangle": [(0, 1), (1, 2), (2, 0)]
}

def splitter(data, train_size=0.8, rng_key=None):
    """
    Split input data into train and test data
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
        
    n = data.shape[0]
    indices = jnp.arange(n)
    shuffled_indices = jax.random.permutation(rng_key, indices)

    split_idx = int(n * train_size)
    train_idx = shuffled_indices[:split_idx]
    test_idx = shuffled_indices[split_idx:]

    return data[train_idx], data[test_idx]

def cells_to_edges(cells, cellConnectivity):

    # Flatten cell connectivity into edges
    edges = []
    for cell in cells:
        for edge_pairs in EDGE_PATTERNS[cellConnectivity]:
            connectivity = [(cell[edge_pairs[0]], cell[edge_pairs[1]])]
            edges.extend(connectivity)

    edges = jnp.array(edges, dtype=jnp.int32)
    senders = jnp.minimum(edges[:, 0], edges[:, 1])
    receivers = jnp.maximum(edges[:, 0], edges[:, 1])
    canonical_edges = jnp.stack([senders, receivers], axis=1)  # shape [N, 2]

    unique_edges = np.unique(np.array(canonical_edges), axis=0)
    unique_edges = jnp.array(unique_edges)

    return unique_edges