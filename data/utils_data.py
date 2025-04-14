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

ELEMENT_VOLUME_BKDOWN = {
    "tetrahedron": [[0, 1, 2, 3]],
    "hexahedron": [[0, 5, 6, 7],
                   [2, 4, 6, 7],
                   [0, 1, 3, 6],
                   [3, 4, 6, 0],
                   [1, 2, 3, 6],
                   [3, 4, 6, 2],],  # Vertical edges
}

class ReferenceGeometry:
    def __init__(self, graph_inputs):
        self.init_node_position = graph_inputs.node_position[0]
        self._n_real_nodes, self._output_dim = self.init_node_position.shape

        self.elements = graph_inputs.mesh_connectivity

        self.elements_vol = np.zeros(self.init_node_position.shape[0])
        for index, element in enumerate(graph_inputs.mesh_connectivity):
            element_vol = 0
            for n_tet in ELEMENT_VOLUME_BKDOWN[graph_inputs.cell_type]:
                element_vol += self.element_volume(self.init_node_position[element[[n_tet]]][0])
            self.elements_vol[index] = element_vol

        # TODO:
        # need constitutive law insertion here 
        from data.constitutive_law import isotropic_elastic
        self.constitutive_law = isotropic_elastic

        # TODO:
        # Virtual nodes implementation? Need for larger meshes

        self._output_dim = self.init_node_position.shape[-1]


    def element_volume(self, tet_vert):

        def _determinant_3x3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                    m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
                    m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))


        def _subtract(a, b):
            return (a[0] - b[0],
                    a[1] - b[1],
                    a[2] - b[2])

        def _tetrahedron_calc_volume(tet_vert):
            return (abs(_determinant_3x3((_subtract(tet_vert[0], tet_vert[1]),
                                        _subtract(tet_vert[1], tet_vert[2]),
                                        _subtract(tet_vert[2], tet_vert[3]),
                                        ))) / 6.0)

        return _tetrahedron_calc_volume(tet_vert)



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