# Utility functions for batching, logging, etc.
import jax
import jax.numpy as jnp
from jax import device_put, random
import numpy as np

from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

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

class ExtForceTemp:
    def __init__(self):
        self.body_force = None
        self.surface_force = None
        
class DataGenerator:
    """Class for generating input theta data points at which the PI-GNN emulator is trained"""

    def __init__(self, data_path: str, lhs_seed: int = 101132, sampler_seed: int = 42, shuffle_seed: int = 420):
        """
        Parameters
         ----------
        data_path: str
               Name of the subdirectory within "/data" where the data is stored
        *_seed: int
               Random seeds for data sampling/shuffling
        """

        # geometry_data_dir = f'data/{data_path}/geometryData'
        stats_dir         = data_path #f'data/{data_path}/normalisationStatistics'

        # sys.path.insert(0, geometry_data_dir)
        from data.constitutive_law import params_lb, params_ub, epoch_size, log_sampling

        # can optionally specify to sample theta inputs on log scale between upper and lower bounds
        if log_sampling:
            self.transform_fn = jnp.log
            self.transform_inv = jnp.exp
        else:
            identity_fn = lambda x: x
            self.transform_fn  = identity_fn
            self.transform_inv = identity_fn

        self.params_lb = self.transform_fn(params_lb)
        self.params_ub = self.transform_fn(params_ub)

        # initialse sampler and shuffler random seeds
        self.lhs_seed = lhs_seed
        self.sampler_key = random.PRNGKey(sampler_seed)
        self.shuffle_key = random.PRNGKey(shuffle_seed)

        self.n_params = len(self.params_lb)

        # array of data point indices that can be iterated over during each epoch
        self.epoch_size = epoch_size
        self.epoch_indices = jnp.arange(self.epoch_size)

        # generate input data points using LHS sample
        self.generate_lhs_points(stats_dir)

    def generate_lhs_points(self, stats_dir):
        """Generate input theta points using Latin HyperCube sampling"""

        sampler = LatinHypercube(d=len(self.params_lb), seed=self.lhs_seed)

        hypercube_samples = sampler.random(n=self.epoch_size)

        theta = self.transform_inv(qmc_scale(hypercube_samples, self.params_lb, self.params_ub))

        self.theta = device_put(theta)

        self.theta_mean = self.theta.mean(0)
        self.theta_std = self.theta.std(0)

        np.save(f'{stats_dir}/theta-mean.npy', self.theta_mean)
        np.save(f'{stats_dir}/theta-std.npy', self.theta_std)

        self.theta_norm = (self.theta - self.theta_mean) / self.theta_std

    def shuffle_epoch_indices(self):
        """Shuffles the order in which the dataset is cycled through

        This is called at the start of each training epoch to randomise the order in which the input data points are seen
        """

        self.shuffle_key, key = random.split(self.shuffle_key)
        self.epoch_indices = random.choice(key, self.epoch_indices, shape=(self.epoch_size,), replace=False)

    def resample_input_points(self):
        """Resample the input points over which the emulator is trained"""

        # reset sampler key
        self.sampler_key, key = random.split(self.sampler_key)

        # uniform random sampling between the lower and upper bounds
        samples = random.uniform(key, shape=(self.epoch_size, self.n_params), minval=self.params_lb, maxval=self.params_ub)

        # save results in original space and normalised results
        self.theta = device_put(self.transform_inv(samples))
        self.theta_norm = device_put((self.theta - self.theta_mean) / self.theta_std)

    def get_data(self, data_idx):
        """Returns input global graph values for specified data point (normalised and unnormalised)"""

        return self.theta_norm[data_idx], self.theta[data_idx]


class ReferenceGeometry:
    def __init__(self, graph_inputs):
        self.init_node_position = graph_inputs.node_position[:,0,:]
        self.init_chosen_node_position = graph_inputs.node_position[:,0,:][graph_inputs.chosen_nodes]
        self._n_real_nodes, self._output_dim = self.init_chosen_node_position.shape

        self.elements = graph_inputs.remapped_chosen_cells

        self.elements_vol = jnp.zeros(self.elements.shape[0])
        for index, element in enumerate(graph_inputs.chosen_cells):
            element_vol = 0
            for n_tet in ELEMENT_VOLUME_BKDOWN[graph_inputs.cell_type]:
                # keep in mind that self.init_node_position is now the size of the training nodeset
                element_vol += self.element_volume(graph_inputs.node_position[:,0,:][element[jnp.array(n_tet)]])
            self.elements_vol = self.elements_vol.at[index].set(element_vol)

        # TODO:
        # need constitutive law insertion here 
        from data.constitutive_law import isotropic_elastic, J_transformation_fn
        self.constitutive_law = isotropic_elastic
        self.Jtransform = J_transformation_fn
        
        # TODO:
        # Virtual nodes implementation? Need for larger meshes

        self._output_dim = self.init_chosen_node_position.shape[-1]

        self._fibre_field = None


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

def splitter(element_data, train_size=0.8, rng_key=None):
    """
    Split input data into train and test data
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
        
    n = element_data.shape[0]
    indices = jnp.arange(n)
    shuffled_indices = jax.random.permutation(rng_key, indices)

    split_idx = int(n * train_size)
    train_idx = shuffled_indices[:split_idx]
    test_idx = shuffled_indices[split_idx:]

    return element_data[train_idx], element_data[test_idx]

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