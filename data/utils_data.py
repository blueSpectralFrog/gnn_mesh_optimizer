# Utility functions for batching, logging, etc.
import jax
import jax.numpy as jnp
import os
import xml.etree.ElementTree as ET
import xmltodict
from jax import device_put, random
import numpy as np
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

import physics.utils_potential_energy as utils_pe

EDGE_PATTERNS = {
    "tetra": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
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

class ExternalForceClass:
    def __init__(self, ref_model):

        if ref_model.boundary_info:
            self.body_force = None #ref_model.boundary_info['']
            self.surface_force = ref_model.boundary_info['Force Magnitude']
            self.surface_element_indices = ref_model.boundary_info['Force1']['surface element indices']
            self.surface_node_indices = ref_model.boundary_info['Force1']['surface node indices']
            self.surface_force_magnitude = ref_model.boundary_info['Force Magnitude']
            self.surface_area_normals = ref_model.boundary_info['Force1']['quad_surface_area_normals']
            self.surface_area_normals_selected = ref_model.boundary_info['Force1']['quad_surface_area_normals_train']
            self.surface_node_normals = jnp.unique(jnp.hstack(ref_model.boundary_info['Force1']['nodes']))
            self.surface_facet_elements = ref_model.boundary_info['Force1']['surface facet elements']

        self.body_force = None
        
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
    def __init__(self, data_dir, graph_inputs, remap):
        self.init_node_position = graph_inputs.node_position[:,0,:]
        self.init_chosen_node_position = graph_inputs.node_position[:,0,:][graph_inputs.nodes_unique_to_training]
        self._n_real_nodes, self._output_dim = self.init_chosen_node_position.shape

        self.elements = graph_inputs.remapped_train_cell_data

        self.elements_vol = jnp.zeros(self.elements.shape[0])
        for index, element in enumerate(graph_inputs.remapped_train_cell_data):
            element_vol = 0
            for n_tet in ELEMENT_VOLUME_BKDOWN[graph_inputs.cell_type]:
                # keep in mind that self.init_node_position is now the size of the training nodeset
                element_vol += self.element_volume(self.init_chosen_node_position[element[jnp.array(n_tet)]])
            self.elements_vol = self.elements_vol.at[index].set(element_vol)

        from data.constitutive_law import isotropic_elastic, J_transformation_fn
        self.constitutive_law = isotropic_elastic
        self.Jtransform = J_transformation_fn
        self.boundary_adjust_fn = dirichlet_bd_coords = lambda U: U*graph_inputs.vertex_data[graph_inputs.nodes_unique_to_training]
        
        # coordinates of dirichlet: self.init_chosen_node_position[jnp.where((jnp.multiply(graph_inputs.nodes_unique_to_training,jnp.hstack(graph_inputs.vertex_data[graph_inputs.nodes_unique_to_training])))!=0)[0]]
        
        # TODO:
        # Virtual nodes implementation? Need for larger meshes

        self._output_dim = self.init_chosen_node_position.shape[-1]

        self._fibre_field = None

        self.read_model_data(data_dir, graph_inputs, remap)

    def add(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

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
    
    def contains_all_elements(self, b_row, a_row):
        return jnp.all(jnp.isin(a_row, b_row))

    def find_rows_in_B_for_each_row_in_A(self, A, B):
        # For each row in A, return a boolean mask over rows in B
        return jax.vmap(lambda a_row: jax.vmap(lambda b_row: self.contains_all_elements(b_row, a_row))(B))(A)
    
    def matched_indices(self, A, B):
        match_matrix = self.find_rows_in_B_for_each_row_in_A(A, B)
        matches = [list(map(int, jnp.where(match_matrix[i])[0])) for i in range(match_matrix.shape[0])]
        return [a[0] for a in matches if a]

    def read_model_data(self, data_dir, graph_inputs, remap):

        # TODO: Eventually, this will have to be semi-auto, needing input from the user to identify which BC
        #       is which, e.g.: Surface1 -> displacement constraint in (x,y,z) etc. 
        file = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.feb')])
        tree = ET.parse(file[-1])
        root = tree.getroot()

        with open(file[-1], "r") as f:
            data_dict = xmltodict.parse(f.read())

        elements = jnp.vstack([list(map(int, nodes['#text'].split(','))) for nodes in data_dict['febio_spec']['Mesh']['Elements']['elem']])
        node_coordinates = jnp.vstack([list(map(float, nodes['#text'].split(','))) for nodes in data_dict['febio_spec']['Mesh']['Nodes']['node']])

        BOUNDARY_DICT = {}
        for boundary in data_dict['febio_spec']['Mesh']['Surface']:
            BOUNDARY_DICT[boundary['@name']] = {}
            BOUNDARY_DICT[boundary['@name']]['nodes'] = jnp.vstack([list(map(int, nodes['#text'].split(','))) 
                                                                    for nodes in boundary['quad4']]) - 1 

            # compute areas and normal vectors of each triangular facet on the pressure surface ($\partial \Omega^\sigma$ in Eq. 1 of the manuscript)
            tri_elements_1 = BOUNDARY_DICT[boundary['@name']]['nodes'][:, :3] 
            tri_elements_2 = BOUNDARY_DICT[boundary['@name']]['nodes'][:, 1:]
            tri_elements = jnp.concatenate([tri_elements_1, tri_elements_2]) 

            # area
            areas, normals = utils_pe.compute_area_N(tri_elements, self.init_node_position)
            normals = self.combine_triangle_normals_to_quad(normals)
            BOUNDARY_DICT[boundary['@name']]['area'] = sum(areas)

            # normals
            node_normals = np.zeros((max(np.hstack(tri_elements))+1, 3)) # zeros array of shape 
            for facet in tri_elements:
                for node in facet:
                    node_normals[node] = node_normals[node] + normals[node]
            row_sums = jnp.sum(node_normals, axis=1, keepdims=True)
            node_normals = jnp.where(row_sums == 0, 0.0, node_normals / row_sums)
            BOUNDARY_DICT[boundary['@name']]['quad_surface_area_normals'] = normals
            BOUNDARY_DICT[boundary['@name']]['quad_surface_node_normals'] = node_normals[~np.all(node_normals == 0, axis=1)]

            # remap surface node indices
            index_and_remap = [i for i in 
                                [facet for facet in 
                                    enumerate(remap[BOUNDARY_DICT[boundary['@name']]['nodes']]) if -1 not in facet[1]]]
            remapped_facets_indices = jnp.array([i[0] for i in index_and_remap])
            remapped_facets  = jnp.vstack([i[1] for i in index_and_remap])

            BOUNDARY_DICT[boundary['@name']]['surface facet elements'] = remapped_facets
            BOUNDARY_DICT[boundary['@name']]['surface facet elements train index'] = remapped_facets_indices
            BOUNDARY_DICT[boundary['@name']]['quad_surface_area_normals_train'] = normals[remapped_facets_indices]

            # find elements which contain surface facets
            surface_elements = self.matched_indices(BOUNDARY_DICT[boundary['@name']]['nodes'], 
                                                                        graph_inputs.mesh_connectivity)
            surface_elements_remapped = jnp.vstack([facet for facet in remap[graph_inputs.mesh_connectivity[surface_elements]] 
                                        if -1 not in facet])
            BOUNDARY_DICT[boundary['@name']]['surface element indices'] = self.matched_indices(surface_elements_remapped, graph_inputs.remapped_train_cell_data)
            
            # find surface node indices 
            BOUNDARY_DICT[boundary['@name']]['surface node indices'] = np.unique(np.hstack(remapped_facets))
            
        BOUNDARY_DICT['Force Magnitude'] = list(map(float, data_dict['febio_spec']['Loads']['surface_load']['force'].split(',')))

        self.add(boundary_info=BOUNDARY_DICT)

    def combine_triangle_normals_to_quad(self, normals):
        return jnp.vstack([np.average((normals[i], normals[i+np.floor(len(normals)/2).astype(int)]), axis=0) for i in range(int(len(normals)/2))])

def splitter(graph_inputs, train_size=0.8, rng_key=None):
    """
    Split input data into train and test data
    """
    def recursive_kl_partition(G, train_ratio=0.8):
        total_nodes = len(G)
        desired_train_size = int(train_ratio * total_nodes)
        train_nodes = set()
        remaining_subgraphs = [G]

        while remaining_subgraphs and len(train_nodes) < desired_train_size:
            # Pop the largest remaining subgraph
            subgraph = max(remaining_subgraphs, key=lambda g: len(g))
            remaining_subgraphs.remove(subgraph)

            # Split it using KL
            try:
                part_a, part_b = kernighan_lin_bisection(subgraph)
            except nx.NetworkXError:
                # If the graph is too small to bisect, just add all its nodes
                part_a = list(subgraph.nodes)
                part_b = []

            # Decide which side to add to the train set
            if len(train_nodes) + len(part_a) <= desired_train_size:
                train_nodes.update(part_a)
                if part_b:
                    remaining_subgraphs.append(subgraph.subgraph(part_b).copy())
            elif len(train_nodes) + len(part_b) <= desired_train_size:
                train_nodes.update(part_b)
                if part_a:
                    remaining_subgraphs.append(subgraph.subgraph(part_a).copy())
            else:
                # If adding either side would overshoot, add the smaller one
                smaller = part_a if len(part_a) < len(part_b) else part_b
                train_nodes.update(smaller)

        # Anything not in train becomes test
        all_nodes = set(G.nodes)
        test_nodes = all_nodes - train_nodes

        return train_nodes, test_nodes
    
    def find_unique_or_union_nodes(arr1, arr2, unique=True):
        """
        Finds nodes unique to arr1 compared to arr2.

        Args:
            arr1: The first JAX array.
            arr2: The second JAX array.

        Returns:
            A new JAX array containing nodes unique to arr1.
        """
        if unique:
            arr1 = jnp.unique(arr1)
            is_unique = ~jnp.isin(arr1, arr2) #~ for not in
            return arr1[is_unique]
        else:
            return arr1[jnp.isin(arr1, arr2)]
        
    element_data = graph_inputs.mesh_connectivity
    n_nodes = len(jnp.unique(element_data.reshape(-1)))
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
        
    G = nx.Graph()

    for edge in graph_inputs.edges.tolist():
        G.add_edge(edge[0], edge[1])

    nodes_train, nodes_test = recursive_kl_partition(G)

    nodes_train_mask = jnp.zeros(n_nodes, dtype=bool).at[jnp.array(list(nodes_train))].set(True)
    nodes_test_mask = jnp.zeros(n_nodes, dtype=bool).at[jnp.array(list(nodes_test))].set(True)

    def filter_elements(elements, allowed_nodes):
        allowed_set = set(allowed_nodes)
        return [elem for elem in elements if all(n in allowed_set for n in elem)]
    
    train_elements = filter_elements(graph_inputs.mesh_connectivity, nodes_train)
    test_elements = filter_elements(graph_inputs.mesh_connectivity, nodes_test)
    
    # n_elems = element_data.shape[0]
    # n_nodes = len(jnp.unique(element_data.reshape(-1)))

    # indices = jnp.arange(n_elems)
    # shuffled_indices = jax.random.permutation(rng_key, indices)

    # split_idx = int(n_elems * train_size)
    # train_idx = shuffled_indices[:split_idx]
    # test_idx = shuffled_indices[split_idx:]

    # nodes_in_training_elements = jnp.unique(element_data[train_idx].reshape(-1))
    # nodes_in_testing_elements = jnp.unique(element_data[test_idx].reshape(-1))

    # nodes_unique_to_training = find_unique_or_union_nodes(nodes_in_training_elements, nodes_in_testing_elements)
    # node_training_mask = jnp.zeros(n_nodes, dtype=bool).at[nodes_unique_to_training].set(True)

    # nodes_unique_to_testing = find_unique_or_union_nodes(nodes_in_testing_elements, nodes_in_training_elements)
    # node_testing_mask = jnp.zeros(n_nodes, dtype=bool).at[nodes_unique_to_testing].set(True)

    # nodes_on_boundary = find_unique_or_union_nodes(nodes_in_training_elements, nodes_in_testing_elements, unique=False)
    # node_boundary_mask = jnp.zeros(n_nodes, dtype=bool).at[nodes_on_boundary].set(True)

    # for elem_idx, element in enumerate(element_data):
    #     if set(element).issubset(train_set):
    #         train_cells.append(elem_idx)
    #     elif set(element).issubset(test_set):
    #         test_cells.append(elem_idx)

    # graph_inputs.add(train_cell_data=element_data[train_idx], 
    #                  test_cell_data=element_data[test_idx], 
    #                  nodes_unique_to_training=nodes_unique_to_training, 
    #                  node_training_mask=node_training_mask,
    #                  nodes_unique_to_testing=nodes_unique_to_testing, 
    #                  node_testing_mask=node_testing_mask,
    #                  nodes_on_boundary=nodes_on_boundary,
    #                  node_boundary_mask=node_boundary_mask)
    
    graph_inputs.add(train_cell_data=jnp.vstack(train_elements), 
                     test_cell_data=jnp.vstack(test_elements), 
                     nodes_unique_to_training=jnp.array(list(nodes_train)), 
                     nodes_train_mask=nodes_train_mask,
                     nodes_unique_to_testing=jnp.array(list(nodes_test)),
                     nodes_test_mask=nodes_test_mask)

    return graph_inputs

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