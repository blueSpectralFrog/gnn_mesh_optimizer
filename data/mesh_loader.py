# Load mesh and boundary conditions

import meshio 
import numpy as np
import jax.numpy as jnp
import os
import data.utils_data as utils_data

class GraphInputs:
        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

        def add(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

def read_simulation_data(data_dir):

    # find all .vtk files in directory
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.vtk')])

    node_position = {}
    node_data = {}
    cell_data = {}
    
    for file_number, file in enumerate(files):
        
        # read file information
        mesh = meshio.read(file)
        
        # split into node and cell data
        node_position[file_number] = mesh.points
        node_data[file_number] = mesh.point_data
        cell_data[file_number] = mesh.cell_data
        
        # cell connectivity
        mesh_connectivity = mesh.cells_dict[mesh.cells[0].type]
        cell_type = mesh.cells[0].type

    # get graph edges
    edges = utils_data.cells_to_edges(mesh_connectivity, cell_type)
    edge_data = jnp.zeros((edges.shape[0],len(node_position),4))
    
    source_idx = edges[:, 0].astype(int)
    target_idx = edges[:, 1].astype(int)

    for key in node_position.keys():
        rel_pos = node_position[key][target_idx] - node_position[key][source_idx]
        rel_norm = jnp.linalg.norm(rel_pos, axis=1, keepdims=True)
        edge_features = jnp.concatenate([rel_pos, rel_norm], axis=1)
        edge_data = edge_data.at[:, key, :].set(edge_features)

    print(f'Node data read:{[key for key in mesh.point_data.keys()]}')
    print(f'Cell data read:{[key for key in mesh.cell_data.keys()]}')
    
    return node_position, node_data, edge_data, cell_data, mesh_connectivity, cell_type, edges

def get_relative_distance_data(node_position, target_idx, source_idx):
    rel_pos = node_position[target_idx] - node_position[source_idx]
    rel_norm = jnp.linalg.norm(rel_pos, axis=1, keepdims=True)
    return jnp.concatenate([rel_pos, rel_norm], axis=1)
        
def stack_simulation_data(data_dict, sim_output_data_label):
    """ 
    Take dictionary data with first-layer keys as timesteps and combine it into one array
    """

    if sim_output_data_label:
        data = jnp.zeros((data_dict[0][sim_output_data_label].shape[0],
                            len(data_dict),
                            data_dict[0][sim_output_data_label].shape[1]))
        
        for frame, key in enumerate(data_dict.keys()):
            for node_index, node_disp in enumerate(data_dict[key][sim_output_data_label]):
                data = data.at[(node_index, frame)].set(node_disp)

    elif sim_output_data_label==None:
        data = jnp.zeros((data_dict[0].shape[0],
                            len(data_dict),
                            data_dict[0].shape[1]))
     
        for frame in data_dict.keys():
            for node_index, node_disp in enumerate(data_dict[frame]):
                data = data.at[(node_index, frame)].set(node_disp)

    print(f'Input {sim_output_data_label} node data has shape: {data.shape}')

    return data

def identify_dirichlet_boundary(node_displacement):
    dirichlet_idx = jnp.zeros((node_displacement.shape[0], 1))
    for node_idx, node in enumerate(node_displacement[:,-1]):
        if np.linalg.norm(node) == 0:
            dirichlet_idx = dirichlet_idx.at[node_idx].set(1)

    return dirichlet_idx

def extract_graph_inputs(data_dir, sim_output_data_label):

    node_position, node_data_dict, edge_sim_data, cell_data, mesh_connectivity, cell_type, edges = read_simulation_data(data_dir)

    node_data = stack_simulation_data(node_data_dict, sim_output_data_label) # displacement data
    node_position = stack_simulation_data(node_position, None) # node position data

    dirichlet_boundary_nodes = identify_dirichlet_boundary(node_data) # vertice data, not including fibre information

    return GraphInputs(vertex_data=dirichlet_boundary_nodes, node_position=node_position, node_data=node_data, edge_sim_data=edge_sim_data, cell_data=cell_data,
                        mesh_connectivity=mesh_connectivity, cell_type=cell_type, edges=edges)