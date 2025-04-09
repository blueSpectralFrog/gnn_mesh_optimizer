# Load mesh and boundary conditions

import meshio 
import numpy as np
import os
import training.utils as utils

def read_data(data_dir):

    # find all .vtk files in directory
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.vtk')])

    node_data = {}
    cell_data = {}

    for file_number, file in enumerate(files):
        
        # read file information
        mesh = meshio.read(file)
        
        # split into node and cell data
        node_data[file_number] = mesh.point_data
        cell_data[file_number] = mesh.cell_data
        
        # cell connectivity
        mesh_connectivity = mesh.cells_dict[mesh.cells[0].type]
        cell_type = mesh.cells[0].type

    print(f'Node data read:{[key for key in mesh.point_data.keys()]}')
    print(f'Cell data read:{[key for key in mesh.cell_data.keys()]}')
    
    return node_data, cell_data, mesh_connectivity, cell_type

def stack_simulation_data(node_data_dict, sim_output_data_label):
    """ 
    Take dictionary data with first-layer keys as timesteps and combine it into one array
    """

    node_data = np.zeros((node_data_dict[0][sim_output_data_label].shape[0],
                        len(node_data_dict),
                        node_data_dict[0][sim_output_data_label].shape[1]))

    for frame, key in enumerate(node_data_dict.keys()):
        for node_index, node_disp in enumerate(node_data_dict[key][sim_output_data_label]):
            node_data[node_index][frame] = node_disp

    print(f'Input {sim_output_data_label} node data has shape: {node_data.shape}')

    return node_data

def extract_graph_inputs(data_dir, sim_output_data_label):

    node_data_dict, _, mesh_connectivity, cell_type = read_data(data_dir)
    node_data = stack_simulation_data(node_data_dict, sim_output_data_label)

    # get graph edges
    edges = utils.cells_to_edges(mesh_connectivity, cell_type)

    return node_data, edges