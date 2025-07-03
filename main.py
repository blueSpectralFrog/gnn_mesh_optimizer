from training.train import run_training, run_evaluation
import data.utils_data as utils_data
import data.mesh_loader as ml
import jax.numpy as jnp

import yaml 
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

################################################################################################
# define globals
data_directory = './data/data_dir/squishy_512_hex'
normalisation_statistics_dir = './data'
task = 'squishy_512.yaml'
train_size = 0.5
################################################################################################

if __name__ == "__main__":

    #################################
    # TRAIN 
    #################################

    ##### Manage simulation data
    graph_inputs = ml.extract_graph_inputs(data_directory, 'displacement')

    # readjust the senders/receivers in graph inputs to account for train/test split:
    graph_inputs.edges = utils_data.cells_to_edges(graph_inputs.mesh_connectivity, graph_inputs.cell_type)
    
    # select the edge data based on chosen cells ONLY
    edge_data = jnp.zeros((graph_inputs.edges.shape[0], 
                                  graph_inputs.node_data.shape[1],
                                  4))
    for interval in range(graph_inputs.node_data.shape[1]):
        edge_data = edge_data.at[:, interval].set(ml.get_relative_distance_data(graph_inputs.node_position[:, interval, :],
                                                                graph_inputs.edges[:,1],
                                                                graph_inputs.edges[:,0]))
    # normalize the edges:
    edge_data = (edge_data-jnp.mean(edge_data, axis=0))/(jnp.std(edge_data, axis=0) + 1e-8)
    graph_inputs.add(edge_data=edge_data)

    config_path = f"./data/configs/{task}"
    config_dict = yaml.safe_load(open(f"{config_path}", 'r'))
    
    ##### Manage initial reference data
    ref_model = utils_data.ReferenceGeometry(data_directory, graph_inputs)
    
    run_training(config_dict, graph_inputs, ref_model, data_directory, normalisation_statistics_dir)

    #################################
    # EVALUATE 
    ################################# 
    
    run_evaluation(graph_inputs, 'squishy_512', config_dict['K'], config_dict['n_epochs'], config_dict['lr'], data_directory, '')

