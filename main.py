from training.train import run_training
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
data_directory = 'C:\\Users\\deangeln\\Documents\\Papers\\gnn_mesh_optimizer\\data\\data_dir'
task = 'squishy_512.yaml'
train_size = 0.5
################################################################################################

if __name__ == "__main__":

    graph_inputs = ml.extract_graph_inputs(data_directory, 'displacement')
    
    # Could also consider using nearest neighbors to find close nodes instead of splitting by element? 
    # Radius adjustable by user for propagation speed.
    train_cell_data, test_cell_data = utils_data.splitter(graph_inputs.mesh_connectivity, train_size)
    graph_inputs.add(chosen_cells=train_cell_data, chosen_nodes=jnp.unique(jnp.hstack(train_cell_data)))

    # readjust the senders/receivers in graph inputs to account for train/test split:
    graph_inputs.edges = utils_data.cells_to_edges(graph_inputs.chosen_cells, graph_inputs.cell_type)
    
    config_path = f"./data/configs/{task}"
    config_dict = yaml.safe_load(open(f"{config_path}", 'r'))
    
    run_training(config_dict, graph_inputs, data_directory)

    graph_inputscell_data=test_cell_data
