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
train_size = 0.8
################################################################################################

if __name__ == "__main__":

    graph_inputs = ml.extract_graph_inputs(data_directory, 'displacement')
    train_data, test_data = utils_data.splitter(graph_inputs.node_data, train_size)

    config_path = f"./data/configs/{task}"
    config_dict = yaml.safe_load(open(f"{config_path}", 'r'))
    
    run_training(config_dict, train_data, graph_inputs, data_directory)
