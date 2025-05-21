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
data_directory = '.\\data\\data_dir'
normalisation_statistics_dir = '.\\data'
task = 'squishy_512.yaml'
train_size = 0.5
################################################################################################

if __name__ == "__main__":

    graph_inputs = ml.extract_graph_inputs(data_directory, 'displacement')

    # Could also consider using nearest neighbors to find close nodes instead of splitting by element? 
    # Radius adjustable by user for propagation speed.
    train_cell_data, test_cell_data = utils_data.splitter(graph_inputs.mesh_connectivity, train_size)
    graph_inputs.add(chosen_cells=train_cell_data, chosen_nodes=jnp.unique(jnp.hstack(train_cell_data)))
    
    # Create a remapping array
    remap = jnp.full(jnp.unique(jnp.hstack(graph_inputs.mesh_connectivity)).shape[0], -1)
    remap = remap.at[graph_inputs.chosen_nodes].set(jnp.arange(len(graph_inputs.chosen_nodes)))

    # Apply the remap
    remapped_chosen_cells = remap[graph_inputs.chosen_cells]
    graph_inputs.add(remapped_chosen_cells=remapped_chosen_cells)

    # readjust the senders/receivers in graph inputs to account for train/test split:
    graph_inputs.edges = utils_data.cells_to_edges(graph_inputs.remapped_chosen_cells, graph_inputs.cell_type)
    
    # select the edge data based on chosen cells ONLY
    chosen_edge_data = jnp.zeros((graph_inputs.edges.shape[0], 
                                  graph_inputs.node_data.shape[1],
                                  4))
    for interval in range(graph_inputs.node_data.shape[1]):
        chosen_edge_data = chosen_edge_data.at[:, interval].set(ml.get_relative_distance_data(graph_inputs.node_position[:, interval, :],
                                                                graph_inputs.edges[:,1],
                                                                graph_inputs.edges[:,0]))
    # normalize the edges:
    chosen_edge_data = (chosen_edge_data-jnp.mean(chosen_edge_data, axis=0))/(jnp.std(chosen_edge_data, axis=0) + 1e-8)

    graph_inputs.add(chosen_edge_data=chosen_edge_data)

    config_path = f"./data/configs/{task}"
    config_dict = yaml.safe_load(open(f"{config_path}", 'r'))
    
    run_training(config_dict, graph_inputs, data_directory, normalisation_statistics_dir)

    graph_inputs.cell_data=test_cell_data