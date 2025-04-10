from training.train import run_training
import data.utils_data as utils_data
import data.mesh_loader as ml

################################################################################################
# define globals
data_directory = 'C:\\Users\\deangeln\\Documents\\Papers\\gnn_mesh_optimizer\\data\\data_dir'
train_size = 0.8
################################################################################################

if __name__ == "__main__":

    ref_geom, node_data, edges = ml.extract_graph_inputs(data_directory, 'displacement')
    train_data, test_data = utils_data.splitter(node_data, train_size)

    config_dict = {}

    run_training(config_dict, train_data, edges, ref_geom, data_directory)
