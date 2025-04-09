from training.train import run_training
import training.utils as utils
import data.mesh_loader as ml

################################################################################################
# define globals
data_directory = 'C:\\Users\\deangeln\\Documents\\Papers\\gnn_mesh_optimizer\\data\\data_dir'
train_size = 0.8
################################################################################################

if __name__ == "__main__":

    node_data, edges = ml.extract_graph_inputs(data_directory, 'displacement')
    train_data, test_data = utils.splitter(node_data, train_size)

    run_training(train_data, edges)
