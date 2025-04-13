import jax
import jax.numpy as jnp
import optax
import haiku as hk

from training.utils import create_emulator
import data.utils_data as utils_data
from physics.loss_terms import physics_residual_loss

def model_fn(graph):
    nodes = encode_node_fn(graph.nodes)
    graph = graph._replace(nodes=nodes)

    gnn = make_gnn_core()
    graph = gnn(graph)

    output = decode_node_fn(graph.nodes)
    return output

def run_training(emulator_config_dict, train_data, graph_inputs, trained_params_dir):

    print("Training stub started...")

    ref_geom = utils_data.ReferenceGeometry(graph_inputs)

    create_emulator(emulator_config_dict, train_data, trained_params_dir, graph_inputs, ref_geom)
    # TODO: Load dataset, initialize model and train
