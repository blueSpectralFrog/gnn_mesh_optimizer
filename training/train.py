import jax
import jax.numpy as jnp
import optax
import haiku as hk

from utils import create_emulator
from models.encoder import encode_node_fn
from models.processor import make_gnn_core
from models.decoder import decode_node_fn
from physics.loss_terms import physics_residual_loss

def model_fn(graph):
    nodes = encode_node_fn(graph.nodes)
    graph = graph._replace(nodes=nodes)

    gnn = make_gnn_core()
    graph = gnn(graph)

    output = decode_node_fn(graph.nodes)
    return output

def run_training(emulator_config_dict, train_data, edges, ref_geom, trained_params_dir):

    print("Training stub started...")

    create_emulator(emulator_config_dict, train_data, edges, trained_params_dir, ref_geom)
    # TODO: Load dataset, initialize model and train
