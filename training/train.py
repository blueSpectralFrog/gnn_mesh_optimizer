import jax
import jax.numpy as jnp
import optax
import haiku as hk

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

def run_training(train_data, edges):
    print("Training stub started...")

    
    # TODO: Load dataset, initialize model and train
