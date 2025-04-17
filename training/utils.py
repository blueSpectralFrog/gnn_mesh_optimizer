# Utility functions for initializing GNN.
import jax
import jax.numpy as jnp
import numpy as np
import models.encoder as encoder
import jax.random as random

from typing import Sequence

def create_config_dict(K: int, n_epochs: int, lr: float, output_dim: int, local_embed_dim: int, mlp_features: Sequence[int], rng_seed: int):
    """Creates dictionary of configuration details for the GNN emulator"""

    return {'K': K,
            'n_train_epochs': n_epochs,
            'learning_rate': lr,
            'output_dim': output_dim,
            'local_embedding_dim': local_embed_dim,
            'mlp_features': mlp_features,
            'rng_seed': rng_seed
            }

def initialise_network_params(node_data_generator, edge_data_generator, ref_geom, model, rng_seed: int):
    key = random.PRNGKey(rng_seed)

    # TODO:
    # What is theta and why is it a 1x4 array
    theta_init = jnp.zeros((4,)) 
    V_init = node_data_generator[:,0]
    E_init = edge_data_generator[:,0]

    params = model.init(key, V_init, E_init, theta_init)
    return params

def init_emulator_full(config_dict: dict, graph_inputs, ref_geom):
    emulator =  encoder.PrimalGraphEmulator(mlp_features=[config_dict['mlp_features']],
                                           latent_size=[config_dict['local_embed_dim']],
                                           K = config_dict['K'],
                                           receivers = graph_inputs.edges[:,1],
                                           senders = graph_inputs.edges[:,0],
                                           n_total_nodes= ref_geom._n_real_nodes,
                                           output_dim= [config_dict['output_dim']],
                                        #    real_node_indices = ref_geom._real_node_indices,
                                        #    boundary_adjust_fn = ref_geom.boundary_adjust_fn
                                        )

    params = initialise_network_params(graph_inputs.node_data[graph_inputs.chosen_nodes], graph_inputs.chosen_edge_data, ref_geom, emulator, config_dict['rng_seed'])

    return emulator, params

def create_emulator(emulator_config_dict, graph_inputs, ref_geom):

    # initialise varying geometry emulator (models.PrimalGraphEmulator) and parameters
    emulator, params = init_emulator_full(emulator_config_dict, graph_inputs, ref_geom)
    emulator_pred_fn = lambda p, theta_norm: emulator.apply(p, ref_geom._node_features, ref_geom._edge_features, theta_norm)