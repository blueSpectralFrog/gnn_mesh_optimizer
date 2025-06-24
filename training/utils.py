# Utility functions for initializing GNN.
import jax
import jax.numpy as jnp
import numpy as np
import models.encoder as encoder
import jax.random as random
from flax.core import freeze, unfreeze

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

def gen_zero_params_gnn(model, params_randL):
    node_decode_first_mlp_index = model.K*2 + 3
    node_decode_last_mlp_index = node_decode_first_mlp_index + model.output_dim[0]

    mlp_depth = len(model.mlp_features)
    index = node_decode_first_mlp_index

    final_weights_layer_decoder_rand = params_randL['params'][f'FlaxMLP_{index}'][f'Dense_{mlp_depth}']['kernel']
    final_weights_layer_decoder_zero = jnp.zeros_like(final_weights_layer_decoder_rand)

    params_zero = unfreeze(params_randL)

    for index in range(node_decode_first_mlp_index, node_decode_last_mlp_index):
        params_zero['params'][f'FlaxMLP_{index}'][f'Dense_{mlp_depth}']['kernel'] = final_weights_layer_decoder_zero

    return freeze(params_zero)

def initialise_network_params(vertex_data_generator, edge_data_generator, ref_model, model, material_data_generator, rng_seed: int):
    key = random.PRNGKey(rng_seed)

    theta_init, _ = material_data_generator.get_data(0)
    V_init = vertex_data_generator[:]
    E_init = edge_data_generator[:,0]

    params = model.init(key, V_init, E_init, theta_init)
    return params

def init_emulator_full(config_dict: dict, graph_inputs, material_data_generator, ref_model):
    emulator =  encoder.PrimalGraphEmulator(mlp_features=config_dict['mlp_features'],
                                           latent_size=[config_dict['local_embed_dim']],
                                           K = config_dict['K'],
                                           receivers = graph_inputs.edges[:,1],
                                           senders = graph_inputs.edges[:,0],
                                           n_total_nodes= ref_model._n_real_nodes,
                                           output_dim= [config_dict['output_dim']],
                                        #    real_node_indices = ref_model._real_node_indices,
                                        #    boundary_adjust_fn = ref_model.boundary_adjust_fn
                                        )

    params = initialise_network_params(graph_inputs.vertex_data[graph_inputs.nodes_unique_to_training], 
                                       graph_inputs.chosen_edge_data, 
                                       ref_model, 
                                       emulator, 
                                       material_data_generator,
                                       config_dict['rng_seed'])

    return emulator, params

def create_emulator(emulator_config_dict, graph_inputs, material_data_generator, ref_model):

    # initialise varying geometry emulator (models.PrimalGraphEmulator) and parameters
    emulator, params = init_emulator_full(emulator_config_dict, graph_inputs, material_data_generator, ref_model)

    emulator_pred_fn = lambda p, theta_norm: emulator.apply(p, 
                                                            graph_inputs.vertex_data[graph_inputs.nodes_unique_to_training], 
                                                            graph_inputs.chosen_edge_data[:, 0], 
                                                            theta_norm)

    return emulator_pred_fn, params, emulator