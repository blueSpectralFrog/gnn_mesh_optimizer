# Utility functions for initializing GNN.
import jax
import jax.numpy as jnp
import numpy as np
import models.encoder as encoder

def init_emulator_full(config_dict: dict, data_generator, trained_params_dir: str, graph_inputs, ref_geom):
    emulator =  encoder.PrimalGraphEmulator(mlp_features=config_dict['mlp_features'],
                                           latent_size=[config_dict['local_embedding_dim']],
                                           K = config_dict['K'],
                                           receivers = graph_inputs.edges[:,1],
                                           senders = graph_inputs.edges[:,0],
                                           n_total_nodes= ref_geom._n_total_nodes,
                                           output_dim= [config_dict['output_dim']],
                                           real_node_indices = ref_geom._real_node_indices,
                                           boundary_adjust_fn = ref_geom.boundary_adjust_fn)

def create_emulator(emulator_config_dict, train_data, trained_params_dir, graph_inputs, ref_geom):

    # initialise varying geometry emulator (models.PrimalGraphEmulator) and parameters
    emulator, params = init_emulator_full(emulator_config_dict, train_data, trained_params_dir, graph_inputs, ref_geom)
    emulator_pred_fn = lambda p, theta_norm: emulator.apply(p, ref_geom._node_features, ref_geom._edge_features, theta_norm)