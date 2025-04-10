# Utility functions for initializing GNN.
import jax
import jax.numpy as jnp
import numpy as np
import models.encoder as encoder

def init_emulator_full(config_dict: dict, data_generator, trained_params_dir: str, ref_geom):
    emulator =  encoder.PrimalGraphEmulator(mlp_features=config_dict['mlp_features'],
                                           latent_size=[config_dict['local_embedding_dim']],
                                           K = config_dict['K'],
                                           receivers = ref_geom._receivers,
                                           senders = ref_geom._senders,
                                           n_total_nodes= ref_geom._n_total_nodes,
                                           output_dim= [config_dict['output_dim']],
                                           real_node_indices = ref_geom._real_node_indices,
                                           boundary_adjust_fn = ref_geom.boundary_adjust_fn)

def create_emulator(emulator_config_dict, data_generator, trained_params_dir, ref_geom_data):

    # initialise varying geometry emulator (models.PrimalGraphEmulator) and parameters
    emulator, params = init_emulator_full(emulator_config_dict, data_generator, trained_params_dir, ref_geom_data)
    emulator_pred_fn = lambda p, theta_norm: emulator.apply(p, ref_geom_data._node_features, ref_geom_data._edge_features, theta_norm)