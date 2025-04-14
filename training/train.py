import jax
import jax.numpy as jnp
import optax
import haiku as hk

from training.utils import create_emulator
import data.utils_data as utils_data
from physics.loss_terms import physics_residual_loss

def run_training(emulator_config_dict, train_data, graph_inputs, trained_params_dir):

    print("Training stub started...")

    ref_geom = utils_data.ReferenceGeometry(graph_inputs)

    emulator_config_dict['mlp_features'] = emulator_config_dict['mlp_width']*emulator_config_dict['mlp_depth']
    emulator_config_dict['output_dim'] = ref_geom._output_dim

    create_emulator(emulator_config_dict, train_data, graph_inputs, ref_geom)
    # TODO: Load dataset, initialize model and train
