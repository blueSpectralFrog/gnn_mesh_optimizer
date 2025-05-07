import jax
import jax.numpy as jnp
from jax import jit
import optax
import haiku as hk

from functools import partial

import numpy as np

import training.utils
import data.utils_data as utils_data
from physics.loss_terms import physics_residual_loss
from physics.utils_potential_energy import total_potential_energy

#################################################################
OPTIMISATION_ALGORITHM = optax.adam

#################################################################

def compute_loss_pinn(params, theta_tuple, pred_fn, ref_geom_data, external_forces):
    """Compute total potential energy from emulator prediction"""

    theta_norm, theta = theta_tuple
    Upred = pred_fn(params, theta_norm)
    return total_potential_energy(Upred, theta, ref_geom_data, external_forces)

def train_step(params, opt_state, theta_tuple, optimiser, loss_fn):
    """Train emulator for one theta input point """

    partial_loss_fn = partial(loss_fn, theta_tuple=theta_tuple)
    grad_fn = jax.value_and_grad(partial_loss_fn)
    loss, grads = grad_fn(params)
    updates, opt_state = optimiser.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def predict_dataset(data_loader, pred_fn):
    """Make predictions for entire dataset"""

    Upred = np.zeros_like(data_loader._displacement)
    for graph_idx in data_loader._epoch_indices:
        input_output_tuple = data_loader.get_graph(graph_idx)
        theta_norm, _, _, _ = input_output_tuple
        Upred[graph_idx] = np.array(pred_fn(theta_norm))
    return Upred

class PhysicsLearner:
    """Class for training PI-GNN emulator and saving learned parameters"""

    def __init__(self, pred_fn, train_dg, params, lr, optim_algorithm, ref_geom_data, external_forces, logging, results_save_dir = None, summary_writer=None):

        self.train_dg = train_dg
        self.params = params
        self.lr = lr
        self.optim_algorithm = optim_algorithm
        self.init_optimiser()
        self.opt_state = self.optimiser.init(params)
        self.n_epochs_trained = 0
        self.offset_idx = 0
        self.logging = logging
        self.summary_writer = summary_writer
        self.results_save_dir = results_save_dir
        self.min_train_loss = 1e7

        # intitialise loss as function of displacement and theta
        self.train_loss_fn = partial(compute_loss_pinn,
                                     pred_fn=pred_fn,
                                     ref_geom_data=ref_geom_data,
                                     external_forces=external_forces)

        # jit the training step function for faster execution
        self.train_step = jit(partial(train_step,
                                      optimiser = self.optimiser,
                                      loss_fn = self.train_loss_fn))

    def train_epoch(self, random_sampling=False):
        """Train network for one epoch"""

        if random_sampling:
            self.train_dg.resample_input_points()
        else:
            self.train_dg.shuffle_epoch_indices()

        loss = 0.
        for graph_idx in self.train_dg.epoch_indices:
            theta_tuple_idx = self.train_dg.get_data(graph_idx)
            self.params, self.opt_state, loss_idx = self.train_step(self.params, self.opt_state, theta_tuple_idx)
            loss += loss_idx
        # train loss for epoch is mean total potential energy
        self.train_loss = loss / self.train_dg.epoch_size

    def fit_pinn(self, n_epochs: int, save_params = False, random_sampling=False):
        """Train network for 'n_epochs' epochs"""

        self.logging.info(f'Beginning training for {n_epochs} epochs')
        for epoch_idx in range(n_epochs):

            # train network for one epoch
            self.train_epoch(random_sampling)

            # keep track of number of training epochs that have been completed
            self.n_epochs_trained += 1

            # offset epoch_idx to account for any previous calls to self.fit_pinn
            epoch_idx_total = self.offset_idx + epoch_idx

            # save trained network parameters based on validation set prediction error
            if save_params:
                self.save_trained_params(self.train_loss)

            # write loss values to tensorboard summary_writer
            if self.summary_writer is not None:
                self.summary_writer.scalar('train_loss', self.train_loss, epoch_idx_total)
                self.summary_writer.scalar('learning_rate', self.opt_state.hyperparams["learning_rate"], epoch_idx_total)

            if (epoch_idx % 250 == 0) or (epoch_idx < 150):
                self.logging.info(f'({epoch_idx_total}): train_loss={self.train_loss:.5f}, lr={self.opt_state.hyperparams["learning_rate"]:.1e}')

        # keep track of number of training epochs that have been performed for reference if training is restarted later
        self.offset_idx = self.n_epochs_trained

    def init_optimiser(self):
        """Initialise the optimiser used for training"""

        self.optimiser = optax.inject_hyperparams(self.optim_algorithm)(learning_rate=self.lr)

    def update_learning_rate(self, new_lr):
        """Update learning rate used for training"""

        self.lr = new_lr
        self.init_optimiser()
        self.opt_state = self.optimiser.init(self.params)
        self.train_step = jit(partial(train_step,
                                      optimiser = self.optimiser,
                                      loss_fn = self.train_loss_fn))


    def save_trained_params(self, epoch_loss):
        """Save network parameters if current loss exceeds minimum loss"""

        if epoch_loss < self.min_train_loss:
            self.min_train_loss = epoch_loss
            with pathlib.Path(self.results_save_dir, f'trainedNetworkParams.pkl').open('wb') as fp:
                pickle.dump(self.params, fp)

def run_training(emulator_config_dict, graph_inputs, trained_params_dir, normalisation_statistics_dir):

    print("Training stub started...")

    ref_geom = utils_data.ReferenceGeometry(graph_inputs)

    emulator_config_dict['mlp_features'] = [emulator_config_dict['mlp_width']]*emulator_config_dict['mlp_depth']
    emulator_config_dict['output_dim'] = ref_geom._output_dim

    train_dg = utils_data.DataGenerator(normalisation_statistics_dir)

    emulator_pred_fn, params, emulator = training.utils.create_emulator(emulator_config_dict, graph_inputs, train_dg, ref_geom)

    # zero out the weights in the last layer of the decoder FCNNs
    params = training.utils.gen_zero_params_gnn(emulator, params)

    learner = PhysicsLearner(emulator_pred_fn, train_dg, params, emulator_config_dict['lr'], OPTIMISATION_ALGORITHM, ref_geom, external_forces, logging, results_save_dir, summary_writer)

