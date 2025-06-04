import jax
import jax.numpy as jnp
from jax import jit
import optax
import haiku as hk

import pdb

from functools import partial
from absl import logging

import numpy as np
import pathlib
import pickle

import training.utils
import data.utils_data as utils_data
from physics.loss_terms import physics_residual_loss
from physics.utils_potential_energy import total_potential_energy

#################################################################
OPTIMISATION_ALGORITHM = optax.adam

#################################################################

# function to create subdir to save emulation results
# create_savedir = partial(utils.create_savedir,
#                                    local_embedding_dim=LOCAL_EMBED_DIM,
#                                    mlp_width=MLP_WIDTH,
#                                    mlp_depth=MLP_DEPTH,
#                                    rng_seed=RNG_SEED)

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
    
    # jax.tree_util.tree_map(lambda g: jnp.isnan(g).any(), grads)
    # jax.debug.print("Grad shape: {}", jax.tree_util.tree_map(lambda x: x.shape, grads))
    
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

    def __init__(self, pred_fn, train_dg, params, lr, optim_algorithm, ref_geom_data, external_forces=None, logging=None, results_save_dir=None, summary_writer=None):

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
        print(f'loss: {self.train_loss}')

    def fit_pinn(self, n_epochs: int, save_params = False, random_sampling=False):
        """Train network for 'n_epochs' epochs"""

        # self.logging.info(f'Beginning training for {n_epochs} epochs')
        for epoch_idx in range(n_epochs):

            print(f'epoch: {epoch_idx}')
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

            # if (epoch_idx % 250 == 0) or (epoch_idx < 150):
            #     self.logging.info(f'({epoch_idx_total}): train_loss={self.train_loss:.5f}, lr={self.opt_state.hyperparams["learning_rate"]:.1e}')

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
    external_forces = utils_data.ExtForceTemp(graph_inputs)

    emulator_pred_fn, params, emulator = training.utils.create_emulator(emulator_config_dict, graph_inputs, train_dg, ref_geom)

    # zero out the weights in the last layer of the decoder FCNNs
    params = training.utils.gen_zero_params_gnn(emulator, params)
    
    learner = PhysicsLearner(emulator_pred_fn, train_dg, params, emulator_config_dict['lr'], OPTIMISATION_ALGORITHM, ref_geom, external_forces, results_save_dir='./data')
    
    # train first half at learning rate lr
    n_epochs_start = max(emulator_config_dict['n_epochs']//2, 1)
    learner.fit_pinn(n_epochs_start, save_params=True, random_sampling=False)

    # finish training at learning rate lr/10
    learner.update_learning_rate(emulator_config_dict['lr']/10.)
    n_epochs_end = max(emulator_config_dict['n_epochs']-n_epochs_start, 1)
    learner.fit_pinn(n_epochs_end, save_params=False, random_sampling=True) # randomly sample material parameters at each step

    # save trained network params
    with pathlib.Path(learner.results_save_dir, f'trainedNetworkParams.pkl').open('wb') as fp:
         pickle.dump(learner.params, fp)

def run_evaluation(graph_inputs, data_path: str, K: int, n_epochs: int, lr: float, trained_params_dir: str, dir_label: str):
    """Evaluate performance of a trained a PI-GNN emulator on simulation data"""

    # directory where results are saved
    results_save_dir = 'C:/Users/ndnde/Documents/Projects/ML/gnn_mesh_optimizer/emulation' #create_savedir(data_path, K, n_epochs, lr, dir_label)

    # logging.get_absl_handler().use_absl_log_file('evaluation', f'{results_save_dir}/logFiles')
    # logging.set_stderrthreshold(logging.DEBUG)

    logging.info('Beginning Evaluation')
    logging.info(f'Data path: {data_path}')
    logging.info(f'Message passing steps (K): {K}')
    logging.info(f'Training epochs: {n_epochs}')
    logging.info(f'Learning rate: {lr}')
    logging.info(f'Trained Params Dir: {trained_params_dir}/n')
    logging.info(f'Results save directory: {results_save_dir}\n')

    # load reference geometry data
    ref_geom = utils_data.ReferenceGeometry(graph_inputs)

    # store external force data (body forces and/or surface forces)
    external_forces = utils_data.ExtForceTemp()

    # load test simulation data
    # test_data = utils_data.DataLoader(data_path, 'test') -> replaced by ml.extract_graph_inputs which already gets node data
    logging.info(f'Number of test data points: {graph_inputs.chosen_node_data.shape[0]}')

    # create dictionary of hyperparameters of the GNN emulator
    config_dict = training.utils.create_config_dict(K, n_epochs, lr, ref_geom._output_dim)

    # if trained_params_dir is not set, parameters are read from results_save_dir
    if trained_params_dir == "None": trained_params_dir = results_save_dir

    # initialise GNN emulator and read trained network parameters
    pred_fn, trained_params, emulator = utils.initialise_emulator(config_dict, test_data, results_save_dir, ref_geom, True, trained_params_dir)

    # vmap to allow total potential energy to be computed for all simulations similtaneously
    pe_vmap = jax.vmap(partial(total_potential_energy, ref_geom_data=ref_geom, external_forces=external_forces))

    # hardcode trained parameters into prediction function and jit for faster execution
    pred_fn_jit = jit(lambda theta_norm: pred_fn(trained_params, theta_norm))

    logging.info('Predicting on test data set using trained emulator')
    Upred = predict_dataset(test_data, pred_fn_jit)

    # calculate total potential energy for test data using emulator prediction
    PEpred = pe_vmap(Upred,test_data._theta)

    # retrieve the corresponding values obtained using the finite-element simulator
    Utrue = test_data._displacement
    PEtrue = test_data._pe_values

    # find error between total potential energy calculated in JAX versus results returned by FEniCS (PEtrue)
    PEtrue_calc = pe_vmap(Utrue,test_data._theta)
    calc_errors = 100.*jnp.abs((PEtrue - PEtrue_calc)/PEtrue)
    logging.info(f'Max PE calc error against FEniCS: {calc_errors.max():.4f}% (sim {calc_errors.argmax()})\n')

    # compute deformation gradient and potential energy from true displacements
    Ftrue, Jtrue, I1true = utils_eval.compute_F_J_I1_vmap(Utrue, ref_geom)

    # compute deformation gradient and potential energy from predicted displacements
    Fpred, Jpred, I1pred = utils_eval.compute_F_J_I1_vmap(Upred, ref_geom)

    # collect true/predicted arrays into tuples
    true_arrs = Utrue, PEtrue, Ftrue, Jtrue, I1true
    pred_arrs = Upred, PEpred, Fpred, Jpred, I1pred

    # print prediction error statistics to console
    utils_eval.print_error_statistics(true_arrs, pred_arrs, results_save_dir, logging)

    if SAVE_PREDICTIONS:
        logging.info('Saving Results')
        np.save(f'{results_save_dir}/predDisplacement.npy', Upred)
        np.save(f'{results_save_dir}/trueDisplacement.npy', Utrue)

    # make Paraview (.vtk) and Augmented Reality (.ply) visualisations of results
    try:
        import paraview.simple
    except:
        logging.warning('3D visualisations cannot be generated, as there was en error in importing paraview. Try typing "from paraview.simple import *" in a Python shell and follow the error printed to the screen.')
        return

    from utils_visualisation import make_3D_visualisations as make_visuals
    logging.info('Generating 3D visualisation files')
    make_visuals(Utrue, Upred, ref_geom.ref_coords, data_path, results_save_dir, logging)
