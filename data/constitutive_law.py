import jax.numpy as jnp
from jax import device_put
import os
import sys

####################################################
## Transformation functions for stabilising strain 
## energy - see Section 2.4.3 of paper for details
####################################################

Jmin = 0.001  # minimum value of J which we allow
Jtrans = 0.05 # point at which transformation function kicks in

Jdiff = Jtrans - Jmin

beta_1 = 1. / Jdiff
beta_2 = jnp.log(Jdiff) - Jtrans/Jdiff
beta_3 = Jmin

exp_fn = lambda J: jnp.exp(beta_1*J + beta_2) + beta_3
J_transformation_fn = lambda J: jnp.where(J > Jtrans, J, exp_fn(J))

I1_trans = 10 # point at which transformation function kicks in

tanh_fn = lambda I1: jnp.tanh(I1 - I1_trans) + I1_trans
I1_trans_fn = lambda I1: jnp.where(I1 < I1_trans, I1, tanh_fn(I1))


def isotropic_elastic(params, F, J, fibres=None):
    """
    Compute the strain energy density W for isotropic linear elasticity.

    Parameters:
    - epsilon: 3x3 strain tensor (numpy array)
    - E: Young's modulus
    - nu: Poisson's ratio

    Returns:
    - W: scalar strain energy density
    """
    E, nu = params

    # Lamé parameters
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    C = jnp.matmul(F.T, F)

    # Compute traces
    trace_eps = jnp.trace(C)
    eps_squared = jnp.dot(C, C)
    trace_eps_squared = jnp.trace(eps_squared)

    # Strain energy density
    W = 0.5 * lam * (trace_eps ** 2) + mu * trace_eps_squared
    return W

####################################################
## Define material parameter boundaries
####################################################

params_lb = jnp.array([5000.]*2)
params_ub = jnp.array([15000.]*2)

####################################################
## Sample parameters on log or uniform scale
####################################################

log_sampling = False

####################################################
## Set number of parameters to sample at each epoch
####################################################

epoch_size = 50
