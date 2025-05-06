import jax.numpy as jnp
from jax import device_put
import os
import sys

def isotropic_elastic(params, epsilon):
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

    # Compute traces
    trace_eps = jnp.trace(epsilon)
    eps_squared = jnp.dot(epsilon, epsilon)
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
