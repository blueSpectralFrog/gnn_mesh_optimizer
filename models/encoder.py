import haiku as hk
import jax.numpy as jnp

def encode_node_fn(node_features):
    mlp = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu
    ])
    return mlp(node_features)
