import haiku as hk
import jax.numpy as jnp

def decode_node_fn(node_embeddings):
    mlp = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1)
    ])
    return mlp(node_embeddings)
