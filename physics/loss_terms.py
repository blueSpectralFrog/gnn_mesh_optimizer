import jax.numpy as jnp

def physics_residual_loss(pred, node_positions, material_properties):
    strain = jnp.gradient(pred, axis=0)
    stress = material_properties['E'] * strain
    residual = jnp.gradient(stress, axis=0)
    return jnp.mean(residual**2)
