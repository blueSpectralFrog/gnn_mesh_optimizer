import jraph
import haiku as hk

def make_gnn_core():
    return jraph.GraphNetwork(
        update_edge_fn=None,
        update_node_fn=lambda n, s, r, g: hk.nets.MLP([64, 64])(n),
        update_global_fn=None
    )
