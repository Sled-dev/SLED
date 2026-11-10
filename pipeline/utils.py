import jax
from flax.training import checkpoints


def restore_checkpoint(state, workdir, step=None):
    # check if passed state is in a sharded state
    # if so, reduce to a single device sharding
    if isinstance(
        jax.tree_map(lambda x: x.sharding, jax.tree_leaves(state.params))[0],
        jax.sharding.PmapSharding,
    ):
        state = jax.tree_map(lambda x: x[0], state)

    # ensuring that we're in a single device setting
    assert isinstance(
        jax.tree_map(lambda x: x.sharding, jax.tree_leaves(state.params))[0],
        jax.sharding.SingleDeviceSharding,
    )
    state = checkpoints.restore_checkpoint(workdir, state, step=step)
    return state
