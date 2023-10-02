import jax.numpy as jnp
import numpy as np


def freeze_array(arr: np.ndarray) -> np.ndarray:
    """Sets an array to read-only mode."""
    arr.setflags(write=False)
    return arr


def load_mask(filename: str, fallback: jnp.ndarray) -> jnp.ndarray:
    """Tries to load a controllable mask from a file and uses the fallback if not found

    Args:
        filename (str): The numpy file to load
        fallback (np.ndarray): A fallback array to use in case there was no controllable mask

    Raises:
        ValueError: Occurs if the loaded array has the wrong shape

    Returns:
        np.ndarray: The controllable mask of shape fallback.shape and dtype bool
    """
    try:
        retval = jnp.load(filename).astype(bool)
    except FileNotFoundError:
        retval = fallback
    if retval.shape != fallback.shape:
        raise ValueError(
            f"Invalid mask shape in {filename}, expected {fallback.shape}, got {retval.shape}"
        )
    return retval
