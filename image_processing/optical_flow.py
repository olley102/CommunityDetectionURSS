"""
# Optical flow

[1] Horn, B., Schunk, B., *Determining Optical Flow* 1981.
"""

import numpy as np
import numpy.typing as npt
from scipy.signal import convolve2d
from typing import Union

"""
## 1. Estimating partial derivatives
"""


def image_seq_partial(arr: npt.ArrayLike, centering: npt.ArrayLike = (0, 0, 0)) -> npt.ArrayLike:
    """
    Estimate derivatives of brightness from the discrete set of image brightness measurements available [1].
    The estimates can be centered to the right, left or symmetrically for both spatial directions and for the temporal
    direction. For symmetric derivatives, the step length is doubled. Unit length is assumed for all directions, so
    both spatial directions and the temporal direction have equal contribution to the total derivative.

    :param arr: rank 3 numpy array of ordering (Y, X, T), storing each image (Y, X, t) with single color channel.
    :param centering: 3 dimensional array-like to store derivative centering preferences in order (X, Y, T). Each entry
    is either 1, 0 or -1. A 1 corresponds to right centering, 0 to symmetric, -1 to left.
    :return: derivative estimates in the same shape as arr.
    """
    # Pad the sequence spatially and temporally.
    # Swap x and y.
    centering_yxz = (centering[1], centering[0], centering[2])
    pad_vecs = np.zeros((3, 2), dtype='int')
    step_lengths = np.array([1, 1, 1])
    for d in range(3):
        if centering_yxz[d] == 0:
            pad_vecs[d] = (1, 1)
            step_lengths[d] = 2
        elif centering_yxz[d] == 1:
            pad_vecs[d] = (0, 1)
            step_lengths[d] = 1
        elif centering_yxz[d] == -1:
            pad_vecs[d] = (1, 0)
            step_lengths[d] = 1
        else:
            raise ValueError(f'Centering entries must be either 1, 0 or -1. Found {centering_yxz[d]}.')
    padded = np.pad(arr, pad_vecs, mode='edge')

    # Shift the entire array.
    # Roll does the trick since we will ignore values in the padded boundary.
    # Shift directions are combinations of pad_vecs entries.
    shift = {}
    pad_vecs[:, 0] = -pad_vecs[:, 0]
    for i in range(8):
        bin_i_str = format(int(bin(i)[2:]), '03d')
        bin_i = np.array(list(bin_i_str), dtype='int')
        direction = np.choose(bin_i, pad_vecs.T)
        shift[bin_i_str] = np.roll(padded, -direction, axis=(0, 1, 2))

    # Approximate derivatives.
    partial_x = 0.25 * (1/step_lengths[1]) * (shift['010'] - shift['000'] + shift['110'] - shift['100'] + shift['011'] - shift['001'] + shift['111'] - shift['101'])
    partial_y = 0.25 * (1/step_lengths[0]) * (shift['100'] - shift['000'] + shift['110'] - shift['010'] + shift['101'] - shift['001'] + shift['111'] - shift['011'])
    partial_t = 0.25 * (1/step_lengths[2]) * (shift['001'] - shift['000'] + shift['101'] - shift['100'] + shift['011'] - shift['010'] + shift['111'] - shift['110'])

    # Concatenate and return at positions available in arr.
    slices = [slice(1, None), slice(1, -1), slice(0, -1)]
    chosen = np.choose(np.array(centering_yxz)+1, slices)
    return np.stack((
        partial_x[chosen[0], chosen[1], chosen[2]],
        partial_y[chosen[0], chosen[1], chosen[2]],
        partial_t[chosen[0], chosen[1], chosen[2]]
    ), axis=0)


"""
## 2. Estimating the Laplacian of the flow velocities.
"""


def image_seq_laplacian(arr: npt.ArrayLike) -> npt.ArrayLike:
    """
    Estimate the Laplacian of a sequence of images.

    :param arr: rank 3 numpy array of ordering (Y, X, T), storing each image (Y, X, t) with single color channel.
    :return: Laplacian estimates in the same shape as arr.
    """
    # Pad all spatial boundaries.
    if len(arr.shape) == 2:
        arr_new = np.expand_dims(arr, axis=-1)
    elif len(arr.shape) == 3:
        arr_new = arr
    else:
        raise ValueError(f'Input dimension must be 2 or 3. Found array shape {arr.shape}.')
    padded = np.pad(arr_new, ((1, 1), (1, 1), (0, 0)), mode='edge')

    # Convolution kernel [1].
    kernel = np.array([
        [0.25, 0.5, 0.25],
        [0.5, -3.0, 0.5],
        [0.25, 0.5, 0.25]
    ])

    # Convolve each frame of padded with kernel and discard boundary.
    lap = np.zeros_like(arr_new, dtype='float')
    for f in range(arr_new.shape[-1]):
        lap[..., f] += convolve2d(padded[..., f], kernel, mode='valid')

    return lap


"""
## 3. Local averages
"""


def image_local_averages(arr: npt.ArrayLike) -> npt.ArrayLike:
    """
    Estimate the local averages of a single image frame.

    :param arr: rank 2 numpy array of ordering (Y, X) with single color channel.
    :return: local average estimates in the same shape as arr.
    """
    if len(arr.shape) != 2:
        raise ValueError(f'Input dimension must be 2. Found array shape {arr.shape}.')
    # Pad all spatial boundaries.
    padded = np.pad(arr, ((1, 1), (1, 1)), mode='edge')

    # Convolution kernel [1].
    kernel = 1/3 * np.array([
        [0.25, 0.5, 0.25],
        [0.5, 0.0, 0.5],
        [0.25, 0.5, 0.25]
    ])

    # Convolve padded with kernel and discard boundary.
    loc = convolve2d(padded, kernel, mode='valid')

    return loc


"""
## 4. Iterative solution
"""


def iteration_step(uv: npt.ArrayLike, e_x: npt.ArrayLike, e_y: npt.ArrayLike, e_t: npt.ArrayLike,
                   alpha: Union[int, float]):
    """
    Iteration step for a single image frame. Iteration scheme explained by [1].

    :param uv: initial values to be passed to iteration. Must be of shape (2, Any, Any)
    :param e_x: derivative estimates in X of same shape as uv[0] and uv[1].
    :param e_y: derivative estimates in X of same shape as uv[0] and uv[1].
    :param e_t: derivative estimates in X of same shape as uv[0] and uv[1].
    :param alpha: corrective variable to prevent dividing by zero.
    :return:
    """
    u_loc = image_local_averages(uv[0])
    v_loc = image_local_averages(uv[1])
    uv_new = np.zeros_like(uv)
    uv_new[0] += u_loc - e_x*(e_x*u_loc + e_y*v_loc + e_t) / (alpha**2 + e_x**2 + e_y**2)
    uv_new[1] += v_loc - e_y*(e_x*u_loc + e_y*v_loc + e_t) / (alpha**2 + e_x**2 + e_y**2)
    return uv_new


def iteration(arr, n, alpha, uv_init=None, use_previous=False, centering=(0, 0, 0)):
    """
    Iteration scheme [1] for a sequence of images to compute optical flow.

    :param arr: rank 3 numpy array of ordering (Y, X, T), storing each image (Y, X, t) with single color channel.
    :param n: number of iterations per frame.
    :param alpha: corrective variable to prevent dividing by zero.
    :param uv_init: initialization of optical flow estimates of a frame with same shape as each arr[..., t].
    :param use_previous: if use_previous=True then the result of each time frame will be copied to the next time frame's
    iteration. Otherwise, uv_init will be used instead.
    :param centering: 3 dimensional array-like to store derivative centering preferences in order (X, Y, T). Each entry
    is either 1, 0 or -1. A 1 corresponds to right centering, 0 to symmetric, -1 to left.
    :return: estimates of optical flow of image sequence with shape (2,) + arr.shape
    """
    if uv_init is None:
        uv_init = np.zeros((2,) + arr.shape[:-1])

    uv = np.zeros_like(uv_init)
    uv += uv_init

    uv_final = np.zeros((2,) + arr.shape)

    e_x, e_y, e_t = image_seq_partial(arr, centering=centering)

    for t in range(arr.shape[-1]):
        for k in range(n):
            uv = iteration_step(uv, e_x[..., t], e_y[..., t], e_t[..., t], alpha=alpha)
        uv_final[..., t] += uv

        if not use_previous:
            uv = np.zeros_like(uv_init)
            uv += uv_init

    return uv_final
