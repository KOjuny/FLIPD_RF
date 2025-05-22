from typing import Iterable, Tuple

import numpy as np
import torch
from kneed import KneeLocator


def fast_regression(
    all_ys: torch.Tensor,  # [data_size, trajectory_size]
    xs: torch.Tensor,  # [trajectory_size]
) -> torch.Tensor:  # [data_size]
    """Perform a rapid 1D linear regression that can utilize GPU as well"""
    column_with_nan = torch.isnan(all_ys).any(dim=0)
    xs = xs[~column_with_nan]
    all_ys = all_ys[:, ~column_with_nan]

    # return a tensor of slopes of size [data_size]
    mean_x = xs.mean()
    x_diff = xs - mean_x
    denomenator = torch.sum(x_diff**2)
    y_diff = all_ys - all_ys.mean(dim=1, keepdim=True)  # [data_size, trajectory_size]

    return torch.sum(y_diff * x_diff, dim=-1) / denomenator


def convex_hull(
    x: Iterable,
    y: Iterable,
) -> Tuple[np.array, np.array]:
    """
    A linear algorithm to find the convex hull of a given curve, it
    returns two numpy arrays denoting the x and y points of the convex hull.
    """

    # a stack will contain the x and y values,
    # when a new (x, y) is encountered, it is compared
    # to the last and second to last element of the
    # convex hull and pops the last element if that point
    # lies above the intercept of the second to last point
    # and the new point
    stack_x = [x[0]]
    stack_y = [y[0]]
    for i in range(1, len(x)):
        if y[i] >= stack_y[-1]:
            stack_x.append(x[i])
            stack_y.append(y[i])
            continue

        while len(stack_x) > 1:
            x_ref, y_ref = stack_x[-2], stack_y[-2]
            x_bef, y_bef = stack_x[-1], stack_y[-1]
            x_cur, y_cur = x[i], y[i]

            theta = (x_bef - x_ref) / (x_cur - x_ref)

            y_compare = y_ref + theta * (y_cur - y_ref)
            if y_bef > y_compare:
                stack_x.pop()
                stack_y.pop()
            else:
                break

        stack_x.append(x[i])
        stack_y.append(y[i])

    # after that, the x values that were popped should return back to the
    # list with linear interpolation.
    final_x = [x[0]]
    final_y = [y[0]]
    pnt = 1
    for i in range(1, len(stack_x)):
        while x[pnt] < stack_x[i]:
            lst_x = final_x[-1]
            lst_y = final_y[-1]
            nxt_x = stack_x[i]
            nxt_y = stack_y[i]

            final_x.append(x[pnt])
            final_y.append((x[pnt] - lst_x) / (nxt_x - lst_x) * (nxt_y - lst_y) + lst_y)
            pnt += 1
        final_x.append(stack_x[i])
        final_y.append(stack_y[i])
        pnt += 1

    return np.array(final_x), np.array(final_y)


def compute_knee(
    timesteps,
    lid_curve,
    ambient_dim,
    return_info: bool = False,
    S: float = 1.0,
    ignore_timesteps_left: float = 0.05,
    ignore_timesteps_right: float = 0.5,
):
    """
    This algorithm first takes the convex hull of the LID curve (and ignores large values of 't' if given)
    then this convex hull is given to the Kneedle algorithm to detect a knee:
    https://github.com/arvkevi/kneed

    Note that using a convex hull is needed because Kneedle assumes that the input curve is convex when
    it is decreasing.

    Args:
        timesteps:
            The timesteps that the lid curve is evaluated on and it takes values from 0 to T
        lid_curve:
            This is the FLIPD values obtained on each timestep
        ambient_dim:
            This is the ambient dimension
        return_info:
            when set to True, in addition to the LID estimate, some extra information from
            the compute_knee algorithm will be logged
        S:
            sensitivity of the kneedle
        ignore_timesteps_threshold:
            Any timestep above this does not make sense for LID estimation
    Returns:
        either a single float value equal to the LID estimate or a dictionary with the following keys:
        {
            'lid': the value it should return,
            'convex_hull': the curve of the convex hull which is passed through
            'knee_timestep': the timestep at which the knee was detected.
        }
    """
    assert len(timesteps) == len(
        lid_curve
    ), f"timesteps and lid_curve should have the same length but got {len(timesteps)} and {len(lid_curve)} respectively."

    if torch.is_tensor(timesteps):
        timesteps = timesteps.numpy()
    if torch.is_tensor(lid_curve):
        lid_curve = lid_curve.numpy()
    filtered_timesteps = (ignore_timesteps_left < timesteps) & (timesteps < ignore_timesteps_right)

    kl = KneeLocator(
        timesteps[filtered_timesteps],
        lid_curve[filtered_timesteps],
        S=S,
        curve="convex",
        direction="decreasing",
    )
    best_knee = kl.knee
    best_knee_y = kl.knee_y

    if best_knee is None:
        best_knee_y = ambient_dim

    if return_info:
        return {
            "lid": best_knee_y,
            "convex_hull": lid_curve[filtered_timesteps],
            "timesteps": timesteps[filtered_timesteps],
            "knee_timestep": best_knee,
        }
    return best_knee_y
