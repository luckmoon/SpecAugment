#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-08-08 15:57
# @Author  : wulala
# @Project : SpecAugment
# @File    : b.py
# @Software: PyCharm

import tensorflow as tf

print(tf.VERSION)
# tf.enable_eager_execution()
# print(tf.executing_eagerly())

import numpy as np
from tensorflow.contrib.image.python.ops import interpolate_spline
from tensorflow.contrib.image.python.ops import dense_image_warp


def _get_grid_locations(image_height, image_width):
    """Wrapper for np.meshgrid."""

    y_range = np.linspace(0, image_height - 1, image_height)
    x_range = np.linspace(0, image_width - 1, image_width)
    y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
    return np.stack((y_grid, x_grid), -1)


def _get_boundary_locations(image_height, image_width, num_points_per_edge):
    """Compute evenly-spaced indices along edge of image."""
    y_range = np.linspace(0, image_height - 1, num_points_per_edge + 2)
    x_range = np.linspace(0, image_width - 1, num_points_per_edge + 2)
    ys, xs = np.meshgrid(y_range, x_range, indexing='ij')
    is_boundary = np.logical_or(
        np.logical_or(xs == 0, xs == image_width - 1),
        np.logical_or(ys == 0, ys == image_height - 1))
    return np.stack([ys[is_boundary], xs[is_boundary]], axis=-1)


def _expand_to_minibatch(np_array, batch_size):
    """Tile arbitrarily-sized np_array to include new batch dimension."""
    tiles = [batch_size] + [1] * np_array.ndim
    return np.tile(np.expand_dims(np_array, 0), tiles)


def _add_zero_flow_controls_at_boundary(control_point_locations,
                                        control_point_flows, image_height,
                                        image_width, boundary_points_per_edge):
    """Add control points for zero-flow boundary conditions.

     Augment the set of control points with extra points on the
     boundary of the image that have zero flow.

    Args:
      control_point_locations: input control points
      control_point_flows: their flows
      image_height: image height
      image_width: image width
      boundary_points_per_edge: number of points to add in the middle of each
                             edge (not including the corners).
                             The total number of points added is
                             4 + 4*(boundary_points_per_edge).

    Returns:
      merged_control_point_locations: augmented set of control point locations
      merged_control_point_flows: augmented set of control point flows
    """

    batch_size = control_point_locations.get_shape()[0].value

    boundary_point_locations = _get_boundary_locations(image_height, image_width,
                                                       boundary_points_per_edge)

    boundary_point_flows = np.zeros([boundary_point_locations.shape[0], 2])

    type_to_use = control_point_locations.dtype
    boundary_point_locations = _expand_to_minibatch(boundary_point_locations, batch_size)

    boundary_point_flows = _expand_to_minibatch(boundary_point_flows, batch_size)

    merged_control_point_locations = np.concatenate(
        [control_point_locations, boundary_point_locations], 1)

    merged_control_point_flows = np.concatenate(
        [control_point_flows, boundary_point_flows], 1)

    return merged_control_point_locations, merged_control_point_flows


if __name__ == '__main__':
    image_height = 5
    image_width = 13
    image = np.random.rand(1, 5, 13, 1)
    image = tf.constant(image)
    boundary_points_per_edge = 1  # 每条边上有几个点
    # control_point_locations = np.array([1, 2, 3])
    # control_point_flows = np.expand_dims([])
    source_control_point_locations = [[2, 1]]
    source_control_point_locations = np.float64(np.expand_dims(source_control_point_locations, 0))
    dest_control_point_locations = [[2, 2]]
    dest_control_point_locations = np.float64(np.expand_dims(dest_control_point_locations, 0))
    batch_size = source_control_point_locations.shape[0]

    control_point_flows = (
            dest_control_point_locations - source_control_point_locations)
    print(control_point_flows)

    print('=======' * 4)
    grid_locations = _get_grid_locations(5, 13)
    print(grid_locations.shape)
    flattened_grid_locations = np.reshape(grid_locations,
                                          [image_height * image_width, 2])
    flattened_grid_locations = _expand_to_minibatch(flattened_grid_locations, batch_size)

    # # local
    # print('=======' * 4)
    # boundary_point_locations = _get_boundary_locations(image_height, image_width,
    #                                                    boundary_points_per_edge)
    # print(boundary_point_locations.shape)
    # print(boundary_point_locations)
    #
    # print('=======' * 4)
    # boundary_point_flows = np.zeros([boundary_point_locations.shape[0], 2])
    # print(boundary_point_flows.shape)
    #
    # print('=======' * 4)
    # boundary_point_locations = _expand_to_minibatch(boundary_point_locations, batch_size)  # 把边上点的坐标tile
    # print(boundary_point_locations)
    # boundary_point_flows = _expand_to_minibatch(boundary_point_flows, batch_size)
    # print(boundary_point_flows.shape)
    #
    # print()
    # merged_control_point_locations = np.concatenate([control_point_flows, boundary_point_flows], 1)
    interpolation_order = 3
    regularization_weight = 0.0
    print("dest_control_point_locations:", dest_control_point_locations)
    print("control_point_flows:", control_point_flows)
    flattened_flows = interpolate_spline.interpolate_spline(
        dest_control_point_locations, control_point_flows,
        flattened_grid_locations, interpolation_order, regularization_weight)
    print(flattened_flows)
    dense_flows = tf.reshape(flattened_flows,
                             [batch_size, image_height, image_width, 2])

    warped_image = dense_image_warp.dense_image_warp(image, dense_flows)
    print(warped_image)