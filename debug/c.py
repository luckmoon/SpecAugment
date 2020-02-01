#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-08-12 17:21
# @Author  : wulala
# @Project : SpecAugment
# @File    : c.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(12)
import tensorflow as tf
tf.enable_eager_execution()
print(tf.executing_eagerly())

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _interpolate_bilinear(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij'):
    """Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
      name: a name for the operation (optional).
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the inputs
        invalid.
    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

    with ops.name_scope(name):
        grid = ops.convert_to_tensor(grid)
        query_points = ops.convert_to_tensor(query_points)
        shape = grid.get_shape().as_list()
        if len(shape) != 4:
            msg = 'Grid must be 4 dimensional. Received size: '
            raise ValueError(msg + str(grid.get_shape()))

        batch_size, height, width, channels = shape
        query_type = query_points.dtype
        grid_type = grid.dtype

        if (len(query_points.get_shape()) != 3 or
                query_points.get_shape()[2].value != 2):
            msg = ('Query points must be 3 dimensional and size 2 in dim 2. Received '
                   'size: ')
            raise ValueError(msg + str(query_points.get_shape()))

        _, num_queries, _ = query_points.get_shape().as_list()

        if height < 2 or width < 2:
            msg = 'Grid must be at least batch_size x 2 x 2 in size. Received size: '
            raise ValueError(msg + str(grid.get_shape()))

        alphas = []
        floors = []
        ceils = []

        index_order = [0, 1] if indexing == 'ij' else [1, 0]
        print('--->1, query_points:', query_points)
        unstacked_query_points = array_ops.unstack(query_points, axis=2)
        print('--->2, unstacked_query_points:', unstacked_query_points)

        for dim in index_order:
            print('======='* 10, dim)
            with ops.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]
                print('queries:', queries)

                size_in_indexing_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                print("max_floor:", max_floor)
                min_floor = constant_op.constant(0.0, dtype=query_type)
                print('math_ops.floor(queries):', math_ops.floor(queries))
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                print("floor:", floor)
                int_floor = math_ops.cast(floor, dtypes.int32)
                print("int_floor:", int_floor)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)
                print("ceils:", ceils)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = math_ops.cast(queries - floor, grid_type)
                print("alpha1:", alpha)
                min_alpha = constant_op.constant(0.0, dtype=grid_type)
                max_alpha = constant_op.constant(1.0, dtype=grid_type)
                alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)
                print("alpha2:", alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = array_ops.expand_dims(alpha, 2)
                alphas.append(alpha)
                print("alphas:", alphas)

        if batch_size * height * width > np.iinfo(np.int32).max / 8:
            error_msg = """The image size or batch size is sufficiently large
                     that the linearized addresses used by array_ops.gather
                     may exceed the int32 limit."""
            raise ValueError(error_msg)

        flattened_grid = array_ops.reshape(grid,
                                           [batch_size * height * width, channels])
        batch_offsets = array_ops.reshape(
            math_ops.range(batch_size) * height * width, [batch_size, 1])

        # This wraps array_ops.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using array_ops.gather_nd.
        def gather(y_coords, x_coords, name):
            with ops.name_scope('gather-' + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                return array_ops.reshape(gathered_values,
                                         [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        print("+++++++"*10)
        print("floors:", floors)
        print("ceils:", ceils)
        top_left = gather(floors[0], floors[1], 'top_left')
        print("top_left:", top_left)
        top_right = gather(floors[0], ceils[1], 'top_right')
        print("top_right:", top_right)
        bottom_left = gather(ceils[0], floors[1], 'bottom_left')
        print('bottom_left:', bottom_left)
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right')
        print('bottom_right:', bottom_right)

        # now, do the actual interpolation
        with ops.name_scope('interpolate'):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp


if __name__ == '__main__':
    grid = np.random.randint(-10, 10, size=(1, 5, 7, 1)).astype(np.float64)
    print("grid:", grid)
    query_points = np.random.randint(-7, 7, size=(1, 3, 2)).astype(np.float64)
    res = _interpolate_bilinear(grid, query_points)
    # print(res)
