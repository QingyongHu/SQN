import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
from os.path import dirname

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
interpolate_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so.so'))

ROOT_DIR = dirname(dirname(dirname(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.three_nn(xyz1, xyz2)


ops.NoGradient('ThreeNN')


def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.three_interpolate(points, idx, weight)


@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]


def knn_search(support_pts, query_pts, k):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """

    neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
    return neighbor_idx.astype(np.int32)


def gather_neighbour(pc, neighbor_idx):
    # gather the coordinates or features of neighboring points
    batch_size = tf.shape(pc)[0]
    # num_points = tf.shape(pc)[1]
    num_points = tf.shape(neighbor_idx)[1]
    d = pc.get_shape()[2].value
    index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
    features = tf.batch_gather(pc, index_input)
    features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
    return features


if __name__ == '__main__':
    import numpy as np
    import time

    np.random.seed(100)
    pts = np.random.random((32, 128, 64)).astype('float32')
    tmp1 = np.random.random((32, 512, 3)).astype('float32')
    tmp2 = np.random.random((32, 128, 3)).astype('float32')
    idx_knn = knn_search(tmp2, tmp1, 3)

    with tf.device('/cpu:0'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        dist, idx = three_nn(xyz1, xyz2)
        weight = tf.ones_like(dist) / 3.0
        interpolated_points = three_interpolate(points, idx, weight)

        # batch_knn
        neigh_idx = tf.py_func(knn_search, [tmp2, tmp1, 3], tf.int32)
        neighbor_xyz = gather_neighbour(xyz2, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz1, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        # relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=False))
        relative_dis = tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=False)

    with tf.Session('') as sess:
        now = time.time()
        for _ in range(100):
            ret = sess.run(interpolated_points)
            print(sess.run(neigh_idx - idx))
            print(sess.run(dist - relative_dis))
            print('dist', sess.run(dist))
            print('dist2', sess.run(relative_dis))
        print(time.time() - now)
        print(ret.shape, ret.dtype)
        # print ret
