""" NPZ Model Loader

Model will be loaded from npz file in format of:
http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
"""
import logging

import numpy as np
import tensorflow._api.v2.compat.v1 as tf

from . import CallbackLoc, OnceCallback
from ..utils.logger import callback_log


class NPZModelLoader(OnceCallback):
    """NPZModelLoader loads a model with weights in npz file.

    Args:
        npz_file (str): name of npz_file
        weights: if provided, only load names in weights from npz file
        variable_scope: if provided, load all weights in this scope,
            otherwise load from default variable scope.

    Examples::

        # TODO

    """

    def __init__(self, npz_file, weights=None,
                 variable_scope=tf.get_variable_scope()):
        def convert_name(name):
            """convert tensorflow variable name to normal model name.
            we assume the name template of tensorflow is like:
                - model/conv1/weights:0
                - model/bn5c_branch2c/variance:0
            """
            name = name[:-2]
            ind = name.rfind('/', 0, name.rfind('/'))
            return name[ind+1:]

        super().__init__(CallbackLoc.train_start)
        self._npz_file = npz_file
        if self._npz_file[-4:] != '.npz':
            assert ValueError('Not a valid .npz file.')
        # get weights
        self._weights = weights
        if self._weights is None:
            self._weights = {}
            for tf_var in tf.global_variables():
                # we assume name template is variable_scope/conv1/weights:0
                if tf_var.name.startswith(variable_scope):
                    name = convert_name(tf_var.name)
                    self._weights[name] = tf_var
        # load npz data
        self._npz_data = np.load(self._npz_file)

    def run(self, sess):
        callback_log('Trigger NPZModelLoader: Load npz model from %s.'
                     % self._npz_file)
        for name in self._weights:
            if name in self._npz_data.keys():
                npy = self._npz_data[name]
                if (list(npy.shape) !=
                        self._weights[name].get_shape().as_list()):
                    logger.warning(
                        'Dimension of weights not equal. Ignored weights of '
                        'name: {}.'.format(self._weights[name].name))
                else:
                    sess.run(self._weights[name].assign(npy))
                    print('Loaded weights of name: {}.'.format(
                        self._weights[name].name))
            else:
                print('Ignored weights of name: {}.'.format(
                    self._weights[name].name))
