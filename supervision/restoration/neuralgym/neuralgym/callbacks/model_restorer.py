"""model_restorer"""
import os
import logging

import tensorflow._api.v2.compat.v1 as tf

from . import CallbackLoc, OnceCallback
from ..utils.logger import callback_log


class ModelRestorer(OnceCallback):
    """Restore model from file either with dump_prefix or ckpt_file.

    Args:
        saver: Tensorflow saver.
        dump_prefix (str): Prefix of model files.
        ckpt_file (str): Exact name of model file.
        optimistic (bool): Only restore weights of same names with model.
    """

    def __init__(self, saver, dump_prefix=None, ckpt_file=None,
                 optimistic=False):
        super().__init__(CallbackLoc.train_start)
        self._saver = saver
        self._no_ckpt_file = False
        self._optimistic = optimistic
        if ckpt_file is None:
            if dump_prefix is None:
                raise ValueError('dump_prefix is None.')
            # get ckpt file.
            self._ckpt_file = tf.train.latest_checkpoint(
                os.path.dirname(dump_prefix))
            if self._ckpt_file is None:
                self._no_ckpt_file = True
                # raise ValueError('no checkpoint file.')
        else:
            self._ckpt_file = ckpt_file

    def run(self, sess):
        def optimistic_restore(sess, ckpt_file):
            reader = tf.train.NewCheckpointReader(ckpt_file)
            saved_shapes = reader.get_variable_to_shape_map()
            var_names = sorted([(var.name, var.name.split(':')[0])
                                for var in tf.global_variables()
                                if var.name.split(':')[0] in saved_shapes])
            restore_vars = []
            name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                                    tf.global_variables()),
                                tf.global_variables()))
            with tf.variable_scope('', reuse=True):
                for var_name, saved_var_name in var_names:
                    curr_var = name2var[saved_var_name]
                    var_shape = curr_var.get_shape().as_list()
                    if var_shape == saved_shapes[saved_var_name]:
                        restore_vars.append(curr_var)
                        print('- restoring variable: {}'
                                    .format(curr_var.name))
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, ckpt_file)

        if not self._no_ckpt_file:
            callback_log('Trigger ModelRestorer: Load model from %s.'
                         % self._ckpt_file)
            if self._optimistic:
                optimistic_restore(sess, self._ckpt_file)
            else:
                print('- restoring all variables.')
                self._saver.restore(sess, self._ckpt_file)
