"""sync model weights from one variable scope to another"""
import time
import logging

import tensorflow._api.v2.compat.v1 as tf

from . import PeriodicCallback, CallbackLoc
from ..utils.logger import callback_log


class ModelSync(PeriodicCallback):
    """ModelSync.

    Currently it only supports sync trainable variables from one namescope to
    another namescope, which is enough for reinforcement learning.

    Args:
        pstep (int): Sync every pstep.
        from_namescope (str): Sync from from_namescope.
        to_namescope (str): Sync to to_namescope.
        step_start: Sync at step_start, otherwise at step_end.

    Examples::

        # TODO

    """

    def __init__(self, pstep, from_namescope, to_namescope, step_start=False):
        if step_start:
            super().__init__(CallbackLoc.step_start, pstep)
        else:
            super().__init__(CallbackLoc.step_end, pstep)
        self.from_namescope = from_namescope
        self.to_namescope = to_namescope
        vars_from = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, from_namescope)
        vars_to = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, to_namescope)
        # ops to sync
        self._ops = []
        callback_log(
            'Add callback: sync model from namescope: {} to namescope: {}'
            .format(from_namescope, to_namescope))
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for var in vars_to:
                name = var.name
                if self.from_namescope == '':
                    # root
                    name = name.replace(
                        self.to_namescope+'/', self.from_namescope)
                else:
                    name = name.replace(
                        self.to_namescope+'/', self.from_namescope+'/')
                name = name.replace(':0', '')
                from_var = tf.get_variable(name)
                print('Add op for syncing from var: {} to var: {}'
                            .format(from_var.name, var.name))
                self._ops.append(var.assign(from_var))

    def run(self, sess, step):
        """Run model sync"""
        sess.run(self._ops)
