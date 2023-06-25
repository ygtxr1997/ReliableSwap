"""WeightsViewer"""
import logging

import tensorflow._api.v2.compat.v1 as tf
import numpy as np

from . import CallbackLoc, OnceCallback
from ..utils.logger import callback_log


class WeightsViewer(OnceCallback):
    """WeightsViewer logs names and size of all weights.

    Args:
        counts (bool): Counting trainalbe weights or not.
        size (bool): Size of trainable weights or not.
        verbose (bool): Display each trainable variable or not.
        hist_summary (bool): Histogram summary of trainable weights or not.

    """

    def __init__(self, counts=True, size=True, verbose=True,
                 hist_summary=True):
        super().__init__(CallbackLoc.train_start)
        self.counts = counts
        self.size = size
        self.verbose = verbose
        self.hist_summary = hist_summary

    def run(self, sess):
        callback_log('Trigger WeightsViewer: logging model weights...')
        total_size = 0
        for var in tf.trainable_variables():
            # counts
            if self.counts or self.size:
                w_size = np.prod(var.get_shape().as_list())
                if self.verbose:
                    print(
                        '- weight name: {}, shape: {}, size: {}'.format(
                            var.name, var.get_shape().as_list(), w_size))
                total_size += w_size
            # histogram summary
            if self.hist_summary:
                tf.summary.histogram(var.name, var)
        if self.counts:
            callback_log('Total counts of trainable weights: %d.' % total_size)
        if self.size:
            # Data is 32-bit datatype in most cases.
            total = total_size * 4
            b_size = total_size % 1024
            k_size = (total_size//1024) % 1024
            m_size = (total_size//(1024 * 1024)) % 1024
            g_size = (total_size//(1024 * 1024 * 1024)) % 1024
            # log
            print(
                'Total size of trainable weights: %dG %dM %dK %dB (Assuming'
                '32-bit data type.)' % (g_size, m_size, k_size, b_size))
