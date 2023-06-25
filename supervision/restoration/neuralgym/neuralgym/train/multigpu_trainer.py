"""Class for multi-GPU trainer."""
import logging
import threading

import tensorflow._api.v2.compat.v1 as tf

from ..utils.logger import ProgressBar
from ..ops.train_ops import average_gradients, process_gradients
from .trainer import Trainer


class MultiGPUTrainer(Trainer):
    """Trainer class for train iterative algorithm on multi GPUs.

    Args:
        num_gpus (int): Number of GPU(s) for training.
        async_train (bool): Asynchronous train or not.

    """

    def __init__(self, **context):
        self.context = context
        self.context['async_train'] = context.pop('async_train', False)
        self._train_op, self._loss = self.train_ops_and_losses()
        super().__init__(**self.context)

    def train(self):
        """Start training with callbacks.

        """
        def train_function(sess, train_op):
            """"""
            while True:
                sess.run(train_op)

        if self.context['async_train']:
            train_threads = []
            for i, train_op in enumerate(self._train_op):
                if i == 0:
                    # main thread
                    pass
                else:
                    train_threads.append(
                        threading.Thread(
                            target=train_function, args=(
                                self.context['sess'], train_op,)))
            # Start the threads, and block on their completion.
            try:
                for t in train_threads:
                    print("Start new thread for async training.")
                    t.start()
                # start main thread
                super().train()
                for t in train_threads:
                    t.join()
            except (KeyboardInterrupt, SystemExit):
                print("Training is stoped.")
        else:
            super().train()

    def train_ops_and_losses(self):
        optimizer = self.context['optimizer']
        loss = self.context.get('loss')
        var_list = self.context.get('var_list')
        graph_def_kwargs = self.context['graph_def_kwargs']
        gradient_processor = self.context.get('gradient_processor')
        tower_grads = []
        tower_losses = []
        for gpu in range(self.context.get('num_gpus')):
            with tf.device('/gpu:%d' % gpu):
                # with tf.name_scope('tower_gpu%d' % gpu) as scope:
                # Reuse variables for the next tower.
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    loss = self.context['graph_def'](
                        gpu_id=gpu, **graph_def_kwargs)
                    tower_losses.append(loss)
                    # Calculate the gradients for the batch of data
                    grads = optimizer.compute_gradients(loss, var_list)
                    if self.context['grads_summary']:
                        for grad, var in grads:
                            if grad is not None:
                                tf.summary.histogram(
                                    'gradients/' + var.name, grad)
                    grads = process_gradients(grads, gradient_processor)
                    tower_grads.append(grads)

        if self.context['async_train']:
            apply_gradient_op = []
            loss = tower_losses[0]  # only monitor loss of first tower
            for i in range(len(tower_grads)):
                apply_gradient_op.append(
                    optimizer.apply_gradients(tower_grads[i]))
        else:
            # average gradients.
            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads)
            loss = tf.reduce_mean(tower_losses)
        return apply_gradient_op, loss
