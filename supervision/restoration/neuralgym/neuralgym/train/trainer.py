import time
import logging

import numpy as np
import tensorflow._api.v2.compat.v1 as tf

from ..utils.logger import ProgressBar
from ..callbacks import CallbackLoc
from ..callbacks import PeriodicCallback, OnceCallback, ScheduledCallback
from ..ops.train_ops import process_gradients


class Trainer(object):
    """Trainer class for train iterative algorithm on single GPU.

    There are two types of trainer in neuralgym: primary trainer and
    secondary trainer. For primary trainer, tensorflow related instances
    and configurations will be initialized, e.g. init all variables, summary
    writer, session, start_queue_runner and others. For the secondary trainer
    only train_ops and losses are iteratively updated/ran.
    """

    def __init__(self, primary=True, **context):
        self.context = context
        self.primary = primary
        self.callbacks = self.context.pop('callbacks', [])
        # contexts
        self.context['feed_dict'] = self.context.pop('feed_dict', {})
        self.context['max_iters'] = int(self.context.pop('max_iters', 999999))
        self.context['log_dir'] = self.context.pop('log_dir', '/tmp/neuralgym')
        self.context['spe'] = self.context.pop('spe', 1)
        # grads summary
        self.context['grads_summary'] = self.context.pop(
            'grads_summary', True)
        # train ops and losses
        self._train_op = self.context.pop('train_op', None)
        if self._train_op is None:
            self._train_op, self._loss = self.train_ops_and_losses()
        else:
            self._loss = self.context.pop('loss', 0)
        # global step
        self.context['log_progress'] = self.context.pop('log_progress', True)
        if self.context['log_progress']:
            self._bar = ProgressBar()
        # total loss, beginning timepoint
        self._log_stats = [0, None]
        # callbacks types
        self._periodic_callbacks = None
        self._once_callbacks = None
        self._scheduled_callbacks = None
        # init primary trainer
        if self.primary:
            self.init_primary_trainer()
        # log context of trainer
        if self.primary:
            print(' Context Of Primary Trainer '.center(80, '-'))
        else:
            print(' Context Of Secondary Trainer '.center(80, '-'))
        for k in self.context:
            print(k + ': ' + str(self.context[k]))
        print(''.center(80, '-'))

    def init_primary_trainer(self):
        """Initialize primary trainer context including:

            * log_dir
            * global_step
            * sess_config
            * allow_growth
            * summary writer
            * saver
            * global_variables_initializer
            * start_queue_runners

        """
        self.context['global_step'] = self.context.pop(
            'global_step', tf.get_variable(
                'global_step', [], dtype=tf.int32,
                initializer=tf.zeros_initializer(), trainable=False))
        self.context['global_step_add_one'] = tf.assign_add(
            self.context['global_step'], 1, name='add_one_to_global_step')
        self.context['sess_config'] = self.context.pop(
            'sess_config', tf.ConfigProto())
        self.context['sess_config'].gpu_options.allow_growth = (
            self.context.pop('allow_growth', True))
        self.context['sess_config'].allow_soft_placement = self.context.pop(
            'allow_soft_placement', True)
        self.context['sess'] = tf.Session(config=self.context['sess_config'])
        self.context['summary_writer'] = tf.summary.FileWriter(
            self.context['log_dir'], self.context['sess'].graph)
        self.context['saver'] = tf.train.Saver(tf.global_variables())
        # queue runner
        self.context['start_queue_runners'] = self.context.pop(
            'start_queue_runner', True)
        if self.context['start_queue_runners']:
            tf.train.start_queue_runners(sess=self.context['sess'])
        # initialization
        self.context['global_variables_initializer'] = self.context.pop(
            'global_variables_initializer', True)
        if self.context['global_variables_initializer']:
            self.context['sess'].run(tf.global_variables_initializer())

    def train(self):
        """Start training with callbacks.

        """
        sess = self.context['sess']
        max_iters = self.context['max_iters']
        self.update_callbacks()
        if self.context.get('global_step') is None:
            step = 0
            global_step_add_one = None
        else:
            step = sess.run(self.context['global_step'])
            global_step_add_one = self.context['global_step_add_one']
        # once_callbacks at train start
        for cb in self._once_callbacks:
            if cb.cb_loc == CallbackLoc.train_start:
                cb.run(sess)
        try:
            while step < max_iters:
                # update and get current step
                step += 1
                if global_step_add_one is not None:
                    sess.run(global_step_add_one)
                # periodic callbacks at step start
                for cb in self._periodic_callbacks:
                    if (cb.cb_loc == CallbackLoc.step_start and
                            step % cb.pstep == 0):
                        cb.run(sess, step)
                # scheduled callbacks at step start
                for cb in self._scheduled_callbacks:
                    if (cb.cb_loc == CallbackLoc.step_start and
                            step in cb.schedule):
                        cb.run(sess, step)
                # run train op
                _, loss_value = sess.run([self._train_op, self._loss],
                                         feed_dict=self.context['feed_dict'])
                # if nan, exist
                assert not np.isnan(loss_value)
                # log one
                if self.context['log_progress']:
                    self.progress_logger(step, loss_value)
                # periodic callbacks at step end
                for cb in self._periodic_callbacks:
                    if (cb.cb_loc == CallbackLoc.step_end and
                            step % cb.pstep == 0):
                        cb.run(sess, step)
                # scheduled callbacks at step end
                for cb in self._scheduled_callbacks:
                    if (cb.cb_loc == CallbackLoc.step_end and
                            step in cb.schedule):
                        cb.run(sess, step)
        except (KeyboardInterrupt, SystemExit):
            print("Training is stoped.")
        except:
            raise
        finally:
            # once_callbacks at exception
            for cb in self._once_callbacks:
                if cb.cb_loc == CallbackLoc.exception:
                    cb.run(sess)
        # once_callbacks at train end
        for cb in self._once_callbacks:
            if cb.cb_loc == CallbackLoc.train_end:
                cb.run(sess)

    def progress_logger(self, step, loss):
        """Progress bar for logging.

        **Note** all statistics are averaged over epoch.
        """
        # init
        if self._log_stats[1] is None:
            self._log_stats[1] = time.time()
            self._log_stats[0] = loss
            return
        # update statistic
        self._log_stats[0] += loss
        # time
        t_start = self._log_stats[1]
        t_now = time.time()
        spe = self.context['spe']
        # after running the session, the step is actually increased.
        step = step + 1
        epoch_end = (step % spe == 0)
        # set update step 0.1%
        log_per_iters = max(int(spe/1000), 10)
        # update progress bar per log_per_iters
        epoch_nums = (step - 1) // spe + 1
        epoch_iters = (step - 1) % spe + 1
        if epoch_iters % log_per_iters == 0 or epoch_end:
            batches_per_sec = epoch_iters / (t_now - t_start)
            texts = ''.join([
                'train epoch {},'.format(epoch_nums),
                ' iter {}/{},'.format(epoch_iters, spe),
                ' loss {:.6f}, {:.2f} batches/sec.'.format(
                    self._log_stats[0]/epoch_iters, batches_per_sec),
            ])
            # progress, if at the end of epoch, 100%; else current progress
            prog = 1 if epoch_end else (step / spe) % 1
            self._bar.progress(prog, texts)
        # reset
        if epoch_end:
            self._log_stats[1] = None
            self._log_stats[0] = 0
        return

    def add_callbacks(self, callbacks):
        """Add callbacks.

        Args:
            callbacks: list of callbacks

        """
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        # keep order
        self.callbacks = self.callbacks + callbacks
        # after add callbacks, update callbacks list.
        self.update_callbacks()

    def update_callbacks(self):
        def _check_type(t, cb):
            return t == cb.__class__ or t in cb.__class__.__bases__

        # clear
        self._periodic_callbacks = []
        self._once_callbacks = []
        self._scheduled_callbacks = []
        # add
        for cb in self.callbacks:
            if _check_type(PeriodicCallback, cb):
                self._periodic_callbacks.append(cb)
            if _check_type(OnceCallback, cb):
                self._once_callbacks.append(cb)
            if _check_type(ScheduledCallback, cb):
                self._scheduled_callbacks.append(cb)

    def train_ops_and_losses(self):
        optimizer = self.context['optimizer']
        loss = self.context.get('loss')
        var_list = self.context.get('var_list')
        graph_def_kwargs = self.context['graph_def_kwargs']
        gradient_processor = self.context.get('gradient_processor')
        if loss is None:
            loss = self.context['graph_def'](**graph_def_kwargs)
        # get gradients
        grads = optimizer.compute_gradients(loss, var_list)
        if self.context['grads_summary']:
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram('gradients/' + var.name, grad)
        grads = process_gradients(grads, gradient_processor)
        # get operations
        apply_gradient_op = optimizer.apply_gradients(grads)
        return apply_gradient_op, loss
