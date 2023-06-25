"""HyperParamScheduler"""
import tensorflow._api.v2.compat.v1 as tf

from . import ScheduledCallback, CallbackLoc
from ..utils.logger import callback_log
from ..ops.summary_ops import scalar_summary


class HyperParamScheduler(ScheduledCallback):
    """Set hyper parameters according to schedule.

    This callback sets hyper parameters with numpy using tf.assign
    according to schedule.

    Examples::

        HyperParamScheduler(
            'lr',
            {
                1: 1e-2,
                150: 1e-3,
                225: 4e-4,
                300: 1e-4,
            },
            scope=None,
        )
    """
    def __init__(self, param_name, schedule, scope=None,
                 cb_loc=CallbackLoc.step_end):
        super().__init__(cb_loc, schedule)
        if scope is None:
            scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=True):
            self._param = tf.get_variable(param_name)
        scalar_summary('hyper_param_scheduler/'+param_name, self._param)

    def run(self, sess, step):
        callback_log(
            'Trigger HyperParamScheduler callback at Step-%d: update '
            'hyper parameters of %s: %s' % (
                step, self._param.name[:-2], self.schedule[step]))
        sess.run(self._param.assign(self.schedule[step]))
