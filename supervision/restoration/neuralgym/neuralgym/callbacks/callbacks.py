"""Callbacks."""
from enum import Enum
from abc import abstractmethod

from ..utils.logger import callback_log


class CallbackLoc(Enum):
    """Enum class for callback location."""

    train_start = 0
    train_end = 1
    step_start = 2
    step_end = 3
    exception = 4


class Callback(object):
    """Callback class.

    Callbacks are functions that execute automatically during
    training/evaluation process (primary trainer). For examples,
    saving/loading models, scheduling learning rate,
    updating network parameters (assigning target
    network in reinforcement learning), summary learning processes,
    saving images, secondary trainer (generative adversarial network), etc.

    Currently there are three types of callbacks:

    * :class:`.OnceCallback`
    * :class:`.PeriodicCallback`
    * :class:`.ScheduledCallback`

    and five types of locations to call (supported in primary trainer):

    * train_start
    * train_end
    * step_start
    * step_end
    * exception

    """

    def __init__(self, cb_loc):
        assert isinstance(cb_loc, CallbackLoc), "cb_loc is not of CallbackLoc."
        self.cb_loc = cb_loc

    @abstractmethod
    def run(self):
        """Abstract method for executing the callback.

        **Note**: ops should be defined in __init__, otherwise when callbacks
        are called multiple times, the ops in graph will continue increasing.
        """
        pass


class PeriodicCallback(Callback):
    """PeriodicCallback executes periodically.

    PeriodicalCallback is executed at:

    1) at the step start, and
    2) at the step end

    of training every p steps periodically.

    Args:
        cb_loc: callback location
        pstep (int): run function every pstep
        func (function): function to call
        \*\*kwargs: kwargs for function

    """

    def __init__(self, cb_loc, pstep, func=None, **kwargs):
        assert cb_loc == CallbackLoc.step_start or \
            cb_loc == CallbackLoc.step_end, 'Callback Location Error: '
        'PeriodicalCallback should be executed 1) at the step start, '
        '2) at the step end of training.'
        super().__init__(cb_loc)
        self.pstep = pstep
        assert isinstance(self.pstep, int)
        self._func = func
        self._func_kwargs = kwargs

    def run(self, sess, step):
        callback_log('Trigger PeriodicCallback at Step-%d' % step)
        if self._func:
            self._func(sess, self._func_kwargs)
        else:
            raise ValueError('No callback function to execute.')


class OnceCallback(Callback):
    """OnceCallback only executes once.

    OnceCallback is executed:

    1) at the train start,
    2) at the train end, and
    3) when exception occurs

    during training process.
    """

    def __init__(self, cb_loc, func=None, **kwargs):
        assert cb_loc == CallbackLoc.train_start or \
            cb_loc == CallbackLoc.train_end or \
            cb_loc == CallbackLoc.exception, 'Callback Location Error: '
        'OnceCallback should be executed 1) at the train start, '
        '2) at the train end, 3) when exception occurs during training.'
        super().__init__(cb_loc)
        self._func = func
        self._func_kwargs = kwargs

    def run(self, sess):
        callback_log('Trigger OnceCallback')
        if self._func:
            self._func(sess, self._func_kwargs)
        else:
            raise ValueError('No callback function to execute.')


class ScheduledCallback(Callback):
    """ScheduledCallback executes according to its schedule.

    ScheduledCallback is executed:

    1) at the step start, and
    2) at the step end

    according to recorded step in schedule.

    Args:
        cb_loc: callback location
        schedule (dict): a dict, with step as its key, funcs as its value:
            e.g. {1: func1, 80: func2}
    """

    def __init__(self, cb_loc, schedule):
        assert cb_loc == CallbackLoc.step_start or \
            cb_loc == CallbackLoc.step_end, 'Callback Location Error: '
        'ScheduledCallback should be executed 1) at the step start, '
        '2) at the step end according to recorded step in schedule.'
        super().__init__(cb_loc)
        self.schedule = schedule

    @abstractmethod
    def run(self, sess, step):
        callback_log('Trigger ScheduledCallback at Step-%d' % step)
        func = self.schedule[step]
        if func:
            func(sess)
        else:
            raise ValueError('No callback function to execute.')
