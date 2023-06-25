import datetime
import logging
import os

from logging.config import dictConfig
from .utils.logger import colorize


__version__ = '0.0.1'
__all__ = ['Config', 'get_gpus', 'set_gpus', 'date_uid', 'unset_logger',
           'get_sess']


def date_uid():
    """Generate a unique id based on date.

    Returns:
        str: Return uid string, e.g. '20171122171307111552'.

    """
    return str(datetime.datetime.now()).replace(
        '-', '').replace(
            ' ', '').replace(
                ':', '').replace('.', '')


from . import callbacks
from . import ops
from . import train
from . import models
from . import data
from . import server

from .utils.gpus import set_gpus, get_gpus
from .utils.tf_utils import get_sess
from .utils.config import Config
