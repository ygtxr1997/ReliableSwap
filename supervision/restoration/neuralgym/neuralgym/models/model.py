"""Base class for Model"""
from abc import abstractmethod


class Model(object):

    """Base class for models."""

    def __init__(self, name=''):
        """Initialize model name."""
        self.name = name
