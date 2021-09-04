import torch
from abc import ABC, abstractmethod


class MetricsMeter(object):
    """Computes and stores the metrics"""
    def __init__(self, metrics, fmt=':f'):
        self.metrics = metrics
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        this function should reset any of the metrics that is being calculated
        """
        self.metrics = None

    @abstractmethod
    def update(self, val, n=1):
        """
        update the metrics that is being calculated
        """
        pass

