import torch

from abc import ABC, abstractmethod
from collections import OrderedDict


class Metrics(OrderedDict):
    """
    class that provides attribute like access to OrderdDict objects
    """
    def __init__(self, *args, **kwargs):
        super(Metrics, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        if isinstance(self.get(key), AverageMeter):
            self.get(key).update(value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Metrics, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)
    
    def add(self, attr_names):
        """
        adds an attribute with AverageMeter value which keeps
        track of the average, sum and current value.
        """
        if not isinstance(attr_names, list):
            attr_names = [attr_names]
        
        for attr in attr_names:
            assert(isinstance(attr, str))
            self[attr] = Accuracy(attr)


class Accuracy(object):
    """Computes and stores the accuracy"""
    def __init__(self, fmt=':f'):
        self.name = 'acc'
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        reset any of the metrics that is being calculated
        """
        self.val = 0
        self.correct = 0
        self.incorrect = 0
        self.total = 0
        self.acc = 0

    def __call__(self, pred, actual):
        """
        computes counter for correctly classified samples.
        """
        self.total = len(pred)
        self.correct = (torch.argmax(pred, dim=1) == actual).sum()
        # calculate accuracy
        self.acc = self.correct / self.total

