import torch
import warnings
import numpy as np

from abc import ABC, abstractmethod

class Metrics(ABC):
    """
    Base class for all Metrics.
    """
    def __init__(self, output_transform = lambda x: x, device = torch.device("cpu")):
        self._output_transform = output_transform\
        # Check device if distributed is initialized:
        self._device = torch.device(device)
        self._is_reduced = False
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric to it's initial state.
        By default, this is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, output) -> None:
        """
        Updates the metric's state using the passed batch output.
        By default, this is called once for each batch.
        Args:
            output: the is the output from the engine's process function.
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Computes the metric based on it's accumulated state.
        By default, this is called at the end of each epoch.
        Returns:
            Any: | the actual quantity of interest. However, if a :class:`~collections.abc.Mapping` is returned,
                 it will be (shallow) flattened into `engine.state.metrics` when
                 :func:`~ignite.metrics.metric.Metric.completed` is called.
        Raises:
            NotComputableError: raised when the metric cannot be computed.
        """
        pass