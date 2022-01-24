import torch
import warnings
import numpy as np

from abc import ABC, abstractmethod

class Metrics(ABC):
    """
    Base class for all Metrics.
    """
    def __init__(self, num_features,
                       feature_extractor = None,
                       output_transform = lambda x: x,
                       device = torch.device("cpu")):
        self._num_features = num_features
        self._device = torch.device(device)
        self._feature_extractor = feature_extractor.to(self._device)
        self._output_transform = output_transform
        self._is_reduced = False
        self.reset()

    def _check_feature_shapes(self, samples) -> None:
        if samples.dim() != 2:
            raise ValueError(f"feature_extractor output must be a tensor of dim 2, got: {samples.dim()}")
        if samples.shape[0] == 0:
            raise ValueError(f"Batch size should be greater than one, got: {samples.shape[0]}")
        if samples.shape[1] != self._num_features:
            raise ValueError(
                f"num_features returned by feature_extractor should be {self._num_features}, got: {samples.shape[1]}"
            )

    def _extract_features(self, inputs) -> torch.Tensor:
        inputs = inputs.detach()
        if inputs.device != torch.device(self._device):
            inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._feature_extractor(inputs)
            print(outputs.shape)
        self._check_feature_shapes(outputs)
        return outputs

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