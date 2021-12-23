import scipy
import torch
import scipy.linalg
import numpy as np
import warnings

from distutils.version import LooseVersion
from metrics.metrics import Metrics
from torchvision.models import inception_v3

class InceptionModel(torch.nn.Module):
    """Inception Model pre-trained on the ImageNet Dataset.
    Args:
        return_features: set it to `True` if you want the model to return features from the last pooling
            layer instead of prediction probabilities.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
    """
    def __init__(self, return_features, device = "cpu"):
        super(InceptionModel, self).__init__()
        self.device = device
        self.model = inception_v3(pretrained=True).to(self.device)
        if return_features:
            self.model.fc = torch.nn.Identity()
        else:
            self.model.fc = torch.nn.Sequential(self.model.fc, torch.nn.Softmax(dim=1))
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {x.dim()}")
        if x.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {x.shape}")
        if x.device != torch.device(self.device):
            x = x.to(self.device)
        return self.model(x)

def fid_score(mu1, mu2, sigma1, sigma2, eps = 1e-6):
    mu1, mu2 = mu1.cpu(), mu2.cpu()
    sigma1, sigma2 = sigma1.cpu(), sigma2.cpu()
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.mm(sigma2), disp=False)
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    if not np.isfinite(covmean).all():
        tr_covmean = np.sum(np.sqrt(((np.diag(sigma1) * eps) * (np.diag(sigma2) * eps)) / (eps * eps)))

    return float(diff.dot(diff).item() + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)

class FID(Metrics):
    """Calculates Frechet Inception Distance.
    __ https://arxiv.org/pdf/1706.08500.pdf
    In addition, a faster and online computation approach can be found in `Chen et al. 2014`__
    __ https://arxiv.org/pdf/2009.14075.pdf
    """
    def __init__(self, num_features = None, feature_extractor = None, output_transform = lambda x: x, device = torch.device("cpu")):
        if num_features is None and feature_extractor is None:
            # defaults to 1000
            self._num_features = 1000
        else:
            self._num_features = num_features

        if feature_extractor is None:
            feature_extractor = InceptionModel(return_features=False, device=device)

        self._eps = 1e-6
        super(FID, self).__init__(output_transform=output_transform, device=device)
        self._feature_extractor = feature_extractor.to(device)
        self._device = device

    @staticmethod
    def _online_update(features, total, sigma):
        total += features
        if LooseVersion(torch.__version__) <= LooseVersion("1.7.0"):
            sigma += torch.ger(features, features)
        else:
            sigma += torch.outer(features, features)

    def _check_feature_shapes(self, samples: torch.Tensor) -> None:
        if samples.dim() != 2:
            raise ValueError(f"feature_extractor output must be a tensor of dim 2, got: {samples.dim()}")
        if samples.shape[0] == 0:
            raise ValueError(f"Batch size should be greater than one, got: {samples.shape[0]}")
        if samples.shape[1] != self._num_features:
            raise ValueError(
                f"num_features returned by feature_extractor should be {self._num_features}, got: {samples.shape[1]}"
            )

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.detach()
        if inputs.device != torch.device(self._device):
            inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._feature_extractor(inputs).to(self._device, dtype=torch.float64)
        self._check_feature_shapes(outputs)
        return outputs

    def _get_covariance(self, sigma, total):
        """
        Calculates covariance from mean and sum of products of variables
        """
        if LooseVersion(torch.__version__) <= LooseVersion("1.7.0"):
            sub_matrix = torch.ger(total, total)
        else:
            sub_matrix = torch.outer(total, total)
        sub_matrix = sub_matrix / self._num_examples
        return (sigma - sub_matrix) / (self._num_examples - 1)

    def reset(self):
        super(FID, self).reset()
        self._train_sigma = torch.zeros(
            (self._num_features, self._num_features), dtype=torch.float64, device=self._device
        )
        self._train_total = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)
        self._test_sigma = torch.zeros(
            (self._num_features, self._num_features), dtype=torch.float64, device=self._device
        )
        self._test_total = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)
        self._num_examples = 0

    def update(self, output):
        train, test = output
        train_features = self._extract_features(train)
        test_features = self._extract_features(test)
        if train_features.shape[0] != test_features.shape[0] or train_features.shape[1] != test_features.shape[1]:
            raise ValueError(
            f"""
            Number of Training Features and Testing Features should be equal ({train_features.shape} != {test_features.shape})
            """
            )
        # Updates the mean and covariance for the train features
        for features in train_features:
            self._online_update(features, self._train_total, self._train_sigma)
        # Updates the mean and covariance for the test features
        for features in test_features:
            self._online_update(features, self._test_total, self._test_sigma)
        self._num_examples += train_features.shape[0]

    def compute(self) -> float:
        fid = fid_score(
            mu1=self._train_total / self._num_examples,
            mu2=self._test_total / self._num_examples,
            sigma1=self._get_covariance(self._train_sigma, self._train_total),
            sigma2=self._get_covariance(self._test_sigma, self._test_total),
            eps=self._eps,
        )
        if torch.isnan(torch.tensor(fid)) or torch.isinf(torch.tensor(fid)):
            warnings.warn("The product of covariance of train and test features is out of bounds.")
        return fid