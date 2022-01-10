import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, n_out,
                       eps=1e-5,
                       affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, latent_dim,
                       num_features):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc_w = nn.Linear(latent_dim, num_features)
        self.fc_b = nn.Linear(latent_dim, num_features)

    def forward(self, x, z):
        w = self.fc_w(z)
        b = self.fc_b(z)
        w = w.view(x.size(0), x.size(1), 1, 1)
        b = b.view(x.size(0), x.size(1), 1, 1)
        return w * self.norm(x) + b