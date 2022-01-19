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
        self.fc = nn.Linear(latent_dim, num_features*2)

    def forward(self, x, z):
        h = self.fc(z)
        h = h.view(h.size(0), h.size(1), 1, 1)
        w, b = torch.chunk(h, chunks=2, dim=1)
        return w * self.norm(x) + b