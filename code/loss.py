import torch
import os
import torch.nn as nn

from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import vgg as vgg
from networks import init_net

NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}

class GANLoss(nn.Module):
    def __init__(self, loss='vanilla'):
        super(GANLoss, self).__init__()
        self.register_buffer('ones', torch.tensor(1.0))
        self.register_buffer('zeros', torch.tensor(0.0))
        if loss == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif loss == 'bce':
            self.loss = nn.BCELoss()
        elif loss == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'Loss {loss} is not implemented')

    def forward(self, inp, trg_is_real):
        if trg_is_real:
            trg = self.ones.expand_as(inp).to(inp.get_device())
        else:
            trg = self.zeros.expand_as(inp).to(inp.get_device())
        return self.loss(inp, trg)

class VGGFeatureExtractor(nn.Module):
    """define pretrained vgg19 network for perceptual loss"""
    def __init__(self, feature_layers, vgg_type='vgg19', requires_grad=False):
        super(VGGFeatureExtractor, self).__init__()
        vgg_net = vgg_net = getattr(vgg, vgg_type)(pretrained=True).features
        self.names = NAMES[vgg_type.replace('_bn', '')]
        self.feature_layers = feature_layers
        # max index of layers to be considered in the model
        max_idx = 0
        for v in feature_layers:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx
        self.vgg_net = vgg_net.features[:max_idx + 1]
        # set requires_grad to False
        for param in self.parameters():
            param.requires_grad = requires_grad
        # the mean is for image with range [0, 1]
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        outputs = []
        x = (x - self.mean) / self.std
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.feature_layers:
                output[key] = x.clone()
        return outputs

class VGGPerceptualLoss(nn.Module):
    def __init__(self, args, layer_weight=[1.0]):
        super(VGGPerceptualLoss, self).__init__()
        self.args = args
        self.layer_weights = layer_weight
        if 'mse' in self.args.vgg_loss:
            self.criterion = torch.nn.MSELoss()
        elif 'l1' in self.args.vgg_loss:
            self.criterion = torch.nn.L1Loss()
        else:
            raise NotImplemented(f"L1 loss and MSE loss are the supported loss types. Got {self.args.vgg_loss}")
        self.model = init_net(VGGFeatureExtractor(args.vgg_layers, args.vgg_type, requires_grad=False),
                              None,
                              gpu_ids=args.gpu_ids)
        self.instancenorm = torch.nn.InstanceNorm2d(512, affine=False)

    def forward(self, x, y):
        x = self.model(x)
        y = self.model(y)
        percep_loss = 0.0
        for k in x.keys():
            percep_loss += self.criterion(x[k], y[k]) * self.layer_weights[k]
        return percep_loss