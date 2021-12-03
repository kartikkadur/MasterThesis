import torch
import os
import torch.nn as nn

from torch.autograd import Variable
from networks import VGG16
from networks import init_net

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

class VGGPerceptualLoss(nn.Module):
    def __init__(self, args):
        super(VGGPerceptualLoss, self).__init__()
        self.args = args
        if 'mse' in self.args.vgg_loss:
            self.loss = torch.nn.MSELoss()
        elif 'l1' in self.args.vgg_loss:
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplemented(f"L1 loss and MSE loss are the supported loss types. Got {self.args.vgg_loss}")
        self.model = init_net(VGG16(self.args.vgg_layers), None, gpu_ids=self.args.gpu_ids)
        if self.args.vgg_checkpoint is not None:
            self.model.load_state_dict(torch.load(self.args.vgg_checkpoint))
        else:
            raise NotImplemented("To compute perceptual loss vgg checkpoint has to be provided")
        
        self.model.eval()
        for param in self.model.parameters():
                param.requires_grad = False
        self.instancenorm = torch.nn.InstanceNorm2d(512, affine=False)

    def preprocess(self, batch, vgg_mean=False):
        tensortype = type(batch.data)
        (r, g, b) = torch.chunk(batch, 3, dim = 1)
        batch = torch.cat((b, g, r), dim = 1)
        batch = (batch + 1) * 255 * 0.5
        if vgg_mean:
            mean = tensortype(batch.data.size())
            mean[:, 0, :, :] = 103.939
            mean[:, 1, :, :] = 116.779
            mean[:, 2, :, :] = 123.680
            batch = batch.sub(Variable(mean))
        return batch

    def forward(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)
        x = self.model(x)
        y = self.model(y)
        loss = 0.0
        for o1, o2 in zip(x, y):
            if self.args.no_vgg_instance:
                loss += self.loss(o1, o2)
            else:
                loss += self.loss(self.instancenorm(o1), self.instancenorm(o2))
        return loss