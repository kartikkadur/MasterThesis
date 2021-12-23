import torch
import os
import functools
import numpy as np

from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
from inspect import isclass

def get_norm_layer(norm_layer ='instance'):
    """Return a normalization layer class if found or None"""
    if isinstance(norm_layer, str):
        if norm_layer == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_layer == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_layer == 'layer':
            norm_layer = functools.partial(LayerNorm)
        elif norm_layer == 'adain':
            norm_layer = functools.partial(AdaIN)
        elif norm_layer == 'sn':
            norm_layer = functools.partial(SpectralNorm)
        else:
            raise NotImplementedError(f"norm type '{norm_layer}' is not supported at the moment")
    elif not (norm_layer == None) and isclass(norm_layer) and not issubclass(norm_layer, nn.Module):
        raise ValueError(f"parameter type of norm_layer should be one of 'str' or 'nn.Module' class, but got {norm_layer}.")
    return norm_layer

def get_activation_layer(activation=None):
    """returns the activation layer class or None"""
    if isinstance(activation, str):
        if activation == 'relu':
            activation = nn.ReLU(inplace=False)
        elif activation == 'leaky_relu':
            activation = nn.LeakyReLU(0.02, inplace=False)
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            raise NotImplementedError(f"activation type '{activation}' is not supported at the moment")
    elif not (activation == None) and isclass(activation) and not issubclass(activation, nn.Module):
        raise ValueError(f"parameter type of activation should be one of 'str' or 'nn.Module', but got {type(activation)}.")
    if activation is not None:
        activation = activation if isinstance(activation, nn.Module) else activation()
    return activation

def get_padding_layer(padding_type=None):
    """returns the padding layer class"""
    if isinstance(padding_type, str):
        # set normal padding parameter to 0
        if padding_type == 'reflect':
            padding_layer = functools.partial(nn.ReflectionPad2d)
        elif padding_type == 'replicate':
            padding_layer = functools.partial(nn.ReplicationPad2d)
        else:
            raise NotImplementedError(f"padding type '{padding_type}' is not supported at the moment")
    elif not (padding_type == None) and isclass(padding_type) and not issubclass(padding_type, nn.Module):
        raise ValueError(f"parameter type of padding_type should be one of 'str' or 'nn.Module', but got {type(padding_type)}.")
    return padding_layer

def get_scheduler(optimizer, args, cur_ep=-1):
    if args.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - args.n_epoch_decay) / float(args.n_epoch - args.n_epoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        raise NotImplementedError(f'Learning rate policy {args.lr_policy} is not implemented')
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1 and classname.find('Conv') == 0:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, device='cpu', gpu_ids=[]):
    """Initialize a network"""
    if 'cuda' in device and len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    else:
        net = net.to(device)
    if init_type:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                            torch.nn.ConvTranspose2d,
                            torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    return torch.nn.utils.spectral_norm(module, name, n_power_iterations, eps, dim)

def conv_block(in_nc, out_nc, kernel_size, stride=1, padding=0, output_padding=0, bias=False,
                norm_layer=None, activation=None, padding_type=None, upsample=False, sn=False):
    block = []
    # add padding layer if passed
    if padding_type:
        padding_layer = get_padding_layer(padding_type)
        block += [padding_layer(padding)]
        padding = 0
    # add conv/deconv layers
    if upsample:
        conv_layer = nn.ConvTranspose2d(in_nc, out_nc, kernel_size, stride, padding, output_padding, bias=bias)
    else:
        conv_layer = nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding, bias=bias)
    if sn:
        conv_layer = spectral_norm(conv_layer)
    block += [conv_layer]
    # add norm layer
    if norm_layer:
        norm_layer = get_norm_layer(norm_layer)
        block += [norm_layer(out_nc)]
    # add activation
    if activation:
        activation = get_activation_layer(activation)
        block += [activation]
    return block

def layernorm_upsample_block(in_ch, out_ch, kernel_size=3, stride=1, padding=0, output_padding=0, layer='conv_transpose'):
        layers = []
        if 'conv_transpose' in layer:
            layers += [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, 
                                                padding=padding, output_padding=output_padding, bias=True)]
        else:
            layers += [nn.Upsample(scale_factor=2),
                       nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1, bias=True)]
        layers += [LayerNorm(out_ch)]
        layers += [nn.ReLU(inplace=True)]
        return layers

####################
### Basic Blocks ###
####################

class AdaIN(nn.Module):
    def __init__(self, latent_dim, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(latent_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

class ConvBlock(nn.Module):
    """Convolution block containing conv, norm layer and activation layer"""
    def __init__(self, input_nc ,output_nc, kernel_size, stride=1, padding=0, output_padding=0, bias=False,
                       norm_layer=None, activation=None, padding_type=None, upsample=False, sn=False):
        super(ConvBlock, self).__init__()
        layers = conv_block(input_nc ,output_nc, kernel_size, stride, padding, output_padding, bias,
                            norm_layer, activation, padding_type, upsample, sn)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class BasicBlock(nn.Module):
    """Basic resnet type block with short cut convolution connection"""
    def __init__(self, input_nc, output_nc, norm_layer='instance', activation='relu'):
        super(BasicBlock, self).__init__()
        layers = []
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        if isinstance(activation, str):
            activation = get_activation_layer(activation)
        if norm_layer is not None:
            layers += [norm_layer(input_nc)]
        layers += [activation]
        layers += conv_block(input_nc, input_nc, 3, 1, padding=1, padding_type='reflect', bias=True)
        if norm_layer is not None:
            layers += [norm_layer(input_nc)]
        layers += [activation]
        layers += conv_block(input_nc, output_nc, 3, 1, padding=1, padding_type='reflect', bias=True)
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(*[nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0, bias=True)])

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class ResnetBlock(nn.Module):
    """resnet block"""
    def __init__(self, input_nc, output_nc, dropout=0.0, norm_layer='instance', padding_type=None, activation='relu'):
        super(ResnetBlock, self).__init__()
        model = []
        model += conv_block(input_nc, output_nc, 3, 1, 1, padding_type=padding_type, norm_layer=norm_layer, activation=activation)
        model += conv_block(output_nc, output_nc, 3, 1, 1, padding_type=padding_type, norm_layer=norm_layer)
        if dropout > 0:
            model += [nn.Dropout(dropout)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResnetBlock2(nn.Module):
    def __init__(self, n_channel, add_channel, stride=1, dropout=0.0):
        super(ResnetBlock2, self).__init__()
        self.conv1 = ConvBlock(n_channel, n_channel, 3, stride=stride, padding=1, padding_type='reflect', norm_layer='instance')
        self.conv2 = ConvBlock(n_channel, n_channel, 3, stride=stride, padding=1, padding_type='reflect', norm_layer='instance')
        block1 = conv_block(n_channel + add_channel, n_channel + add_channel, 1, stride=stride, padding=0, activation='relu')
        block1 += conv_block(n_channel + add_channel, n_channel, 1, stride=1, padding=0, activation='relu')
        self.block1 = nn.Sequential(*block1)
        block2 = conv_block(n_channel + add_channel, n_channel + add_channel, 1, stride=1, padding=0, activation='relu')
        block2 += conv_block(n_channel + add_channel, n_channel, 1, stride=1, padding=0, activation='relu')
        self.block2 = nn.Sequential(*block2)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        out = self.conv1(x)
        out = self.block1(torch.cat([out, z_expand], dim=1))
        out = self.conv2(out)
        out = self.block2(torch.cat([out, z_expand], dim=1))
        out += residual
        return out

class AdaINResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, latent_dim=64, activation='leaky_relu', upsample=False):
        super(AdaINResnetBlock, self).__init__()
        self.activation = get_activation_layer(activation)
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.model = []
        self.model += conv_block(dim_in, dim_out, 3, 1, 1, activation=activation, norm_layer='adain')
        self.model += conv_block(dim_out, dim_out, 3, 1, 1, activation=activation, norm_layer='adain')
        if self.learned_sc:
            self.conv1x1 += nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.model = nn.Sequential(*self.model)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.model(x)

    def forward(self, x, s):
        out = (self._residual(x, s) + self._shortcut(x)) / np.sqrt(2)
        return out

class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if not self.training:
            return x
        noise = Variable(torch.randn(x.size()).to(x.get_device()))
        return x + noise

############################
### Normalization layers ###
############################

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
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

class AdaptiveInstanceNorm2d(nn.Module):
    """applies adaptive instance normalization to input image"""
    def __init__(self) -> None:
        super(AdaptiveInstanceNorm2d, self).__init__()

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content, style):
        assert (content.size()[:2] == style.size()[:2])
        size = content.size()
        style_mean, style_std = self.calc_mean_std(style)
        content_mean, content_std = self.calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

################
### Networks ###
################

class ContentEncoder(nn.Module):
    """Encoder model for encoding image content to a latent representation"""
    def __init__(self, input_nc, ngf=64, num_downs=2, n_blocks=4, norm_layer='instance', padding_type='reflect'):
        super(ContentEncoder, self).__init__()
        layers = []
        layers += [ConvBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3, padding_type=padding_type, activation='leaky_relu', bias=True)]
        for i in range(num_downs):
            layers += [ConvBlock(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, padding_type=padding_type, norm_layer=norm_layer, activation='relu', bias=True)]
            ngf *= 2
        for i in range(n_blocks):
            layers += [ResnetBlock(ngf, ngf)]
        layers += [GaussianNoiseLayer()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AttributeEncoder(nn.Module):
    """Encoder model for encoding image attributes to a latent representation"""
    def __init__(self, input_nc, output_nc=8, ngf=64, num_downs=4, num_domains=2, padding_type='reflect'):
        super(AttributeEncoder, self).__init__()
        layers = conv_block(input_nc+num_domains, ngf, 7, 1, padding=3, padding_type=padding_type, activation='relu')
        max_mult=4
        for n in range(num_downs):
            input_ngf = ngf * min(max_mult, 2**n)
            output_ngf = ngf * min(max_mult, 2**(n+1))
            layers += conv_block(input_ngf, output_ngf, 4, 2, padding=1, padding_type=padding_type, activation='relu')
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Conv2d(output_ngf, output_nc, 1, 1, 0)]
        self.model = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)
        output = self.model(x_c)
        return output.view(output.size(0), -1)

class AttributeEncoderConcat(nn.Module):
    def __init__(self, input_nc, output_nc=8, ndf=64, n_blocks=4, num_domains=2, norm_layer=None, activation=None):
        super(AttributeEncoderConcat, self).__init__()
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        if isinstance(activation, str):
            activation = get_activation_layer(activation)
        max_ndf_mult = 4
        layers = []
        layers += conv_block(input_nc+num_domains, ndf, kernel_size=4, stride=2, padding=1, padding_type='reflect', bias=True)
        for n in range(1, n_blocks+1):
            input_ndf = ndf * min(max_ndf_mult, n)
            output_ndf = ndf * min(max_ndf_mult, n+1)
            layers += [BasicBlock(input_ndf, output_ndf, norm_layer, activation)]
        layers += [activation, nn.AdaptiveAvgPool2d(1)]
        self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)
        x_conv = self.conv(x_c)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        outputVar = self.fcVar(conv_flat)
        return output, outputVar

class Generator(nn.Module):
    def __init__(self, output_nc, ngf=256, num_domains=2, latent_dim=8, upsample_layer='conv_transpose'):
        super(Generator, self).__init__()
        init_nch = ngf
        nch_add = init_nch
        nch = init_nch
        self.nch_add = nch_add
        self.dec1 = ResnetBlock2(nch, nch_add)
        self.dec2 = ResnetBlock2(nch, nch_add)
        self.dec3 = ResnetBlock2(nch, nch_add)
        self.dec4 = ResnetBlock2(nch, nch_add)
        dec5 = []
        dec5 += layernorm_upsample_block(nch, nch//2, kernel_size=3, stride=2, padding=1, output_padding=1, layer=upsample_layer)
        nch = nch//2
        dec5 += layernorm_upsample_block(nch, nch//2, kernel_size=3, stride=2, padding=1, output_padding=1, layer=upsample_layer)
        nch = nch//2
        if 'conv_transpose' in upsample_layer:
            dec5 += [nn.ConvTranspose2d(nch, output_nc, kernel_size=1, stride=1, padding=0)]
        else:
            dec5 += [nn.Upsample(scale_factor=1)]
            dec5 += [nn.Conv2d(nch, output_nc, kernel_size=1, stride=1, padding=0)]
        dec5 += [nn.Tanh()]
        self.dec5 = nn.Sequential(*dec5)

        self.linear = nn.Sequential(
                        nn.Linear(latent_dim + num_domains, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, nch_add*4))

    def forward(self, x, z, c):
        z_c = torch.cat([c, z], 1)
        z_c = self.linear(z_c)
        z1, z2, z3, z4 = torch.split(z_c, self.nch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out = self.dec1(x, z1)
        out = self.dec2(out, z2)
        out = self.dec3(out, z3)
        out = self.dec4(out, z4)
        out = self.dec5(out)
        return out

class GeneratorConcat(nn.Module):
    def __init__(self, output_nc, ngf=256, n_blocks=3, num_domains=2, latent_dim=8, upsample_layer='conv_transpose'):
        super(GeneratorConcat, self).__init__()
        dec_share = []
        dec_share += [ResnetBlock(ngf, ngf)]
        self.dec_share = nn.Sequential(*dec_share)
        nch = 256 + latent_dim + num_domains
        dec1 = []
        for i in range(n_blocks):
            dec1 += [ResnetBlock(nch, nch)]
        self.dec1 = nn.Sequential(*dec1)
        nch = nch + latent_dim
        dec2 = layernorm_upsample_block(nch, nch//2, kernel_size=3, stride=2, padding=1, output_padding=1, layer=upsample_layer)
        self.dec2 = nn.Sequential(*dec2)
        nch = nch//2
        nch = nch + latent_dim
        dec3 = layernorm_upsample_block(nch, nch//2, kernel_size=3, stride=2, padding=1, output_padding=1, layer=upsample_layer)
        self.dec3 = nn.Sequential(*dec3)
        nch = nch//2
        nch = nch + latent_dim
        if 'conv_transpose' in upsample_layer:
            dec4 = [nn.ConvTranspose2d(nch, output_nc, kernel_size=1, stride=1, padding=0)]
        else:
            dec4 = [nn.Upsample(scale_factor=1),
                    nn.Conv2d(nch, output_nc, kernel_size=1, stride=1, padding=0)]
        dec4 += [nn.Tanh()]
        self.dec4 = nn.Sequential(*dec4)

    def forward(self, x, z, c):
        out = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, out.size(2), out.size(3))
        x_c_z = torch.cat([out, c, z_img], 1)
        out = self.dec1(x_c_z)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out.size(2), out.size(3))
        x_z = torch.cat([out, z_img], 1)
        out = self.dec2(x_z)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out.size(2), out.size(3))
        x_z = torch.cat([out, z_img], 1)
        out = self.dec3(x_z)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out.size(2), out.size(3))
        x_z = torch.cat([out, z_img], 1)
        out = self.dec4(x_z)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=None, sn=False, num_domains=3, image_size=216):
        super(Discriminator, self).__init__()
        layers = []
        layers += [ConvBlock(input_nc, ndf, kernel_size=3, stride=2, padding=1, padding_type='reflect', norm_layer=norm_layer, sn=sn, activation='leaky_relu')]
        nch = ndf
        for i in range(n_layers-2):
            layers += [ConvBlock(nch, nch * 2, kernel_size=3, stride=2, padding=1, padding_type='reflect', norm_layer=norm_layer, sn=sn, activation='leaky_relu')]
            nch *= 2
        layers += [ConvBlock(nch, nch, kernel_size=3, stride=2, padding=1, padding_type='reflect', sn=sn, activation='leaky_relu')]
        self.model = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(nch, 1, kernel_size=1, stride=1, padding=1, bias=False)
        kernal_size = int(image_size/np.power(2, n_layers))
        self.conv2 = nn.Conv2d(nch, num_domains, kernel_size=kernal_size, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        h = self.model(x)
        out = self.conv1(h)
        out_cls = self.conv2(h)
        out_cls = self.pool(out_cls)
        return out, out_cls.view(out_cls.size(0), out_cls.size(1))

class ContentDiscriminator(nn.Module):
    def __init__(self, num_domains=3, ndf=256):
        super(ContentDiscriminator, self).__init__()
        layers = []
        for i in range(3):
            layers += [ConvBlock(ndf, ndf, kernel_size=7, stride=2, padding=1, padding_type='reflect', norm_layer='instance', activation='leaky_relu')]
        layers += [ConvBlock(ndf, ndf, kernel_size=4, stride=1, padding=0, padding_type='reflect', activation='leaky_relu')]
        layers += [nn.Conv2d(ndf, num_domains, kernel_size=1, stride=1, padding=0)]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        out = self.pool(out)
        out = out.view(out.size(0), out.size(1))
        return out

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=None, activation='leaky_relu',
                                        padding_type=None, num_domains=2, num_scales=3, sn=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.model = []
        self.model += [ConvBlock(input_nc, ndf, 4, 2, 1, norm_layer=None, activation=activation, padding_type=padding_type, sn=sn)]
        for i in range(n_layers - 1):
            self.model += [ConvBlock(ndf, ndf * 2, 4, 2, 1, norm_layer=norm_layer, activation=activation, padding_type=padding_type, sn=sn)]
            ndf *= 2
        self.model = nn.Sequential(*self.model)
        self.dis = nn.Conv2d(ndf, 1, 1, 1, 0)
        self.cls = nn.Conv2d(ndf, num_domains, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        outputs = []
        for i in range(self.num_scales):
            h = self.model(x)
            dis = self.dis(h)
            c = self.pool(self.cls(h))
            outputs.append((dis, c.view(c.size(0), c.size(1))))
            x = self.downsample(x)
        return outputs

############################
#### Network generators ####
############################

class ResnetGenerator(nn.Module):
    """Returns resnet varient model"""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer='instance', dropout=0.0, n_blocks=6, num_downs=2, padding_type=None, sn=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_layer)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # initial conv block
        layers = conv_block(input_nc, ngf, kernel_size=7, padding=3, padding_type=padding_type,
                                bias=use_bias, norm_layer=norm_layer, activation='relu', sn=sn)
        # num downsampling layers
        for i in range(num_downs):
            input_ngf = ngf * (2**i)
            output_ngf = ngf * (2**(i+1))
            layers += conv_block(input_ngf, output_ngf, kernel_size=3, stride=2, padding=1, padding_type=padding_type,
                                                            bias=use_bias, norm_layer=norm_layer, activation='relu', sn=sn)
        # resnet blocks
        for i in range(n_blocks):
            layers += [ResnetBlock(output_ngf, output_ngf, dropout=dropout)]
        # upsample blocks
        for i in range(num_downs):
            layers += conv_block(output_ngf, output_ngf // 2, kernel_size=3, stride=2, padding=1, padding_type=padding_type, output_padding=1,
                                        bias=use_bias, norm_layer=norm_layer, activation='relu', upsample=True, sn=sn)
            output_ngf = output_ngf // 2
        # final layers
        layers += conv_block(ngf, output_nc, kernel_size=7, padding=3, padding_type=padding_type, sn=sn)
        layers += [nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MultiDomainResnetGenerator(ResnetGenerator):
    """resnet generator for multi domain output"""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer='instance', 
                    dropout=0, n_blocks=6, num_downs=2, padding_type=None, sn=False, num_domains=2):
        super().__init__(input_nc+num_domains, output_nc, ngf=ngf, norm_layer=norm_layer, dropout=dropout, 
                            n_blocks=n_blocks, num_downs=num_downs, padding_type=padding_type, sn=sn)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat((x,c), dim=1)
        return self.model(x_c)

class UnetDownBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False, dropout=False, norm_layer=None, activation=None):
        super(UnetDownBlock, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm_layer = get_norm_layer(norm_layer)
        self.activation = get_activation_layer(activation)
        if self.norm_layer:
            self.norm_layer = self.norm_layer(output_nc)
        if self.activation:
            self.activation = self.activation
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_layer:
            x = self.norm_layer(x)
        if self.dropout:
            x = self.dropout(x)
        if self.activation:
            x = self.activation(x)
        return x

class UnetUpBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False, dropout=False, norm_layer=None, activation=None):
        super(UnetUpBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm_layer = get_norm_layer(norm_layer)
        self.activation = get_activation_layer(activation)
        if self.norm_layer:
            self.norm_layer = self.norm_layer(output_nc)
        if self.activation:
            self.activation = self.activation
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x, residual):
        x = self.deconv(x)
        if self.norm_layer:
            x = self.norm_layer(x)
        if self.dropout:
            x = self.dropout(x)
        if self.activation:
            x = self.activation(x)
        x = torch.cat((x, residual), dim=1)
        return x

class UnetGenerator(nn.Module):
    """Define a U-Net Generator"""
    def __init__(self, input_nc, output_nc, ngf=64, max_ngf=512, norm_layer='instance', use_dropout=False, input_size=256, num_domains=2):
        super(UnetGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_layer)
        # bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # calculate num downs based on input size
        num_downs = int(np.log2(input_size))
        # define encoder and decoder blocks       
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # create partial functions
        Down = functools.partial(UnetDownBlock, kernel_size=4, stride=2, padding=1, bias=use_bias, activation='leaky_relu')
        Up = functools.partial(UnetUpBlock, kernel_size=4, stride=2, padding=1, bias=use_bias, activation='relu')
        self.encoder.append(Down(input_nc+num_domains, ngf, norm_layer=None))
        # num channels
        nch = ngf
        max_iter = int(np.log2(max_ngf // ngf))
        for i in range(max_iter):
            self.encoder.append(Down(nch, nch*2, norm_layer=norm_layer))
            self.decoder.insert(0, Up(nch*4, nch, norm_layer=norm_layer))
            nch *= 2

        for i in range(num_downs-max_iter-1):
            if i+1 == (num_downs-max_iter-1):
                self.encoder.append(Down(nch, nch, norm_layer=None))
                self.decoder.insert(0, Up(nch, nch, norm_layer=norm_layer))
            else:
                self.encoder.append(Down(nch, nch, norm_layer=norm_layer))
                self.decoder.insert(0, Up(nch*2, nch, norm_layer=norm_layer))
        self.final_conv = nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, c):
        # replicate and concatinate c to input x
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        out = torch.cat([x, c], dim=1).type(torch.cuda.FloatTensor)
        enc_out = []
        for i, layer in enumerate(self.encoder):
            out = layer(out)
            enc_out.insert(0, out)
        for i, layer in enumerate(self.decoder):
            if i == 0:
                out = layer(enc_out[i], enc_out[i+1])
            else:
                out = layer(out, enc_out[i+1])
        out = self.final_conv(out)
        return self.tanh(out)

class NLayerDiscriminator(nn.Module):
    """defines a n-layer patch gan discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer='instance', use_sigmoid=False, sn=False) -> None:
        super(NLayerDiscriminator, self).__init__()
        max_mult=4
        layers = conv_block(input_nc, ndf, kernel_size=4, stride=2, padding=1, activation='leaky_relu', sn=sn)
        for n in range(0, n_layers-1):
            input_nch = ndf * min(2**(n), max_mult)
            self.output_nch = ndf * min(2**(n+1), max_mult)
            layers += conv_block(input_nch, self.output_nch, kernel_size=4, stride=2, padding=1, norm_layer=norm_layer, activation='leaky_relu', sn=sn)
        #layers += conv_block(self.output_nch, self.output_nch, kernel_size=4, padding=1, norm_layer=norm_layer, activation='leaky_relu', sn=sn) 
        self.model = nn.Sequential(*layers)
        # discriminator output
        layers = conv_block(self.output_nch, 1, kernel_size=4, stride=1, padding=1, sn=sn)
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.dis_out = nn.Sequential(*layers)

    def forward(self, x):
        h = self.model(x)
        return self.dis_out(h)

class NoNormDiscriminator(NLayerDiscriminator):
    def __init__(self, input_nc, ndf=64, n_layers=6, num_domains=2, use_sigmoid=False, image_size=256, sn=False):
        super(NoNormDiscriminator, self).__init__(input_nc, ndf, n_layers, norm_layer=None, use_sigmoid=use_sigmoid, sn=sn)
        kernel_size = int(image_size / np.power(2, n_layers))
        self.cls_out = nn.Sequential(*conv_block(self.output_nch, num_domains, kernel_size=kernel_size, bias=False, sn=sn))

    def forward(self, x):
        h = self.model(x)
        d_out = self.dis_out(h)
        c_out = self.cls_out(h)
        return d_out, c_out.view(c_out.size(0), c_out.size(1))

class UpsampleAttn(nn.Module):
    def __init__(self, input_nc, ngf=64, n_blocks=3, num_downs=2, norm_layer='instance', padding_type=None):
        super(UpsampleAttn, self).__init__()
        layers =  conv_block(input_nc, ngf, 7, stride=1, padding=3, padding_type=padding_type, norm_layer=norm_layer, activation='relu')
        for i in range(num_downs):
            input_ngf = ngf * 2**i
            output_ngf = ngf * 2**(i+1)
            layers += conv_block(input_ngf, output_ngf, 3, stride=2, padding=1, padding_type=padding_type, norm_layer=norm_layer, activation='relu')
        for i in range(n_blocks):
            layers += [ResnetBlock(output_ngf, output_ngf)]
        for i in range(num_downs):
            layers += [nn.UpsamplingNearest2d(scale_factor=2)]
            layers += conv_block(output_ngf, output_ngf//2, 3, stride=1, padding=1, padding_type=padding_type, norm_layer=norm_layer, activation='relu')
            output_ngf = output_ngf // 2
        layers += conv_block(output_ngf, 1, 7, stride=1, padding=3, padding_type=padding_type, activation='sigmoid')
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)