import torch
import functools
import torch.nn as nn
import numpy as np

from inspect import isclass
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.models.vgg import VGG, cfgs, make_layers, model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


########################
### Helper functions ###
########################

def get_norm_layer(norm_layer ='instance'):
    """Return a normalization layer
    Parameters:
        norm_layer (str) : the name of the normalization layer: batch | instance | none
    """
    if isinstance(norm_layer, str):
        if norm_layer == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_layer == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_layer == 'layer':
            norm_layer == functools.partial(nn.LayerNorm)
        else:
            raise NotImplementedError(f"norm type '{norm_layer}' is not supported at the moment")
    elif not (norm_layer == None) and isclass(norm_layer) and not issubclass(norm_layer, nn.Module):
        raise ValueError(f"parameter type of norm_layer should be one of 'str' or 'nn.Module' class, but got {norm_layer}.")
    return norm_layer

def get_activation_layer(activation=None):
    """
    returns the action layer object or None (by default)
    Parameter:
        activation: type of activation as str or nn.Module class
    """
    if isinstance(activation, str):
        if activation == 'relu':
            activation = nn.ReLU(inplace=False)
        elif activation == 'leaky_relu':
            activation = nn.LeakyReLU(0.02, inplace=False)
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'sigmoid':
            activation == nn.Sigmoid()
        else:
            raise NotImplementedError(f"activation type '{activation}' is not supported at the moment")
    elif not (activation == None) and isclass(activation) and not issubclass(activation, nn.Module):
        raise ValueError(f"parameter type of activation should be one of 'str' or 'nn.Module', but got {type(activation)}.")
    activation = activation if isinstance(activation, nn.Module) else activation()
    return activation

def get_padding_layer(padding_type=None):
    """
    return the padding layer object
    Parameters:
        padding_type (str/nn.Module) : the type of padding. eaither str : 'reflect'/'replicate' or padding layer object
    """
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

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          : the optimizer of the network
        args (Agumrnt class) : stores all the experiment flags; needs to be a subclass of Arguments.
                              args.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - args.epoch_decay) / float(args.n_epochs - args.epoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   : network to be initialized
        init_type (str) : the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    : scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
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
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      : the network to be initialized
        init_type (str)    : the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       : scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) : which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args):
    """
    Create a generator
    """
    net = None
    netG = args.netG.__name__

    if netG == 'ResnetGenerator':
        net = args.netG(args.input_nc,
                            args.output_nc,
                            args.ngf,
                            norm_layer=args.norm_layer,
                            use_dropout=not args.no_dropout,
                            n_blocks=args.n_blocks_G,
                            attention=args.attention,
                            feature=args.feature)

    elif netG == 'UnetGenerator':
        net = args.netG(args.input_nc,
                            args.output_nc,
                            ngf=args.ngf,
                            norm_layer=args.norm_layer,
                            use_dropout=not args.no_dropout)
    else:
        raise NotImplementedError('Generator model [%s] is not recognized' % netG)
    return init_net(net, args.init_type, args.init_gain, args.gpu_ids)


def define_D(args):
    """
    Create a discriminator
    """
    net = None
    netD = args.netD.__name__
    if netD == 'NLayerDiscriminator':
        net = args.netD(args.input_nc, args.ndf, n_layers=args.n_layers_D, norm_layer=args.norm_layer)
    elif netD == 'MultiClassNLayerDiscriminator':
        if args.num_classes != -1:
            net = args.netD(args.num_classes, args.input_nc, args.ndf, args.n_layers_D, norm_layer=args.norm_layer)
        else:
            raise ValueError(f'provide a value for num_classes argument. Current value is {args.num_classes}')
    elif netD == 'PixelDiscriminator':     # classify if each pixel is real or fake
        net = args.netD(args.input_nc, args.ndf, norm_layer=args.norm_layer)
    elif netD == 'AttNLayerDiscriminator':
        net = args.netD(args.input_nc, args.ndf, norm_layer=args.norm_layer, n_domain=args.num_domains, input_size=args.crop_size)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, args.init_type, args.init_gain, args.gpu_ids)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              : discriminator network
        real_data (tensor array)    : real images
        fake_data (tensor array)    : generated images from the generator
        device (str)                : GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  : if we mix real and fake data or not [real | fake | mixed].
        constant (float)            : the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           : weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def init_gaussian(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        module.weight.data.normal_(0.0, 0.02)

####################
### Basic Blocks ###
####################

class Identity(nn.Module):
    def forward(self, x):
        return x

class Conv2dBlock(nn.Module):
    """
    Convolution block containing conv, norm layer and activation layer
    """
    def __init__(self, input_nc:int,
                       output_nc:int,
                       kernel_size:int,
                       stride:int = 1,
                       padding:int = 0,
                       output_padding:int = 0,
                       bias:bool = False,
                       norm_layer = None,
                       activation = None,
                       padding_type = None,
                       upsample:bool = False,
                       spectral_norm:bool = False
                       ):
        super(Conv2dBlock, self).__init__()
        # empty model
        model = []
        # add padding layer if passed
        if padding_type:
            padding_layer = get_padding_layer(padding_type)
            model += [padding_layer(padding)]
            padding = 0
        # add conv/deconv layers
        if upsample:
            conv_layer = nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride, padding, output_padding, bias=bias)
        else:
            conv_layer = nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding, bias=bias)
        if spectral_norm:
            conv_layer = spectral_normalization(conv_layer)
        model += [conv_layer]
        # add norm layer
        if norm_layer:
            norm_layer = get_norm_layer(norm_layer)
            model += [norm_layer(output_nc)]
        # add activation
        if activation:
            activation = get_activation_layer(activation)
            model += [activation]
        # generate final block
        self.block = nn.Sequential(*model)

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self,
                in_channels:int,
                out_channels:int,
                downsample:nn.Module = None,
                norm_layer = None,
                dropout:bool = False,
                padding:int = 0,
                padding_type:str = None,
                activation:str = 'relu'
                ) -> None:
        """
        ----------- order of layer operations ------------
        |-- convolution -- normalization -- activation --|
        """
        super(ResnetBlock, self).__init__()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(0.5)
        if padding_type:
            self.padding = get_padding_layer(padding_type)(1)
            padding = 0
        else:
            self.padding = None
            padding = 1
        self.norm_layer = get_norm_layer(norm_layer)
        activation = get_activation_layer(activation)
        self.activation = activation
        self.downsample = downsample
        self.skip_conv = in_channels != out_channels
        if self.skip_conv:
            self.conv0 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding)
        if self.norm_layer:
            self.norm1 = self.norm_layer(in_channels)
            self.norm2 = self.norm_layer(out_channels)

    def _shortcut(self, x):
        if self.skip_conv:
            x = self.conv0(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _conv(self, x):
        if self.padding:
            x = self.padding(x)
        x = self.activation(self.norm1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        if self.padding:
            x = self.padding(x)
        x = self.activation(self.norm2(self.conv2(x)))
        return x

    def forward(self, x):
        out = self._conv(x) + self._shortcut(x)
        return out


class ResnetBlockV2(nn.Module):
    expansion: int = 1
    def __init__(self,
                in_channels:int,
                out_channels:int,
                downsample:bool = False,
                norm_layer:nn.Module = None,
                dropout:bool = False,
                padding:str = 'zero',
                activation:str = 'relu'
                ) -> None:
        """
        ----------- order of layer operations ------------
        |-- normalization -- activation -- convolution --|
        """
        super(ResnetBlockV2, self).__init__()
        # get layers
        self.norm_layer = get_norm_layer(norm_layer)
        if padding == 'zero':
            padding = 1
            self.padding = None
        else:
            padding = 0
            self.padding = get_padding_layer(padding)
        activation = get_activation_layer(activation)
        self.activation = activation
        self.skip_conv = in_channels != out_channels
        if self.skip_conv:
            self.conv0 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding)
        if self.norm_layer:
            self.norm1 = self.norm_layer(in_channels)
            self.norm2 = self.norm_layer(in_channels)
        self.downsample = downsample

    def _shortcut(self, x):
        if self.skip_conv:
            x = self.conv0(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _conv(self, x):
        if self.norm_layer:
            x = self.norm1(x)
        x = self.activation(x)
        if self.padding:
            x = self.padding(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.norm_layer:
            x = self.norm2(x)
        x = self.activation(x)
        if self.padding:
            x = self.padding(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        out = self._conv(x) + self._shortcut(x)
        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, dim:int, num_channels:int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        self.fc = nn.Linear(dim, num_channels*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class GaussianNoiseLayer(nn.Module):
    def __init__(self, isTrain:bool = False):
        super(GaussianNoiseLayer, self).__init__()
        self.isTrain = isTrain

    def forward(self, x):
        if self.isTrain == False:
            return x
        noise = Variable(torch.randn(x.size()).to(x.get_device()))
        return x + noise


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

####################
### U-Net blocks ###
####################
class UnetDownBlock(nn.Module):
    def __init__(self, input_nc:int,
                       output_nc:int,
                       kernel_size:int = 4,
                       stride:int = 2,
                       padding:int = 1,
                       bias:bool = False,
                       dropout:bool = False,
                       norm_layer:str = None,
                       activation:str = None):
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
    def __init__(self, input_nc:int,
                       output_nc:int,
                       kernel_size:int = 4,
                       stride:int = 2,
                       padding:int = 1,
                       bias:int = False,
                       dropout:bool = False,
                       norm_layer:str = None,
                       activation:str = None):
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

    def forward(self, x, c):
        x = self.deconv(x)
        if self.norm_layer:
            x = self.norm_layer(x)
        if self.dropout:
            x = self.dropout(x)
        if self.activation:
            x = self.activation(x)
        x = torch.cat((x,c), dim=1)
        return x


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc:int,
                       inner_nc:int,
                       input_nc:int = None,
                       submodule:nn.Module = None,
                       outermost:bool = False,
                       innermost:bool = False,
                       norm_layer:nn.Module = nn.BatchNorm2d,
                       use_dropout:bool = False) -> None:
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) : the number of filters in the outer conv layer
            inner_nc (int) : the number of filters in the inner conv layer
            input_nc (int) : the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) : previously defined submodules
            outermost (bool)    : if this module is the outermost module
            innermost (bool)    : if this module is the innermost module
            norm_layer          : normalization layer
            use_dropout (bool)  : if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

#############################
### Resnet block varients ###
#############################

class BasicBlock(nn.Module):
    def __init__(self, input_nc:int, output_nc:int, norm_layer:str = 'instance', activation:str = 'relu'):
        """
        Resnet basic block
        """
        super(BasicBlock, self).__init__()
        # get norm and activation layers
        norm_layer = get_norm_layer(norm_layer)
        activation = get_activation_layer(activation)
        # create block layers
        layers = []
        if norm_layer:
            layers += [norm_layer(input_nc)]
        if activation:
            layers += [activation]
        layers += [nn.ReflectionPad2d(1), 
                   nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=0, bias=True)]
        if norm_layer:
            layers += [norm_layer(input_nc)]
        if activation:
            layers += [activation]
        layers += [nn.ReflectionPad2d(1), 
                   nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=0, bias=True),
                   nn.AvgPool2d(kernel_size=2, stride=2)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(*[nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0, bias=True)])

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class AdaINResnetBlock(nn.Module):
    def __init__(self, in_channels:int,
                       out_channels:int,
                       style_dim:int = 64,
                       w_hpf:int = 0,
                       activation:nn.Module = nn.LeakyReLU(0.2),
                       upsample:bool = False):
        super().__init__()
        self.w_hpf = w_hpf
        self.activation = activation
        self.upsample = upsample
        self.skip_conv = in_channels != out_channels
        if self.skip_conv:
            self.conv0 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = AdaptiveInstanceNorm(style_dim, in_channels)
        self.norm2 = AdaptiveInstanceNorm(style_dim, out_channels)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.skip_conv:
            x = self.conv0(x)
        return x

    def _conv(self, x, s):
        x = self.norm1(x, s)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._conv(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x))
        return out


class DritResnetBlock(nn.Module):
    """resnet block that takes in a latent representation"""
    def __init__(self, input_nc, output_nc, stride=1, dropout=0.0, init=True):
        super(DritResnetBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=stride),
                                   nn.InstanceNorm2d(input_nc))
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=stride),
                                   nn.InstanceNorm2d(input_nc))
        block_ch = input_nc + output_nc
        self.block1 = nn.Sequential(nn.Conv2d(block_ch, block_ch, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(block_ch, input_nc, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=False)
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(block_ch, block_ch, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(block_ch, input_nc, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=False)
                                    )
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.block1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.block2(torch.cat([o3, z_expand], dim=1))
        out = out + residual
        return out

#####################
### Base networks ###
#####################

class ResnetGenerator(nn.Module):
    """
    Generator model for GAN
    """
    def __init__(self, input_nc:int,
                       output_nc:int,
                       ngf:int = 64,
                       max_ngf=512,
                       n_downsampling:int = 2,
                       n_blocks:int = 6,
                       num_domains:int = 2,
                       norm_layer:nn.Module = nn.InstanceNorm2d,
                       use_dropout:bool = False,
                       padding_type:str = 'reflect',
                       feature:bool = False) -> None:
        super(ResnetGenerator, self).__init__()
        assert(n_blocks >= 0)
        self.num_domains = num_domains
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # initial conv block
        self.init_conv_block = Conv2dBlock(input_nc + num_domains, ngf, kernel_size=7, padding=3, 
                                            bias=use_bias, norm_layer=norm_layer,
                                            activation='relu')
        # downsampling blocks
        nch = ngf
        downsample_blocks = []
        for i in range(n_downsampling):
            nch = min(max_ngf, nch)
            downsample_blocks += [Conv2dBlock(nch, nch * 2, kernel_size=4, stride=2, 
                                            padding=1, bias=use_bias, norm_layer=norm_layer, activation='relu')]
            nch *= 2
        self.downsample_blocks = nn.Sequential(*downsample_blocks)
        # ResNet blocks
        resnet_blocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet_blocks += [ResnetBlock(nch, nch, norm_layer=norm_layer, dropout=use_dropout, padding_type=padding_type)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # upsampling blocks
        upsample_blocks = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            upsample_blocks += [Conv2dBlock(nch, nch//2,
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias,
                                         norm_layer=norm_layer,
                                         activation='relu',
                                         upsample=True)]
            nch = nch//2
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        # final conv block
        self.final_conv_block = Conv2dBlock(nch, output_nc, kernel_size=7, padding=3, norm_layer=None, activation='tanh')

        # return feature if required
        self.feature = feature
        if feature:
            self.feature_block = nn.Sequential(*upsample_blocks[:1])

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        # encoder
        enc = self.init_conv_block(x)
        enc = self.downsample_blocks(enc)
        enc = self.resnet_blocks(enc)
        # decoder
        dec = self.upsample_blocks(enc)
        out = self.final_conv_block(dec)

        output = [out]
        if self.feature:
            fea = self.feature_block(enc)
            output += [fea]
        return output


class UnetGenerator(nn.Module):
    """Define a U-Net Generator"""
    def __init__(self, input_nc:int,
                       output_nc:int,
                       ngf:int = 64,
                       max_ngf=512,
                       norm_layer:nn.Module = nn.BatchNorm2d,
                       use_dropout:bool = False,
                       input_size:int = 256,
                       num_domains:int = 2,
                       feature:bool = False):
        super(UnetGenerator, self).__init__()
        # for calculating feature loss
        self.feature = feature
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
                if self.feature:
                    d1_out = out
            else:
                out = layer(out, enc_out[i+1])
        out = self.final_conv(out)
        if self.feature:
            return self.tanh(out), d1_out
        return self.tanh(out)


class PrograssiveUnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc:int,
                       output_nc:int,
                       num_downs:int,
                       ngf:int = 64,
                       norm_layer:nn.Module = nn.BatchNorm2d,
                       use_dropout:bool = False,
                       attention:bool = False) -> None:
        """
        Parameters:
            input_nc (int)  : the number of channels in input images
            output_nc (int) : the number of channels in output images
            num_downs (int) : the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       : the number of filters in the last conv layer
            norm_layer      : normalization layer
        """
        super(PrograssiveUnetGenerator, self).__init__()
        # attention
        self.attention = attention
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.translator_net = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        
        if self.attention:
            self.attention_net = UnetSkipConnectionBlock(1, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        """Standard forward"""
        if self.attention:
            att = self.attention_net(x)
            out = self.translator_net(x)
            return att * out + (1 - att) * x, att
        else:
            return self.translator_net(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc:int,
                       ndf:int = 64,
                       max_ndf:int = 512,
                       n_layers:int = 3,
                       norm_layer:nn.Module = nn.BatchNorm2d,
                       n_domain:int = 2,
                       input_size:int = 256):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (norm_layer == 'instance')
        # max downs possible
        max_downs = int(np.log2(input_size)) - 1
        n_layers = min(max_downs, n_layers)
        # model
        layers = [Conv2dBlock(input_nc, ndf, kernel_size=4, stride=2, padding=1, 
                                    norm_layer=None, activation='leaky_relu')]
        nch = ndf
        for i in range(n_layers-1):
            nch = min(max_ndf, nch)
            layers += [Conv2dBlock(nch, nch*2, kernel_size=4, stride=2, padding=1,
                          bias=use_bias, norm_layer=norm_layer, activation='leaky_relu')]
            nch *= 2
        layers += [Conv2dBlock(nch, nch, kernel_size=4, stride=1, padding=1,
                                    bias=use_bias, norm_layer=norm_layer, activation='leaky_relu')]
        self.conv_ops = nn.Sequential(*layers)
        # discriminator output predicting real/fake sample
        self.conv_dis = nn.Conv2d(nch, 1, kernel_size=4, stride=1, padding=1)
        # classification output
        self.conv_cls = nn.Sequential(nn.Conv2d(nch, n_domain, kernel_size=4, stride=1, padding=1),
                                      nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.conv_ops(x)
        # discriminator out
        d_out = self.conv_dis(x)
        # classification out
        c_out = self.conv_cls(x)
        return d_out, c_out.view(c_out.size(0), c_out.size(1))


class AttContentDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator with attention"""

    def __init__(self, input_nc:int,
                       ndf:int = 64,
                       max_ndf:int = 512,
                       norm_layer:nn.Module = nn.BatchNorm2d,
                       n_domain:int = 2,
                       input_size:int=256):
        super(AttContentDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (norm_layer == 'instance')
        # define num downs based on input image size so that you have a 8x8 patch for attention output
        num_downs = int(np.log2(input_size)) - 4
        # model
        layers = [Conv2dBlock(input_nc, ndf, kernel_size=4, stride=2, padding=1, 
                                       norm_layer=None, activation='leaky_relu')]
        nch = ndf
        for i in range(num_downs):
            nch = min(max_ndf, nch)
            layers += [Conv2dBlock(nch, nch*2, kernel_size=4, stride=2, padding=1,
                                 bias=use_bias, norm_layer=norm_layer, activation='leaky_relu')]
            nch *= 2
        self.conv_blocks = nn.Sequential(*layers)
        # upsample block to get attention map
        self.upsample = nn.Upsample(input_size, mode='bilinear', align_corners=True)
        self.conv2 = Conv2dBlock(nch, nch, kernel_size=4, stride=1, padding=1,
                                    bias=use_bias, norm_layer=norm_layer, activation='leaky_relu')
        self.conv3 = nn.Conv2d(nch, n_domain, kernel_size=4, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_blocks(x)
        # calculate attention maps
        att = torch.sum(x, dim=1).unsqueeze(1)
        att = self.upsample(att/torch.max(att))
        x = self.conv2(x)
        c_out = self.pool(self.conv3(x))
        return c_out.view(c_out.size(0), c_out.size(1)), att


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc:int, ndf:int = 64, norm_layer:nn.Module = nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  : the number of channels in input images
            ndf (int)       : the number of filters in the last conv layer
            norm_layer      : normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.model = [
            Conv2dBlock(input_nc, ndf, kernel_size=1, stride=1, padding=0, 
                      norm_layer=None, activation='leaky_relu'),
            Conv2dBlock(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias,
                      norm_layer=norm_layer, activation='leaky_relu'),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)

#####################
### DRIT networks ###
#####################

class ContentEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, num_domains=5, num_downs=3, n_blocks=3):
        super(ContentEncoder, self).__init__()
        layers = []
        activation = nn.LeakyReLU(inplace=True)
        layers += [Conv2dBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3, padding_type='reflect', activation=activation)]
        nch = ngf
        for i in range(1, num_downs):
            layers += [Conv2dBlock(nch, nch * 2, kernel_size=3, stride=2, padding=1, padding_type='reflect', activation='relu')]
            nch *= 2
        for i in range(0, n_blocks):
            layers += [ResnetBlock(nch, nch, norm_layer='instance', activation='relu', padding_type='reflect')]

        for i in range(0, 1):
            layers += [ResnetBlock(nch, nch, norm_layer='instance', activation='relu', padding_type='reflect')]
            layers += [GaussianNoiseLayer()]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class AttributeEncoder(nn.Module):
    def __init__(self, input_nc, output_nc=8, num_domains=3, ngf=64):
        super(AttributeEncoder, self).__init__()
        max_mult=4
        layers = [Conv2dBlock(input_nc+num_domains, ngf, 7, 1, padding=3, padding_type='reflect', activation='relu')]
        for i in range(5):
            input_ngf = ngf * (2**min(max_mult, i))
            output_ngf = ngf * (2**min(max_mult, i+1))
            layers += [Conv2dBlock(input_ngf, output_ngf, 4, 2, padding=1, padding_type='reflect', activation='relu')]
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Conv2d(output_ngf, output_nc, 1, 1, 0)]
        self.model = n.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)
        output = self.model(x_c)
        return output.view(output.size(0), -1)


class AttributeEncoderConcat(nn.Module):
    def __init__(self, input_nc, output_nc=8, ndf=64, n_blocks=4, num_domains=3, norm_layer=None, activation=None):
        super(AttributeEncoderConcat, self).__init__()
        max_ndf = 4
        activation = get_activation_layer(activation)
        layers = [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(input_nc + num_domains, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for i in range(n_blocks):
            input_ndf = ndf * (2**min(max_ndf, i))
            output_ndf = ndf * (2**min(max_ndf, i+1))
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


class DomainGeneratorConcat(nn.Module):
    def __init__(self, output_nc, ngf=256, num_domains=3, latent_dim=8):
        super(DomainGeneratorConcat, self).__init__()
        self.latent_dim = latent_dim
        self.num_domains = num_domains
        shared_decoder = []
        shared_decoder += [ResnetBlock(ngf, ngf, norm_layer='instance', activation='relu', padding_type='reflect')]
        self.shared_decoder = nn.Sequential(*shared_decoder)
        nch = 256 + self.latent_dim + self.num_domains
        decoder1 = []
        for i in range(0, 3):
            decoder1 += [ResnetBlock(nch, nch, norm_layer='instance', activation='relu', padding_type='reflect')]
        nch = nch + self.latent_dim
        norm_layer = LayerNorm
        decoder2 = [Conv2dBlock(nch, nch//2, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu', norm_layer=norm_layer, upsample=True)]
        nch = nch//2
        nch = nch + self.latent_dim
        decoder3 = [Conv2dBlock(nch, nch//2, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu', norm_layer=norm_layer, upsample=True)]
        nch = nch//2
        nch = nch + self.latent_dim
        decoder4 = [nn.ConvTranspose2d(nch, output_nc, kernel_size=1, stride=1, padding=0)]
        decoder4 += [nn.Tanh()]
        self.decoder1 = nn.Sequential(*decoder1)
        self.decoder2 = nn.Sequential(*decoder2)
        self.decoder3 = nn.Sequential(*decoder3)
        self.decoder4 = nn.Sequential(*decoder4)

    def forward(self, x, z, c):
        out0 = self.shared_decoder(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, out0.size(2), out0.size(3))
        x_c_z = torch.cat([out0, c, z_img], 1)
        out1 = self.decoder1(x_c_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.decoder2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.decoder3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.decoder4(x_and_z4)
        return out4


class DomainGenerator(nn.Module):
    def __init__(self, output_nc, ngf=256, num_domains=3, latent_dim=8):
        super(DomainGenerator, self).__init__()
        self.latent_dim = latent_dim
        nch = ngf
        nch_add = ngf
        self.nch_add = nch_add
        self.decoder1 = DritResnetBlock(nch, nch_add)
        self.decoder2 = DritResnetBlock(nch, nch_add)
        self.decoder3 = DritResnetBlock(nch, nch_add)
        self.decoder4 = DritResnetBlock(nch, nch_add)
        norm_layer = LayerNorm
        decoder5 = []
        decoder5 += [Conv2dBlock(nch, nch//2, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu', norm_layer=norm_layer, upsample=True)]
        nch = nch//2
        decoder5 += [Conv2dBlock(nch, nch//2, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu', norm_layer=norm_layer, upsample=True)]
        nch = nch//2
        decoder5 += [nn.ConvTranspose2d(nch, output_nc, kernel_size=1, stride=1, padding=0)]
        decoder5 += [nn.Tanh()]
        self.decoder5 = nn.Sequential(*decoder5)

        self.fc = nn.Sequential(
                        nn.Linear(latent_dim + num_domains, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, nch_add*4))

    def forward(self, x, z, c):
        z_c = torch.cat([c, z], 1)
        z_c = self.fc(z_c)
        z1, z2, z3, z4 = torch.split(z_c, self.nch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decoder1(x, z1)
        out2 = self.decoder2(out1, z2)
        out3 = self.decoder3(out2, z3)
        out4 = self.decoder4(out3, z4)
        out = self.decoder5(out4)
        return out


class DomainDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=None, num_domains=5, sn=False, image_size=256):
        super(DomainDiscriminator, self).__init__()
        model = []
        model += [Conv2dBlock(input_nc, ndf, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, spectral_norm=sn, activation='leaky_relu', padding_type='reflect')]
        nch = ndf
        for i in range(1, n_layers-1):
            model += [Conv2dBlock(nch, nch * 2, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, spectral_norm=sn, activation='leaky_relu', padding_type='reflect')]
            nch *= 2
        model += [Conv2dBlock(nch, nch, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, spectral_norm=sn, activation='leaky_relu', padding_type='reflect')]
        self.model = nn.Sequential(*model)
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
    def __init__(self, ngf=256, num_downs=5, num_domains=5):
        super(ContentDiscriminator, self).__init__()
        model = []
        for i in range(num_downs):
            model += [Conv2dBlock(ngf, ngf, kernel_size=3, stride=2, padding=1, norm_layer='instance', activation='leaky_relu', padding_type='reflect')]
        model += [nn.Conv2d(ngf, num_domains, kernel_size=1, stride=2, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), out.size(1))
        return out

#######################
### StarGAN-V2 nets ###
#######################

class StarGANGenerator(nn.Module):
    def __init__(self, style_dim:int = 64,
                       max_conv_dim:int = 512,
                       n_blocks:int = 2,
                       num_downs:int = 0,
                       img_size:int = 256,
                       w_hpf:int = 1):
        super(StarGANGenerator, self).__init__()
        init_nc = 2**14 // img_size
        self.from_rgb = nn.Conv2d(3, init_nc, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(init_nc, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(init_nc, 3, 1, 1, 0))

        # down/up-sampling blocks
        if not num_downs:
            num_downs = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            num_downs += 1
        for _ in range(num_downs):
            output_nc = min(init_nc*2, max_conv_dim)
            self.encode.append(
                ResnetBlockV2(init_nc, output_nc, norm_layer='instance', downsample=True))
            self.decode.insert(
                0, AdaINResnetBlock(output_nc, init_nc, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            init_nc = output_nc

        # bottleneck blocks
        for _ in range(n_blocks):
            self.encode.append(
                ResnetBlockV2(output_nc, output_nc, norm_layer='instance'))
            self.decode.insert(
                0, AdaINResnetBlock(output_nc, output_nc, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        out = self.to_rgb(x)
        return out


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim:int = 16,
                       style_dim:int = 64,
                       num_domains:int = 2):
        super(MappingNetwork, self).__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, style_dim:int = 64,
                       n_blocks:int = 0,
                       num_domains:int = 2,
                       img_size:int = 256,
                       max_conv_dim:int = 512):
        super(StyleEncoder, self).__init__()
        init_ngf = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, init_ngf, 3, 1, 1)]
        if not n_blocks:
            n_blocks = int(np.log2(img_size)) - 2
        for _ in range(n_blocks):
            ngf = min(init_ngf*2, max_conv_dim)
            blocks += [ResnetBlockV2(init_ngf, ngf, downsample=True)]
            init_ngf = ngf

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(ngf, ngf, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(ngf, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StarGANDiscriminator(nn.Module):
    def __init__(self, num_domains:int = 2,
                       n_blocks:int = 0,
                       img_size:int = 256,
                       max_conv_dim:int = 512):
        super(StarGANDiscriminator, self).__init__()
        init_ndf = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, init_ndf, 3, 1, 1)]
        if not n_blocks:
            n_blocks = int(np.log2(img_size)) - 2
        for _ in range(n_blocks):
            ndf = min(init_ndf*2, max_conv_dim)
            blocks += [ResnetBlockV2(init_ndf, ndf, downsample=True)]
            init_ndf = ndf

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(ndf, ndf, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(ndf, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


#################
### VGG stuff ###
#################

class VGGGenerator(VGG):
    """
    constructs VGG net from torchvision and modifies the classification layer to output
    num_classes classes
    """
    cfg = {'vgg11' : 'A',
            'vgg13' : 'B',
            'vgg16' : 'D',
            'vgg19' : 'E'}
    def __init__(self, model:str, num_classes:int = 1000, checkpoint=None, batch_norm:bool = True):
        """
        Parameters:
            model : type of vgg model
            num_classes : number of classes in the model output
            batch_norm : bool indicating if batchnorm should be used.
        """
        if model not in VGGGenerator.cfg.keys():
            raise NotImplementedError(f'Implemented model choices are : {model.keys()}')

        super(VGGGenerator, self).__init__(features = make_layers(cfgs[VGGGenerator.cfg[model]], batch_norm=batch_norm),
                                           num_classes = num_classes)
        
        if checkpoint:
            self.load_state_dict(torch.load(checkpoint), strict=False)


class FeatureExtractor(nn.Module):
    """
    Constructs a VGG 19 model used as a loss network
    """
    def __init__(self, model:nn.Module, layer_count:int, checkpoint:str = None, url:str = None):
        """
        Parameters:
            model : the model from which features has to be extracted.
            layer_count : the layer number from which the feature has to be returned
        """
        super(FeatureExtractor, self).__init__()
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
        self.feature = nn.Sequential(
                        *self.unwrap_model(model)[:layer_count]
                        )

    def unwrap_model(self, model, layers=[]):
        """
        unwraps the nn.Sequential layers and returns individual layers of the model.
        """
        for module in model.children():
            # recursively unwrap if the module is nn.Sequential
            if isinstance(module, nn.Sequential):
                self.unwrap_model(module, layers)
            # if its a child module then add to the final list
            if list(module.children()) == []:
                layers.append(module)
        return layers
    
    def forward(self, x):
        return self.feature(x)


##############################
### Spectral Normalization ###
##############################

class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                        'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                    *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn

def spectral_normalization(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                            torch.nn.ConvTranspose2d,
                            torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

def remove_spectral_normalization(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))