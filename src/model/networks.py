import torch
import functools
import torch.nn as nn

from inspect import isclass
from torch.nn import init
from torch.autograd import Variable
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
            activation = functools.partial(nn.ReLU, inplace=True)
        elif activation == 'leaky_relu':
            activation = functools.partial(nn.LeakyReLU, inplace=True)
        elif activation == 'tanh':
            activation = functools.partial(nn.Tanh)
        elif activation == 'sigmoid':
            activation == functools.partial(nn.Sigmoid)
        else:
            raise NotImplementedError(f"activation type '{activation}' is not supported at the moment")
    elif not (activation == None) and isclass(activation) and not issubclass(activation, nn.Module):
        raise ValueError(f"parameter type of activation should be one of 'str' or 'nn.Module', but got {type(activation)}.")
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
    if torch.cuda.is_available():
        net.to('cuda')

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
                            num_downs=8,
                            ngf=args.ngf,
                            norm_layer=args.norm_layer,
                            use_dropout=not args.no_dropout,
                            attention=True)
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
                       upsample:bool = False
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
            model += [nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride, padding, output_padding, bias=bias)]
        else:
            model += [nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding, bias=bias)]
        # add norm layer
        if norm_layer:
            norm_layer = get_norm_layer(norm_layer)
            model += [norm_layer(output_nc)]
        # add activation
        if activation:
            activation = get_activation_layer(activation)
            activation = activation if isinstance(activation, nn.Module) else activation()
            model += [activation]
        # generate final block
        self.block = nn.Sequential(*model)

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self,
                channels:int,
                norm_layer = None,
                dropout:bool = False,
                padding:int = 0,
                padding_type:str = None
                ) -> None:
        super(ResnetBlock, self).__init__()
        p = 0
        self.padding = None
        self.dropout = None
        if padding_type:
            self.padding = get_padding_layer(padding_type)(1)
            padding = 0
        elif padding == 0:
            self.padding = nn.ReflectionPad2d(1)
        # get norm layer
        norm_layer = get_norm_layer(norm_layer)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding)
        self.norm1 = norm_layer(channels)
        self.relu = nn.ReLU(True)

        if dropout:
            self.dropout = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding)
        self.norm2 = norm_layer(channels)

    def forward(self, x):
        out = x
        # initial padding
        if self.padding:
            out = self.padding(out)

        out = self.relu(self.norm1(self.conv1(out)))
        # dropout
        if self.dropout:
            out = self.dropout(out)
        
        if self.padding:
            out = self.padding(out)
        # residual connections
        out = x + self.norm2(self.conv2(out))
        return out


class ResNextBlock(nn.Module):
    expansion: int = 1
    def __init__(self,
                in_channels:int,
                out_channels:int,
                stride:int = 1,
                downsample:nn.Module = None,
                groups:int = 1,
                base_width:int = 64,
                norm_layer:nn.Module = None,
                dropout:bool = False,
                padding:str = 'zero'
                ) -> None:
        """
        Implementation of Resnet bottleneck block. Adapted from the pytorch github code.
        Arguments:
        in_channels  : the number of input channels
        out_channels : the number of output channels
        stride       : stride to use for Conv2d operation
        downsample   : a nn.Module class for downsampling the input dimention
        groups       : number of groups used for network width
        base_width   : the starting width of initial conv layer
        norm_layer   : a nn.Module class for handling normalization
        """
        super(ResNextBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.padding = None
        self.dropout = None
        # initialize padding
        p=0
        if padding == 'reflect':
            self.padding = nn.ReflectionPad2d(1)
        elif padding == 'replicate':
            self.padding = nn.ReplicationPad2d(1)
        else:
            p=1
        # initialize dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)
        # assign a downsample layer
        if (in_channels != out_channels or stride > 1) and downsample is None:
            self.downsample = nn.Sequential(
                                 nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                                 norm_layer(out_channels * self.expansion)
                            )
        else:
            self.downsample = downsample
        # calculate initial filter width based on base_width and groups
        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=p, groups=groups)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.padding:
            out = self.padding(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


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


class GaussianNoiseLayer(nn.Module):
    def __init__(self, isTrain=False):
        super(GaussianNoiseLayer, self).__init__()
        self.isTrain = isTrain

    def forward(self, x):
        if self.isTrain == False:
            return x
        noise = Variable(torch.randn(x.size()).to(x.get_device()))
        return x + noise


#########################
### Model definitions ###
#########################

class ResnetGenerator(nn.Module):
    """
    Generator model for GAN
    """
    def __init__(self, input_nc:int,
                       output_nc:int,
                       ngf:int = 64,
                       n_downsampling:int = 2,
                       n_blocks:int = 6,
                       norm_layer:nn.Module = nn.InstanceNorm2d,
                       use_dropout:bool = False,
                       padding_type:str = 'reflect',
                       attention:bool = False,
                       feature:bool = False) -> None:
        """Construct a Resnet-based generator with attention
        Parameters:
            input_nc (int)      : the number of channels in input images
            output_nc (int)     : the number of channels in output images
            ngf (int)           : the number of filters in the last conv layer
            n_downsampling      : the number of downsampling layers to be used
            n_blocks (int)      : the number of ResNet blocks
            norm_layer          : normalization layer
            use_dropout (bool)  : if use dropout layers
            padding_type (str)  : the name of padding layer in conv layers: reflect | replicate | zero
            attention (int)    : a number indicating the type of attention network to be used. 0:no attention, 1: seperate attention, 2:only decoder attention
        """
        super(ResnetGenerator, self).__init__()
        assert(n_blocks >= 0)
        self.attention = attention
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.padding = nn.ReflectionPad2d(3)
        # initial conv block
        self.init_conv_block = Conv2dBlock(input_nc, ngf, kernel_size=7, padding=0, 
                                            bias=use_bias, norm_layer=norm_layer,
                                            activation='relu')
        # downsampling blocks
        downsample_blocks = []
        for i in range(n_downsampling):
            mult = 2 ** i
            downsample_blocks += [Conv2dBlock(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, 
                                            padding=1, bias=use_bias, norm_layer=norm_layer, activation='relu')]
        self.downsample_blocks = nn.Sequential(*downsample_blocks)

        # ResNet blocks
        resnet_blocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet_blocks += [ResnetBlock(ngf * mult, norm_layer=norm_layer, dropout=use_dropout, padding_type=padding_type)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # upsampling blocks
        upsample_blocks = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            upsample_blocks += [Conv2dBlock(ngf * mult, (ngf * mult)//2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias,
                                         norm_layer=norm_layer,
                                         activation='relu',
                                         upsample=True)]
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        # final conv block
        self.final_conv_block = Conv2dBlock(ngf, output_nc, kernel_size=7, padding=0, norm_layer=None, activation='tanh')
        
        # initialize attention model
        if self.attention:
            # attention layer with a single channel attention mask output
            self.att_resnet_blocks = nn.Sequential(*resnet_blocks[:len(resnet_blocks)//2])
            self.attention_block = Conv2dBlock(ngf, 1, kernel_size=1, padding=0, norm_layer=None, activation='tanh')
        
        # return feature if required
        self.feature = feature
        if feature:
            self.feature_block = nn.Sequential(*upsample_blocks[:1])

    def forward(self, x):
        # encoder
        enc = self.init_conv_block(self.padding(x))
        enc = self.downsample_blocks(enc)
        enc = self.resnet_blocks(enc)
        # decoder
        dec = self.upsample_blocks(enc)
        out = self.final_conv_block(self.padding(dec))

        # attention
        att = torch.ones(x.size()).to(x.get_device())
        if self.attention == 1:
            att = self.init_conv_block(self.padding(x))
            att = self.downsample_blocks(att)
            att = self.att_resnet_blocks(att)
            att = self.upsample_blocks(att)
            att = self.attention_block(att)
        
        # only decoder attention
        if self.attention == 2:
            att = self.upsample_blocks(enc)
            att = self.attention_block(att)
        
        # return feature map of 1st layer of decoder
        if self.feature:
            fea = self.feature_block(enc)
            return att * out + (1 - att) * x, att, fea

        return att * out + (1 - att) * x, att


class UnetGenerator(nn.Module):
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
        super(UnetGenerator, self).__init__()
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


class ContentEncoder(nn.Module):
    """Defines a encoder model for enconding image -> content space (DRIT)"""
    def __init__(self, input_nc:int,
                       ngf:int=64,
                       num_downs:int = 3,
                       n_blocks:int = 3,
                       norm_layer = nn.InstanceNorm2d,
                       use_dropout:bool = False,
                       padding_type:str = 'reflect'):
        super(ContentEncoder, self).__init__()
        layers = []
        layers += [Conv2dBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3,
                                norm_layer=norm_layer, activation='leaky_relu', padding_type='reflect')]
        # downsample blocks
        for i in range(num_downs):
            mult = 2**i
            layers += [Conv2dBlock(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer, activation='leaky_relu', padding_type='reflect')]
        # residual blocks
        for i in range(n_blocks):
            layers += [ResnetBlock(ngf, norm_layer='instance', padding=1, padding_type='reflect')]

        for i in range(0, 1):
            layers += [ResnetBlock(ngf, norm_layer='instance', padding=1, padding_type='reflect')]
            layers += [GaussianNoiseLayer()]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttributeEncoder(nn.Module):
    """Defines a encoder model for enconding image -> attribute space"""
    def __init__(self, input_nc:int, output_nc:int, ngf:int = 64, num_downs:int = 4, content_nc:int = 3):
        super(AttributeEncoder, self).__init__()
        layers = [Conv2dBlock(input_nc + content_nc , ngf, 7, 1, padding=1,
                                padding_type='reflect', activation='relu')]
        for i in range(num_downs):
            mult = 2**i
            layers += [Conv2dBlock(ngf * mult , ngf * mult * 2, 4, 2, padding=1,
                                padding_type='reflect', activation='relu')]
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Conv2d(ngf*4, output_nc, 1, 1, 0)]
        # final model
        self.model = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)
        output = self.model(x_c)
        return output.view(output.size(0), -1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc:int,
                       ndf:int = 64,
                       n_layers:int = 3,
                       norm_layer:nn.Module = nn.BatchNorm2d,
                       n_domain:int = 2):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  : the number of channels in input images
            ndf (int)       : the number of filters in the last conv layer
            n_layers (int)  : the number of conv layers in the discriminator
            norm_layer      : normalization layer
            n_domain (int)  : Number of classes used for discrimination
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.LeakyReLU(0.2, True)
        kw = 4
        padw = 1
        self.init_block = Conv2dBlock(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, 
                                    norm_layer=None, activation=activation)
        nf_mult = 1
        nf_mult_prev = 1
        blocks = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            blocks += [Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                          bias=use_bias, norm_layer=norm_layer, activation=activation)]

        self.downsample_blocks = nn.Sequential(*blocks)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.conv_block = Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw,
                                    bias=use_bias, norm_layer=norm_layer, activation=activation)
        # discriminator output predicting real/fake sample
        self.out_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        # classification output
        self.n_domains = n_domain
        if self.n_domains > 2:
            self.out_cls = nn.Conv2d(ndf * nf_mult, self.n_domains, kernel_size=kw, stride=1, padding=padw)
            self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.init_block(x)
        x = self.downsample_blocks(x)
        x = self.conv_block(x)
        # discriminator out
        d_out = self.out_conv(x)
        # classification out
        if self.n_domains > 2:
            c_out = self.pool(self.out_cls(x))
            return d_out, c_out.view(c_out.size(0), c_out.size(1))
        return d_out

class AttNLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator with attention"""

    def __init__(self, input_nc:int,
                       ndf:int = 64,
                       n_layers:int = 3,
                       norm_layer:nn.Module = nn.BatchNorm2d,
                       n_domain:int = 2,
                       input_size:int=256):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  : the number of channels in input images
            ndf (int)       : the number of filters in the last conv layer
            n_layers (int)  : the number of conv layers in the discriminator
            norm_layer      : normalization layer
            input_size      : the input image size
        """
        super(AttNLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        activation = nn.LeakyReLU(0.2, True)
        kw = 4
        padw = 1
        self.init_block = Conv2dBlock(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, 
                                       norm_layer=None, activation=activation)
        nf_mult = 1
        nf_mult_prev = 1
        blocks = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            blocks += [Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                 bias=use_bias, norm_layer=norm_layer, activation=activation)]

        self.downsample_blocks = nn.Sequential(*blocks)
        # upsample block to get attention map
        self.upsample_block = nn.Upsample(input_size, mode='bilinear')

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.conv_block = Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw,
                                    bias=use_bias, norm_layer=norm_layer, activation=activation)
        # discriminator output predicting real/fake sample
        self.out_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        # classification output
        self.n_domains = n_domain
        if self.n_domains > 2:
            self.out_cls = nn.Conv2d(ndf * nf_mult, self.n_domains, kernel_size=kw, stride=1, padding=padw)
            self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.init_block(x)
        x = self.downsample_blocks(x)
        # calculate attention maps
        att = torch.mean(self.upsample_block(x), dim=1)
        x = self.conv_block(x)
        d_out = self.out_conv(x)
        if self.n_domains > 2:
            c_out = self.pool(self.out_cls(x))
            return d_out, att, c_out.view(c_out.size(0), c_out.size(1))
        return d_out, att


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