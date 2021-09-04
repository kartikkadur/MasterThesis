import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x

########################
### Helper functions ###
########################

def get_norm_layer(norm_type : str ='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) : the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          : the optimizer of the network
        opt (option class) : stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
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
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator
    Parameters:
        input_nc (int) : the number of channels in input images
        output_nc (int) : the number of channels in output images
        ngf (int) : the number of filters in the last conv layer
        netG (str) : the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) : the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) : if use dropout layers.
        init_type (str)    : the name of our initialization method.
        init_gain (float)  : scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) : which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     : the number of channels in input images
        ndf (int)          : the number of filters in the first conv layer
        netD (str)         : the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   : the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         : the type of normalization layers used in the network.
        init_type (str)    : the name of the initialization method.
        init_gain (float)  : scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) : which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


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


#########################
### Model definitions ###
#########################

class ResnetGenerator(nn.Module):
    """
    Generator model for GAN
    """
    def __init__(self, input_nc : int,
                       output_nc : int,
                       ngf : int = 64,
                       n_downsampling : int = 2,
                       n_blocks : int = 6,
                       norm_layer : nn.Module = nn.BatchNorm2d,
                       use_dropout : bool = False,
                       padding_type : str = 'reflect',
                       use_attention : bool = True) -> None:
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
            attention (bool)    : weather to add attention block or not
        """
        super(ResnetGenerator, self).__init__()
        assert(n_blocks >= 0)

        self.attention = use_attention

        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d
        # initial conv block
        init_conv_block = [nn.ReflectionPad2d(3),
                            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                            norm_layer(ngf),
                            nn.ReLU(True)]
        # downsampling blocks
        for i in range(n_downsampling):
            mult = 2 ** i
            downsample_blocks = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet_blocks = [ResnetBlock(ngf * mult, ngf * mult, norm_layer=norm_layer, dropout=use_dropout, padding=padding_type]
        # upsampling blocks
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            upsample_blocks = [nn.ConvTranspose2d(ngf * mult, (ngf * mult)//2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        # final conv block
        final_conv_block = [nn.ReflectionPad2d(3),
                            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                            nn.Tanh()]
        
        # initialize the translator network model
        self.translator_net = nn.Sequential(*(init_conv_block + downsample_blocks + resnet_blocks + upsample_blocks + final_conv_block))
        
        if self.attention:
            # final attention layer with a single channel attention mask output
            att_final_conv_block = [nn.ReflectionPad2d(3),
                                    nn.Conv2d(ngf, 1, kernel_size=7, padding=0),
                                    nn.Tanh()]
            self.attention_net = nn.Sequential(*(init_conv_block 
                                                + downsample_blocks
                                                + resnet_blocks[:n_blocks//2]
                                                + upsample_blocks
                                                + att_final_conv_block))
    
    def forward(self, x):
        if self.attention:
            att = self.attention_net(x)
            out = self.translator_net(x)
            return att * out + (1 - att) * x
        else:
            return self.translator_net(x)

class ResnetBlock(nn.Module):
    expansion: int = 1
    def __init__(self,
                in_channels : int,
                out_channels : int,
                stride : int = 1,
                downsample : nn.Module = None,
                groups : int = 1,
                base_width : int = 64,
                norm_layer : nn.Module = None,
                dropout : bool = False,
                padding : str = 'zero'
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
        super(ResnetBlock, self).__init__()
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


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc : int,
                       output_nc : int,
                       num_downs : int,
                       ngf : int = 64,
                       norm_layer : nn.Module = nn.BatchNorm2d,
                       use_dropout : bool = False,
                       use_attention : bool = False) -> None:
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
        self.attention = use_attention
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
            return att * out + (1 - att) * x
        else:
            return self.translator_net(x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc : int,
                       inner_nc : int,
                       input_nc : int = None,
                       submodule : UnetSkipConnectionBlock = None,
                       outermost : bool = False,
                       innermost : bool = False,
                       norm_layer : nn.Module = nn.BatchNorm2d,
                       use_dropout : bool = False) -> None:
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


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc : int, ndf : int = 64, n_layers : int = 3, norm_layer : nn.Module = nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  : the number of channels in input images
            ndf (int)       : the number of filters in the last conv layer
            n_layers (int)  : the number of conv layers in the discriminator
            norm_layer      : normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc : int, ndf : int = 64, norm_layer : nn.Module = nn.BatchNorm2d):
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
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)