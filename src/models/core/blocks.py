import torch
import torch.nn as nn

from models.core.norm import AdaptiveInstanceNorm
from models.core.functions import get_padding_layer
from models.core.functions import get_activation_layer
from models.core.functions import get_norm_layer
from models.core.functions import spectral_norm

class ConvBlock(nn.Module):
    """Convolution block containing conv, norm layer and activation layer"""
    def __init__(self, input_dim:int,
                       output_dim:int,
                       kernel_size:int,
                       stride:int = 1,
                       padding:int = 0,
                       bias:bool = False,
                       norm_layer:str = None,
                       activation:str = None,
                       padding_type:str = None,
                       sn:bool = False):
        super(ConvBlock, self).__init__()
        self.block = []
        # add padding, norm and activation layers if passed
        padding_layer = get_padding_layer(padding_type)
        activation = get_activation_layer(activation)
        norm_layer = get_norm_layer(norm_layer)
        # add layers to block
        if padding_layer is not None:
            self.block += [padding_layer(padding)]
            padding = 0
        # add conv layer
        if sn:
            self.block += [spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias))]
        else:
            self.block += [nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias)]
        # add norm layer
        if norm_layer is not None:
            self.block += [norm_layer(output_dim)]
        # add activation
        if activation is not None:
            self.block += [activation()]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)

class UpsampleBlock(nn.Module):
    """Transpopse Convolution block containing transpose conv, norm layer and activation layer"""
    def __init__(self, input_dim:int,
                       output_dim:int,
                       kernel_size:int,
                       stride:int = 1,
                       padding:int = 0,
                       output_padding:int = 0,
                       bias:bool = False,
                       norm_layer:str = None,
                       activation:str = None,
                       padding_type:str = None,
                       sn:bool = False,
                       up_type:str = 'transpose'):
        super(UpsampleBlock, self).__init__()
        self.block = []
        # add padding, norm and activation layers if passed
        padding_layer = get_padding_layer(padding_type)
        activation = get_activation_layer(activation)
        norm_layer = get_norm_layer(norm_layer)
        # add layers to block
        if 'transpose' in up_type:
            if sn:
                self.block += [spectral_norm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, output_padding, bias=bias))]
            else:
                self.block += [nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, output_padding, bias=bias)]
        elif 'nearest' in up_type:
            self.block += [nn.Upsample(scale_factor=2, mode='nearest'),
                    ConvBlock(input_dim, output_dim, kernel_size, 1, padding, padding_type=padding_type, bias=bias, sn=sn)]
        elif 'pixelshuffle' in up_type:
            self.block += [ConvBlock(input_dim, output_dim, kernel_size, 1, padding, padding_type=padding_type, bias=bias, sn=sn),
                           nn.PixelShuffle(2)]
        else:
            raise NotImplementedError(f"Mode {up_type} is not supported at the moment")
        # add norm layer
        if norm_layer is not None:
            self.block += [norm_layer(output_dim)]
        # add activation
        if activation is not None:
            self.block += [activation()]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)

class DownResnetBlock(nn.Module):
    """Basic resnet type block with short cut convolution connection"""
    def __init__(self, input_dim:int,
                       output_dim:int,
                       norm_layer:str = 'instance',
                       activation:str = 'lrelu',
                       padding_type:str = 'reflect',
                       bias:bool = True):
        super(DownResnetBlock, self).__init__()
        self.conv = []
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        if isinstance(activation, str):
            activation = get_activation_layer(activation)
        if norm_layer is not None:
            self.conv.append(norm_layer(input_dim))
        self.conv.append(activation())
        self.conv.append(ConvBlock(input_dim, input_dim, 3, 1, padding=1, padding_type=padding_type,
                                    norm_layer=norm_layer, activation=activation, bias=bias))
        self.conv.append(ConvBlock(input_dim, output_dim, 3, 1, padding=1, padding_type=padding_type, bias=bias))
        self.conv.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv = nn.Sequential(*self.conv)
        self.shortcut = nn.Sequential(*[nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(input_dim, output_dim, 1, 1, 0, bias=bias)])

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class ResnetBlock(nn.Module):
    """resnet block"""
    def __init__(self, input_dim:int,
                       output_dim:int,
                       dropout:bool = False,
                       norm_layer:str = 'instance',
                       padding_type:str = 'reflect',
                       activation:str = 'relu'):
        super(ResnetBlock, self).__init__()
        self.model = []
        self.model += [ConvBlock(input_dim, output_dim, 3, 1, 1, padding_type=padding_type, norm_layer=norm_layer, activation=activation)]
        self.model += [ConvBlock(output_dim, output_dim, 3, 1, 1, padding_type=padding_type, norm_layer=norm_layer)]
        if dropout:
            self.model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return x + self.model(x)

class AdaINResnetBlock(nn.Module):
    """resnet block"""
    def __init__(self, input_dim:int,
                       output_dim:int,
                       dropout:bool = False,
                       latent_dim:int = 256,
                       padding_type:str = 'reflect',
                       activation:str = 'relu'):
        super(AdaINResnetBlock, self).__init__()
        self.activation = get_activation_layer(activation)()
        self.conv1 = ConvBlock(input_dim, output_dim, 3, 1, 1, padding_type=padding_type)
        self.conv2 = ConvBlock(output_dim, output_dim, 3, 1, 1, padding_type=padding_type)
        self.norm = AdaptiveInstanceNorm(latent_dim, output_dim)
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

    def forward(self, x, z):
        print(z.shape)
        residual = x
        x = self.conv1(x)
        x = self.norm(x, z)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x, z)
        x = self.dropout(x)
        x += residual
        return x

class DecResnetBlock(nn.Module):
    def __init__(self, n_channel:int,
                       add_channel:int,
                       norm_layer:str = 'instance',
                       padding_type:str = 'reflect',
                       stride:int = 1,
                       dropout:bool = False):
        super(DecResnetBlock, self).__init__()
        self.conv1 = ConvBlock(n_channel, n_channel, 3, stride=stride, padding=1, padding_type=padding_type)
        self.conv2 = ConvBlock(n_channel, n_channel, 3, stride=stride, padding=1, padding_type=padding_type)
        self.norm = get_norm_layer(norm_layer)(n_channel)
        block1 = [ConvBlock(n_channel + add_channel, n_channel + add_channel, 1, stride=stride, padding=0, activation='relu')]
        block1 += [ConvBlock(n_channel + add_channel, n_channel, 1, stride=1, padding=0, activation='relu')]
        self.block1 = nn.Sequential(*block1)
        block2 = [ConvBlock(n_channel + add_channel, n_channel + add_channel, 1, stride=1, padding=0, activation='relu')]
        block2 += [ConvBlock(n_channel + add_channel, n_channel, 1, stride=1, padding=0, activation='relu')]
        self.block2 = nn.Sequential(*block2)
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        out = self.conv1(x)
        out = self.norm(out)
        out = self.block1(torch.cat([out, z_expand], dim=1))
        out = self.conv2(out)
        out = self.norm(out)
        out = self.block2(torch.cat([out, z_expand], dim=1))
        out = self.dropout(out)
        out += residual
        return out