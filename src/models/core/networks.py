import torch
import numpy as np
from torch import nn

from models.core.blocks import *
from models.core.misc import *

class ContentEncoder(nn.Module):
    def __init__(self, input_dim:int,
                       dim:int = 64,
                       num_downs:int = 2,
                       n_blocks:int = 4,
                       norm_layer:str = 'instance',
                       padding_type:str = 'reflect',
                       bias:bool = True):
        """
        Content encoder encodes the domain invarient content such as edges and lines
        into a feature map.
        Arguments:
            input_dim   : number of channels in the input image, Usually 3
            dim         : dimention to be increased at each downsampling layer. Default : 64
            num_downs   : number of downsampling layers. Default : 2
            n_blocks    : number of resnet blocks to be added. Default : 4
            norm_layer  : type of normalization layer to be used. Default : instance
            padding_type: type of padding layer to be used. Default : reflect
            bias        : use bias or not. Default : True
        """
        super(ContentEncoder, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(ConvBlock(input_dim, dim, 7, 1, 3, padding_type=padding_type,
                                    norm_layer=norm_layer, activation='lrelu', bias=bias))
        for i in range(num_downs):
            self.model.append(ConvBlock(dim, dim * 2, 3, 2, 1, padding_type=padding_type,
                                        norm_layer=norm_layer, activation='relu', bias=bias))
            dim *= 2
        for i in range(n_blocks):
            self.model.append(ResnetBlock(dim, dim, norm_layer=norm_layer, activation='relu'))
        self.model.append(GaussianNoiseLayer())
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class StyleEncoder(nn.Module):
    def __init__(self, input_dim:int,
                       output_dim:int = 8,
                       dim:int = 64,
                       num_downs:int = 4,
                       num_domains:int = 2,
                       padding_type:str = 'reflect',
                       activation:str = 'relu'):
        """
        Encodes domain invarient features in to a style code.
        A global avg pooling operation on the channel dim is performed
        to extract the style code.
        Arguments:
            input_dim   : number of channels in the input image, Usually 3
            output_dim  : the size of the style code, Default : 8
            dim         : dimention to be increased at each downsampling layer. Default : 64
            num_downs   : number of downsampling layers. Default : 4
            padding_type: type of padding layer to be used. Default : reflect
            activation  : type of activation layer to be used. Default : relu
        """
        super(StyleEncoder, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(ConvBlock(input_dim+num_domains, dim, 7, 1, padding=3, 
                                    padding_type=padding_type, activation=activation))
        max_filter_size=256
        for n in range(num_downs):
            in_dim = min(max_filter_size, dim)
            out_dim = min(max_filter_size, dim*2)
            self.model.append(ConvBlock(in_dim, out_dim, 4, 2, padding=1,
                                        padding_type=padding_type, activation=activation))
            dim *= 2
        self.model.append(nn.AdaptiveAvgPool2d(1))
        self.model.append(nn.Conv2d(out_dim, output_dim, 1, 1, 0))
        self.model = nn.Sequential(*self.model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)
        output = self.model(x_c)
        return output.view(output.size(0), -1)

class ReparameterizedStyleEncoder(nn.Module):
    def __init__(self, input_dim:int,
                       output_dim:int = 8,
                       dim:int = 64,
                       n_blocks:int = 4,
                       num_domains:int = 2,
                       norm_layer:str = None,
                       activation:str = None,
                       bias:bool = True):
        """
        Encodes domain invarient features in to a style code.
        A global avg pooling operation on the channel dim is performed
        to extract the style code.
        Arguments:
            input_dim   : number of channels in the input image, Usually 3
            output_dim  : the size of the style code, Default : 8
            dim         : dimention to be increased at each downsampling layer. Default : 64
            n_blocks    : number of resnet blocks to be added. Default : 4
            num_domains : number of target domains. Default : 2
            norm_layer  : type of normalization layer to be used. Default : None
            activation  : type of activation layer to be used. Default : None
            bias        : use bias or not. Default : True
        """
        super(ReparameterizedStyleEncoder, self).__init__()
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        if isinstance(activation, str):
            activation = get_activation_layer(activation)
        max_filter_size = 256
        self.model = nn.ModuleList()
        self.model.append(ConvBlock(input_dim+num_domains, dim, 4, 2, 1, padding_type='reflect', bias=bias))
        for n in range(1, n_blocks):
            in_dim = min(max_filter_size, dim)
            out_dim = min(max_filter_size, dim*2)
            self.model.append(DownResnetBlock(in_dim, out_dim, norm_layer, activation, bias=bias))
            dim *= 2
        self.model.append(activation())
        self.model.append(nn.AdaptiveAvgPool2d(1))
        self.model = nn.Sequential(*self.model)
        self.out_nch = out_dim
        self.fc = nn.Linear(out_dim, output_dim)
        self.fcVar = nn.Linear(out_dim, output_dim)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        #eps = torch.randn(std.size()).to(mu.device)
        z_style = eps.mul(std).add_(mu)
        return z_style

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)
        x_conv = self.model(x_c)
        conv_flat = x_conv.view(x.size(0), -1)
        mu = self.fc(conv_flat)
        logvar = self.fcVar(conv_flat)
        z_style = self.reparameterize(mu, logvar)
        return z_style, mu, logvar

class Decoder(nn.Module):
    def __init__(self, output_dim:int,
                       dim:int = 256,
                       n_blocks:int = 4,
                       num_domains:int = 2,
                       num_ups:int = 2,
                       latent_dim:int = 8,
                       up_type:str = 'transpose',
                       dropout:bool = False,
                       norm_layer:str = 'layer',
                       activation:str = 'relu',
                       bias:bool = True):
        """
        Given a encoded content and style, the decoder maps the code
        back to the image space.
        Arguments:
            output_dim  : the size of the output feature, usually 3.
            dim         : style code dimention. Default : 256
            n_blocks    : number of resnet blocks to be added. Default : 4
            num_domains : number of target domains. Default : 2
            num_ups     : number of upsampling layers. Default : 2
            latent_dim  : size of the latent code. Default : 8
            up_type     : type of upsampling to be used. Options [transpose, nearest, pixel]
            dropout     : use dropout or not. Default : False
            norm_layer  : type of normalization layer to be used. Default : None
            activation  : type of activation layer to be used. Default : None
            bias        : use bias or not. Default : True
        """
        super(Decoder, self).__init__()
        self.dim_add = dim
        self.dec1 = nn.ModuleList()
        for i in range(n_blocks):
            self.dec1.append(DecResnetBlock(dim, self.dim_add, dropout=dropout))
        self.dec2 = nn.ModuleList()
        for i in range(num_ups):
            self.dec2.append(UpsampleBlock(dim, dim//2, 3, 2, 1, 1, norm_layer=norm_layer, activation=activation, up_type=up_type, bias=bias))
            dim = dim//2
        if 'transpose' in up_type:
            self.dec2.append(UpsampleBlock(dim, output_dim, 1, 1, 0, activation='tanh', up_type='transpose'))
        else:
            self.dec2.append(ConvBlock(dim, output_dim, 7, 1, 3, activation='tanh'))
        self.dec2 = nn.Sequential(*self.dec2)
        self.linear = nn.Sequential(
                    nn.Linear(latent_dim + num_domains, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, self.dim_add*n_blocks))

    def forward(self, x, z, c):
        z_c = torch.cat([c, z], 1)
        z_c = self.linear(z_c)
        out = x
        z_splits = torch.split(z_c, self.dim_add, dim=1)
        for dec, z in zip(self.dec1, z_splits):
            out = dec(out, z.contiguous())
        out = self.dec2(out)
        return out

class AdaINDecoder(nn.Module):
    def __init__(self, output_dim:int,
                       dim:int = 256,
                       n_blocks:int = 4,
                       num_domains:int = 2,
                       num_ups:int = 2,
                       latent_dim:int = 8,
                       up_type:str = 'transpose',
                       res_norm:str = 'adain',
                       dropout:bool = False,
                       norm_layer:str = 'layer',
                       activation:str = 'relu',
                       bias:bool = True):
        """
        Given a encoded content and style, the decoder maps the code
        back to the image space.
        Arguments:
            output_dim  : the size of the output feature, usually 3.
            dim         : style code dimention. Default : 256
            n_blocks    : number of resnet blocks to be added. Default : 4
            num_domains : number of target domains. Default : 2
            num_ups     : number of upsampling layers. Default : 2
            latent_dim  : size of the latent code. Default : 8
            up_type     : type of upsampling to be used. Options [transpose, nearest, pixel]
            res_norm    : type of resnet normalization
            dropout     : use dropout or not. Default : False
            norm_layer  : type of normalization layer to be used. Default : None
            activation  : type of activation layer to be used. Default : None
            bias        : use bias or not. Default : True
        """
        super(AdaINDecoder, self).__init__()
        self.dim_add = dim
        self.dec1 = nn.ModuleList()
        if 'adain' in res_norm:
            for i in range(n_blocks):
                self.dec1.append(AdaINResnetBlock(dim, self.dim_add, style_dim=self.dim_add, dropout=dropout))
        else:
            for i in range(n_blocks):
                self.dec1.append(ResnetBlock(dim, self.dim_add, dropout=dropout))
        self.dec2 = nn.ModuleList()
        for i in range(num_ups):
            self.dec2.append(UpsampleBlock(dim, dim//2, 3, 2, 1, 1, norm_layer=norm_layer, activation=activation, up_type=up_type, bias=bias))
            dim = dim//2
        if 'transpose' in up_type:
            self.dec2.append(UpsampleBlock(dim, output_dim, 1, 1, 0, activation='tanh', up_type='transpose'))
        else:
            self.dec2.append(ConvBlock(dim, output_dim, 7, 1, 3, activation='tanh'))
        self.dec2 = nn.Sequential(*self.dec2)
        if 'adain' in res_norm:
            self.linear = nn.Sequential(
                            nn.Linear(latent_dim + num_domains, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, self.dim_add))

    def forward(self, x, z, c):
        z_c = torch.cat([c, z], 1)
        z_c = self.linear(z_c)
        out = x
        for dec in self.dec1:
            out = dec(out, z_c)
        out = self.dec2(out)
        return out

class DecoderConcat(nn.Module):
    def __init__(self, output_dim:int,
                       dim:int = 256,
                       n_blocks:int = 3,
                       num_domains:int = 2,
                       latent_dim:int = 8,
                       up_type:str = 'transpose',
                       dropout:bool = False,
                       norm_layer:str = 'layer',
                       activation:str = 'relu',
                       bias:bool = True):
        """
        Given a encoded content and style, the decoder maps the code
        back to the image space.
        Arguments:
            output_dim  : the size of the output feature, usually 3.
            dim         : style code dimention. Default : 256
            n_blocks    : number of resnet blocks to be added. Default : 4
            num_domains : number of target domains. Default : 2
            latent_dim  : size of the latent code. Default : 8
            up_type     : type of upsampling to be used. Options [transpose, nearest, pixel]
            dropout     : boolian indicatig weather to use dropout or not. Default : False
            norm_layer  : type of normalization layer to be used. Default : None
            activation  : type of activation layer to be used. Default : None
            bias        : use bias or not. Default : True
        """
        super(DecoderConcat, self).__init__()
        self.dec_share = ResnetBlock(dim, dim)
        nch = dim + latent_dim + num_domains
        self.dec1 = nn.ModuleList()
        for i in range(n_blocks):
            self.dec1.append(ResnetBlock(nch, nch, dropout=dropout))
        self.dec1 = nn.Sequential(*self.dec1)
        nch = nch + latent_dim
        self.dec2 = UpsampleBlock(nch, nch//2, 3, 2, 1, 1, norm_layer=norm_layer, activation=activation, up_type=up_type, bias=bias)
        nch = nch//2
        nch = nch + latent_dim
        self.dec3 = UpsampleBlock(nch, nch//2, 3, 2, 1, 1, norm_layer=norm_layer, activation=activation, up_type=up_type, bias=bias)
        nch = nch//2
        nch = nch + latent_dim
        if 'transpose' in up_type:
            self.dec4 = UpsampleBlock(nch, output_dim, 1, 1, 0, activation='tanh', up_type='transpose')
        else:
            self.dec4 = ConvBlock(nch, output_dim, 7, 1, 3, activation='tanh')

    def forward(self, x, z, c):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, out0.size(2), out0.size(3))
        x_c_z = torch.cat([out0, c, z_img], 1)
        out1 = self.dec1(x_c_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.dec2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.dec3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.dec4(x_and_z4)
        return out4

class Discriminator(nn.Module):
    def __init__(self, input_dim:int,
                       dim:int = 64,
                       n_layers:int = 6,
                       num_domains:int = 2,
                       norm_layer:str = None,
                       activation:str = 'lrelu',
                       padding_type:str = 'reflect',
                       bias:bool = True,
                       sn:bool = False,
                       image_size:int = 256):
        """
        Discriminator discriminates between real and fake samples.
        It also classifies the input to beong to a perticular class.
        Arguments:
            input_dim  : the size of the input feature, usually 3.
            dim         : style code dimention. Default : 256
            n_layers    : number of layers to be added. Default : 4
            num_domains : number of target domains. Default : 2
            norm_layer  : type of normalization layer to be used. Default : None
            activation  : type of activation layer to be used. Default : lrelu
            padding_type: type of padding layer to be used. Default : reflect
            bias        : use bias or not. Default : True
            sn          : use spectral normalization or not. Default : False
            image_size  : input image size. Default:256
        """
        super(Discriminator, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(ConvBlock(input_dim, dim, kernel_size=3, stride=2, padding=1, padding_type=padding_type,
                                                        norm_layer=norm_layer, sn=sn, activation=activation, bias=bias))
        nch = dim
        for i in range(n_layers-2):
            self.model.append(ConvBlock(nch, nch * 2, kernel_size=3, stride=2, padding=1, padding_type=padding_type,
                                                        norm_layer=norm_layer, sn=sn, activation=activation, bias=bias))
            nch *= 2
        self.model.append(ConvBlock(nch, nch, kernel_size=3, stride=2, padding=1, padding_type=padding_type,
                                                                sn=sn, activation=activation, bias=bias))
        self.model = nn.Sequential(*self.model)
        self.conv1 = nn.Conv2d(nch, 1, kernel_size=1, stride=1, padding=1, bias=False)
        kernal_size = int(image_size/np.power(2, n_layers))
        self.conv2 = nn.Conv2d(nch, num_domains, kernel_size=kernal_size, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = nch

    def forward(self, x):
        h = self.model(x)
        out = self.conv1(h)
        out_cls = self.conv2(h)
        out_cls = self.pool(out_cls)
        return out, out_cls.view(out_cls.size(0), out_cls.size(1))

class ContentDiscriminator(nn.Module):
    def __init__(self, dim:int = 256,
                       num_domains:int = 3,
                       norm_layer:str = 'instance',
                       activation:str = 'lrelu',
                       padding_type:str = 'reflect',
                       bias:bool = True):
        """
        classifies a given encoded content feature map to belong to a
        perticular domain.
        Arguments:
            dim         : number of channels in the feature map. Default : 256
            num_domains : number of target domains. Default : 2
            norm_layer  : type of normalization layer to be used. Default : instance
            activation  : type of activation layer to be used. Default : lrelu
            padding_type: type of padding layer to be used. Default : reflect
            bias        : use bias or not. Default : True
        """
        super(ContentDiscriminator, self).__init__()
        self.model = nn.ModuleList()
        for i in range(3):
            self.model.append(ConvBlock(dim, dim, kernel_size=7, stride=2, padding=1, padding_type=padding_type, 
                                                norm_layer=norm_layer, activation=activation, bias=bias))
        self.model.append(ConvBlock(dim, dim, kernel_size=4, stride=1,
                            padding=0, padding_type=padding_type, activation=activation, bias=bias))
        self.model.append(nn.Conv2d(dim, num_domains, kernel_size=1, stride=1, padding=0))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        out = self.pool(out)
        out = out.view(out.size(0), out.size(1))
        return out

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_dim:int,
                       dim:int = 64,
                       n_layers:int = 6,
                       num_domains:int = 2,
                       norm_layer:str = None,
                       activation:str = 'lrelu',
                       padding_type:str = None,
                       num_scales:int = 3,
                       sn:bool = False):
        """
        discriminates between real and fake samples on multiple scales.
        It also classifies the input to beong to a perticular class.
        Arguments:
            input_dim  : the size of the input feature, usually 3.
            dim         : style code dimention. Default : 256
            n_layers    : number of layers to be added. Default : 4
            num_domains : number of target domains. Default : 2
            norm_layer  : type of normalization layer to be used. Default : None
            activation  : type of activation layer to be used. Default : lrelu
            padding_type: type of padding layer to be used. Default : reflect
            num_scales  : number of downsample operation to be performed. Default : 3
            sn          : use spectral normalization or not. Default : False
        """
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.model = nn.ModuleList()
        self.model.append(ConvBlock(input_dim, dim, 4, 2, 1, norm_layer=None, activation=activation, padding_type=padding_type, sn=sn))
        for i in range(n_layers - 1):
            self.model.append(ConvBlock(dim, dim * 2, 4, 2, 1, norm_layer=norm_layer, activation=activation, padding_type=padding_type, sn=sn))
            dim *= 2
        self.model = nn.Sequential(*self.model)
        self.dis = nn.Conv2d(dim, 1, 1, 1, 0)
        self.cls = nn.Conv2d(dim, num_domains, 1, 1, 0)
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

class ResnetGenerator(nn.Module):
    def __init__(self, input_dim:int,
                       output_dim:int,
                       dim:int = 64,
                       num_downs:int = 2,
                       n_blocks:int = 6,
                       norm_layer:str = None,
                       activation:str = None,
                       padding_type:str = None):
        """
        A resnet generator
        Arguments:
            input_dim   : the size of the input feature, usually 3.
            output_dim  : the size of the output feature, usually 3.
            dim         : style code dimention. Default : 256
            num_downs   : number of downsampling layers. Default : 2
            n_blocks    : number of resenet blocks, Default: 6
            norm_layer  : type of normalization layer to be used. Default : None
            activation  : type of activation layer to be used. Default : None
            padding_type: type of padding layer to be used. Default : None
        """
        super(ResnetGenerator, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # initial convolution
        self.encoder.append(ConvBlock(input_dim, dim, 7, 1, 3,
            norm_layer=norm_layer, padding_type=padding_type, activation=activation))
        # downsample layers
        for i in range(num_downs):
            in_dim = dim * 2**(i)
            out_dim = dim * 2**(i+1)
            self.encoder.append(ConvBlock(in_dim, out_dim, 3, 2, 1,
                padding_type=padding_type, norm_layer=norm_layer, activation=activation))
            self.decoder.insert(0, UpsampleBlock(out_dim, in_dim, 3, 2, 1, 1,
                padding_type=padding_type, norm_layer=norm_layer, activation=activation))
        # final conv
        self.decoder.append(ConvBlock(dim, output_dim, 7, 1, 3,
                padding_type=padding_type, norm_layer=norm_layer, activation='tanh'))
        # make sequential
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)