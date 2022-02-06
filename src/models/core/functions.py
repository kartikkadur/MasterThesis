import torch
import torch.nn as nn
import functools

from inspect import isclass
from torch.optim import lr_scheduler
from torch.nn import init
from models.core.norm import AdaptiveInstanceNorm
from models.core.norm import LayerNorm

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
            norm_layer = functools.partial(AdaptiveInstanceNorm)
        else:
            raise NotImplementedError(f"norm type '{norm_layer}' is not supported at the moment")
    elif not (norm_layer == None) and isclass(norm_layer) and not issubclass(norm_layer, nn.Module):
        raise ValueError(f"parameter type of norm_layer should be one of 'str' or 'nn.Module' class, but got {norm_layer}.")
    return norm_layer

def get_activation_layer(activation=None):
    """returns the activation layer class or None"""
    if isinstance(activation, str):
        if activation == 'relu':
            activation = functools.partial(nn.ReLU, inplace=True)
        elif activation == 'lrelu':
            activation = functools.partial(nn.LeakyReLU, inplace=True)
        elif activation == 'tanh':
            activation = functools.partial(nn.Tanh)
        elif activation == 'sigmoid':
            activation = functools.partial(nn.Sigmoid)
        else:
            raise NotImplementedError(f"activation type '{activation}' is not supported at the moment")
    elif not (activation == None) and isclass(activation) and not issubclass(activation, nn.Module):
        raise ValueError(f"parameter type of activation should be one of 'str' or 'nn.Module', but got {type(activation)}.")
    return activation

def get_padding_layer(padding_type=None):
    """returns the padding layer class"""
    padding_layer=None
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

def get_scheduler(optimizer, args, cur_it=-1):
    if args.lr_policy == 'lambda':
        def lambda_rule(it):
            lr_l = 1.0 - max(0, it - args.n_iter_decay) / float(args.n_iters - args.n_iter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_it)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.n_iter_decay, gamma=0.1, last_epoch=cur_it)
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