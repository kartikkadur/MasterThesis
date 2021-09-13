import os
import time
import torch
import random
import inspect
import re

import numpy as np
import matplotlib.cm as cm

from inspect import isclass
from collections import OrderedDict
from PIL import Image
#from scipy.misc import imresize


######################
### Helper methods ###
######################
def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if x not in exclude and isclass(getattr(module, x))
                 and getattr(module, x) not in exclude])

def param_to_str(**kwargs):
    """
    returns a string with [parameter_name1: value1, ...] format
    """
    return str([f"{key}: {value}" for key, value in kwargs.items()]).strip('[]')

def tensor_to_image(tensor, imtype=np.uint8):
    """
    converts a torch tensor to a numpy image
    """
    if isinstance(tensor, torch.Tensor):
        image_numpy = tensor.squeeze().data.cpu().float().numpy()
    else:
        image_numpy = tensor
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def tensor_to_mask(tensor, imtype=np.uint8):
    """
    converts a torch tensor to a segmentation mask
    """
    if len(tensor.shape) == 4:
        image_numpy = tensor.squeeze().data.cpu().float().numpy()
    else:
        image_numpy = tensor.data.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)

def mask_to_image(img):
    """
    converts a mask to an image
    """
    return np.stack((img[:,:,0],img[:,:,0],img[:,:,0]),axis=2)

def mask_to_heatmap(tensor, imtype=np.uint8, size=None):
    """
    converts a mask to a heatmap
    """
    image_numpy = tensor[0,0].cpu().float().numpy()
    if size:
        image_numpy=resize(image_numpy,size)
    heatmap_marked = cm.jet(image_numpy)[..., :3] * 255.0
    return heatmap_marked.transpose((2,0,1)).astype(imtype)

def overlay(seed_img, heatmap_marked, alpha=0.5, imtype=np.uint8):
    """
    overlays heatmap on the seed image
    """
    if isinstance(seed_img, torch.Tensor):
        seed_img = seed_img.squeeze().data.cpu().float().numpy()
    img = seed_img * alpha + heatmap_marked * (1. - alpha)
    return img.astype(imtype)

def resize(img, size):
    """
    resize image
    """
    image = Image.fromarray(img)
    image = image.resize(size)
    return np.array(image)

def save_image(image_numpy, image_path):
    """
    saves image to disc as a PIL image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

#########################
#### Helper classes #####
#########################

class AttributeDict(OrderedDict):
    """
    class that provides attribute like access to OrderdDict objects
    """
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        if isinstance(self.get(key), AverageMeter):
            self.get(key).update(value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(AttributeDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)
    
    def add(self, attr_names):
        """
        adds an attribute with AverageMeter value which keeps
        track of the average, sum and current value.
        """
        if not isinstance(attr_names, list):
            attr_names = [attr_names]
        
        for attr in attr_names:
            assert(isinstance(attr, str))
            self[attr] = AverageMeter(attr)

class TimerBlock:
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print("  [{:.3f}{}] {}".format(duration, units, string), flush = True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images