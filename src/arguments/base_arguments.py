import argparse
import os
import torch
import models
import dataset

from datetime import datetime
from models import networks
from inspect import isclass
from dataset.base_dataset import Dataset
from models.base_model import Model
from utils.tools import module_to_dict


def get_modules(module, superclass=None, filter=None):
    if superclass:
        modules = dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x)) and issubclass(getattr(module, x), superclass)]).keys()
    else:
        modules = dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x))]).keys()
    if filter:
        modules = [m for m in modules if filter in m]
    return modules


class Arguments():
    """
    This class defines arguments used during both training and test time.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common arguments that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='root folder of the dataset')
        parser.add_argument('--name', type=str, default=f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='AttentionGANModel', choices=get_modules(models, superclass=Model), help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, choices=get_modules(networks, filter='Discriminator'), default='NLayerDiscriminator', help='specify discriminator architecture.')
        parser.add_argument('--netG', type=str, choices=get_modules(networks, filter='Generator'), default='ResnetGenerator', help='specify generator architecture.')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==NLayerDiscriminator')
        parser.add_argument('--n_blocks_G', type=int, default=6, help='number of blocks to be used in Generator model')
        parser.add_argument('--norm_layer', type=str, default='InstanceNorm2d', choices= ['BatchNorm2d', 'InstanceNorm2d'], help='instance normalization or batch normalization.')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization.')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        parser.add_argument('--dataset', type=str, default='MultiClassDataset', choices=get_modules(dataset, superclass=Dataset), help='chooses how datasets are loaded.')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--shuffle', action='store_true', help='if true, takes the batches randomly, else takes them in serial fashion')
        parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: args.name = args.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_arguments(self):
        """
        gathers the commandline arguments and returnes the parsed arguments
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic arguments
        args, _ = parser.parse_known_args()

        # modify model-related parser arguments
        model_argument_setter = module_to_dict(models)[args.model].modify_commandline_arguments
        parser = model_argument_setter(parser, self.isTrain)
        args, _ = parser.parse_known_args()

        # modify dataset-related parser arguments
        dataset_argument_setter = module_to_dict(dataset)[args.dataset].modify_commandline_arguments
        parser = dataset_argument_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_arguments(self, args):
        """
        Print and save arguments
        """
        message = ''
        message += '----------------- arguments ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(args.checkpoints_dir, args.name)
        #os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, '{}_args.txt'.format(args.mode))
        #with open(file_name, 'wt') as args_file:
        #    args_file.write(message)
        #    args_file.write('\n')

    def parse(self):
        """Parse our arguments, create checkpoints directory suffix, and set up gpu device."""
        args = self.gather_arguments()
        args.isTrain = self.isTrain

        # process args.suffix
        if args.suffix:
            suffix = ('_' + args.suffix.format(**vars(args))) if args.suffix != '' else ''
            args.name = args.name + suffix

        self.print_arguments(args)

        # set gpu ids
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])

        # set model and dataset classes
        args.netG = module_to_dict(networks)[args.netG]
        args.netD = module_to_dict(networks)[args.netD]
        args.model = module_to_dict(models)[args.model]
        args.dataset = module_to_dict(dataset)[args.dataset]
        args.norm_layer = module_to_dict(torch.nn)[args.norm_layer]

        self.args = args
        return self.args