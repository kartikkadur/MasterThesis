import os
import argparse
import model
import dataset
import tools
import torch

class BaseArguments():
    def __init__(self):
        self.initialized=False
        self.model_names = sorted(name for name in model.__dict__
                            if not name.startswith("__") and callable(model.__dict__[name]))
        self.dataset_names = sorted(name for name in dataset.__dict__
                            if not name.startswith("__"))
    
    def initialize(self, parser):
        parser.add_argument('--model', metavar='MODEL', default='DCGAN',
                            choices=self.model_names, help='model architecture: ' + ' | '.join(self.model_names) + ' (default: dcgan)')
        parser.add_argument('--dataset', default='Cifar10', type=str, metavar='DATASET_CLASS', choices=self.dataset_names,
                            help='Specify dataset class for loading (Default: Cifar10)')
        parser.add_argument('--checkpoint', default='', type=str, metavar='CHECKPOINT_PATH',
                            help='path to checkpoint (default: none)')
        parser.add_argument('--save_root', default=os.getcwd(), type=str, metavar='SAVE_ROOT',
                            help='directory to save log files (default: cwd)')
        # learning rate parameters
        parser.add_argument('--lr', '--learning_rate', default=0.0002, type=float,
                            metavar='LR', help='initial learning rate')
        # dataset parameters
        parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='BATCH_SIZE',
                            help='mini batch size (default = 4)')
        parser.add_argument('--num_workers', default=2, type=int, metavar='NUM_WORKERS',
                            help='number of workers to be used for loading batches (default = 2)')
        # optimizer and loss
        parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPTIMIZER',
                            help='Specify optimizer from torch.optim (Default: Adam)')
        parser.add_argument('--loss', default='BCELoss', type=str, metavar='LOSS',
                            help='Specify loss function from torch.nn (Default: BCELoss)')

        parser.add_argument('--print_freq', default=100, type=int, metavar="PRINT_FREQ",
                            help='frequency of printing training status (default: 100)')

        parser.add_argument('--save_freq', type=int, default=20, metavar="SAVE_FREQ",
                            help='frequency of saving intermediate models (default: 20)')
        # dataset root
        parser.add_argument('--root',  metavar="DATASET_ROOT", required=True,
                            help='path to root folder of dataset')
        parser.add_argument('--input_shape',  metavar="INPUT_SHAPE", default=100,
                            help='Shape of latent input')
        parser.add_argument('--preprocess',  metavar="PREPROCESS", nargs='+', default=None,
                            help='preprocessing that is to be applied to the data')
        
        # visualization
        parser.add_argument('--viz', action='store_true',
                            help='weather to save visualization or not')
        self.initialized = True
        return parser

    def parse(self, logger=None):
        if not self.initialized:
            parser = argparse.ArgumentParser(description='Pytorch GAN implementation')
            parser = self.initialize(parser)

        args = parser.parse_args()

        os.makedirs(args.save_root, exist_ok=True)
        defaults, input_arguments = {}, {}
        for key in vars(args):
            defaults[key] = parser.get_default(key)

        for argument, value in sorted(vars(args).items()):
            if value != defaults[argument] and argument in vars(parser.parse_args()).keys():
                input_arguments['--' + str(argument)] = value
                if logger:
                    logger.log('{}: {}'.format(argument, value))
                else:
                    print('{}: {}'.format(argument, value))
        return args