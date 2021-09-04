import argparse
from .base_arguments import BaseArguments


class TrainArguments(BaseArguments):

    def initialize(self, parser):
        parser = BaseArguments.initialize(self, parser)
        # learning rate scheduler
        parser.add_argument('--lr_scheduler', default='MultiStepLR', type=str,
                            metavar='LR_Scheduler', help='Scheduler for learning' +
                                                        ' rate (only ExponentialLR, MultiStepLR, PolyLR supported.')
        parser.add_argument('--lr_gamma', default=0.1, type=float,
                            help='learning rate will be multipled by this gamma')
        parser.add_argument('--lr_step', default=200, type=int,
                            help='stepsize of changing the learning rate')
        parser.add_argument('--lr_milestones', type=int, nargs='+',
                            default=[250, 450], help="Spatial dimension to " +
                                                    "crop training samples for training")
        # other training parameters
        parser.add_argument('--wd', '--weight_decay', default=0.001, type=float, metavar='WEIGHT_DECAY',
                            help='weight_decay (default = 0.001)')
        parser.add_argument('--seed', default=1234, type=int, metavar="SEED",
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=100, type=int, metavar="EPOCHES",
                            help='number of total epochs to run (default: 100)')
        parser.add_argument('--start_epoch', default=0, type=int, metavar="STARTEPOCH",
                            help='Epoch number to start training with (default: 0)')
        parser.add_argument('--is_training', action='store_false',
                            help='training or not')
        return parser