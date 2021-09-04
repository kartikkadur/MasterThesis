import argparse
from .base_arguments import BaseArguments

class TestArguments(BaseArguments):

    def initialize(self, parser):
        parser = BaseArguments.initialize(self, parser)
        # other training parameters
        parser.add_argument('--seed', default=1234, type=int, metavar="SEED",
                            help='seed for initializing training. ')
        return parser