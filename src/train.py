import time
from arguments.train_arguments import TrainArguments
from dataset import create_dataset
from models import create_model
#from util.visualizer import Visualizer

from models.attentionGAN import AttentionGANModel

def main():
    args = TrainArguments().parse()   # get training options
    dataset = create_dataset(args)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    #model = create_model(opt)      # create a model given opt.model and other options
    model = AttentionGANModel(args)
    #model.setup(opt)               # regular setup: load and print networks; create schedulers
    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    model.fit()

if __name__ == '__main__':
    args = TrainArguments().parse()