import os
import torch
import torchvision
from arguments import TestArguments
from utils import TimerBlock, save_images
from dataset import ImageFolder

class Sampler(object):
    '''Applies the model to a sample set of images or a video'''
    def __init__(self, args):
        self.args = args
        if args.trg_class is None:
            self.args.num_domains = 5
        else:
            self.args.num_domains = 1


    def load_data(self):
        with TimerBlock('Loading data') as block:
            dataset = ImageFolder(args)
            block.log('Create dataloader')
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                                                    num_workers=self.args.num_workers)

    def create_model(self):
        with TimerBlock('Creating model') as block:
            self.model = self.args.model(args)
            block.log('Initialize model')
            self.model.initialize()
            if self.args.resume:
                block.log('Load pretrained weights')
                self.model.load(self.args.resume)

    def run(self):
        with TimerBlock('Generating Images') as block:
            for domin in range(self.args.num_domains):
                for i, batch in enumerate(self.dataloader):
                    img = batch
                    with torch.no_grad():
                        images = self.model.forward_random(img, domain)
                        names = [str(i) for i in range(self.args.num_domains)]
                    
