import os
import tools
import torch

from arguments.train_arguments import TrainArguments

def create_dataloaders(block, args):
    block.log("Creating train and valid  data loaders")
    trainset = args.dataset_class(args, is_training=True)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True)
    valset = args.dataset_class(args, is_training=False)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            drop_last=True)
    return trainloader, valloader

def set_random_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def train(block, args):
    # build dataset
    trainloader, valloader = create_dataloaders(block, args)
    block.log(f"Number of training samples : {len(trainloader)}")
    block.log(f"Number of validation samples : {len(valloader)}")
    # build and setup model
    model = args.network_class(args)
    model.setup()

def main():
    with tools.TimerBlock("Get Arguments") as block:
        args = TrainArguments().parse(block)

    with tools.TimerBlock("Setting random seed"):
        set_random_seed(args)
    
    with tools.TimerBlock("Start training") as block:
        args.logger = block
        train(block, args)

if __name__ == '__main__':
    main()