import time
import torch

from arguments.test_arguments import TestArguments
from dataset import create_dataset
from utils.visualizer import Visualizer

def main():
    args = TestArguments().parse()
    visualizer = Visualizer(args)
    test_dataset = args.dataset(args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    model = args.model(args)
    if model.is_compiled:
        if args.load_checkpoint:
            model.load(args.load_checkpoint)
        model.evaluate(val_data=test_loader, visualizer=visualizer)

if __name__ == '__main__':
    main()