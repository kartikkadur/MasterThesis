import time
from arguments.test_arguments import TestArguments
from dataset import create_dataset
from utils.visualizer import Visualizer

def main():
    args = TestArguments().parse()
    visualizer = Visualizer(args)
    test_dataset = create_dataset(args)
    model = args.model(args)
    if model.is_compiled:
        if args.epoch:
            model = model.load(args.epoch)
        model.evaluate(val_data=test_dataset, visualizer=visualizer)

if __name__ == '__main__':
    main()