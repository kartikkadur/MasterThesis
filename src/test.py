import time
from arguments.train_arguments import TrainArguments
from dataset import create_dataset
from utils.visualizer import Visualizer

def main():
    args = TrainArguments().parse()
    visualizer = Visualizer(args)
    test_dataset = create_dataset(args)
    model = args.model(args)
    if model.is_compiled:
        if args.epoch:
            model = model.load(args.epoch)
        model.evaluate(val_data=test_dataset, visualizer=visualizer)

if __name__ == '__main__':
    main()