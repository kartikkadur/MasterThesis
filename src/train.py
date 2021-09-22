import time
from arguments.train_arguments import TrainArguments
from dataset import create_dataset
from utils.visualizer import Visualizer

def main():
    args = TrainArguments().parse()
    visualizer = Visualizer(args)
    train_dataset = create_dataset(args)
    model = args.model(args)
    if args.load_checkpoint:
        model.load(args.load_checkpoint)
    if model.is_compiled:
        model.fit(train_data=train_dataset, visualizer=visualizer)

if __name__ == '__main__':
    main()