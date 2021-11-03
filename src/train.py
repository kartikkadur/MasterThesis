import time
from arguments.train_arguments import TrainArguments
from dataset import create_dataset
from model import create_model
from utils.visualizer import Visualizer

def main():
    args = TrainArguments().parse()
    #visualizer = Visualizer(args)
    train_dataset = create_dataset(args)
    model = create_model(args)
    if model.is_compiled:
        model.fit(train_data=train_dataset)

if __name__ == '__main__':
    main() 