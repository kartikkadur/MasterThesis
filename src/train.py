import os
import torch
import torchvision
from arguments import TrainArguments, TestArguments
from utils import TimerBlock

class Trainer(object):
    """class used for training"""
    def __init__(self):
        pass

    def load_dataset(self, args):
        with TimerBlock('Loading Dataset and creating dataloaders') as block:
            block.log('Create dataset object')
            dataset = args.dataset(args)
            block.log('Create dataloader')
            dataloader = torch.utils.data.DataLoader(dataset, 
                                                     batch_size=args.batch_size, 
                                                     shuffle=False,
                                                     num_workers=args.num_workers)
        return dataloader

    def create_model(self, args):
        with TimerBlock('Creating model') as block:
            model = args.model(args)
            block.log('Initialize model')
            model.initialize()
        return model
    
    def train(self, args, model, dataloader):
        with TimerBlock('Training model') as block:
            global_iter = args.last_iter + 1 if args.resume_opt is not None else 0
            iterations = min(args.n_iters, args.max_iter)
            block.log(f"Running for {iterations} iterations")
            while True:
                for it, batch in enumerate(dataloader):
                    # update learning rate
                    model.update_lr()
                    # set inputs
                    model.set_inputs(batch)
                    # optimize parameters by doing backprop
                    model.optimize_parameters(global_iter)
                    if global_iter % args.print_freq == 0:
                        block.log('\n')
                        block.log(f"Iteration: {global_iter}, LR : {model.get_current_lr()}")
                        model.write_loss(global_iter)
                        block.log(model.print_losses())
                    if global_iter % args.save_freq == 0:
                        block.log(f'Saving model inside : {args.checkpoint_dir}')
                        model.save(global_iter)
                    if global_iter % args.display_freq == 0 and global_iter % args.d_iter == 0:
                        block.log('Writing images')
                        model.save_images(global_iter)
                    global_iter += 1
                    if global_iter > iterations:
                        block.log(f'Saving model inside : {args.checkpoint_dir}')
                        model.save(global_iter)
                        block.log("Finished training")
                        return

    def run(self, args):
        # create dataloader
        dataloader = self.load_dataset(args)
        # create model
        model = self.create_model(args)
        # start training
        self.train(args, model, dataloader)

if __name__ == '__main__':
    args = TrainArguments().parse()
    trainer = Trainer()
    trainer.run(args)