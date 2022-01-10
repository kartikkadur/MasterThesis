import os
import torch
import torchvision
from arguments import TrainArguments, TestArguments
from utils import TimerBlock

class Trainer(object):
    """class used for training"""
    def __init__(self, args):
        self.args = args

    def load_dataset(self):
        with TimerBlock('Loading Dataset and creating dataloaders') as block:
            block.log('Create dataset object')
            dataset = self.args.dataset(args)
            block.log('Create dataloader')
            self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                                        shuffle=True, num_workers=args.num_workers)

    def create_model(self):
        with TimerBlock('Creating model') as block:
            self.model = self.args.model(args)
            block.log('Initialize model')
            self.model.initialize()
            if self.args.resume:
                block.log('Load pretrained weights')
                self.model.load(self.args.resume, self.args.resume_opt)
            if 'train' in self.args.mode:
                block.log('Creating lr schedulers')
                self.model.init_scheduler(args)
    
    def train(self):
        with TimerBlock('Training model') as block:
            global_iter = 0
            total_iters = int(min(self.args.train_n_batch, float(len(self.train_loader))))
            for epoch in range(self.args.start_epoch, self.args.n_epoch):
                loader = iter(self.train_loader)
                for i in range(total_iters):
                    batch = next(loader)
                    # update global iteration
                    global_iter += 1
                    self.model.set_inputs(batch)
                    # optimize parameters by doing backprop
                    self.model.optimize_parameters(global_iter)
                    if global_iter % self.args.print_freq == 0:
                        block.log('\n')
                        block.log(f"Epoch : {epoch}, Total iters: {global_iter}, LR : {self.model.get_current_lr()}")
                        self.model.write_loss(global_iter)
                        loss = self.model.print_losses()
                        block.log(loss)
                    if global_iter % self.args.save_freq == 0:
                        block.log(f'Saving model inside : {self.args.checkpoint_dir}')
                        self.model.save(epoch, global_iter)
                    if global_iter % self.args.display_freq == 0 and global_iter % self.args.d_iter == 0:
                        block.log('Writing images')
                        self.model.save_images(epoch, global_iter)
                    if global_iter % self.args.max_iter == 0.0:
                        return
                if self.args.n_epoch_decay > -1:
                    self.model.update_lr()
            block.log(f'Saving model inside : {self.args.checkpoint_dir}')
            self.model.save(epoch, global_iter)

    def test(self):
        with TimerBlock('testing model') as block:
            testset = self.args.dataset(args)
            test_loader = torch.utils.data.DataLoader(testset)
            for i, batch in enumerate(test_loader):
                self.set_inputs(batch)
                self.model.forward()
                self.save_images(0, i)
                self.compute_metrics()

    def run(self):
        # create dataset
        self.load_dataset()
        # create model
        self.create_model()
        # start training
        if 'train' in self.args.mode:
            self.train()
        else:
            with torch.no_grad():
                self.test()

if __name__ == '__main__':
    args = TrainArguments().parse()
    trainer = Trainer(args)
    trainer.run()
