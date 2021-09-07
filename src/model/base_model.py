import torch
import os
import utils.tools as tools
import sys
import tqdm

from models import networks
from abc import ABC, abstractmethod
from collections import OrderedDict
from utils.tools import AverageMeter


class Model(torch.nn.Module, ABC):
    """
    Base model class that provides the functionality to
    train a model
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.is_compiled = False

        self.args = args
        self.isTrain = args.isTrain
        self.gpu_ids = args.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(args.checkpoints_dir, args.name)
        # create save directories
        os.makedirs(self.save_dir, exist_ok=True)
        # optimizer and criterion lists
        self.criterion = []
        self.optimizers = []
        # train and val losses
        self.train_loss_values = OrderedDict()
        self.val_loss_values = OrderedDict()
    
    @staticmethod
    def modify_commandline_arguments(parser, isTrain=True):
        """
        method to update specific command line arguments
        """
        return parser

    @abstractmethod
    def set_inputs(self):
        """
        set initial inputs to the model
        """
        pass

    @abstractmethod
    def forward(self):
        """
        handle single forward pass on the input
        """
        pass

    def optimize_parameters(self, is_train=True):
        """
        handle backward pass and weights update.
        Note: This is a default implementation.
        Override this method for specific implementation of optimizing the parameters.
        """
        if self.is_compiled:
            self.optimizers[0].zero_grad()
            pred = self.forward()
            loss = self.criterion[0](pred, self.labels)
            if is_train:
                loss.backward()
                self.optimizers[0].step()
            self.update_losses(self.loss_names[0], loss, is_train)
        else:
            sys.exit('call model.compile() method before calling fit')
    
    def _init_optimizer(self, optimizer):
        """
        sets up optimizers
        """
        for opt in optimizer:
            if isinstance(opt, str):
                try:
                    self.optimizers += [tools.module_to_dict(torch.optim)[opt](self.parameters())]
                except KeyError as exp:
                    tqdm.tqdm.write(f'{opt} is not a valied optimizer')
                    sys.exit(1)
            elif isinstance(opt, torch.optim.Optimizer):
                self.optimizers += [opt]
    
    def _init_criterion(self, criterion):
        """
        set up losses
        """
        for loss in criterion:
            if isinstance(loss, str):
                try:
                    self.criterion += [tools.module_to_dict(torch.nn)[loss]()]
                except KeyError as exp:
                    tqdm.tqdm.write(f"{loss} is not a valied criterion")
                    sys.exit(1)
            elif isinstance(loss, torch.nn.Module):
                self.criterion += [loss]
    
    def _init_losses(self, losses):
        """
        initalize the losses that is being logged
        """
        for loss in losses:
            self.train_loss_values[loss] = AverageMeter(loss)
            self.val_loss_values[loss] = AverageMeter(loss)

    def compile(self, optimizer=None,
                      criterion=None,
                      loss_names = None,
                      model_names=None,
                      metrics=None,
                      loss_weights=None,
                      weighted_metrics=None):
        """
        setup criterion and optimizer
        """
        assert optimizer != None, "parameter 'optimizer' cannot be None type"
        assert criterion != None, "parameter 'criterion' cannot be None type"
        # convert to a list
        criterion = [criterion] if not isinstance(criterion, list) else criterion
        optimizer = [optimizer] if not isinstance(optimizer, list) else optimizer
        
        metrics = metrics if metrics is not None else ['acc', 'loss']
        # initialize optimizer, schedulers and criterion
        self._init_optimizer(optimizer)
        self._init_criterion(criterion)
        
        self.schedulers = [networks.get_scheduler(optimizer, self.args) for optimizer in self.optimizers]

        # assign model and loss names
        self.model_names = model_names if model_names is not None else []
        self.loss_names = loss_names if loss_names is not None else ['loss']
        self._init_losses(self.loss_names)
        # set compiled as true
        self.is_compiled = True

    def load(self, path):
        """
        load weights from checkpoint
        """
        assert os.path.exists(path), f"The path {path} does't exist. Provide a correct checkpoint path"
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)

    def save(self, save_path):
        """
        save model weights
        """
        tqdm.tqdm.write(f"Model saved in : {save_path}")
        torch.save(self.state_dict(), save_path)
    
    def update_losses(self, name, value, is_train=True):
        """
        updates metrics
        """
        if is_train:
            self.train_loss_values[name].update(value)
        else:
            self.val_loss_values[name].update(value)

    def update_learning_rate(self):
        """
        Update learning rates for all the networks;
        """
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        tqdm.tqdm.write('learning rate = %.7f' % lr)

    def fit(self, train_data=None,
                validation_data=None,
                epochs=None,
                start_epoch=0,
                steps_per_epoch=None,
                val_freq=None,
                print_freq=1,
                save_freq=None,
                save_path=None):
        """
        performes training loop operations
        """
        assert train_data != None, f'train_data cannot be of type None'
        assert epochs != None, 'epoch cannot be of type None'

        self.train()
        steps_per_epoch = steps_per_epoch if steps_per_epoch else len(train_data)
        tqdm_iter = tqdm.tqdm(train_data)
        for epoch in range(start_epoch, epochs):
            iteration = 0
            for batch in tqdm_iter:
                iteration += 1
                if iteration == steps_per_epoch:
                    break
                # set inputs
                self.set_inputs(batch)
                # do forward ,backword and update loss values in a loss meter
                self.optimize_parameters()
                if iteration % print_freq == 0:
                    tqdm_iter.set_description(f"Epoch: {epoch}, Losses : {tools.param_to_str(**self.get_losses())}")
                iteration += 1
            # do validation
            if validation_data and val_freq and epoch % val_freq == 0:
                self.evaluate(validation_data)
            # save model
            if save_freq and epoch % save_freq == 0:
                save_path = save_path if save_path is not None else os.getcwd()
                self.save(save_path)
            #self.update_learning_rate()

    def evaluate(self, val_data, num_val_step=None):
        """
        evaluation function
        """
        self.eval()
        num_val_step = num_val_step if not num_val_step else len(val_data)
        for iteration in tqdm(range(num_val_step)):
            batch = val_data[iteration]
            iteration += 1
            # set inputs
            self.set_inputs(batch)
            # do one forward only
            self.optimize_parameters(is_train=False)
            dataset.set_description(f"Validation Loss : {tools.param_to_str(**self.get_losses(is_train=False))}")
    
    def get_losses(self, is_train=True):
        """
        returns a OrderedDict of losses that is being logged in a LossMeter
        """
        errors = OrderedDict()
        if is_train:
            loss_values = self.train_loss_values
        else:
            loss_values = self.val_loss_values

        for loss in loss_values:
            errors[loss] = loss_values[loss].avg
        return errors
    
    def set_requires_grad(self, models, grad=False):
        """
        sets the gradient parameter
        """
        models = models if isinstance(models, list) else [models]
        for net in models:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = grad

            
