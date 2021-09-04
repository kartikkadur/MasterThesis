import torch
import os
import utils.tools as tools
import sys
import tqdm

from abc import ABC, abstractmethod
from collections import OrderedDict
from utils.tools import AverageMeter


class Model(torch.nn.Module, ABC):
    """
    Base model class that provides the functionality to
    train a model
    """
    def __init__(self):
        super(Model, self).__init__()
        self.is_compiled = False
        self.criterion = OrderedDict()
        self.optimizer = OrderedDict()
        self.train_loss_values = OrderedDict()
        self.val_loss_values = OrderedDict()
    
    @staticmethod
    def add_commandline_arguments(arguments):
        """
        method to update specific command line arguments
        """
        pass

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
            self.optimizer[next(iter(self.optimizer))].zero_grad()
            pred = self.forward()
            loss = self.criterion[next(iter(self.criterion))](pred, self.labels)
            if is_train:
                loss.backward()
                self.optimizer[next(iter(self.optimizer))].step()
            self.update_losses(next(iter(self.criterion)), loss, is_train)
        else:
            sys.exit('call model.compile() method before calling fit')
    
    def _init_optimizer(self, optimizer):
        """
        sets up optimizers
        """
        for opt in optimizer:
            if isinstance(opt, str):
                try:
                    self.optimizer[opt] = tools.module_to_dict(torch.optim)[opt](self.parameters())
                except KeyError as exp:
                    print(f'{opt} is not a valied optimizer')
                    sys.exit(1)
            elif isinstance(optimizer, torch.optim.Optimizer):
                self.optimizer[opt.__name__] = opt
    
    def _init_criterion(self, criterion):
        """
        initialize losses
        """
        for loss in criterion:
            if isinstance(loss, str):
                try:
                    self.criterion[loss] = tools.module_to_dict(torch.nn)[loss]()
                except KeyError as exp:
                    print(f"{loss} is not a valied criterion")
                    sys.exit(1)
            elif isinstance(loss, torch.nn.Module):
                self.criterion[loss.__name__] = loss
    
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
        # initialize optimizer and criterion
        self._init_optimizer(optimizer)
        self._init_criterion(criterion)

        self.loss_names = loss_names if loss_names is not None else [crt if isinstance(crt, str) else crt.__name__ for crt in criterion]
        self._init_losses(self.loss_names)
        self.loss_weights = loss_weights
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
        running_loss = 0
        for epoch in range(start_epoch, epochs):
            dataset = tqdm.tqdm(train_data)
            iteration=0
            for batch in dataset:
                # set inputs
                self.set_inputs(batch)
                # do forward ,backword and update loss values in a loss meter
                self.optimize_parameters()
                if iteration % print_freq == 0:
                    dataset.set_description(f"Epoch: {epoch}, Losses : {tools.param_to_str(**self.get_losses())}")
                iteration += 1
            # do validation
            if validation_data and val_freq and epoch % val_freq == 0:
                self.evaluate(validation_data)
            # save model
            if save_freq and epoch % save_freq == 0:
                save_path = save_path if save_path is not None else os.getcwd()
                self.save(save_path)

    def evaluate(self, val_data, num_val_step=None):
        """
        evaluation function
        """
        self.eval()
        dataset = tqdm.tqdm(val_data)
        iteration = 0
        loss=0
        for batch in dataset:
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

            
