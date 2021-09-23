import torch
import os
import utils.tools as tools
import sys
import tqdm

from model import networks
from abc import ABC, abstractmethod
from collections import OrderedDict
from utils import metrics
from utils.tools import AttributeDict

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
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_dir = os.path.join(args.checkpoints_dir, args.name)
        # create save directories
        os.makedirs(self.save_dir, exist_ok=True)
        # networks used
        self.models = AttributeDict()
        self.criterion = AttributeDict()
        self.optimizer = AttributeDict()
        # losses that is calculated
        self.loss = AttributeDict()
        self.val_loss = AttributeDict()
        # metrics
        self.metrics = metrics.Metrics()
        #lossese to print
        self.print_losses = []
        # visuals
        self.visuals = []
    
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
    
    @abstractmethod
    def optimize_parameters(self, is_train=True):
        """
        handle one backward pass and weights update.
        The training loop is automatically handled by model.fit() function.
        """
        pass

    def _init_optimizer(self, optimizer):
        """
        sets up optimizers
        """
        for i, opt in enumerate(optimizer):
            if isinstance(opt, str):
                try:
                    self.optimizer[opt] = tools.module_to_dict(torch.optim)[opt](self.parameters(),
                                                                                 lr=self.args.lr)
                except KeyError as exp:
                    tqdm.tqdm.write(f'{opt} is not a valied optimizer')
                    sys.exit(1)
            elif isinstance(opt, torch.optim.Optimizer):
                if self.optimizer.has_key(opt.__name__):
                    self.optimizer[f'{opt.__name__}_{i}'] = opt(self.parameters(), lr=self.args.lr)
                else:
                    self.optimizer[opt.__name__] = opt(self.parameters(), lr=self.args.lr)
    
    def _init_criterion(self, criterion):
        """
        set up losses
        """
        for i, loss in enumerate(criterion):
            if isinstance(loss, str):
                try:
                    self.criterion[loss] = tools.module_to_dict(torch.nn)[loss]()
                except KeyError as exp:
                    tqdm.tqdm.write(f"{loss} is not a valied criterion")
                    sys.exit(1)
            elif isinstance(loss, torch.nn.Module):
                if self.criterion.has_key(loss.__name__):
                    self.criterion[f'{loss.__name__}_{i}'] = loss
                else:
                    self.criterion[loss.__name__] = loss
    
    def _init_loss(self, losses):
        """
        initalize the losses that is being logged
        """
        if not losses:
            losses = ['loss']
        # initialize losses
        self.loss.add(losses)
        self.val_loss.add(losses)
    
    def _init_metrics(self, metrics):
        """
        initialize metrics that are calculated.
        Currently supported metrics : [acc]
        """
        self.metrics.add(metrics)

    def compile(self, optimizer=None,
                      criterion=None,
                      loss_names = None,
                      metrics=None,
                      loss_weights=None,
                      weighted_metrics=None):
        """
        setup criterion and optimizer
        """
        # initialize optimizer and criterion if passed
        if optimizer:
            optimizer = [optimizer] if not isinstance(optimizer, list) else optimizer
            self._init_optimizer(optimizer)
        
        if criterion:
            criterion = [criterion] if not isinstance(criterion, list) else criterion
            self._init_criterion(criterion)
        # check if optimizer and criterion has been added
        assert(len(self.optimizer.keys()) > 0)
        assert(len(self.criterion.keys()) > 0)
        # inti schedulers
        self.schedulers = [networks.get_scheduler(self.optimizer[opt], self.args) for opt in self.optimizer]
        # assign model and loss names
        if loss_names:
            self._init_loss(loss_names)
        # initialize metrics
        if metrics:
            self._init_metrics(metrics)
        # set compiled as true
        self.is_compiled = True

    def load(self, path):
        """
        load weights from checkpoint
        """
        assert os.path.exists(path), f"The path {path} does't exist. Provide a correct checkpoint path"
        checkpoint = torch.load(path)
        for name, model in self.models.items():
            model.load_state_dict(checkpoint[name])

    def save(self, epoch):
        """
        save model weights
        """
        checkpoint = dict()
        for name, model in self.models.items():
            checkpoint[name] = model.state_dict()
        torch.save(checkpoint, os.path.join(self.args.checkpoints_dir, f"{epoch}"))
        tqdm.tqdm.write(f"Model saved in : {self.args.checkpoints_dir}")

    def update_learning_rate(self):
        """
        Update learning rates for all the networks;
        """
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()
        curr_lr = str([f"{key}: {value.param_groups[0]['lr']}" \
                        for key, value in self.optimizer.items()]).strip('[]')
        return curr_lr

    def fit(self, train_data=None, validation_data=None, visualizer=None):
        """
        performes training loop operations
        """
        assert train_data != None, f'train_data cannot be of type None'
        # start training
        self.train()
        # get current learning rate for displaying
        curr_lr = str([f"{key}: {value.param_groups[0]['lr']}" \
                        for key, value in self.optimizer.items()]).strip('[]')
        # get steps per epoch
        steps_per_epoch = self.args.steps_per_epoch if self.args.steps_per_epoch else len(train_data)
        # training loop
        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            with tqdm.tqdm(total=steps_per_epoch, position=0) as discription_bar:
                with tqdm.tqdm(train_data, position=1) as data_bar:
                    for iteration, batch in enumerate(data_bar):
                        # set inputs
                        self.set_inputs(batch)
                        # do forward ,backword and update loss values in a loss meter
                        self.optimize_parameters()
                        # update discription bar
                        if iteration % self.args.print_freq == 0:
                            message = self.print_current_infos(epoch, iteration, curr_lr)
                            discription_bar.set_description(message)
                        # save visuals
                        if iteration % self.args.display_freq == 0 and visualizer:
                            visualizer.reset()
                            self.compute_visuals()
                            visualizer.display_current_results(self.get_visuals(),
                                                                epoch,
                                                                iteration % self.args.update_html_freq == 0)
                        data_bar.set_description(f"Epoch: {epoch}, "\
                            f"Learning Rate: {curr_lr}")
                    # do validation
                    if validation_data and self.args.val_freq and epoch % self.args.val_freq == 0:
                        self.evaluate(validation_data)
                    # save model
                    if self.args.save_freq and epoch % self.args.save_freq == 0:
                        self.save(epoch)
                    curr_lr = self.update_learning_rate()

    def evaluate(self, val_data, num_val_step=None, visualizer=None):
        """
        evaluation function
        """
        self.eval()
        num_val_step = num_val_step if not num_val_step else len(val_data)
        with tqdm.tqdm(val_data, position=0) as data_bar:
            iteration=0
            for batch in data_bar:
                iteration += 1
                # set inputs
                self.set_inputs(batch)
                # do one forward only
                self.optimize_parameters(is_train=False)
                if visualizer:
                    visualizer.reset()
                    self.compute_visuals()
                    visualizer.display_current_results(self.get_visuals(),
                                                        iteration,
                                                        False)
                data_bar.set_description(f"Validation Loss : {tools.param_to_str(**self.get_losses(is_train=False))}")
    
    def get_losses(self, is_train=True):
        """
        returns a OrderedDict of losses that is being logged in a LossMeter
        """
        errors = OrderedDict()
        if is_train:
            loss_values = self.loss
        else:
            loss_values = self.val_loss

        for loss in self.print_losses:
            errors[loss] = loss_values[loss].avg
        return errors
    
    def get_metrics(self, is_train=True):
        """
        return metrics
        """
        metrics = OrderedDict()

        for metric in self.metrics:
            metrics[metric] = self.metrics[metric].acc
        return metrics

    def get_visuals(self):
        """
        returns the visuals that is added to self.visuals attribte and saves them.
        """
        visuals = OrderedDict()
        for image in self.visuals:
            if isinstance(image, str):
                visuals[image] = getattr(self, image)
        return visuals

    def print_current_infos(self, epoch, iteration, curr_lr):
        loss = ''
        metrics = ''
        if self.print_losses is not []:
            loss = tools.param_to_str(**self.get_losses())
        if self.print_metrics is not []:
            metrics = tools.param_to_str(**self.get_metrics())
        message = f'Loss: {loss} | Metrics: {metrics}'
        # do logging if required
        if self.args.save_logs:
            infos = f'Epoch: {epoch}, Iteration: {iteration}, LR: {curr_lr} '\
                    f'{message}'
            self.args.logger.info(infos)
        return message

    def compute_visuals(self):
        """
        method for computing additional visuals that is to be saved.
        """
        pass

    def set_requires_grad(self, models, grad=False):
        """
        sets the gradient parameter
        """
        models = models if isinstance(models, list) else [models]
        for net in models:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = grad

            
