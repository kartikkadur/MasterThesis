import os
import torch
import torch.nn as nn
import numpy as np
import torchvision

from PIL import Image
from abc import ABC, abstractmethod
from utils import AttributeDict, save_image
from models.core import networks
from models.core.functions import init_net
from models.core.functions import get_scheduler
from tensorboardX import SummaryWriter

class Model(ABC, nn.Module):
    """
    This is the base model class. Subclass this class for create different model implemntations.
    This class provides attributes for adding models, optimizers and schedulers.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AttributeDict()
        self.optimizer = AttributeDict()
        self.scheduler = AttributeDict()
        self.loss = AttributeDict()
        if 'train' in args.mode:
            self.writer = SummaryWriter(log_dir=args.logdir)
        self.print_loss = []

    @abstractmethod
    def set_inputs(self, inputs):
        """this method is overloaded to set batch inputs"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """method for optimizing one batch of inputs"""
        pass

    def initialize(self):
        init_type = None if self.args.resume else self.args.init_type
        # initialize model
        for net in self.model:
            self.model[net] = init_net(self.model[net], init_type=init_type, gpu_ids=self.args.gpu_ids, device=self.device)
        # load checkpoint if provided
        if self.args.resume or self.args.resume_opt:
            self.load(self.args.resume, self.args.resume_opt)
        # initialize lr scheduler
        self.args.last_iter = -1 if self.args.resume_opt is None else self.args.last_iter
        if 'train' in self.args.mode:
            self.init_scheduler()

    def init_scheduler(self):
        for opt in self.optimizer:
            self.scheduler[opt] = get_scheduler(self.optimizer[opt], self.args, self.args.last_iter)

    def get_current_lr(self):
        curr_lrs = {}
        for opt in self.optimizer:
            curr_lrs[opt] = self.optimizer[opt].param_groups[0]['lr']
        return curr_lrs

    def update_lr(self):
        for net in self.model:
            self.scheduler[net].step()

    def save(self, it):
        model_state = {}
        opt_state = {}
        # model state
        for net in self.model:
            model_state[net] = self.model[net].state_dict()
        path = os.path.join(self.args.checkpoint_dir, f"model_{it}.ckpt")
        torch.save(model_state, path)
        # opt state
        for opt in self.optimizer:
            opt_state[opt] = self.optimizer[opt].state_dict()
        path = os.path.join(self.args.checkpoint_dir, f"opt_{it}.ckpt")
        torch.save(opt_state, path)

    def load(self, checkpoint, opt_ckpt=None):
        if checkpoint is not None:
            ckpt = torch.load(checkpoint)
            for net in ckpt:
                if net in self.model.keys():
                    print(f"Loading checkpoint for : {net}")
                    self.model[net].load_state_dict(ckpt[net])
                else:
                    print(f"Checkpoint for {net} network is not found.")
        if opt_ckpt is not None:
            ckpt = torch.load(opt_ckpt)
            for opt in ckpt:
                if opt in self.optimizer.keys():
                    print(f"Loading checkpoint for {opt} optimizer.")
                    self.optimizer[opt].load_state_dict(ckpt[opt])
                else:
                    print(f"Checkpoint for {opt} optimizer is not found.")

    def save_images(self, it):
        visuals = self.compute_visuals()
        img_filename = os.path.join(self.args.display_dir, f'gen_{it}.jpg')
        if isinstance(visuals, torch.Tensor):
            torchvision.utils.save_image(visuals / 2 + 0.5, img_filename, nrow=1)
        else:
            save_image(visuals, img_filename)

    def write_loss(self, global_iter):
        for loss in self.loss:
            self.writer.add_scalar(loss, self.loss[loss], global_iter)

    def print_losses(self):
        loss_to_print = {}
        for loss in self.loss:
            if loss in self.print_loss:
                loss_to_print[loss] = self.loss[loss]
        return loss_to_print

    def compute_metrics(self):
        pass