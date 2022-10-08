import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, model, lr=0.0001, weight_decay=0.0001, amsgrad=False):
        self.model = model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, amsgrad=amsgrad)

    def zero_grad(self):
        self.opt.zero_grad()

    def load_state_dict(self, state_dic, train=True):
        if train:
            self.opt.load_state_dict(state_dic['opt'])

    def state_dict(self):
        return {
            'opt': self.opt.state_dict()
        }
