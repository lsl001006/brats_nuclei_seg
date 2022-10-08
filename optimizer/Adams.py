import torch
from torch.optim.optimizer import Optimizer


class Adams(Optimizer):

    def __init__(self, model, lr=0.01, weight_decay=0.0001, amsgrad=False):
        self.G = model.G
        self.D = model.D
        G_trainable_params = filter(lambda p: p.requires_grad, self.G.parameters())
        D_trainable_params = filter(lambda p: p.requires_grad, self.D.parameters())
        self.gen_opt = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay, amsgrad=amsgrad)
        self.dis_opt = torch.optim.Adam(self.D.parameters(), lr=lr*0.01, betas=(0.5, 0.999), weight_decay=weight_decay, amsgrad=amsgrad)

    def zero_grad(self):
        self.gen_opt.zero_grad()
        self.dis_opt.zero_grad()

    def load_state_dict(self, state_dic, train=True):
        if train:
            self.gen_opt.load_state_dict(state_dic['gen_opt'])
            self.dis_opt.load_state_dict(state_dic['dis_opt'])

    def state_dict(self):
        return {
            'gen_opt': self.gen_opt.state_dict(),
            'dis_opt': self.dis_opt.state_dict()
        }
