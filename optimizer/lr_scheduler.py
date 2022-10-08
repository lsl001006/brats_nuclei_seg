from torch.optim.lr_scheduler import _LRScheduler, StepLR


class StepLRs(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma):
        self.g_scheduler = StepLR(optimizer.gen_opt, step_size=step_size, gamma=gamma)
        self.d_scheduler = StepLR(optimizer.dis_opt, step_size=step_size, gamma=gamma)



    def step(self, step):
        self.g_scheduler.step(step)
        self.d_scheduler.step(step)
