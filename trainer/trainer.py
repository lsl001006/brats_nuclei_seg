import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', 'discriminator_loss', 'generator_loss', writer=self.writer)

        self.valid_metrics = MetricTracker('loss', 'discriminator_loss', 'generator_loss', writer=self.writer)

        # if config.resume is None:
        #     self.model.initialize()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            data = data.cuda().detach()

            self.optimizer.zero_grad()
            d_loss = self.model.update_D(data, target)
            self.optimizer.dis_opt.step()

            g_loss = self.model.update_G()
            self.optimizer.gen_opt.step()

            # output = self.model(data)
            # loss = self.criterion(output, target)
            # loss.backward()
            # self.optimizer.step()
            loss = d_loss.item() + g_loss.item()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            self.train_metrics.update('loss', loss)
            self.train_metrics.update('discriminator_loss', d_loss.item())
            self.train_metrics.update('generator_loss', g_loss.item())

            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=
                image_dis = torchvision.utils.make_grid(self.model.image_display,
                                                        nrow=self.model.image_display.size(0) // 2) / 2 + 0.5

                self.writer.add_image('image', image_dis, batch_idx)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step(epoch)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
