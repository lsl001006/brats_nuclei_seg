import argparse
import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm
import data_loader.data_loaders as module_data
import loss.loss as module_loss
import metric.metric_seg as module_metric
from metric.metric_seg import hausdorff95
import models.UNet as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['test_data_loader']['type'])(
        config['test_data_loader']['args']['h5_filepath'],
        batch_size=config['test_data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    test_metric_names = config['test_metrics']
    hausdorff_flag = False
    if 'hausdorff95' in test_metric_names:
        hausdorff_flag = True
        hd95_metric = 0.0
        N_nonzero_pred = 0
        test_metric_names.remove('hausdorff95')
    metric_fns = [getattr(module_metric, met) for met in test_metric_names]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    save_dir = '{:s}/test_results'.format(str(config.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, (data, target, img_names) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if target.max() == 255:
                target = target / 255
                # target /= 255

            # computing loss, metrics on test set
            loss = loss_fn(torch.sigmoid(output), target.float())
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

            for k, metric in enumerate(metric_fns):
                total_metrics[k] += metric(output, target) * batch_size
            

            if hausdorff_flag:
                hd95_val, n = hausdorff95(output, target)
                # print('{:.2f}\t{:d}'.format(hd95_val, n))
                hd95_metric += hd95_val
                N_nonzero_pred += n

            pred = torch.sigmoid(output) > 0.5
            for k in range(data.size(0)):
                for t, m, s in zip(data[k], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):
                    t.mul_(s).add_(m)
                img_concat = torch.cat((data[k][0], target[k][0].float(), pred[k][0].float()), dim=1)
                save_img_name = '{:s}/{:s}'.format(save_dir, img_names[k])
                # for i, met in enumerate(metric_fns):
                    # save_img_name += '-{:s}-{:.4f}'.format(met.__name__, total_metrics[i].item())
                save_image(img_concat, save_img_name+'.png')

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    result_dict = {met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)}
    if hausdorff_flag:
        result_dict['hausdorff95'] = hd95_metric / N_nonzero_pred
        print('N_total: {:d}\nN_haus: {:d}'.format(n_samples, N_nonzero_pred))
    log.update(result_dict)
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
