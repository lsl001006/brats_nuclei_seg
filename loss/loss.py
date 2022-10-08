import torch.nn.functional as F


def nll_loss(output, target, reduction='mean'):
    return F.nll_loss(output, target, reduction=reduction)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)


def dice_loss(output, target):
    smooth = 1.

    oflat = output.view(-1)
    tflat = target.view(-1)
    intersection = (oflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (oflat.sum() + tflat.sum() + smooth))


def bce_and_dice_loss(output, target):
    return bce_loss(output, target) + dice_loss(output, target)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def bce_loss_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output, target)



if __name__ == '__main__':
    import torch
    torch.manual_seed(32)
    output = torch.rand([2, 2, 3, 4])
    target = torch.tensor([[[0, 1, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0]],
                           [[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]]])

    print('dice loss: {:.4f}'.format(dice_loss(output[:,1,:,:], target.float())))
    print('ce loss: {:.4f}'.format(cross_entropy_loss(output, target)))
    print('logsoftmax + nll loss: {:.4f}'.format(nll_loss(F.log_softmax(output, dim=1).float(), target)))
    print('bce loss: {:.4f}'.format(bce_loss(F.softmax(output, dim=1)[:,1,:,:], target.float())))
    print('bce loss with digits: {:.4f}'.format(bce_loss_with_logits(output[:,1,:,:], target.float())))