def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


import cv2
import torch
import numpy as np

    
def heatmap(x_show, name=None):
    x_show = torch.mean(x_show, dim=1, keepdim=True).data.cpu().numpy().squeeze()
    x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)

    x_show = np.uint8(255 * x_show)
    x_show = cv2.applyColorMap(x_show, cv2.COLORMAP_JET)
    x_show = cv2.resize(x_show, (320, 320))

    if name is not None:
        cv2.imwrite('./heatmap/' + name + '.jpg', x_show)
        print('Save heat map in:', './heatmap/' + name + '.jpg')
