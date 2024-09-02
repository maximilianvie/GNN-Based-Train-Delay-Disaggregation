import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

@register_loss('l2_losses')
def l2_losses(pred, true):
    if cfg.model.loss_fun == 'l2':
        l2_loss = nn.MSELoss()  # L2 Loss
        loss = l2_loss(pred, true)
        return loss, pred
    elif cfg.model.loss_fun == 'smoothl2':
        smooth_l2_loss = nn.SmoothL1Loss(beta=1.0)  # Setting beta=1.0 approximates L2 Loss
        loss = smooth_l2_loss(pred, true)
        return loss, pred