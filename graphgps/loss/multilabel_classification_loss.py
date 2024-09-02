import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('multilabel_cross_entropy')
def multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss.
    """
    if cfg.dataset.task_type == 'classification_multilabel':
        if cfg.model.loss_fun != 'cross_entropy':
            raise ValueError("Only 'cross_entropy' loss_fun supported with "
                             "'classification_multilabel' task_type.")
        bce_loss = nn.BCEWithLogitsLoss()
        is_labeled = true == true  # Filter our nans.

        
        return bce_loss(pred[is_labeled], true[is_labeled].float()), pred

"""
@register_loss('binary_cross_entropy')
def binary_cross_entropy(pred, true):
    
    if cfg.dataset.task_type == 'classification_binary':
        if cfg.model.loss_fun != 'cross_entropy':
            raise ValueError("Only 'cross_entropy' loss_fun supported with "
                             "'classification_binary' task_type.")
        bce_loss = nn.BCEWithLogitsLoss()
        is_labeled = true == true  # Filter out NaNs if necessary

        print("in loss pred")
        print(pred)
        print("in loss true")
        print(true)
        return bce_loss(pred[is_labeled], true[is_labeled].float()), pred
"""