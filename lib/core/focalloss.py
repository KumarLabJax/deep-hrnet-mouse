import torch
import torch.nn as nn
import torch.nn.functional as nnfunc

# An implementation of focal loss (Focal Loss for Dense Object Detection; Lin et al)
# Note that in their paper the authors state that initialization of the final layer
# is critical. They also find that using a feature pyramid network (FPN) was important
# to the success of thier approach

def ce_focal_loss(input, target, weight=None, gamma=2.0, reduction='elementwise_mean'):

    """ a simple multinomial implementation of focal loss """

    cross_entropy_loss = nnfunc.cross_entropy(input, target, weight=weight, reduction='none')

    probs = nnfunc.softmax(input, 1)
    true_probs = torch.squeeze(
        torch.gather(probs, 1, torch.unsqueeze(target, 1)),
        1)

    loss = (1 - true_probs) ** gamma * cross_entropy_loss

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'elementwise_mean':
        return loss.mean()
    else:
        raise ValueError(
                'bad reduction value. Valid values are:'
                ' "none", "sum" or "elementwise_mean"')


class CEFocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=2.0, reduction='elementwise_mean'):
        self.register_buffer('weight', weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        return ce_focal_loss(
            input, target,
            weight=self.weight, gamma=self.gamma, reduction=self.reduction)


def bce_focal_loss(input, target, pos_weight=None, gamma=2.0, reduction='elementwise_mean'):

    bce_loss = nnfunc.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='none')
    probs = torch.sigmoid(input)

    abs_prob_diff = torch.abs(target - probs)
    loss = abs_prob_diff ** gamma * bce_loss

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'elementwise_mean':
        return loss.mean()
    else:
        raise ValueError(
                'bad reduction value. Valid values are:'
                ' "none", "sum" or "elementwise_mean"')
