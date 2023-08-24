
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


def segmentation_metrics(logits, targets, activation='0-1', eps=1e-7, reduction='mean'):
    # convert targets to one hot encoding
    if len(targets.shape) == len(logits.shape) - 1:
        if logits.shape[1] == 1: # binary classification
            y_true = targets[:, None].to(logits.dtype)
        else:                    # mulit-class classification
            y_true = F.one_hot(targets, num_classes=logits.shape[1])
            y_true = y_true.to(logits.dtype).to(logits.device)
            y_true = y_true[:, None].transpose(1, -1)[..., 0]
    else:
        y_true = targets
    # logits to probability
    if activation == 'softmax':
        y_pred = torch.softmax(logits, dim=1)
    elif activation == 'sigmoid':
        y_pred = torch.sigmoid(logits)
    elif activation == '0-1':
        if logits.shape[1] == 1: # binary classification
            y_pred = (logits > 0).to(logits.dtype).to(logits.device)
        else:                    # mulit-class classification
            y_pred = torch.argmax(logits, axis=1)
            y_pred = F.one_hot(y_pred, num_classes=logits.shape[1])
            y_pred = y_pred.to(logits.dtype).to(logits.device)
            y_pred = y_pred[:, None].transpose(1, -1)[..., 0]

    axis = list(range(2, len(logits.shape))) # height and width
    # compute true postive, false positive and false negative
    # use mean reduction instead of sum to ensure the numerical stability under float16
    tp = torch.sum(y_true * y_pred, dim=axis)
    fp = torch.sum(y_pred, dim=axis) - tp
    fn = torch.sum(y_true, dim=axis) - tp

    pixel_acc = (torch.sum(tp, dim=1) + eps) / (torch.sum(tp, dim=1) + torch.sum(fp, dim=1) + eps)
    iou = (tp + eps) / (fp + fn + tp + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    
    if reduction == 'mean':
        pixel_acc, iou, dice, precision, recall = pixel_acc.mean(), iou.mean(), dice.mean(), precision.mean(), recall.mean()
    return pixel_acc, iou, dice, precision, recall

''' Compute proper groups '''
def smart_group_norm(num_features, max_groups=32):
    groups = 1
    while groups < max_groups and num_features % (2 * groups) == 0:
        groups *= 2
    return nn.GroupNorm(groups, num_features)

''' Replace BatchNorm2d with InstanceNorm2d or GroupNorm '''
def replace_batchnorm2d(model, norm_layer):
    for key, module in model._modules.items():
        if isinstance(module, nn.BatchNorm2d):
            norm = norm_layer(module.num_features)
            norm.weight.data.copy_(module.weight.data)
            norm.bias.data.copy_(module.bias.data)
            model._modules[key] = norm
        else:
            replace_batchnorm2d(module, norm_layer)

# from https://huggingface.co/spaces/Roll20/pet_score/blob/b258ef28152ab0d5b377d9142a23346f863c1526/lib/timm/utils/agc.py
def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)
def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)

# from https://github.com/kekmodel/MPL-pytorch/blob/main/models.py
class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def forward(self, input):
        return self.module(input)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)




