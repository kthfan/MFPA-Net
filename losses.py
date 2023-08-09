'''pytorch implementation of https://github.com/mlyg/unified-focal-loss/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# channels first format, get height, width axis
def identify_axis(shape):
    return list(range(2, len(shape)))

# convert to one hot encoding
def convert_prob(y_pred, y_true, activation):
    # convert to [batch, classes, ...]
    if len(y_true.shape) == len(y_pred.shape) - 1:
        if y_pred.shape[1] == 1: # binary classification
            y_true = y_true[:, None].to(y_pred.dtype)
        else:                    # mulit-class classification
            y_true = F.one_hot(y_true, num_classes=y_pred.shape[1])
            y_true = y_true.to(y_pred.dtype).to(y_pred.device)
            y_true = y_true[:, None].transpose(1, -1)[..., 0]
    if activation == 'softmax':
        y_pred = torch.softmax(y_pred, dim=1)
    elif activation == 'sigmoid':
        y_pred = torch.sigmoid(y_pred)
    return y_pred, y_true

def apply_reduction(tensor, reduction):
    if reduction == 'mean':
        tensor = torch.mean(tensor)
    elif reduction == 'sum':
        tensor = torch.sum(tensor)
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Invalid reduction: {reduction}.')
    return tensor

################################
#       Dice coefficient       #
################################
def dice_coefficient(logits, targets, smooth = 0.000001, activation='sigmoid', reduction='mean'):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.
    Parameters
    ----------
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    y_pred, y_true = convert_prob(logits, targets, activation)
    axis = identify_axis(y_true.shape)

    intersection = (y_pred * y_true).sum(dim=axis)
    dice_class = (2.*intersection + smooth) / (y_pred.sum(dim=axis) + y_true.sum(dim=axis) + smooth)

    if reduction == 'none':
        return dice_class
    else: # Average class scores
        return apply_reduction(dice_class, reduction)

################################
#       Tversky index       #
################################
def tversky_index(logits, targets, delta = 0.7, smooth = 0.000001, activation='sigmoid', reduction='mean'):
    """Generalization of the Sørensen–Dice coefficient and the Jaccard index.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    y_pred, y_true = convert_prob(logits, targets, activation)
    axis = identify_axis(y_true.shape)
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = torch.sum(y_true * y_pred, dim=axis)
    fn = torch.sum(y_true * (1-y_pred), dim=axis)
    fp = torch.sum((1-y_true) * y_pred, dim=axis)
    tversky_class = (tp + smooth) / (tp + delta*fn + (1-delta)*fp + smooth)

    if reduction == 'none':
        return tversky_class
    else: # Average class scores
        return apply_reduction(tversky_class, reduction)

################################
#         Cross Entropy        #
################################
def cross_entropy(logits, targets, class_weight=None, sample_weight=None, activation='sigmoid', reduction='mean'):
    loss = None
    _, y_true = convert_prob(logits, targets, None)
    if isinstance(class_weight, (float, int)):
        if activation == 'sigmoid':
            class_weight = torch.FloatTensor([1 - class_weight, class_weight]).to(logits.dtype).to(logits.device)
            class_weight = 2 * class_weight / class_weight.sum()
        elif activation == 'softmax':
            class_weight = None

    if sample_weight == None:
        if activation == 'sigmoid':
            loss = F.binary_cross_entropy_with_logits(logits, y_true, class_weight, reduction=reduction)
        elif activation == 'softmax':
            loss = F.cross_entropy(logits, targets, class_weight, reduction=reduction)
        else:
            raise ValueError(f'Invalid activation {activation}.')
    else:
        if activation == 'sigmoid':
            if class_weight is None:
                class_weight = [1, 1]
            if isinstance(sample_weight, (list, tuple)):
                negative_weight, positive_weight = sample_weight
            else:
                negative_weight, positive_weight = sample_weight, sample_weight

            loss = - class_weight[1] * y_true * positive_weight * F.logsigmoid(logits) + \
                    - class_weight[0] * (1 - y_true) * negative_weight * F.logsigmoid(-logits)

        elif activation == 'softmax':
            if class_weight is None:
                class_weight = 1.

            loss = - class_weight * y_true * sample_weight * F.log_softmax(logits, dim=1)
            loss = torch.sum(loss, dim=1)
        else:
            raise ValueError(f'Invalid activation {activation}.')

    return apply_reduction(loss, reduction)

################################
#           Dice loss          #
################################
class DiceLoss(nn.Module):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def __init__(self, smooth = 0.000001, activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.smooth = smooth
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return dice_loss(logits, targets, smooth=self.smooth,
                         activation=self.activation, reduction=self.reduction)

def dice_loss(logits, targets, smooth = 0.000001, activation='sigmoid', reduction='mean'):
    dice_class = dice_coefficient(logits, targets, smooth=smooth,
                                  activation=activation, reduction='none')
    dice_class = 1 - dice_class
    return apply_reduction(dice_class, reduction)


################################
#         Tversky loss         #
################################
class TverskyLoss(nn.Module):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
	Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def __init__(self, delta = 0.7, smooth = 0.000001, activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.delta = delta
        self.smooth = smooth
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return tversky_loss(logits, targets, delta=self.delta, smooth=self.smooth,
                            activation=self.activation, reduction=self.reduction)

def tversky_loss(logits, targets, delta = 0.7, smooth = 0.000001, activation='sigmoid', reduction='mean'):
    tversky_class = tversky_index(logits, targets, delta=delta, smooth=smooth,
                                  activation=activation, reduction='none')
    tversky_class = 1 - tversky_class
    return apply_reduction(tversky_class, reduction)


################################
#          Combo loss          #
################################
class ComboLoss(nn.Module):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
    eps :
        Small fractional to ensure numerical stability. by default 1e-7
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=0.000001, activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return combo_loss(logits, targets, alpha=self.alpha, beta=self.beta, smooth=self.smooth,
                            activation=self.activation, reduction=self.reduction)

def combo_loss(logits, targets, alpha=0.5, beta=0.5, smooth=0.000001, activation='sigmoid', reduction='mean'):
    y_pred, y_true = convert_prob(logits, targets, activation)
    dice = dice_coefficient(y_pred, y_true, smooth=smooth, activation=None, reduction=reduction)
    ce = cross_entropy(logits, targets, class_weight=beta, activation=activation, reduction=reduction)
    combo_loss = (alpha * ce) - ((1 - alpha) * dice)
    return combo_loss


################################
#      Focal Tversky loss      #
################################
class FocalTverskyLoss(nn.Module):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def __init__(self, delta=0.7, gamma=0.75, smooth=0.000001, activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return focal_tversky_loss(logits, targets, delta=self.delta, gamma=self.gamma, smooth=self.smooth,
                                  activation=self.activation, reduction=self.reduction)

def focal_tversky_loss(logits, targets, delta=0.7, gamma=0.75, smooth=0.000001, activation='sigmoid', reduction='mean'):
    tversky_class = tversky_index(logits, targets, delta=delta, smooth=smooth,
                                  activation=activation, reduction='none')
    focal_tversky = torch.pow((1-tversky_class), gamma)
    return apply_reduction(focal_tversky, reduction)


################################
#          Focal loss          #
################################
class FocalLoss(nn.Module):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. alpha > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """
    def __init__(self, alpha=None, gamma_f=2., activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.alpha = alpha
        self.gamma_f = gamma_f
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return focal_loss(logits, targets, alpha=self.alpha, gamma_f=self.gamma_f,
                          activation=self.activation, reduction=self.reduction)

def focal_loss(logits, targets, alpha=None, gamma_f=2., activation='sigmoid', reduction='mean'):
    y_pred, _ = convert_prob(logits, targets, activation)
    if activation == 'sigmoid':
        focal = cross_entropy(logits, targets, class_weight=alpha,
                                sample_weight=[torch.pow(y_pred, gamma_f), torch.pow(1 - y_pred, gamma_f)],
                                activation=activation, reduction=reduction)
    elif activation == 'softmax':
        focal = cross_entropy(logits, targets, class_weight=alpha,
                                sample_weight=torch.pow(1 - y_pred, gamma_f),
                                activation=activation, reduction=reduction)
    else:
        raise ValueError(f'Invalid activation {activation}.')
    return focal


################################
#       Symmetric Focal loss      #
################################
class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def __init__(self, delta=0.7, gamma=2., activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.delta = delta
        self.gamma = gamma
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return symmetric_focal_loss(logits, targets, delta=self.delta, gamma=self.gamma,
                                    activation=self.activation, reduction=self.reduction)

def symmetric_focal_loss(logits, targets, delta=0.7, gamma=2., activation='sigmoid', reduction='mean'):
    return focal_loss(logits, targets, alpha=delta, gamma_f=gamma,
                        activation=activation, reduction=reduction)


#################################
# Symmetric Focal Tversky loss  #
#################################
class SymmetricFocalTverskyLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def __init__(self, delta=0.7, gamma=0.75, smooth=0.000001, activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return symmetric_focal_tversky_loss(logits, targets, delta=self.delta, gamma=self.gamma, smooth=self.smooth,
                                            activation=self.activation, reduction=self.reduction)

def symmetric_focal_tversky_loss(logits, targets, delta=0.7, gamma=0.75, smooth=0.000001, activation='sigmoid', reduction='mean'):
    if activation == 'sigmoid':
        _, y_true = convert_prob(logits, targets, None)
        tversky_back = tversky_index(-logits, 1-y_true, delta=delta, smooth=smooth,
                                        activation=activation, reduction='none')
        tversky_fore = tversky_index(logits, y_true, delta=delta, smooth=smooth,
                                        activation=activation, reduction='none')
        focal_tversky_back = (1 - tversky_back) * torch.pow(1 - tversky_back, -gamma)
        focal_tversky_fore = (1 - tversky_fore) * torch.pow(1 - tversky_fore, -gamma)
        focal_tversky = 0.5 * (focal_tversky_back + focal_tversky_fore)
    elif activation == 'softmax':
        tversky = tversky_index(logits, targets, delta=delta, smooth=smooth,
                                        activation=activation, reduction='none')
        focal_tversky = (1 - tversky) * torch.pow(1 - tversky, -gamma)
    else:
        raise ValueError(f'Invalid activation {activation}.')

    return apply_reduction(focal_tversky, reduction)


################################
#     Asymmetric Focal loss    #
################################
class AsymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets, only suppressing background class.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def __init__(self, delta=0.7, gamma=2, background_axis=None, activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.delta = delta
        self.gamma = gamma
        self.background_axis = background_axis
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return asymmetric_focal_loss(logits, targets, delta=self.delta, gamma=self.gamma, background_axis=self.background_axis,
                                     activation=self.activation, reduction=self.reduction)

def asymmetric_focal_loss(logits, targets, delta=0.7, gamma=2., background_axis=None, activation='sigmoid', reduction='mean'):
    y_pred, _ = convert_prob(logits, targets, activation)
    if activation == 'sigmoid':
        n_gamma = torch.full([1, logits.shape[1]] + (len(logits.shape) - 2) * [1], gamma, dtype=logits.dtype, device=logits.device)
        p_gamma = torch.zeros([1, logits.shape[1]] + (len(logits.shape) - 2) * [1], dtype=logits.dtype, device=logits.device)
        if background_axis is not None:
            n_gamma[:, background_axis] = 0
            p_gamma[:, background_axis] = gamma
        focal = cross_entropy(logits, targets, class_weight=delta,
                                sample_weight=[torch.pow(y_pred, n_gamma), torch.pow(1 - y_pred, p_gamma)],
                                activation=activation, reduction=reduction)
    elif activation == 'softmax':
        gamma = torch.zeros([1, logits.shape[1]] + (len(logits.shape) - 2) * [1], dtype=logits.dtype, device=logits.device)
        if background_axis is not None:
            gamma[:, background_axis] = gamma
        focal = cross_entropy(logits, targets, class_weight=delta,
                                sample_weight=torch.pow(1 - y_pred, gamma),
                                activation=activation, reduction=reduction)
    else:
        raise ValueError(f'Invalid activation {activation}.')
    return focal


#################################
# Asymmetric Focal Tversky loss #
#################################
class AsymmetricFocalTverskyLoss(nn.Module):
    """
    Only enhancing foreground class.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def __init__(self, delta=0.7, gamma=0.75, smooth=0.000001, background_axis=None, activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.background_axis = background_axis
        self.activation = activation
        self.reduction = reduction

    def forward(self, logits, targets):
        return asymmetric_focal_tversky_loss(logits, targets, delta=self.delta, gamma=self.gamma, smooth=self.smooth,
                                             background_axis=self.background_axis, activation=self.activation, reduction=self.reduction)

def asymmetric_focal_tversky_loss(logits, targets, delta=0.7, gamma=0.75, smooth=0.000001,
                                    background_axis=None, activation='sigmoid', reduction='mean'):
    if activation == 'sigmoid':
        n_gamma = torch.zeros([1, logits.shape[1]], dtype=logits.dtype, device=logits.device)
        p_gamma = torch.full([1, logits.shape[1]], gamma, dtype=logits.dtype, device=logits.device)
        if background_axis is not None:
            n_gamma[:, background_axis] = gamma
            p_gamma[:, background_axis] = 0

        _, y_true = convert_prob(logits, targets, None)
        tversky_back = tversky_index(-logits, 1-y_true, delta=delta, smooth=smooth,
                                        activation=activation, reduction='none')
        tversky_fore = tversky_index(logits, y_true, delta=delta, smooth=smooth,
                                        activation=activation, reduction='none')
        focal_tversky_back = (1 - tversky_back) * torch.pow(1 - tversky_back, -n_gamma)
        focal_tversky_fore = (1 - tversky_fore) * torch.pow(1 - tversky_fore, -p_gamma)
        focal_tversky = 0.5 * (focal_tversky_back + focal_tversky_fore)

    elif activation == 'softmax':
        gamma = torch.full([1, logits.shape[1]], gamma, dtype=logits.dtype, device=logits.device)
        if background_axis is not None:
            gamma[:, background_axis] = 0
        tversky = tversky_index(logits, targets, delta=delta, smooth=smooth, 
                                        activation=activation, reduction='none')
        focal_tversky = (1 - tversky) * torch.pow(1 - tversky, -gamma)
    else:
        raise ValueError(f'Invalid activation {activation}.')
    
    return apply_reduction(focal_tversky, reduction)


###########################################
#      Symmetric Unified Focal loss       #
###########################################
class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, smooth=0.000001, activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.activation = activation
        self.reduction = reduction
    
    def forward(self, logits, targets):
        return symmetric_unified_focal_loss(logits, targets, weight=self.weight, delta=self.delta, gamma=self.gamma, 
                                            smooth=self.smooth, activation=self.activation, reduction=self.reduction)

def symmetric_unified_focal_loss(logits, targets, weight=0.5, delta=0.6, gamma=0.5, smooth=0.000001, 
                            activation='sigmoid', reduction='mean'):
    symmetric_ftl = symmetric_focal_tversky_loss(logits, targets, delta=delta, gamma=gamma, smooth=smooth,
                                                 activation=activation, reduction=reduction)
    symmetric_fl = symmetric_focal_loss(logits, targets, delta=delta, gamma=gamma,
                                        activation=activation, reduction=reduction)
    loss = (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)
    return loss

###########################################
#      Asymmetric Unified Focal loss      #
###########################################
class AsymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, smooth=0.000001, background_axis=None, 
                 activation='sigmoid', reduction='mean'):
        super().__init__()
        assert activation in ['sigmoid', 'softmax']
        assert reduction in ['none', 'mean', 'sum']
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.background_axis = background_axis
        self.activation = activation
        self.reduction = reduction
    
    def forward(self, logits, targets):
        return asymmetric_unified_focal_loss(logits, targets, weight=self.weight, delta=self.delta, gamma=self.gamma, smooth=self.smooth, 
                                            background_axis=self.background_axis, activation=self.activation, reduction=self.reduction)

def asymmetric_unified_focal_loss(logits, targets, weight=0.5, delta=0.6, gamma=0.5, smooth=0.000001, 
                                    background_axis=None, activation='sigmoid', reduction='mean'):
    asymmetric_ftl = asymmetric_focal_tversky_loss(logits, targets, delta=delta, gamma=gamma, smooth=smooth,
                                                    background_axis=background_axis, activation=activation, reduction=reduction)
    asymmetric_fl = asymmetric_focal_loss(logits, targets, delta=delta, gamma=gamma,
                                            background_axis=background_axis, activation=activation, reduction=reduction)
    loss = (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)
    return loss
