#######################################참고#############################################
## https://github.com/oikosohn/compound-loss-pytorch/blob/main/jaccard_ce_loss_smp.py ##
########################################################################################
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.base as base
import torch.nn as nn

class JaccardCELoss(base.Loss):
    def __init__(self, alpha=1.0, beta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

        self.jaccardloss=smp.losses.JaccardLoss(mode='multiclass')
        self.jaccardloss.__name__ = 'jaccard_loss'

        self.celoss = nn.CrossEntropyLoss()
        self.celoss.__name__ = 'ce_loss'

    def forward(self, y_pred, y_true):
        return self.alpha * self.celoss.forward(y_pred, y_true) - self.beta * self.jaccardloss.forward(y_pred, y_true)



class DiceFocalLoss(base.Loss):
    def __init__(self, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

        self.diceloss=smp.losses.DiceLoss(mode='multiclass')
        self.diceloss.__name__ = 'dice_loss'

        self.focalloss = smp.losses.FocalLoss(mode='multiclass')
        self.focalloss.__name__ = 'focal_loss'

    def forward(self, y_pred, y_true):
        return self.alpha * self.diceloss.forward(y_pred, y_true) + self.beta * self.focalloss.forward(y_pred, y_true)


"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the CDB-loss/CIFAR-LT directory.
=======
"""

import numpy as np
import torch
import torch.nn as nn


def sigmoid(x):
  return (1/(1+np.exp(-x)))


class CDB_loss(nn.Module):
  
    def __init__(self, class_difficulty, tau='dynamic', reduction='none'):
        
        super(CDB_loss, self).__init__()
        self.class_difficulty = class_difficulty
        if tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/(1 - np.max(class_difficulty) + 0.01)
            tau = sigmoid(bias)
        else:
            tau = float(tau) 
        self.weights = self.class_difficulty ** tau
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weights), reduction=self.reduction).cuda()
        

    def forward(self, input, target):

        return self.loss(input, target)