"""

    Created on 03/05/21 6:35 PM 
    @author: Kartik Prabhu

"""

import torch
import torch.nn as nn
import torch.utils.data

class Dice(nn.Module):
    """
    Class used to get dice_loss and dice_score
    """

    def __init__(self, smooth=1):
        super(Dice, self).__init__()
        self.smooth = smooth
        self.dicescore = 0

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        self.dicescore = dice_score
        return dice_loss

    def get_dice_score(self):
        return self.dicescore


class IOU(nn.Module):
    def __init__(self, smooth=1):
        super(IOU, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f) - intersection
        score = (intersection + self.smooth) / (union + self.smooth)
        return score


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        pt_1 = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return pow((1 - pt_1), self.gamma)

class BCE(nn.Module):
    """
    Class used to get bce_loss
    """

    def __init__(self, smooth=1):
        super(BCE, self).__init__()

    def forward(self, y_pred, y_true):
        bce_loss = torch.nn.BCELoss()
        return bce_loss(y_pred,y_true)