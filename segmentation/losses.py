"""Weighted deep-supervision Dice and cross-entropy loss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import parameter, validate_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


class DeepSupervisionLoss(nn.Module):

    def __init__(self, weights=None, smooth=None):
        super().__init__()
        if smooth is None:
            smooth = parameter("segmentation.loss_smoothing")
        self.base_loss = DiceCELoss(
            softmax=True,
            to_onehot_y=True,
            squared_pred=True,
            smooth_nr=smooth,
            smooth_dr=smooth,
        )
        weights = list(
            parameter("segmentation.loss_weights") if weights is None else weights
        )
        weight_sum = sum(weights)
        self.weights = [w / weight_sum for w in weights]

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target.float())
        total_loss = 0.0
        for i, pred in enumerate(outputs):
            if i >= len(self.weights):
                break
            if pred.shape[2:] != target.shape[2:]:
                t = F.interpolate(target.float(), size=pred.shape[2:], mode="nearest")
            else:
                t = target.float()
            total_loss += self.weights[i] * self.base_loss(pred, t)
        return total_loss
