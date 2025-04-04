"""
Loss module for registration

Uses MONAI's DiceLoss
Implements Dice similarity loss for segmentation alignment after spatial deformation. 
Uses MONAI's `Warp` layer to apply displacement fields to one-hot encoded moving labels.

Key Features:
    - `loss(inputs, ddf)`: differentiable Dice loss for training
    - `metric(inputs, ddf)`: non-differentiable Dice score for monitoring
    - Per-label Dice computed within each anatomical label group
    - Supports multi-label supervision via `params["additional_labels"]`

Notes:
    - Loss uses bilinear warping for smooth gradients. Also, need to one-hot and use "bilinear" interpolation,
    since "nearest" interpoaltion is differentiable but is zero almost everywhere.
    - Metric uses nearest-neighbor warping to preserve label integrity, and no need to backpropagate,
"""

import torch
import numpy as np
from monai.losses import DiceLoss
from monai.networks.blocks import Warp


def onehot(seg, labels):
    onehot = [(seg==0)]
    for label in labels:
        onehot.append((seg==label))
    onehot = torch.cat(onehot, dim=1).type(seg.dtype)
    return onehot

class Dice:
    def __init__(self,params, **kwargs):
        self.loss_func = DiceLoss(include_background=False, to_onehot_y=False, softmax=False, reduction="mean")

        self.warp_layer = Warp(mode="bilinear")
        self.warp_layer_nearest = Warp(mode="nearest")
        self.device = None
        self.label_name = params["additional_labels"].keys()
        self.label_values = [params["additional_labels"][name] for name in self.label_name]
    
    def set_device(self,curr_dev):
        if self.device is None:
            self.device = curr_dev
            self.warp_layer.to(self.device)
            self.warp_layer_nearest.to(self.device)

    def metric(self, inputs, ddf):
        """
            Compute evaluation metrics for model monitoring.

            This function calculates interpretable, often label-wise, performance metrics 
            (e.g., Dice score, foldings, etc.) used for validation and testing.
            
            Returns interpretable metrics for evaluation and logging.

            These are not used for optimization.
        """
        self.set_device(ddf.device)
        metrics = {}
        for name, label_values in zip(self.label_name,self.label_values):
            y_true = inputs[f"fixed_{name}"].to(self.device)
            y_moving = inputs[f"moving_{name}"].to(self.device)
            # Warp :
            y_moved = self.warp_layer_nearest(y_moving, ddf)
            curr_dice_values = []
            for label in label_values:
                curr_y_true = (y_true==label)
                curr_y_moved = (y_moved==label)
                curr_dice = DiceLoss(include_background=False)(curr_y_true,curr_y_moved).item()
                metrics[f"dice_{name}_{label}"] = curr_dice
                curr_dice_values.append(curr_dice)
            metrics[f"avg_dice_{name}"] = np.mean(curr_dice_values)
        return metrics
    
    def loss(self, inputs,ddf):
        """
        Returns the loss used for backpropagation during training.

        This is a differentiable value minimized by the optimizer.
        """
        self.set_device(ddf.device)
        loss = 0.
        num_labels = 0
        for name, label_values in zip(self.label_name,self.label_values):
            y_true = inputs[f"fixed_{name}"].to(self.device)
            y_moving = inputs[f"moving_{name}"].to(self.device)

            # Get labels in given images.. -> useful when working on ROi datasets where not all labels appear, for memory efficiency:
            curr_labels = [a for a in label_values if (a in y_true or a in y_moving)]
            f_onehot = onehot(y_true, curr_labels)#[:,1,:,:,:]
            m_onehot = onehot(y_moving, curr_labels)#[:,1,:,:,:]

            moved_onehot = self.warp_layer(m_onehot, ddf)
            curr_loss= self.loss_func(moved_onehot,f_onehot)
            
            loss += curr_loss
            num_labels+=1
        return loss/num_labels