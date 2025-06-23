"""
Loss module for registration

Implements Mean Squared Error (MSE) loss between fixed and warped moving images.
Captures voxel-wise intensity differences and penalizes large mismatches.

Key Features:
    - `loss(inputs, ddf)`: differentiable MSE loss for training
    - `metric(inputs, ddf)`: reports MSE value for evaluation
    - Warps moving image using MONAI's `Warp` layer

Notes:
    - Works best for mono-modal registration tasks.
    - Only image intensities ("fixed", "moving") are used — no labels required.
"""

import torch.nn.functional as F
from monai.networks.blocks import Warp


class MSE:
    def __init__(self, params, **kwargs):
        self.params = params
        self.warp_layer = Warp()
        self.device = None

    def set_device(self, curr_dev):
        if self.device is None:
            self.device = curr_dev
            self.warp_layer.to(self.device)

    def metric(self, inputs, ddf):
        self.set_device(ddf.device)

        y_true = inputs["fixed"].to(self.device)
        y_moving = inputs["moving"].to(self.device)
        y_pred = self.warp_layer(y_moving, ddf)
        value = F.mse_loss(y_true, y_pred, reduction="mean").item()
        return {"MSE_metric": value}

    def loss(self, inputs, ddf):
        self.set_device(ddf.device)
        y_true = inputs["fixed"].to(self.device)
        y_moving = inputs["moving"].to(self.device)
        y_pred = self.warp_layer(y_moving, ddf)
        return F.mse_loss(y_true, y_pred, reduction="mean")
