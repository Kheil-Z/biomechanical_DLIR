"""
Wraps MONAI's LocalNormalizedCrossCorrelationLoss to provide a consistent interface for 
both loss computation and metric reporting.

Key Features:
    - `loss(inputs, ddf)`: Computes the LNCC loss between the warped moving image and the fixed image.
    - `metric(inputs, ddf)`: Returns the scalar LNCC value as an interpretable similarity metric.
    - Supports both 2D and 3D images.
    - Automatically manages device placement and warping operations via MONAI.

Notes:
    - Inputs should include `"fixed"` and `"moving"` image tensors.
    - The deformation field `ddf` is assumed to be in the format (B, 2, H, W) or (B, 3, D, H, W).
"""

from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.networks.blocks import Warp

class NCC:
    def __init__(self,params,**kwargs):
        self.loss_func = LocalNormalizedCrossCorrelationLoss(**kwargs)
        self.warp_layer = Warp()
        self.device = None

    def set_device(self,curr_dev):
        if self.device is None:
            self.device = curr_dev
            self.warp_layer.to(self.device)

    def metric(self, inputs,ddf):
        self.set_device(ddf.device)

        y_true = inputs["fixed"].to(self.device)
        y_moving = inputs["moving"].to(self.device)
        y_pred = self.warp_layer(y_moving, ddf)
        value = self.loss_func(y_pred,y_true).item()
        return {"NCC_metric":value}

    def loss(self, inputs,ddf):
        self.set_device(ddf.device)

        y_true = inputs["fixed"].to(self.device)
        y_moving = inputs["moving"].to(self.device)
        y_pred = self.warp_layer(y_moving, ddf)
        return self.loss_func(y_pred,y_true)
    